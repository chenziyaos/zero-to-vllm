# VLLM-18 可观测与 profiling：指标口径、torch profiler 与 nsys

运行环境：模拟器、抓取脚本和单测只需要 CPU 和 Python 3 标准库。模拟器复用 `../06-chunked-prefill/sim_chunked.py` 的步进模型。`profile.sh` 需要一张 24 GB 以上的卡（默认 Qwen3-8B）和装好的 vLLM。nsys 段需要 Nsight Systems CLI，proton 段需要 Triton 3.7 以上。SGLang 对照另装 sglang。源码基线：vllm 主干 0d6f92a，sglang 主干 8fabf5e。

| 文件 | 内容 |
|---|---|
| `sim_metrics.py` | 八个场景。intervals：一条请求的事件时间线怎样拆成 TTFT / queue / prefill / decode / inference / TPOT，以及抢占落在哪一段。itl：同一批 token 时间戳下，vLLM 按事件加权的 ITL、SGLang 按 token 加权的 ITL、按请求加权的 TPOT，逐 token 解码和投机解码各算一遍。buckets：histogram_quantile 在两家默认桶上的插值误差，以及自定义桶修正后的误差。hitrate：日志的 1000 条请求窗口与 PromQL rate[1m] / rate[5m] 在高低流量下的滞后。kvusage：kv_cache_usage_perc 不把空闲的前缀缓存块算作占用。throughput：日志 prompt 吞吐（computed）与 counter（total）的差距。profiler：delay / max_iterations / wait / warmup / active 分别落到第几个 engine step。cardinality：直方图的时间序列数 |
| `scrape_metrics.py` | 隔一段时间抓两次 `/metrics`（或读两个文本文件），按 `sum by (le)` 合并各 engine 的桶，再求窗口内的 p50 / p99。另外输出窗口命中率、total 与 computed 两种 prompt tok/s、KV usage。p99 落进 +Inf 桶时会提示。加 `--prefix sglang:` 可读 SGLang 的同名延迟直方图 |
| `test_sim_metrics.py` | 21 个单测：区间拆分（tokenize 计入 TTFT；prefill 中抢占进 TTFT；decode 中抢占进 TPOT；单 token 时 TPOT 为 0）；逐 token 时两家 ITL 相同，投机解码时总时长不变、样本数不同，TPOT 不变；六组默认桶的档数；1-2-5 桶；PromQL 插值与 +Inf 桶返回最后一个上界；1000 条窗口、reset、空 stats 不入队、低流量滞后；三种 profiler 窗口；每 engine 308 条直方图序列；Prometheus 文本解析、跨 engine 合并与窗口 delta |
| `profile.sh` | 真机命令清单：metrics、buckets（自定义桶）、torch-serve / torch-window / torch-bench、nsys-serve / nsys-stats、proton-serve，以及 SGLang 的 sgl-torch / sgl-nsys |
| `metrics_matrix.md` | 时间戳、区间公式、缓存与 token 计数、桶族、profiler 的源码行号，以及与 SGLang 的逐项对照 |

## 用法

```bash
python3 sim_metrics.py --scenario intervals              # 区间拆分与抢占
python3 sim_metrics.py --scenario itl --rate 10          # 三种 ITL / TPOT 口径
python3 sim_metrics.py --scenario buckets --rate 10      # 默认桶的分位误差
python3 sim_metrics.py --scenario hitrate                # 命中率窗口滞后
python3 sim_metrics.py --scenario kvusage
python3 sim_metrics.py --scenario throughput
python3 sim_metrics.py --scenario profiler --delay 2 --max-iters 4 --warmup 1 --active 3
python3 sim_metrics.py --scenario cardinality --dp 8
python3 sim_metrics.py --scenario all

python3 -m unittest test_sim_metrics -v                  # 21 个单测

python3 scrape_metrics.py --url http://127.0.0.1:8000/metrics --interval 30
python3 scrape_metrics.py --files before.txt after.txt --interval 30

bash profile.sh metrics          # 先看指标
bash profile.sh torch-serve      # 起服时带 --profiler-config
SECS=60 bash profile.sh torch-window
bash profile.sh nsys-serve       # 另一终端：bash profile.sh torch-bench
bash profile.sh nsys-stats report1.nsys-rep
```

## 模拟器的假设

- **intervals**：只有一条请求。tokenize 10 ms，排队 400 ms，prefill 300 ms，decode 每 token 20 ms，共 200 个输出 token；抢占暂停 1.5 s。arrival 和 engine 时间戳放在同一条时间轴上，不模拟两个时钟域之间的偏差。
- **itl / buckets**：负载是 06 章的混合负载，rate 10，跑 120 s，预算 8192。投机解码这样模拟：每步先出 1 个 token，再以 0.7 的概率逐个接受草稿 token，最多接受 3 个。这里只是把已有时间戳重新分组，总时长不变，所以不能用来估计投机解码真实的加速比。
- **hitrate**：泊松到达，每条 prompt 2000 token，每条请求的命中比例取 N(p, 0.1)，再按 16 token 一块向下取整。t = 600 s 时 p 从 0.8 突变为 0.2。PromQL 列直接用窗口内 hits / queries 的比值，没有建模 Prometheus 抓取间隔带来的外推。
- **kvusage**：沿用 ch02 的 24957 块。四个时刻的块分布是手工设定的，只用来演示公式。
- **profiler**：逐行移植 `WorkerProfiler.step()` 和 torch schedule 的状态机，不调用真实的 torch.profiler。

数字只用来判断方向。与真机结果不一致时，以真机日志为准。

## 期望输出

`sim_metrics.py --scenario intervals`：

```
场景                  TTFT     queue   prefill    decode  inference       e2e      TPOT   ITL max
无抢占           710.0 ms  400.0 ms  300.0 ms    3.98 s     4.28 s    4.69 s   20.0 ms   20.0 ms
prefill 中抢占     2.21 s  400.0 ms    1.80 s    3.98 s     5.78 s    6.19 s   20.0 ms   20.0 ms
decode 中抢占    710.0 ms  400.0 ms  300.0 ms    5.48 s     5.78 s    6.19 s   27.5 ms    1.52 s
```

`sim_metrics.py --scenario itl`（节选）：

```
投机 k=3    vLLM ITL（事件加权）              133,497   48.1 ms   25.5 ms  555.3 ms
投机 k=3    SGLang ITL（token 加权）        336,773   19.1 ms    9.3 ms  245.6 ms
投机 k=3    TPOT（请求加权）                    1,150   19.6 ms   18.2 ms   40.3 ms
```

`sim_metrics.py --scenario buckets`（节选）：

```
TTFT    p99     1.08 s       1.64 s (+51%)       1.43 s (+31%)        1.10 s (+1%)
ITL     p50     8.8 ms       7.6 ms (-13%)        9.1 ms (+3%)        8.8 ms (+0%)
```

`sim_metrics.py --scenario hitrate`（节选）：

```
     2/s      120     64.5%     19.7%     54.9%
    20/s       30     42.7%     49.0%     73.5%
```

`sim_metrics.py --scenario profiler`（节选）：

```
  delay=3 max_iters=5                      step 1..14: ..RRRRRS......   落盘的 step：[3, 4, 5, 6, 7]
  delay=0 max_iters=0 wait=1 warmup=2 active=5 step 1..14: wwRRRRRwwwwwww   落盘的 step：[3, 4, 5, 6, 7]
```
