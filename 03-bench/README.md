# VLLM-03 指标与压测

对应文档：VLLM-03｜上手｜指标与压测：vllm bench serve、TTFT、TPOT 与 goodput

| 文件 | 运行环境 | 做什么 |
|---|---|---|
| `bench.sh` | GPU + vllm | `vllm bench serve` 的固定参数组：smoke、开环扫 `--request-rate`、闭环扫 `--max-concurrency`、ShareGPT、突发度、`vllm bench sweep serve` 扫服务端参数。每个点前后各抓一次 `/metrics` |
| `sim_goodput.py` | 仅 Python 3 标准库 | 两部分：`metrics()` 逐行对照 `vllm/benchmarks/serve.py` 的 `calculate_metrics`；`simulate()` 是 H100 + Qwen3-8B 的粗粒度连续批处理模拟器，用来演示指标的形状 |

源码基线：vllm 主干 0d6f92a；SGLang 对照 sgl-project/sglang 主干 8fabf5e。

## 1. 不用 GPU：看指标的形状

```bash
python3 sim_goodput.py                         # 开环 / 闭环 / 突发度三组扫描，约 5 秒
python3 sim_goodput.py --goodput ttft:500 tpot:30
python3 sim_goodput.py --demo-itl              # 一条 32K prompt 插进来，TPOT 和 ITL 差多少
python3 sim_goodput.py --steady 0              # 全程吞吐 vs SGLang 式稳态窗口吞吐（0 = 不设并发闸门）
python3 sim_goodput.py --isl 1024 --osl 512    # 换负载
```

默认输出（SLO ttft:1000 tpot:40，节选）：

```
[开环] --request-rate 扫描（--max-concurrency 不设）
             req/s out tok/s  p50 TTFT  p99 TTFT    mTPOT  p99TPOT  p99 ITL  goodput
         1    1.00       128        32        32      7.6      8.3      7.7     1.00
         8    7.94     1,017        32        94     10.0     12.0     32.6     7.94
        16   15.75     2,016        33       126     15.7     20.5     64.4    15.75
        20   19.59     2,507        35       188     22.1     30.4     96.1    19.59
        24   23.27     2,978        67       223     38.8     60.2    161.1    14.33
        32   23.02     2,945      3170     10378     98.5    128.2    247.9     0.09  抢占 157
       inf   22.35     2,859     19657     41886    105.1    144.9    247.9     0.00  抢占 217

[闭环] --request-rate inf + --max-concurrency 扫描
       c=1    1.03       132        32        32      7.4      7.4      7.4     1.03
      c=16   10.39     1,330       217       248     10.5     11.6     63.3    10.39
      c=64   19.11     2,447       219      1735     23.9     25.4    247.9    18.48
     c=128   22.34     2,859       248      3718     40.8     43.5    247.9     2.50
     c=256   24.54     3,141       248      7932     71.3     79.3    247.9     2.13
     c=512   22.35     2,859      6693     19657    105.1    144.9    247.9     0.00

[突发度] 同一 --request-rate 16，只改 --burstiness
    b=0.25   15.66     2,004        96       496     20.0     39.7    189.5    15.58   99.5%
       b=1   15.75     2,016        33       126     15.7     20.5     64.4    15.75  100.0%
     b=inf   15.74     2,015        33        33     14.5     14.7     32.9    15.74  100.0%
```

读法：

- 开环里 req/s 跟着 rate 走，直到 24 左右碰到天花板；再往上 TTFT 爆炸、KV 不够开始抢占，吞吐反而掉了。
- 闭环里 TTFT 不会爆炸，因为排队发生在客户端的 Semaphore 里，**不计入 TTFT**（脚本另打了“客户端排队均值”）。
- goodput 在 c=64 到 c=128 之间断崖：TPOT 均值越过 40 ms 那一刻，几乎所有请求同时出局。
- 突发度只改到达分布，不改平均速率：吞吐几乎不变，尾延迟变了。

`--demo-itl`：

```
第 20 个请求是 32K 的长 prompt；被它拖得最狠的是请求 17：
  TPOT = (latency − TTFT)/(n − 1) = 18.5 ms
  最大 ITL = 247.9 ms，中位 ITL = 7.8 ms
  全体 p99 TPOT = 20.5 ms，p99 ITL = 32.8 ms
```

`--steady 0`：

```
c=0：全程 44.7 s，峰值并发 1000，阈值 900（0.9 × 峰值），稳态窗口 [0.0, 17.7] s
  全程 out tok/s = 2,859；窗口内 out tok/s = 2,390；窗口内完整请求 104 条
```

模拟器只为说明指标的形状。它的 step 时间是 `max(读字节/带宽, 算 FLOP/算力) + 1.2 ms`，没有 CUDA Graph 档位、没有 attention 的二次项、没有采样开销，绝对数字以真机为准。

## 2. 真机压测

```bash
# 终端 1
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-8B --gpu-memory-utilization 0.92
# 终端 2
bash bench.sh smoke
bash bench.sh conc
python3 sim_goodput.py --from-json results/conc/c64.json --goodput ttft:1000 tpot:40
```

`VLLM_SERVER_DEV_MODE=1` 只用来打开 `/reset_prefix_cache`：random 数据集在固定 `--seed` 下每轮生成同一批 prompt，不清缓存第二轮会因为前缀命中而虚高。生产环境不要开。

`--from-json` 用 `--save-detailed` 的逐请求字段重算一遍，结果应该和 `vllm bench serve` 打印的数逐位一致（百分位都是 numpy 默认的线性插值）。不一致时先查是否有失败请求：失败请求不进任何分布，但 SGLang 的 `slo_attainment` 分母会算上它。
