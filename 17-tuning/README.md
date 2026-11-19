# VLLM-17 调参手册：预算、显存比例、编译档位与负载画像

运行环境：模拟器只需要 CPU 和 Python 3 标准库，复用 `../common/specs.py` 和 `../06-chunked-prefill/sim_chunked.py`。`sweep.sh` 需要一张 24 GB 以上的卡（默认 Qwen3-8B），装好 vLLM；`vllm bench sweep` 子命令要求较新的 vLLM。SGLang 对照另装 sglang。源码基线：vllm 主干 0d6f92a，sglang 主干 8fabf5e。

| 文件 | 内容 |
|---|---|
| `sim_tuning.py` | 八个场景。defaults：逐行移植 `get_batch_defaults`、后处理四步和 `_set_cudagraph_sizes`，与 SGLang `memory_hook` 的分档并排对照。mem：沿用 ch02 的显存账本，算 util 0.80–0.97 对应的 KV 块数，演示 `kv_cache_memory_bytes` 跳过 profiling，以及 SGLang 的 reserved / mem_fraction_static。cudagraph：两家捕获网格的个数、平均 padding、超出上限走 eager 的比例。budget：在 06 章的混合负载上扫 `max_num_batched_tokens`，goodput 按 DistServe 口径算。seqs：突发 2000 条请求时扫 `max_num_seqs`。queue：开 `max_num_queued_tokens` 准入（超限返回 503）后的 TTFT。opt：O0–O3 对照。cpu：官方的 CPU 核数公式 |
| `test_sim_tuning.py` | 19 个单测：各卡默认档位；A100 卡名判断；throughput 模式只翻倍未设的项；四步截断；默认网格 51 个；上限 = 2 × max_num_seqs，Blackwell 为 1024；预算低于上限时补进网格；interactivity 76 个；padding 与 eager 比例；SGLang 分档；H100 mem_fraction 0.830；10 GiB 下限；网格 36 / 58；ch02 的 24957 块 / 399312 token；kv_bytes 绕过 util；单调性；goodput 边界；过载时准入能降低 TTFT p99；CPU 公式得 17 |
| `sweep.sh` | 真机实验：抓默认值日志；util 逐档起服；`vllm bench sweep serve` 跑网格；`serve_workload` 画延迟-吞吐曲线；固定 rate 用 goodput 扫预算；`sweep startup` 比较 O0 / O2 / eager；准入过载压测；画图；SGLang 默认值与网格对照 |
| `serve_params.json` / `bench_params.json` | sweep 网格：6 组引擎参数（预算 × seqs，外加一组 long_prefill_token_threshold 1024）乘 3 种负载（chat 1500/300、rag 8000/200、gen 300/1500） |
| `tuning_matrix.md` | 默认值、捕获网格、显存、准入相关的源码行号，以及与 SGLang 的参数对照 |
| `tuning_log_template.md` | 调参记录模板：环境、默认值快照、单条记录格式、推荐顺序 |

## 用法

```bash
python3 sim_tuning.py --scenario defaults                  # 各卡默认值对照
python3 sim_tuning.py --scenario mem                       # util 对应的 KV token 数
python3 sim_tuning.py --scenario cudagraph                 # 捕获网格与 padding
python3 sim_tuning.py --scenario budget --rate 10          # 预算扫描（约 10 秒）
python3 sim_tuning.py --scenario seqs                      # max_num_seqs 扫描
python3 sim_tuning.py --scenario queue                     # 准入阀门
python3 sim_tuning.py --scenario opt                       # 编译档位
python3 sim_tuning.py --scenario cpu --tp 2 --dp 4         # CPU 核数
python3 sim_tuning.py --scenario all

python3 -m unittest test_sim_tuning -v                     # 19 个单测

bash sweep.sh defaults          # 先看引擎替你选了什么
bash sweep.sh util              # 显存
bash sweep.sh goodput 2000 30   # 预算
bash sweep.sh grid              # 预算 × seqs × 负载
bash sweep.sh queue 36000       # 准入
bash sweep.sh startup           # 编译档位与启动时间
bash sweep.sh plot
bash sweep.sh sgl_defaults && bash sweep.sh sgl_grid
```

每改一个参数，按 `tuning_log_template.md` 记一条。

## 模拟器的假设

- **defaults**：每张卡的显存记两份，`torch.cuda.mem_get_info` 的 GiB 给 vLLM 用，nvidia-smi 的 MiB 给 SGLang 用。只覆盖 CUDA，不含 ROCm / TPU / CPU 分支。
- **mem**：沿用 ch02 的 H100 + Qwen3-8B 账本：权重 15.27 GiB，非 torch 0.62 GiB，峰值激活 1.43 GiB，CUDA graph 估计 0.71 GiB，block_size 16。假设这几项不随 util 变化。
- **budget / seqs / queue**：复用 06 章的步进模拟器（roofline 步长模型）。负载为泊松到达，输入 1500±80%，输出 300±60%。SGLang 列沿用 06 章的默认调度：有新 prefill 就先跑 prefill batch，否则跑 decode batch，两者交替；不开 `--enable-mixed-chunk`，也不建模 `schedule_conservativeness`。
- **queue**：prefill 吞吐取预算 8192、64 条 decode 占位时的模拟值。排队 token 按整条 prompt 记账，与 scheduler.py 的注释一致。
- **cudagraph**：假设 decode batch 大小在 1..max_running 之间均匀分布，据此算 padding。真实负载的 batch 偏向高并发段，padding 会更小。

数字只用来判断方向。与真机结果不一致时，以真机日志为准。

## 期望输出

`sim_tuning.py --scenario defaults`（节选）：

```
卡               vLLM API     vLLM LLM 类  API+throughput   vLLM 图上限    图个数 | SGL chunked SGL dec max_bs  dec 图  pre 图 mem_frac
a100-80g    2048/256       8192/256        4096/512            512     51 |        8192            256     36     58    0.830
h100-sxm    8192/1024     16384/1024      16384/2048           512     51 |        8192            256     36     58    0.830
b200       16384/1024     16384/1024      32768/2048          1024     83 |       16384            512     52     74    0.854
```

`sim_tuning.py --scenario budget --rate 10`（节选）：

```
    预算 模式        输出 tok/s  TTFT p50  TTFT p99  ITL p50  ITL p99    达标率  goodput
   512 vllm         2,735  605.5 ms    2.27 s  20.1 ms  26.3 ms  97.2%     9.05
  2048 vllm         2,744  182.7 ms    1.13 s   9.0 ms  80.2 ms  91.8%     8.57
  8192 vllm         2,745  118.8 ms    1.08 s   8.8 ms 265.8 ms  89.0%     8.31
  8192 sglang       2,742  110.5 ms    1.08 s   9.5 ms 387.0 ms  77.9%     7.27
```

`sim_tuning.py --scenario queue`（节选）：

```
 rate        上限     接收   503  TTFT p50  TTFT p99
   16         —    936     0    3.29 s   11.06 s
   16     61192    888    48    1.71 s    2.93 s
   16     36715    905    31  818.0 ms    1.62 s
   20         —   1150     0   12.85 s   27.22 s
   20     36715   1038   112    1.15 s    2.07 s
```
