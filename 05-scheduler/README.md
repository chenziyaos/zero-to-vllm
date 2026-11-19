# VLLM-05 调度器

运行环境：CPU，只依赖 Python 3 标准库。基线 vllm 主干 0d6f92a，sglang 8fabf5e。

| 文件 | 内容 |
|---|---|
| `sim_scheduler.py` | vLLM V1 `Scheduler.schedule()` 的逐步复刻 + SGLang `PrefillAdder` / `retract_decode` 简化对照 |
| `trace_schedule.md` | 一次 `schedule()` 调用的逐行对照表（行号对应主干） |

## 用法

```bash
python3 sim_scheduler.py                                       # unified：R0、R1 在 decode，R2 是一条 5,000 token 的 prompt
python3 sim_scheduler.py --threshold 512                       # 打开 long_prefill_token_threshold
python3 sim_scheduler.py --scenario preempt                    # KV 只有 64 块，FCFS 抢占
python3 sim_scheduler.py --scenario preempt --policy priority  # PRIORITY 选受害者
python3 sim_scheduler.py --scenario preempt --async            # 异步调度：输出晚一步回来
python3 sim_scheduler.py --scenario preempt --engine sglang    # 同负载走 SGLang 预留 + retract
python3 sim_scheduler.py --scenario h100 --compare             # 24,957 块、预算 8192、600 条同时到达
python3 sim_scheduler.py --scenario h100 --watermark 0.05
python3 sim_scheduler.py --scenario h100 --budget 2048
python3 sim_scheduler.py --scenario h100 --engine sglang --conservativeness 0.3
```

时间模型：读权重 16.4 GB + 读 KV（144 KiB/token）按 3.35 TB/s，算力按 989 TFLOPS × 60%，取大者再加 1.2 ms。只用来比较相对快慢，绝对值别拿去报数。

## 期望输出

unified（节选）：

```
step  本步调度（请求:token 数）                             run wait   KV%  事件
   1  R0:300 R1:200                                  2    0   0.8
   2  R0:1 R1:1 R2:2046                              3    0   3.9
   3  R0:1 R1:1 R2:2046                              3    0   7.0
   4  R0:1 R1:1 R2:908                               3    0   8.4
   5  R0:1 R1:1 R2:1                                 3    0   8.4
[vLLM] 步数 40，抢占/retract 0 次（单条最多 0 次），重算 0 token，前向总 token 5,597（有效 5,600）
```

preempt，FCFS（节选）：

```
  30  R0:1 R1:1 R2:1 R3:1 R4:1 R5:1 R6:1             7    1  89.1  抢占 R7(丢 121 tok 的 KV)
  47  R0:1 R1:1 R2:1 R3:1 R4:1 R5:1                  6    2  87.5  抢占 R6(丢 139 tok 的 KV)
  66  R0:1 R1:1 R2:1 R3:1 R4:1                       5    3  85.9  抢占 R5(丢 159 tok 的 KV)
  98  R0:1 R1:1 R2:1 R3:1                            4    4  81.2  抢占 R4(丢 192 tok 的 KV)
 158  R0:1 R1:1 R2:1                                 3    5  76.6  抢占 R3(丢 253 tok 的 KV)
 201  R1:1 R2:1 R3:254                               2    4  54.7  恢复 R3；完成 R1
 202  R2:1 R3:1 R4:193 R5:160                        3    2  60.9  恢复 R4；恢复 R5；完成 R2
 203  R3:1 R4:1 R5:1 R6:140 R7:122                   5    0  89.1  恢复 R6；恢复 R7
 226  R3:1 R4:1 R5:1 R6:1                            4    1  85.9  抢占 R7(丢 144 tok 的 KV)
[vLLM] 步数 406，抢占/retract 7 次（单条最多 3 次），重算 1,209 token，前向总 token 3,601（有效 2,400）
         模型耗时 2.5 s，输出 640 tok/s，TTFT p50 31 ms / p99 43 ms
```

同负载的其他变体：

```
--policy priority  步数 404，抢占 6 次，重算 1,013 token，644 tok/s
--async            步数 410，抢占 7 次，重算 1,208 token，634 tok/s
--engine sglang    步数 603，retract 1 次，重算 256 token，433 tok/s，TTFT p50 1253 ms
```

h100 --compare：

```
[vLLM ] 步数 3982，抢占/retract 182 次（单条最多 4 次），重算 346,095 token，前向总 token 2,053,448（有效 1,707,953）
         模型耗时 116.0 s，输出 4,388 tok/s，TTFT p50 46.9 s / p99 97.2 s
[SGLang] 步数 4744，抢占/retract 0 次（单条最多 0 次），重算 0 token，前向总 token 1,707,353（有效 1,707,953）
         模型耗时 116.0 s，输出 4,389 tok/s，TTFT p50 45.3 s / p99 97.6 s
```

| 配置 | 抢占/retract | 模型耗时 | 输出 tok/s |
|---|---|---|---|
| vLLM 默认 | 182 | 116.0 s | 4,388 |
| vLLM `--watermark 0.05` | 7 | 109.4 s | 4,654 |
| vLLM `--budget 2048` | 152 | 107.7 s | 4,726 |
| SGLang 默认 | 0 | 116.0 s | 4,389 |
| SGLang `--budget 2048` | 0 | 117.5 s | 4,332 |
| SGLang `--conservativeness 0.3` | 0 | 114.7 s | 4,440 |

## 读法

- vLLM 是乐观准入：新请求只要整条 prompt 放得下就收（`full_sequence_must_fit`），不给后续 decode 留余量，所以 h100 场景 12–96 s 之间 KV 一直在 99–100%，靠抢占兜底。SGLang 收新请求前先按 `min(max_new − 已生成, 4096) × new_token_ratio` 给 running 留位，KV 稳在 93% 上下，几乎不 retract。
- 一步里只要抢过人，这一步就不收新请求（scheduler.py:1050）；被抢的请求排在 waiting 队首，它释放的块随即被其余请求的 decode 陆续用掉，整条序列始终放不下（`full_sequence_must_fit`），所以 preempt 日志里 30–200 步 wait 只增不减。
- 恢复的请求要重新 prefill「prompt + 已生成的输出」：R3 在 201 步一次排了 254 = 100 + 154 个 token。这部分就是“重算 token”。
- 线上对应指标：`vllm:num_preemptions`（计数器）和日志里的 `Preemptions: N`；SGLang 是日志 `KV cache pool is full. Retract requests.`。
