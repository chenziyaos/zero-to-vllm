# 调参记录模板

每次改参数都单独记一条，写清楚改动前后的负载、指标和结论。一次只改一个变量；要同时改两个，就走 `vllm bench sweep serve` 跑笛卡尔积。

## 0. 环境（每次换环境时重写）

| 项 | 值 |
|---|---|
| vLLM 版本 / commit | |
| 卡型 × 数量 / TP / PP / DP | |
| 物理核数（是否开超线程） | 下限 A + DP + N + (DP>1)：___ |
| 模型 / dtype / KV dtype | |
| max_model_len | |
| 负载画像 | 输入长度 p50/p99：___；输出长度 p50/p99：___；前缀共享率：___；到达方式：泊松 / 突发 / 固定并发 |
| SLO | TTFT p99 ≤ ___ ms；TPOT p99 ≤ ___ ms；（或 e2el ≤ ___ ms） |

## 1. 引擎默认值快照（不加任何调参参数起一次）

从启动日志抄，不要凭记忆：

| 日志行 | 值 |
|---|---|
| `Defaulting max_num_batched_tokens to`（需 `VLLM_LOGGING_LEVEL=DEBUG`） | |
| max_num_seqs（只有显式传了才出现在 INFO 级 `non-default args`；没传就按 sim_tuning.py --scenario defaults 推，或开 DEBUG 看 config dump） | |
| `Capturing CUDA graphs ... N/N` | |
| `Available KV cache memory: X GiB` | |
| `GPU KV cache size: N tokens, Maximum concurrency ...: Kx` | |
| CUDA graph util 提示行（建议的 util） | |

## 2. 单条记录

```text
#编号  日期  改动人
改动：            （例：--max-num-batched-tokens 8192 → 2048）
假设：            （例：长 prompt 让 ITL p99 超标，缩小预算把 prefill 切细）
依据：            （论文 / 文档 / 源码行号，例：Sarathi-Serve §4；docs/configuration/optimization.md:64-66）
负载：            （bench 命令全文；数据集、request_rate 或 max_concurrency、num_prompts、seed）
重复次数：        （≥3；vllm bench sweep 默认 --num-runs 3）

指标              改前        改后        变化
输出 tok/s
TTFT p50 / p99
TPOT p50 / p99
ITL p99
goodput（--goodput ttft:X tpot:Y）
KV usage 峰值（/metrics vllm:kv_cache_usage_perc）
抢占次数（vllm:num_preemptions_total）
503 次数（vllm:admission_rejections_total）

结论：            保留 / 回滚 / 需要更多数据
副作用：          （例：TTFT p99 +12%）
```

## 3. 推荐顺序（每一步都要回到第 2 节记一条）

1. **显存**：`--gpu-memory-utilization` 从默认 0.92 起，看 KV token 数是否够 `max_num_seqs × 平均序列长度`；不够先考虑 FP8 KV、TP 或缩短 `max_model_len`，最后才调 util。
2. **预算**：`--max-num-batched-tokens` 在 2048 / 4096 / 8192 / 16384 之间用固定 request_rate 比 goodput；ITL 敏感往小走，离线吞吐往大走。
3. **并发**：`--max-num-seqs` 不超过 KV 能装下的条数太多；想压 decode batch 又不想缩小图，用 `--max-num-active-seqs`。
4. **捕获网格**：`max_num_seqs` 超过 256 时检查 `Capturing CUDA graphs` 上限（非 Blackwell 是 512）；需要时显式 `-cc.max_cudagraph_capture_size`。
5. **准入**：过载要快速失败时开 `--max-num-queued-tokens ≈ 目标 TTFT × 实测 prefill 吞吐 × 0.6–0.8`。
6. **启动**：编译缓存 `VLLM_CACHE_ROOT` 打进镜像或挂盘；启动敏感时比较 `-O0 / -O1 / -O2`。
