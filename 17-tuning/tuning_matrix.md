# VLLM-17 调参：源码行号与 SGLang 对照

基线：vllm 主干 0d6f92a，sglang 主干 8fabf5e。

## vLLM：默认值从哪来

| 环节 | 位置 | 要点 |
|---|---|---|
| 按卡选默认预算 | `engine/arg_utils.py:2946` `get_batch_defaults` | ≥160 GiB：16384/1024（两种 usage 相同）；≥70 GiB 且卡名不含 a100、非 GB10：LLM 类 16384/1024，API server 8192/1024；其余：8192/256 与 2048/256 |
| A100 例外 | `engine/arg_utils.py:2976` NOTE(Kuntai) | A100 上大预算反而降吞吐，PR #17885 |
| 后处理四步 | `engine/arg_utils.py:3157` | throughput 模式把未设的两项翻倍；关 chunked 时 tokens ≥ max_model_len；tokens ≤ seqs × max_model_len；seqs ≤ tokens |
| 兜底值 | `config/scheduler.py` | `DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048`，`DEFAULT_MAX_NUM_SEQS = 128` |
| performance_mode | `config/vllm.py:162, 456` | balanced / interactivity / throughput |
| 捕获网格 | `config/vllm.py:2416-2620` `_set_cudagraph_sizes` | 上限 min(max_num_seqs × q_len × 2, 512；cc 10.x 为 1024)，再与 max_num_batched_tokens 取小；1,2,4 + 8 步长到 256 + 16 步长；interactivity 时 1..32 逐个；超出上限走 eager |
| 编译档位 | `config/vllm.py:147-359` | O0：无编译无图；O1：Inductor + PIECEWISE；O2（默认）：FULL_AND_PIECEWISE + allreduce_rms / rope_kvcache 等融合；O3 = O2 |
| 显存比例 | `config/cache.py:215` | `gpu_memory_utilization = 0.92` |
| block_size | `config/cache.py:185, 456-466` | 默认 16 |
| KV 直给 | `config/cache.py:374` `kv_cache_memory_bytes` | 设了就跳过 profiling，util 不参与 |
| 可用 KV | `v1/worker/gpu_worker.py:648-776` | requested − non_kv − cudagraph 估计；`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 默认 1 |
| 准入阀门 | `config/scheduler.py:69, 92, 105` | `max_num_active_seqs`（只限 RUNNING，不缩图）；`max_num_queued_reqs` / `max_num_queued_tokens`（API server 503，跨 DP 计数） |
| 503 计数 | `v1/engine/async_llm.py:371`；`v1/metrics/loggers.py:47, 848` | `vllm:admission_rejections{reason=...}` |
| 自适应长 prefill 阈值 | `config/scheduler.py:86` | 阈值下限 = 预算 / (排队 + 运行中请求数) |
| CPU 核数 | `docs/configuration/optimization.md:463-507` | A + DP + N + (DP>1)；超线程 ×2 |
| 预算取舍 | `docs/configuration/optimization.md:64-66` | 2048 时 ITL 更好；追吞吐设 >8192 |
| 扫参工具 | `docs/benchmarking/sweeps.md`；`benchmarks/auto_tune/auto_tune.sh` | sweep serve / serve_workload / startup / plot；auto_tune 先降 util 找能起的最大值，再网格扫，用 `--goodput e2el:` |

## SGLang：默认值从哪来

| 环节 | 位置 | 要点 |
|---|---|---|
| 按显存分档 | `srt/arg_groups/memory_hook.py:90-171` | <20G 2048/8；<35G 4096/48(tp≥4:160)；<60G 4096/32(160)；<90G、<160G 8192/256(512)；更大 16384/512；格式 chunked_prefill_size / decode max_bs |
| decode max_bs 截断 | 同上 181 附近 | 不超过 max_running_requests |
| prefill 图上限 | 同上 | = chunked_prefill_size；MLA 封顶 8192；llama-2 封顶 4096 |
| 静态显存比例 | 同上 281-325 | reserved = 512 + max(chunked, 2048) × 1.5 + tp × pp / 8 × 1024 + 图预留；>60G 至少 10240 MiB；mem_frac = (gpu − reserved) / gpu |
| 图预留 | 同上 357 `reserve_for_graph_mb` | decode max_bs × 2 + prefill 网格个数 × 8（MLA 改为 1.5 GiB） |
| 捕获网格 | `srt/arg_groups/cuda_graph_hook.py:480-532` | decode：1,2,4,8,12 + 8 步长到 256 + 16 步长到 512 + 32 步长；投机时更细 |
| 调度参数 | `srt/arg_groups/fields/schedule.py` | `max_prefill_tokens=16384`；`schedule_policy=fcfs`；`schedule_conservativeness=1.0` |
| 官方调参指南 | `docs/.../hyperparameter_tuning.mdx` | #queue-req 100–2000、token usage > 0.9；频繁 retract 调 1.3；排队多但 usage 低调 0.3；prefill OOM 降 chunked，decode OOM 降 max_running |

## 参数对照

| 意图 | vLLM | SGLang |
|---|---|---|
| 每步 token 上限 | `--max-num-batched-tokens` | `--chunked-prefill-size`（prefill 段）+ `--max-prefill-tokens` |
| 同时运行请求上限 | `--max-num-seqs`（也决定图上限） | `--max-running-requests` |
| 只限运行不缩图 | `--max-num-active-seqs` | 无直接对应 |
| KV 显存 | `--gpu-memory-utilization`（整卡比例，含权重） | `--mem-fraction-static`（权重 + KV 的静态比例） |
| 直接指定 KV 大小 | `--kv-cache-memory-bytes` | `--max-total-tokens` |
| 图上限 | `-cc.max_cudagraph_capture_size` / `-cc.cudagraph_capture_sizes` | `--cuda-graph-max-bs-decode` / `--cuda-graph-max-bs-prefill` |
| 关图 | `--enforce-eager` / `-O0` | `--disable-decode-cuda-graph` / `--disable-prefill-cuda-graph`（旧 `--disable-cuda-graph` 仍兼容） |
| 编译档位 | `-O0` … `-O3` | `--enable-torch-compile` |
| 排队上限 | `--max-num-queued-reqs` / `--max-num-queued-tokens` | `--max-queued-requests` |
| 长 prompt 切分 | `--long-prefill-token-threshold` | chunked_prefill 每步最多一条在切 |
| 准入保守度 | 无（V1 按块即时分配，不够就抢占） | `--schedule-conservativeness` |
| 偏向延迟/吞吐的一键档 | `--performance-mode interactivity/throughput` | 无，靠分档默认值 |
