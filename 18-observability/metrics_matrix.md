# VLLM-18 可观测：源码行号与 SGLang 对照

基线：vllm 主干 0d6f92a，sglang 主干 8fabf5e。路径均相对 `vllm/` 与 `python/sglang/`。

## vLLM：一条请求的时间戳

| 时间戳 | 谁打的 | 时钟 | 位置 |
|---|---|---|---|
| `arrival_time` | 前端（AsyncLLM / InputProcessor），tokenize 之前 | wall-clock | `v1/metrics/stats.py` RequestStateStats |
| `queued_ts` | engine core，进入 waiting 队列 | monotonic | `v1/engine/__init__.py` EngineCoreEventType.QUEUED |
| `scheduled_ts` | engine core，**第一次**被调度 | monotonic | `stats.py:577`（只在 0 时赋值） |
| `first_token_ts` / `last_token_ts` | engine core 输出 | monotonic | `stats.py:515-560` |

| 指标 | 公式 | 位置 | 坑 |
|---|---|---|---|
| TTFT | first_token − arrival | `stats.py:515-530` | 含 tokenize 与前端排队；跨时钟域，用前端时间相减 |
| ITL | 本次输出事件 − 上次输出事件 | `stats.py:548-560` | **每个输出事件一个样本**，投机解码一次吐多个 token 也只记一次 |
| queue | scheduled − queued | `stats.py:594-613` | 只记第一次排队，抢占后的再排队进 prefill 或 decode |
| prefill | first_token − scheduled | 同上 | prefill 期间被抢占的暂停也算在内 |
| decode | last_token − first_token | 同上 | decode 期间被抢占的暂停也算在内 |
| inference | last_token − scheduled | 同上 | = prefill + decode |
| TPOT | decode / (n − 1)；n ≤ 1 记 0 | `stats.py:613` | Prometheus 记 0，`vllm bench serve` 直接剔除（`benchmarks/serve.py:628-661`） |

## vLLM：缓存与 token 计数

| 指标 | 口径 | 位置 |
|---|---|---|
| 日志 "Prefix cache hit rate" | 最近 1000 条请求的 hits / queries（token 加权）；空 stats 不入队；`reset_prefix_cache` 清空 | `stats.py:39-118` CachingMetrics；`loggers.py:322` |
| `vllm:prefix_cache_queries/hits` | counter，token 数 | `loggers.py:662, 673` |
| `vllm:external_prefix_cache_*` | KV connector 侧的命中 | `loggers.py:686, 698` |
| 日志 "Avg prompt throughput" | 只累加 computed（真正跑过 prefill 的 token） | `loggers.py:163-164` |
| `vllm:prompt_tokens` | 累加 total = computed + local_cache_hit + external | `loggers.py:748, 1262`；`stats.py:501` |
| `vllm:prompt_tokens_by_source{source}` | local_compute / local_cache_hit / external_kv_transfer | `loggers.py:758`；`stats.py:442-467` |
| `vllm:prompt_tokens_cached_by_source{source}` | device / host / disk / p2p / external_unspecified | `loggers.py:782` |
| `vllm:iteration_tokens_total` | 每步 computed + generation | `loggers.py:888, 1272` |
| `vllm:kv_cache_usage_perc` | 1 − free / (num_gpu_blocks − 1)；ref_cnt = 0 的缓存块算 free | `v1/core/block_pool.py:921` |
| `vllm:kv_block_*` | 块生命周期 / 驱逐前空闲 / 复用间隔，`--kv-cache-metrics` 打开，按 `kv_cache_metrics_sample`（默认 0.01）采样 | `loggers.py:1042-1068`；`config/observability.py:62-67` |

## vLLM：直方图桶族

| 族（`--custom-histogram-buckets` 的键） | 默认桶 | 用到它的直方图 |
|---|---|---|
| `request_tokens` | `build_1_2_5_buckets(max_model_len)`，40960 时 14 档 | prompt / generation / max_num_generation / params_max_tokens / prefill_kv_computed |
| `iteration_tokens` | 1, 8, 16 … 16384（13） | iteration_tokens_total |
| `request_params_n` | 1, 2, 5, 10, 20 | request_params_n |
| `time_to_first_token` | 0.001 … 2560（22） | time_to_first_token_seconds |
| `inter_token_latency` | 0.01 … 80（19） | inter_token_latency_seconds、request_time_per_output_token_seconds |
| `request_latency` | 0.3 … 7680（21） | e2e / queue / inference / prefill / decode |
| `request_num_preemptions` | 1, 2, 3, 4, 5, 10, 20 | request_num_preemptions |
| `kv_cache_residency` | 0.001 … 1800 s | `kv_block_*` 三个直方图，只在 `--kv-cache-metrics` 时注册（`loggers.py:1037-1075`） |

位置：`v1/metrics/buckets.py`；映射 `loggers.py:862-1030`；参数 `config/observability.py:70, 167-180`。每个直方图每组 label 的序列数 = 桶数 + 3（+Inf、_count、_sum）。多进程 API server 下用 `prometheus_client.multiprocess`（`v1/metrics/prometheus.py:17`）。

## vLLM：profiler

| 环节 | 位置 | 要点 |
|---|---|---|
| 配置 | `config/profiler.py:150-240` | `profiler`: torch / cuda / proton；`torch_profiler_dir` 必填；`delay_iterations` / `max_iterations` 默认 0；`warmup_iterations` 0、`active_iterations` 5、`wait_iterations` 0 |
| 迭代控制 | `profiler/wrapper.py:130-160` `step()` | 计数先 +1；delay 到了才真正 start；记满 max 后在下一个计数 step 前 stop |
| torch schedule | `profiler/wrapper.py:279-321` | 只有 wait > 0 或 warmup > 0 才建 schedule；丢弃 wait + warmup − 1 步（start 占掉第 0 步） |
| 落表 | `profiler/wrapper.py:273-277, 415-435` `_stop` | `rank<N>.profiler_out.txt`：GPU 上按 self_device_time_total 排序（profiler 默认行数）；只记 CPU 时才按 self_cpu_time_total 出 50 行 |
| 调用点 | `v1/worker/gpu_worker.py:1222, 1362` | 每次 execute_model 调一次 step |
| HTTP | `docs/contributing/profiling.md:83-130` | `/start_profile` body：profile_prefix、delay_iterations、max_iterations、profiler_kwargs；未 stop 再 start 返回 409 |
| nsys | 同上 175-260 | `--trace-fork-before-exec=true --cuda-graph-trace=node`；服务端 `--capture-range=cudaProfilerApi --capture-range-end repeat` + `--profiler-config.profiler cuda`；建议 `VLLM_WORKER_MULTIPROC_METHOD=spawn` |
| proton | 同上 132-175 | CUDA graph 开着时要 `proton_graph_attribution: true`，需 Triton ≥ 3.7 |

## SGLang 对照

| 话题 | vLLM | SGLang | 位置（SGLang） |
|---|---|---|---|
| 开关 | 默认开 `/metrics` | `--enable-metrics` | `srt/arg_groups/fields/observability.py:70` |
| 前缀 | `vllm:` | `sglang:` | `production_metrics.mdx` |
| TTFT 桶 | 22 档，0.001–2560 | 18 档，0.1–400；带 `is_streaming` label | `srt/observability/metrics_collector.py:1965`、`2154-2159` |
| ITL 样本 | 每个输出事件一个 | 间隔 / 新 token 数，按 token 数重复 observe（token 加权） | `metrics_collector.py:2161` |
| TPOT | decode/(n−1)，n ≤ 1 记 0 | 剔除 completion < 2、首包即结束的流式请求、abort、PD prefill | `production_metrics.mdx:127` |
| 改桶 | `--custom-histogram-buckets` JSON，8 个族 | `--bucket-time-to-first-token` / `--bucket-inter-token-latency` / `--bucket-e2e-request-latency` | `fields/observability.py:104-112` |
| 命中率 | 日志：最近 1000 条请求；Prometheus：hits/queries counter | `sglang:cache_hit_rate` gauge：最近一个 prefill batch；另有 15 s 时间窗（`SGLANG_CACHE_HIT_RATE_WINDOW_SECONDS`）；`sglang:cached_tokens_total{cache_source}` | `srt/managers/scheduler_components/metrics_reporter.py:79-97, 826-865`；`srt/environ.py:659` |
| prompt 计数 | counter 累加 total；by_source 拆分 | `sglang:prompt_tokens_total` 请求结束时累加；cached 单独计 | `metrics_collector.py:2100-2136` |
| KV 使用率 | `kv_cache_usage_perc`，不含 ref_cnt=0 缓存块 | `sglang:token_usage`，多个池取 max；另有 `sglang:kv_evictable_tokens` gauge 单独报可驱逐缓存 | `srt/managers/scheduler_components/pool_stats_observer.py` |
| profiler 入口 | `/start_profile` + `--profiler-config` | `/start_profile`，`SGLANG_TORCH_PROFILER_DIR` | `benchmark_and_profiling.mdx:85-270` |
| 跳过步数 | `delay_iterations`（第 delay 步开始记） | `start_step`（含该步） | 同上 199-204 |
| 记几步 | `max_iterations`，记满仍要 `/stop_profile` 才写表 | `num_steps`，记满自动 stop | 同上 198、517 |
| nsys 区间 | `profiler=cuda` → cudaProfilerStart/Stop | `activities: ["CUDA_PROFILER"]` | 同上 478-523 |
| 分层 NVTX | `--enable-layerwise-nvtx-tracing`（`config/observability.py:86`，不支持 CUDA graph） | `--enable-layerwise-nvtx-marker`（需关 CUDA graph） | 同上 488 |
