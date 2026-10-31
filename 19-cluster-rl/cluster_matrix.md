# VLLM-19 集群与 RL：源码行号与 SGLang 对照

基线：vllm 主干 0d6f92a，sglang 主干 8fabf5e。路径相对仓库根。

## 多副本路由

| 项 | vLLM | SGLang |
|---|---|---|
| 入口 | `vllm serve --data-parallel-size N`，API server 进程里内置 LB，一个端口（`docs/serving/data_parallel_deployment.md:23-29`） | 独立的 `sgl-router`（Rust），挂在若干 `sglang.launch_server` 前面（`sgl-router/README.md:60-76`） |
| 默认策略 | 打分扫描：`score = max(client_count × inflight, waiting + running)`，有排队时加 `waiting × 6 × max(0, kv − 0.5)`（`vllm/v1/engine/core_client.py:1789-1815`） | 不加 `--chat-routing` 是 round_robin；`--chat-routing reorg` 下是 power_of_two（`sgl-router/src/config/cli.rs:243-256`） |
| 看不看前缀 | 不看。文档原话 "could be made more sophisticated in future by incorporating KV cache aware logic"（`data_parallel_deployment.md:77`） | `--policy cache_aware`：外部 indexer 或本地 radix 树给出各 worker 的命中前缀，过门槛（默认 1024 token，可选比例）后按命中降序、负载升序排（`sgl-router/src/policies/cache_aware.rs:28-150`） |
| 负载与亲和的权衡 | 无 | `--affinity-mode prefer`（默认）：首选能收就送；`balanced`：首选负载 > 备选 × 2 且 > 备选 + 1024 token（或 4 条请求）才改投（`README.md:143-150`，`cli.rs` AffinityArgs） |
| 负载信号多久更新 | coordinator 最快 100 ms 发一次快照（`vllm/v1/engine/coordinator.py:152`）；两次快照之间靠本地 `waiting += client_count`（`core_client.py:1817-1820`）和精确的 inflight 计数兜底（`core_client.py:1797-1804`） | engine 推送 load report 和 KV 事件，router 自己记 in-flight（`README.md:36-41`）；冷前缀突发靠 `--cache-pending-prefix-ttl-ms` |
| 多个 API server | `--api-server-count`；第 k 个 client 的扫描起点 `N × k // client_count`（`core_client.py:1767`） | router 本身可多副本，K8s 下 `--kv-peer-selector` 从兄弟副本拷 KV 树 |
| 多节点 | `--data-parallel-size-local` / `--data-parallel-start-rank` / `--headless`，或 `--data-parallel-backend=ray`；`--data-parallel-hybrid-lb` 每节点各自一个 LB（`data_parallel_deployment.md:36-100`） | `--service-discovery --selector …` 自动发现 Pod |
| 外部路由要的 KV 事件 | `--kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", ...}'`（`examples/disaggregated/disaggregated_serving/kv_events.sh:43`） | 同名 `--kv-events-config`，加 `replay_endpoint` 支持丢事件后重取（`README.md:312-314`） |

## K8s 部署

| 项 | 值 | 出处 |
|---|---|---|
| 探针默认值 | periodSeconds 10，failureThreshold 3，initialDelaySeconds 0，timeoutSeconds 1 | k8s `concepts/workloads/pods/probes.md:248-260` |
| 最晚被杀 | initialDelay + (failureThreshold − 1) × period | 同上（连续 failureThreshold 次失败） |
| startupProbe | 成功前 liveness / readiness 不跑；`failureThreshold × periodSeconds` 覆盖最坏启动时间 | k8s `tasks/.../configure-liveness-readiness-startup-probes.md`（30 × 10 = 300 s） |
| 重启退避 | 10 s 起翻倍，封顶 300 s | k8s `pod-lifecycle.md` |
| emptyDir | 容器崩溃后数据还在；容器可写层在崩溃后丢失 | k8s `concepts/storage/volumes.md:38-42, 170-171` |
| vLLM 示例 | `/health`，initialDelay 60，liveness period 10 / readiness 5；gRPC 版 120 s；HF 缓存挂 PVC，`/dev/shm` 挂 Memory emptyDir | `docs/deployment/k8s.md:205-255, 400-423` |
| 探针太紧的症状 | 日志 "KeyboardInterrupt: terminated"，events 里 "failed startup probe, will be restarted" | `docs/deployment/k8s.md:436-443` |
| 编译缓存位置 | `VLLM_CACHE_ROOT/torch_compile_cache/<hash>`，默认 `~/.cache/vllm` | `vllm/compilation/backends.py:1096` |

## sleep mode

| 项 | vLLM | SGLang |
|---|---|---|
| 开关 | `--enable-sleep-mode`，HTTP 端点要 `VLLM_SERVER_DEV_MODE=1` | `--enable-memory-saver` |
| 释放 | `POST /sleep?level=1`（权重进 CPU pinned 内存，丢 KV）、`level=2`（全丢）、`level=0&mode=keep` + `/release_kv_cache_memory`（只放 KV） | `POST /release_memory_occupation {"tags": ["kv_cache", "weights"]}`，要求引擎空闲 |
| 恢复 | `POST /wake_up[?tags=weights\|kv_cache]`；level 2 之后 `collective_rpc reload_weights` | `POST /resume_memory_occupation` |
| 引擎侧 | `EngineCore.sleep(level, mode)` 先 `pause_scheduler`（`vllm/v1/engine/core.py:1019-1060`） | — |
| 出处 | `docs/features/sleep_mode.md:19-21, 192-240` | `docs/docs/advanced_features/sglang_for_rl.mdx:24-104` |

## RL 暂停与权重同步

| 项 | vLLM | SGLang |
|---|---|---|
| 暂停 | `POST /pause?mode=abort\|wait\|keep`（默认 abort，`clear_cache` 默认 True，已标 DEPRECATED，keep 时忽略）（`vllm/entrypoints/rl/online/api_router.py:31-75`） | `POST /pause_generation {"mode": "abort\|retract\|in_place"}`，默认 abort（`sglang_for_rl.mdx:620-656`） |
| 新请求 | abort / wait：503 直到 resume，客户端重试；keep：照收照排（`core.py:2283-2290`，`docs/training/async_rl.md:24-36`） | 文档未写 |
| 在飞请求 | abort：`finish_requests(None, FINISHED_ABORTED)` 带部分结果返回；wait：跑完；keep：PAUSED_ALL 冻结（`core.py:2238-2280`） | abort：waiting + running 全返回；retract：running 退回 waiting，KV 可 flush；in_place：原地冻结，之后 flush_cache 会失败 |
| 清缓存 | `_finish_pause`：先 `synchronize_device`，再 `_reset_caches`（前缀缓存含 running、connector、mm、encoder）（`core.py:938-972`） | `update_weights_from_*` 的 `flush_cache` 默认 True |
| 恢复 | `POST /resume` | `POST /continue_generation` |
| 换权重 | `/init_weight_transfer_engine` → `/start_weight_update` → `/update_weights` → `/finish_weight_update {"weight_version": ...}`（`docs/training/weight_transfer/README.md:155-177`） | `/update_weights_from_disk`、`/update_weights_from_tensor`、`/init_weights_update_group` + `/update_weights_from_distributed`（`sglang_for_rl.mdx:106-597`） |
| 传输后端 | nccl（rank 0 广播，packed 1 GiB × 2）、ipc（同卡共置）、sparse_nccl、sharded_rdt（NIXL / Ray Direct Transport）、nccl_m2n（换切分）、ModelExpress（`weight_transfer/README.md:30-39`，`nccl.md:77-108`） | disk / tensor / distributed（NCCL 组） |
| world_size | 训练端广播 rank 数 + `GET /get_world_size`（TP × PP × DP）；worker rank = `rank_offset + dp_index × TP × PP + rank`（`nccl.md:197-210`） | `init_weights_update_group` 的 `world_size` / `rank_offset` |
| 版本段 | `weight_versions` 列表，`[start, end)`，请求带 `return_weight_versions: true`（`async_rl.md:73-90`） | `meta_info.weight_versions`，同样 `[start, end)`（`sglang_for_rl.mdx:598-610`） |
| DP 下 | 内置 LB（ray 后端）一次调用覆盖所有 rank；外部 LB 要对每个实例分别 pause / resume / 设版本（`async_rl.md:59`） | router 前面的每个 engine 分别调 |
| 更新失败 | `/update_weights` 失败后模型处于半更新状态，`/health` 仍是 200，`/finish_weight_update` 被拒；必须重走 start → update → finish（`weight_transfer/README.md:173-177`） | 返回 `success: false` |
