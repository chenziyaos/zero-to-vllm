# PD 分离：源码位置与参数对照

基线 vllm 主干 0d6f92a，sglang 主干 8fabf5e。表里的"行"都是这两个提交的行号。vllm 路径相对 `vllm/`，NIXL 连接器在 `distributed/kv_transfer/kv_connector/v1/nixl/`（下文简称 `nixl/`）。sglang 路径相对 `python/sglang/srt/`。

## 1. 入口：一份 kv_transfer_config 怎么变成连接器

| 位置 | 代码 | 说明 |
|---|---|---|
| config/kv_transfer.py 67 | `kv_buffer_device` | `cuda`（默认，GPU 直连）/ `cpu`（经主机缓冲，`use_host_buffer`） |
| config/kv_transfer.py 71 | `kv_role` | `kv_producer`（P）/ `kv_consumer`（D）/ `kv_both`；NIXL 连接器在运行时按请求参数判断 P/D，角色只影响默认行为 |
| config/kv_transfer.py 64、111 | `engine_id` | 不写就 `uuid4()`；push 模式 proxy 要提前知道 P 的 engine_id，所以 P 必须显式写 |
| config/kv_transfer.py 88 | `kv_load_failure_policy` | `fail`（默认，D 读失败直接报错）/ `recompute`（失败块重算） |
| kv_connector/factory.py 126–130 | `transfer_mode` | `NixlConnector` + `extra_config.transfer_mode` = `pull`（默认）/ `push`，改写成 `NixlPullConnector` / `NixlPushConnector` |
| envs.py 236–237 | `VLLM_NIXL_SIDE_CHANNEL_HOST/PORT` | `localhost` / `5600`；每个 DP rank 在此基础上加 dp_rank；单机两实例必须错开 |

## 2. 调度侧（`nixl/base_scheduler.py`、`pull_scheduler.py`、`push_scheduler.py`）

| 位置 | 说明 |
|---|---|
| base_scheduler 82–88 | `kv_lease_duration` 默认 30；`_heartbeat_interval = lease // 6` = 5 s |
| base_scheduler 168–187 | `kv_recompute_threshold` 64、`bidirectional_kv_xfer` False、`decoder_kv_blocks_ttl` 480 |
| base_scheduler 202–241 | `on_new_request`：只有带 `do_remote_prefill` 的请求（D 侧）登记心跳，按远端 engine 聚合 |
| base_scheduler 385–409 | `_prefill_backoff = max(1 if mamba 或 bounded_replay else 0, lookahead − 1)`；`_get_remote_prefill_token_count` = prompt − backoff |
| base_scheduler 411–436 | `_truncate_request_for_prefill`：P 侧把 prompt 截掉 backoff 那段，只算前面 |
| pull_scheduler 34–106 | `get_num_new_matched_tokens`：D 侧 `(prompt − backoff − 本地命中, True)`；P 侧第二轮（双向）`min(remote_num_tokens, prompt) − 本地命中`，小于 `kv_recompute_threshold` 返回 `(0, False)` |
| pull_scheduler 108 | `update_state_after_alloc`：记下要读的本地块，进 `_reqs_need_recv` |
| pull_scheduler 191–292 | `request_finished`：P 侧延迟释放，到期 now + lease；双向模式 D 侧也延迟释放，到期 now + `decoder_kv_blocks_ttl`；返回 `remote_block_ids / remote_engine_id / remote_request_id / remote_host / remote_port / tp_size / remote_num_tokens / remote_blocks_expiry_time` |
| push_scheduler 96–100 | `push_registration_timeout` 默认取 `decoder_kv_blocks_ttl` |
| push_scheduler 128–205 | `update_state_after_alloc`：D 侧把自己的块和 P 的坐标打成 PUSH_REG，`remote_block_ids=()`，`do_remote_prefill=False` |
| v1/core/sched/scheduler.py 1523 | 异步外部命中 → `WAITING_FOR_REMOTE_KVS` |
| v1/core/sched/scheduler.py 3465–3514 | `_update_waiting_for_remote_kv`：收齐后 `num_computed == num_tokens` 时置 `num_tokens − 1`，D 下一步只算 1 个 token |

## 3. Worker 侧（`nixl/base_worker.py`、`pull_worker.py`、`push_worker.py`）

| 位置 | 说明 |
|---|---|
| base_worker 631 | `Initializing NIXL worker %s`（engine_id） |
| base_worker 643–647 | `_lease_extension = lease · 2 // 3` = 20 s |
| base_worker 1009 | `Detected kv cache layout`；异构 TP 切头要求 HND（LBHNC，头在前） |
| base_worker 1513–1881 | `register_kv_caches`：每层按 `cache.nbytes // (num_blocks · block_stride)` 切 region，`num_descs = sum(region_num_blocks)` |
| base_worker 1134 / 1479 | `_nixl_handshake` / `_background_nixl_handshake`：经 side channel（ZMQ）拿远端 agent 元数据 |
| base_worker 2421、2557 | `add_remote_agent`；`Transfer plan:` 日志 |
| base_worker 3384–3410 | `_reap_expired_send_leases`：P 侧租约到期，`Releasing expired KV blocks` |
| base_worker 3412–3430 | `_handle_heartbeat`：`HB:` 通知里的请求 expiry = max(旧, now + 20) |
| tp_mapping.py 70 | `compute_head_offset`：D TP > P TP 时本 rank 的头在远端块里的偏移 |
| tp_mapping.py 84–128 | `compute_tp_mapping`：MLA 或 D TP ≥ P TP → `[rank · P_tp // D_tp]`；P TP > D TP → 连续 `abs_tp` 个 rank，按头 `np.unique` 去重 |
| pull_worker 33 | `_KV_BLOCKS_EXPIRY_SAFETY_MARGIN = 5.0`：双向第二轮，离 D 侧块到期不足 5 s 就不读了 |
| pull_worker 138 / 401 | `_read_blocks_for_req` / `_read_blocks`：D 在 engine 线程上准备描述符、发 READ |
| pull_worker 676 | `Potentially invalid KV blocks`：完成通知晚于租约到期 |
| push_worker 153、253 | `nixl-push-writer` 后台线程：合并相邻块，发 WRITE |
| push_worker 317 / 337 | `_handle_push_reg_notif`（P 收登记）/ `_send_registration_to_p`（D 发登记） |
| v1/metrics.py 108 | `KV Transfer metrics:` 每 10 秒一行 |
| stats.py 188–276 | `vllm:nixl_xfer_time_seconds`、`_post_time_seconds`、`_bytes_transferred`、`_num_descriptors`、`_num_failed_transfers`、`_num_failed_notifications`、`_num_kv_expired_reqs`、`_num_notifications_after_expiry` |

## 4. `kv_connector_extra_config`（docs/features/nixl_connector_usage.md）

| 键 | 默认 | 说明 |
|---|---|---|
| `transfer_mode` | `pull` | `push`：P 写 D，D 必须在 prefill 之前确定 |
| `backends` | `["UCX"]` | `LIBFABRIC`（EFA）等；UCX 由 `UCX_TLS` / `UCX_NET_DEVICES` 控制 |
| `kv_lease_duration` | 30 | P 侧块的租约秒数，D 心跳续期 |
| `bidirectional_kv_xfer` | false | P、D 两边都要开；第二轮 P 从 D 读上一轮的 KV |
| `kv_recompute_threshold` | 64 | 第二轮差额小于这个 token 数就重算 |
| `decoder_kv_blocks_ttl` | 480 | 双向模式 D 侧块保留秒数，**不续期** |
| `enforce_handshake_compat` | true | 握手时校验 vLLM 版本、模型、dtype、块大小等 |
| `ucx_backend_params` | — | 例如 `{"ucx_error_handling_mode":"none"}` |

## 5. 代理（proxy）

| 文件 | 端口 / 参数 | 流程 |
|---|---|---|
| tests/v1/kv_connector/nixl_integration/toy_proxy_server.py | `--prefiller-hosts/ports`、`--decoder-hosts/ports`，默认 8100 / 8200 | 先发 P（`do_remote_decode`、`max_tokens=1`、非流式），拿到 `kv_transfer_params` 再发 D；P、D 各自轮询 |
| examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py | `--prefill host:port`、`--decode host:port` | pull，同上 |
| …/disagg_proxy_pushconnector_demo.py | `--prefill-engine-id`、`--prefill-kv-host`、`--prefill-side-channel-port`（5600）、`--prefill-tp-size`、`--prefill-pp-size` | push：P、D 同时发，D 拿着 P 的坐标去登记 |
| …/disagg_proxy_multiturn.py | `--prefiller-host/port`、`--decoder-host/port` | 双向：按请求体顶层 `conversation_id` 缓存 D 返回的 `kv_transfer_params`，下一轮带给 P |

## 6. 与 SGLang PD 分离对照

| 需求 | vLLM | SGLang |
|---|---|---|
| 打开 | `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'` | `--disaggregation-mode prefill / decode` |
| 传输引擎 | NIXL（UCX / LIBFABRIC 等），另有 Mooncake、MoRI-IO、P2pNccl 等连接器 | `--disaggregation-transfer-backend mooncake`（默认）/ `nixl` / … |
| 方向 | pull：D 读 P；push：P 写 D | P 写 D（D 先把 kv_indices 发给 P） |
| 握手 | side channel 5600 + dp_rank（ZMQ） | bootstrap server，`--disaggregation-bootstrap-port 8998` |
| 请求配对 | proxy 透传 `kv_transfer_params` | proxy 生成 `bootstrap_host / port / room`；D 去 P 的 `room % dp_size`（decode.py 898） |
| 代理 | toy proxy / disagg_proxy_*（串行发 P 再发 D）；vllm-router | mini_lb（sgl-model-gateway/…/mini_lb.py 103–111 随机选对，176 `asyncio.gather` 同时发）；sglang_router `--pd-disaggregation` |
| D 侧准入 | 按已有 token 分配，不预留输出 | decode.py 511–520：max(已分配 + 512, prompt − 前缀 + min(max_new, 4096) − 可回收) |
| P 等 D 超时 | 租约 30 s，心跳续 20 s | `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` 300（environ.py 763） |
| D 等 KV 超时 | 无 | `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` 300（767） |
| 探活 | NIXL 通知 `HB:req_id` | `HEARTBEAT_INTERVAL` 5.0 × `HEARTBEAT_MAX_FAILURE` 2（765–766） |
| 异构 TP | tp_mapping 自动映射；D 切得更细时要求 HND 布局 | 默认按 token × 头切片；Mooncake 可开 staging buffer（文档 2–5 倍） |
| 多轮复用 | `bidirectional_kv_xfer`：P 从 D 读回上一轮 KV | decode 侧 radix cache 默认关；`--disaggregation-decode-enable-offload-kvcache` 卸到 HiCache |
