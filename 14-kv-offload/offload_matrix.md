# KV 卸载：源码位置与参数对照

基线 vllm 主干 0d6f92a，sglang 主干 8fabf5e。表里的“行”都是这两个提交的行号。vllm 路径相对 `vllm/`，sglang 路径相对 `python/sglang/`。

## 1. 入口：两个 CLI 参数怎么变成连接器（vllm）

| 位置 | 代码 | 说明 |
|---|---|---|
| config/cache.py 383 | `kv_offloading_size` | GiB，**TP 组内各 rank 合计** |
| config/cache.py 389 | `kv_offloading_backend` | `native` / `lmcache` |
| config/vllm.py 1168–1196 | 改写 `kv_transfer_config` | native → `OffloadingConnector`（`VLLM_USE_SIMPLE_KV_OFFLOAD=1` 时 `SimpleCPUOffloadConnector`），`cpu_bytes_to_use = size · 2^30`；lmcache → `LMCacheMPConnector`，默认 `tcp://localhost:5555`，**size 不往下传**（容量由 `lmcache server --l1-size-gb` 决定）；`kv_role` 强制 `kv_both` |
| config/kv_transfer.py 88 | `kv_load_failure_policy` | `fail`（默认）/ `recompute` |
| distributed/kv_transfer/kv_connector/factory.py | 注册表 | Offloading 219、Mooncake 231、MooncakeStore 236、SimpleCPUOffload 251、LMCacheV1 171、LMCacheMP 177、Nixl 183、Multi 201 |

## 2. 连接器 API（vllm `distributed/kv_transfer/kv_connector/v1/base.py`）

| 侧 | 方法 | 行 | 调用时机 |
|---|---|---|---|
| scheduler | `get_num_new_matched_tokens` | 516 | 排一个 WAITING 请求时，本地前缀命中之后；返回 `(tokens 或 None, load_async)` |
| scheduler | `update_state_after_alloc` | 569 | 给外部命中的 token 分完 GPU 块之后 |
| scheduler | `build_connector_meta` | 595 | 每步末尾，打包本步要搬的块交给 worker |
| scheduler | `request_finished` | 631 | 请求结束；返回 True 表示块要等异步保存完才能释放 |
| worker | `start_load_kv` / `wait_for_layer_load` | 333 / 350 | 前向开始前发起回迁；每层 attention 前等这一层 |
| worker | `save_kv_layer` / `wait_for_save` | 364 / 386 | 每层算完存；前向结束等保存 |
| worker | `get_finished` | 398 | 报告 finished_sending / finished_recving |

## 3. 调度器怎么用连接器（vllm `v1/core/sched/scheduler.py`）

| 行 | 说明 |
|---|---|
| 1104–1175 | 先本地前缀命中；`partial_tail = local % block_size`，交给连接器的是按块对齐的命中；返回 None → `skip_request`；远端命中超过尾巴 → 丢掉本地不满块 |
| 3465–3513 | `_update_waiting_for_remote_kv`：失败时只 cache 有效前缀，否则 `cache_blocks`；`num_computed == num_tokens` 时置 `num_tokens - 1` |
| 3517 | `_handle_blocked_waiting_request`：`WAITING_FOR_REMOTE_KVS` 等到 finished_recving，转回 WAITING 或 PREEMPTED |

## 4. OffloadingConnector 调度侧（`distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`）

| 行 | 说明 |
|---|---|
| 1258–1325 | `get_num_new_matched_tokens`：还有在途传输 → `(None, False)`；`skip_reading_external_kv` 或 `load_failed` → 命中 0；`_lookup` 计时进 `vllm:kv_offload_lookup_sync_delay_seconds`；返回 `(hit, bool(hit) and not sync_load)` |
| 1327–1351 | `_reserve_sync_load`：本步 `sync_load_budget_bytes` 预算，至少留 1 个 token 计算 |
| 229 / 236 | 投机解码：EAGLE/MTP 草稿组的最后一个 chunk 不卸载 |
| multi_connector.py 451–470 | 第一个命中 > 0 的子连接器胜出；任一子连接器返回 None 就整体 `(None, False)` |

## 5. 淘汰与写入（vllm `v1/kv_offload/cpu/`）

| 位置 | 说明 |
|---|---|
| policies/lru.py | 请求粒度 recency，惰性堆 |
| policies/arc.py 52–190 | T1/T2/B1/B2；新块进 T1，`on_request_finished` 把被复用的就绪块升 T2；B1 ghost 命中 `target_t1_size += max(1, |B2|/|B1|)`，B2 反之；`|T1| ≥ int(target)` 先淘汰 T1 |
| manager.py 66 / 287 | `store_threshold`：同一 key 被 lookup 的次数达到阈值才进入写候选（已驻留的不受影响），跳过数记入 `STORES_SKIPPED`；计数表 `max_tracker_size` 默认 64000（spec.py 163） |
| policies/factory.py | `CachePolicyFactory`，预注册 lru / arc；自定义用 `cache_policy_module_path` |
| gpu_worker.py 881 | `Allocating %d CPU tensors...` |

## 6. `kv_connector_extra_config`（docs/features/kv_offloading_usage.md）

| 键 | 默认 | 说明 |
|---|---|---|
| `spec_name` | CPUOffloadingSpec | `TieringOffloadingSpec` 开多级 |
| `cpu_bytes_to_use` | 必填 | 所有 worker 合计 |
| `block_size` / `blocks_per_chunk` | GPU 块 / 1 | 卸载粒度，必须是 GPU 块的倍数 |
| `eviction_policy` | `lru` | `arc` 或自定义 |
| `store_threshold` | 0 | 被查询（lookup）达到次数才卸载；只在单级，≥ 2 时 Tiering 拒绝 |
| `secondary_tiers` | `[]` | `fs` / 对象存储 / p2p，按顺序查询；GPU↔二级都经 CPU 中转 |
| `tier_poll_interval_s` | 0.001 | p2p 层的轮询间隔 |
| `offload_prompt_only` | true | 只卸载 prompt 部分 |
| `sync_load_budget_bytes` | 0 | 每步每卡同步回迁预算 |
| `canonical_layout` | false | 跨 TP 共享、复制 KV 只存一份；只支持 TP |
| 每请求 `kv_transfer_params` | — | `max_load_tokens` / `max_offload_tokens`（实验） |

指标：`vllm:kv_offload_config_info`、`vllm:kv_offload_cpu_cache_usage_perc` / `_write_usage_perc` / `_read_usage_perc`、`vllm:kv_offload_stores_skipped`（cpu/common.py 13–17）、`vllm:kv_offload_load_bytes` / `_load_time` / `_store_bytes` / `_store_time`、`vllm:kv_offload_lookup_sync_delay_seconds`、`vllm:external_prefix_cache_queries` / `_hits`（offloading/metrics.py 27–55、v1/metrics/loggers.py 685）。

## 7. 与 SGLang HiCache 的参数对照

| 需求 | vLLM | SGLang |
|---|---|---|
| 打开 CPU 层 | `--kv-offloading-backend native --kv-offloading-size N` | `--enable-hierarchical-cache` |
| 容量 | `N` GiB，TP 组合计；不写就没有 | `--hicache-ratio`（cache 模式默认 2.0）或 `--hicache-size` GB(1e9)/rank |
| 写入 | 算完就写；`store_threshold` 过滤 | `--hicache-write-policy write_through`（默认）/ `write_through_selective` / `write_back` |
| 淘汰 | `eviction_policy` lru / arc / 自定义 | host 侧 LRU |
| 拷贝 | DMA `cudaMemcpyAsync` | `--hicache-io-backend kernel`（默认）/ `direct` |
| 主机布局 | GPU 页原样，`canonical_layout` 可选 | `--hicache-mem-layout page_first`（默认）/ `page_first_direct` / `page_head` |
| 第三级 | `secondary_tiers`；或 LMCache / MooncakeStore 连接器 | `--hicache-storage-backend file / mooncake / hf3fs / nixl / aibrix …` |
| 跨 TP 共享 | `canonical_layout: true` | `--hicache-storage-backend-extra-config '{"tp_lcm_size": 8}'` |
| 阈值 | 无硬阈值 | `load_back_threshold=10`（unified_radix_cache.py 588）、`prefetch_threshold=256`（cache_controller.py 309）、selective 命中 2 次（570） |
| 预取等待 | 不调度直到 finished_recving | `--hicache-storage-prefetch-policy timeout`：代码默认 1 + 0.25·tokens/1024 秒、无上限（文档写 min(30, 2 + 0.1·tokens/1024)） |
| 失败 | `kv_load_failure_policy` fail / recompute | 预取超时放弃，剩余重算 |
| PD + 共享池 | `MultiConnector[MooncakeConnector, MooncakeStoreConnector]` | `--disaggregation-mode` + HiCache；decode 侧 `--disaggregation-decode-enable-offload-kvcache` |
