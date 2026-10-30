# 前缀缓存与混合 KV 管理：逐行对照

基线 vllm 主干 0d6f92a，sglang 主干 8fabf5e。

## 1. 块哈希（vllm/v1/core/kv_cache_utils.py、vllm/utils/hashing.py）

| 行 | 代码 | 说明 |
|---|---|---|
| 60–69 | `make_block_hash_with_group_id` | `block_hash + group_id.to_bytes(4, "big")`，哈希表的键带组号，同一段 token 在不同 KV 组里是不同的条目 |
| 96–105 | 注释 | 不可预测的种子让攻击者没法离线构造碰撞块（issue #12621）；`PYTHONHASHSEED` 在两种情况下都能覆盖种子 |
| 108 | `DEFAULT_NONE_HASH_SEED = "vllm-none-hash"` | sha256 / sha256_cbor 这类密码学哈希没设 `PYTHONHASHSEED` 时用固定种子，多实例之间哈希一致，KV 事件和外部缓存才对得上 |
| 146–160 | `init_none_hash` | xxhash / xxhash_cbor 不抗碰撞，没设种子时用 `os.urandom` 随机种子并打警告；`NONE_HASH = hash_fn(seed)` |
| hashing.py:26 | `sha256` | `pickle.dumps(obj, HIGHEST_PROTOCOL)` 后求 sha256；`sha256_cbor` 用 CBOR 序列化，跨语言可复现 |
| 590–607 | `_gen_lora_extra_hash_keys` | `("lora", lora_name, lora_path)`；路径也进哈希，同名适配器换了权重不会复用旧 KV |
| 637–671 | `generate_block_hash_extra_keys` | 顺序 lora + 多模态 + `("cache_salt", salt)` + prompt_embeds；salt 只在 `start_token_idx == 0` 时加；全空返回 None |
| 707 | `hash_block_tokens` | `hash_fn((parent_block_hash or NONE_HASH, tuple(curr_block_token_ids), extra_keys))`，链式 |
| 953 | `get_request_block_hasher` | 请求创建和追加 token 时增量算块哈希，只算满块 |

## 2. 哈希表与块池（vllm/v1/core/block_pool.py）

| 行 | 代码 | 说明 |
|---|---|---|
| 36–56 | `BlockHashToBlockMap` docstring | NOTE #1：不去重。一个块满了就缓存，不检查表里有没有相同的块，因为块表只能追加，已分配的块号不能换 |
| 62–71 | `get_one_block` | 值是单个块或 `{block_id: block}`；有重复时取第一个 |
| 84–96 | `insert` | 同键第二个块进来时把值从单块升级成 dict（减少 GC 压力） |
| 202–228 | `get_cached_block` | 对 `kv_cache_group_ids` 每个组拼键查，任何一组缺失就算未命中 |
| 230–300 | `cache_full_blocks` | 给 `blocks[num_cached:num_full]` 设哈希并插表；`block_mask` 为 False 的块跳过（SWA 稀疏保留） |
| 706 | `get_new_blocks` | 从队头弹块；块上还有哈希就 `_maybe_evict_cached_block`，这一刻才真正驱逐 |
| 769–790 | `_maybe_evict_cached_block` | 从表里删掉 (hash, block_id)，发 BlockRemoved 事件 |
| 792 | `touch` | 命中块 ref_cnt 为 0 时从空闲队列中间摘掉 |
| 816–849 | `free_blocks` | ref_cnt 归零的块：无哈希的 prepend（"LIFO reuse … better GPU locality"），有哈希的 append（"FIFO reuse … LRU eviction behavior"） |
| 871 | `reset_prefix_cache` | 只有所有块都空闲时才能清，RL 换权重后要调用 |

## 3. 命中与缓存入口（vllm/v1/core/kv_cache_manager.py）

| 行 | 代码 | 说明 |
|---|---|---|
| 266–328 | `get_computed_blocks` | 上限 `request.get_max_cache_hit_length()`，即 `num_tokens - 1`：全命中时最后一个 token 也要重算拿 logits，块对齐时会重算整块 |
| 323 | `shared_prefix_boundary` | 有未缓存部分时等于 `命中 + num_uncached`，记录和别的请求分叉的位置（Marconi 式），SWA 稀疏保留时要给这里留窗口尾巴 |
| 379 | `allocate_slots` | 分配完立刻缓存满块，同一批后面的请求就能命中（prefix_caching.md:148） |
| 612–626 | 尾部 | `delay_cache_blocks`（P/D 远端 KV 未到）直接返回；否则 `min(总已算 + 新 token, request.num_tokens)`，草稿 token 不缓存 |

## 4. 单组查找（vllm/v1/core/single_type_kv_cache_manager.py）

| 行 | 代码 | 说明 |
|---|---|---|
| 477–540 | `cache_blocks` | `num_full = min(num_tokens, 哈希数 × hash_bs) // bs`；`reachable_boundaries = replay_boundaries + shared_prefix_boundary` |
| 743 | `FullAttentionManager.find_longest_cache_hit` | 从左往右，第一次失配就 break |
| 956 | `_contiguous_blocks_for_hit` | `cdiv(w - 1, bs)`，EAGLE 再加 1 |
| 969–1064 | `SlidingWindowManager.find_longest_cache_hit` | 预填 null block，从右往左；凑够连续块数就截掉右边多余部分；全程没凑够时保留前 `num_contiguous` 块；TODO 注明复杂度 O(max_num_blocks) |
| 1066–1145 | `reachable_block_mask` | retention None 稠密、0 只留重放边界、正数每段留一个尾巴；`need >= per_segment` 时返回 None（全缓存） |
| 1148 | `get_num_skipped_tokens` | `max(0, n - w + 1)`：窗口外的块可以还给池子 |

## 5. 多组协调与分组（kv_cache_coordinator.py、kv_cache_planning.py）

| 行 | 代码 | 说明 |
|---|---|---|
| coordinator 33–62 | retention 校验 | 必须非负且是 scheduler block size 的倍数 |
| coordinator 65 | `replay_boundaries` | 不用 EAGLE 时是 `(num_prompt - 1,)`，相同 prompt 重发的命中位置 |
| coordinator 709 | `_cache_hit_alignment_tokens` | 默认 scheduler block size（各组块大小的公倍数） |
| coordinator 806–941 | `HybridKVCacheCoordinator.find_longest_cache_hit` | 不动点迭代：各组轮流把命中长度往下压；full 组只查一次再截断；full + 一种其他类型一轮就收敛 |
| planning 522–532 | group_size | 取最少的那类层数；`max < 1.5 × min` 时取最多的（gpt-oss-20b + EAGLE 12 sw + 13 full → 13/13） |
| planning 535–538 | padding 警告 | "Add %d padding layers, may waste at most %.2f%% KV cache memory"，百分比 = padding / 该类层数 |
| planning 552 | `layers[i::num_groups]` | 轮转分配，PP 切分后每个 stage 的组仍然齐整 |

## 6. 指标（vllm/v1/metrics/stats.py）

| 行 | 代码 | 说明 |
|---|---|---|
| 121 | `PrefixCacheStats` | queries / hits / preempted_queries / preempted_hits；Prometheus 上是 `vllm:prefix_cache_queries` 与 `vllm:prefix_cache_hits` |

## 7. SGLang 对照

| 位置 | 代码 | 说明 |
|---|---|---|
| radix_cache.py:250–277 | `TreeNode`、`__lt__` | 节点有 key、value（KV 槽位张量）、lock_ref、last_access_time、hit_count、priority；堆按 last_access_time 比较 |
| radix_cache.py:364 | `match_prefix` | page 对齐后沿树走，返回命中的槽位和最后一个节点 |
| radix_cache.py:651 | `_match_prefix_helper` | 更新访问时间；命中停在节点中间时 split |
| radix_cache.py:677 | `_split_node` | 新父节点继承 priority、hit_count、lock_ref |
| radix_cache.py:428 / 705 | `insert` / `_insert_helper` | priority 取 max 向上传；新节点计入 evictable_size_ |
| radix_cache.py:450 | `_insert_cache` | value 来自 `req_to_token` 行；已在树里的部分 `free_segment(kv_indices[cache_protected_len:prefix_len])`——去重 |
| radix_cache.py:510 | `checkpoint` | 每个 extend chunk 后由 `common.py:271 checkpoint_kv_cache` 调用 |
| radix_cache.py:552 | `evict` | 收集可驱逐叶子建堆，弹出后父节点变叶子再入堆 |
| radix_cache.py:582 / 597 | `inc_lock_ref` / `dec_lock_ref` | 沿路径到根加减锁，0↔1 时增减 evictable_size_ |
| common.py:418 | `release_kv_cache` | retract 走 `checkpoint=False`，不把被抢占请求插树（TODO 里写着以后可以插） |
| f_memory.py:25–42 | `--radix-eviction-policy` | 默认 lru；可选 lfu、slru、priority、tlru、fifo、mru、filo |
| f_schedule.py:86–102 | `--schedule-policy` | 默认 fcfs；lpm、dfs-weight、hrrn 等是 cache-aware |
| schedule_policy.py:106–118 | in-batch 阈值 | `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = 32`，`..._DEPRIORITIZE_THRESHOLD = 32` |
| schedule_policy.py:314–325 | `_determine_active_policy` | 等待队列超过 128 条时 LPM / HRRN 退回 FCFS，匹配太贵 |
| schedule_policy.py:356–406 | `_compute_prefix_matches` | 已命中 ≤ 32 的请求在等待队列临时树里再匹配，≥ 32 就降级，否则插入临时树 |
| f_schedule.py:150 | `--swa-full-tokens-ratio` | SWA 层 token 数 / full 层 token 数，默认 0.8，范围 (0, 1] |
| pool_configurator.py:798–803、881 | 池子切分 | 每 token 字节 = full_cell + ratio × swa_cell；`swa = align(int(full × ratio))` |
| schedule_batch.py:4075 | `maybe_evict_swa` | 每 `SGLANG_SWA_EVICTION_INTERVAL = 128` 个 decode token 回收一次窗口外的 SWA 槽位 |
