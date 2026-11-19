# PagedAttention 块分配：逐行对照

基线 vllm 主干 0d6f92a，sglang 主干 8fabf5e。

## 1. 块与空闲队列（vllm/v1/core/kv_cache_utils.py）

| 行 | 代码 | 说明 |
|---|---|---|
| 162 | `class KVCacheBlock` | `block_id`、`ref_cnt`、`_block_hash`（只有满块且开前缀缓存时才有）、`prev_free_block` / `next_free_block`、`is_null` |
| 258 | `class FreeKVCacheBlockQueue` | 手写双向链表，不用 `deque`，因为前缀命中时要从中间 O(1) 摘块；操作时只改块上的两个指针，不分配 Python 对象 |
| 258 docstring | 驱逐顺序 | 队头是最久未用的块（LRU）。同一请求释放的块里，链尾（哈希覆盖 token 更多的块）排在前面。这个顺序靠调用方把块列表反转后再交回来维持 |
| 297 | `fake_free_list_head / tail` | 假头假尾，保证真实块总有前驱和后继，少写分支 |
| 438 | `prepend_n` | 一次把 n 块插到队头 |
| 459 | `append_n` | 一次把 n 块接到队尾 |
| 1061 | `get_max_concurrency_for_kv_cache_config` | 对每个 KV 组按 `cdiv(max_memory_usage_bytes, page_size_bytes)` 求一条请求最多要几块，池块数除以它就是日志里的并发倍数 |
| 1121 / 1147–1154 | `get_kv_cache_capacity` / `update_kv_cache_capacity` | 启动日志：`%s KV cache size: %s tokens, Maximum concurrency for %s tokens per request: %.2fx` |

## 2. 块池（vllm/v1/core/block_pool.py）

| 行 | 代码 | 说明 |
|---|---|---|
| 137 | `class BlockPool` | 管理所有块、空闲队列和哈希表 `cached_block_hash_to_block` |
| 176–182 | `self.blocks = [KVCacheBlock(idx) ...]`、`FreeKVCacheBlockQueue(self.blocks)` | 建池时所有块按 id 顺序入队 |
| 189–192 | `self.null_block = self.free_block_queue.popleft()` | block 0 弹出作为 null block，表示“这个位置不需要 KV”（滑动窗口外、跳过的块）。注释：`ref_cnt of null_block is not maintained` |
| 706 | `get_new_blocks(n)` | `popleft_n(n)` 从队头取；开缓存时对每块调 `_maybe_evict_cached_block`，把它旧的哈希从表里删掉；`ref_cnt` 0 → 1 |
| 769 | `_maybe_evict_cached_block` | 驱逐发生在“这个块被重新分出去”的那一刻，不是释放时 |
| 792 | `touch(blocks)` | 前缀命中：`ref_cnt == 0` 的块在空闲队列里，先 `remove` 摘出，再 `ref_cnt += 1` |
| 816 | `free_blocks(ordered_blocks)` | 跳过 null；属于其他池的块转交；`ref_cnt -= 1` 到 0 时分两类 |
| 838–843 | 无哈希或关缓存 → `blocks_to_evict_first`；有哈希 → `blocks_to_evict_last` | 注释原文：`LIFO reuse of non-cached blocks for better GPU locality`、`FIFO reuse of cached blocks for LRU eviction behavior` |
| 845–847 | `prepend_n(first)`、`append_n(last)` | 没用的块马上复用，可能被命中的块尽量晚复用 |
| 921 | `get_usage` | 用量分母是 `num_gpu_blocks − 1`，扣掉 null block |

## 3. 请求级释放（vllm/v1/core/single_type_kv_cache_manager.py）

| 行 | 代码 | 说明 |
|---|---|---|
| 591 | `free(request_id)` | `free_blocks(reversed(blocks))`，注释 `tail blocks are freed first`。对无哈希的块，反转后再 prepend，结果是尾块排在队头；对有哈希的块，反转后 append，链尾在前、先被驱逐，链头（最可能被别人命中的前缀）留得最久 |

## 4. 分配入口（vllm/v1/core/kv_cache_manager.py）

| 行 | 代码 | 说明 |
|---|---|---|
| 210 | `self.watermark_blocks = int(watermark * num_blocks)` | 水位，默认 0.0（config/scheduler.py:196） |
| 379 | `allocate_slots(...)` | 调度器每次给请求加 token 都走这里；docstring 画了 computed / new_computed / external / new / lookahead 几段的布局 |
| 517–524 | 水位只对 WAITING / PREEMPTED 生效，且本步已经排了别的请求 | running 请求续块不受水位限制；空闲系统里第一条请求也不受限制 |
| 530–547 | `full_sequence_must_fit` | 接收 waiting 请求时先按整条 prompt 算块数，放不下就不收。调用方传 `scheduler_reserve_full_isl`（config/scheduler.py:190，默认 True；scheduler.py:1449），防止 chunked prefill 下只看第一段就收进来、后面再抢占 |
| 563 | `remove_skipped_blocks` | 滑动窗口外的块先释放、换成 null block，再算要几块，减少驱逐 |
| 583–587 | `available = free − reserved_blocks`；`required = n + watermark_blocks` | 不够就返回 None，调度器据此抢占（running）或停止接收（waiting） |

## 5. 抢占（vllm/v1/core/sched/scheduler.py）

| 行 | 代码 | 说明 |
|---|---|---|
| 911–975 | running 循环 `allocate_slots` 返回 None | 默认 FCFS：抢 `running[-1]`（最后进来的）；priority 策略抢优先级最低的。被抢的若就是自己，本步不排 |
| 1799 | `_preempt_request` | 释放全部块，`num_computed_tokens = 0`，状态 PREEMPTED，放回 waiting 队头。V1 只有重算，没有 swap（`swap_space` / `preemption_mode` 已删除） |

## 6. worker 侧块表（vllm/v1/worker/block_table.py）

| 行 | 代码 | 说明 |
|---|---|---|
| 29 | `get_block_table_width` | 宽度按 128 token 对齐：`block_alignment = 128 // gcd(128, bs)`。bs 16 时 8 块对齐 |
| 44 | `SlotMappingMode` | `TOKEN_TO_KV_SLOT`；Mamba/GDN 组用 `NONE`，块表当状态索引用 |
| 49 | `class BlockTable` | `block_table` int32 `[max_num_reqs, width]`，CPU pinned + GPU 各一份；`slot_mapping` int64 `[max_num_batched_tokens]` |
| 153 | `compute_slot_mapping` | 调 Triton kernel，在 GPU 上算 |
| 181 | `commit_block_table` | 只拷贝前 num_reqs 行到 GPU |
| 208 | `class MultiGroupBlockTable` | 每个 KV 组一张表（混合模型、滑动窗口 + 全注意力） |
| 305–375 | `ComputeSlotMappingKernel` | grid = num_reqs + 1。每个请求一个 program：`slot = block_table[row, pos // vbs] * bs + local_offset`，`vbs = bs × CP world size`，不属于本 rank 的位置写 PAD_ID。最后一个 program 把 `[num_tokens, max_num_tokens)` 填 PAD_ID，注释 `for CUDA graph compatibility` |

## 7. 写 KV 与读 KV

| 位置 | 代码 | 说明 |
|---|---|---|
| vllm/v1/attention/backends/utils.py:54 | `PAD_SLOT_ID = -1` | |
| csrc/cache_kernels.cu:279/350/444/510/560 | `if (slot_idx < 0) return;` | reshape_and_cache 系列 kernel 跳过填充位，CUDA graph 按最大 token 数捕获也不会写坏 KV |
| v1/attention/backends/flash_attn.py:1202 | kv_cache 形状 `[num_blocks, num_kv_heads, block_size, 2*head_size]` | |
| flash_attn.py:357 | `get_supported_kernel_block_sizes` | `MultipleOf(16)`，XPU `MultipleOf(64)`，FA4 另有限制 |
| flash_attn.py:372 | `get_preferred_block_size` | |
| config/cache.py:185 | `DEFAULT_BLOCK_SIZE = 16` | |
| platforms/interface.py:754 | `update_block_size_for_backend` | 后端不支持当前块大小时改掉，日志 `Setting kv cache block size to %d` |
| worker/utils.py:630 | `prepare_kernel_block_sizes` | 管理器块大小与 kernel 块大小不同时，把一个逻辑块拆成多个 kernel 块（虚拟块） |
| gpu_model_runner.py:6914 | `initialize_kv_cache_tensors` | 按 KVCacheConfig 一次性分配整个池 |

## 8. SGLang 对照

| 位置 | 代码 | 说明 |
|---|---|---|
| srt/mem_cache/memory_pool.py:376 | `class ReqToTokenPool` | `req_to_token` int32 `[(size+1), max_context_len]`，第 0 行是 CUDA graph 填充 batch 的哑行；`free_slots = range(1, size+1)` |
| srt/mem_cache/allocator/token.py:28 | `TokenToKVPoolAllocator` | page_size 1：`free_pages = arange(1, size+1)` int64，槽位 0 是哑写目标 |
| token.py:54–63 | `alloc` | 切 `free_pages[:need]`，没有链表 |
| token.py:65–75 | `free` | `torch.cat` 接回队尾；`need_sort`（PD 等场景）时先进 `release_pages`，下次分配前 `merge_and_sort_free`（allocator/base.py:227）排序 |
| srt/mem_cache/allocator/paged.py:117–128 | `alloc_decode_naive` | `(seq_lens − 1) % page_size == 0` 时才要新页 |
| paged.py:136 / 204 | `PagedTokenToKVPoolAllocator.alloc_extend` | 三段：补齐上一页剩余、整页、新页里的尾巴 |
| srt/arg_groups/overrides.py:1341 | `_page_size_default` | CUDA 默认 1；MUSA 64；HIP vectorized_5d 64。FlashMLA 强制 64，TRT-LLM MLA / CuteDSL MLA 32/64，TRT-LLM MHA 16/32/64/128 否则 64，FA4 非 MLA 128 |
| srt/mem_cache/kv_cache_configurator.py:3119 | `resolve_max_num_reqs` | `clamp(int(cap / context_len × 512), 2048, 4096)`，再 `min(·, cap // 2)`；`max_context_len = context_len + 4 + 草稿 token` |
| srt/managers/schedule_batch.py:3458 | `retract_decode` | KV 不够时撤回：输出最少的先撤，同样多时输入更长的先撤；配合 `new_token_ratio`（初值 0.7，600 步衰减到 0.7×0.14）估未来需求 |
