# Chunked prefill 与异步调度：逐行对照

基线 vllm 主干 0d6f92a，sglang 主干 8fabf5e。

## 1. 配置（vllm/config/scheduler.py、vllm/engine/arg_utils.py）

| 位置 | 代码 | 说明 |
|---|---|---|
| scheduler.py:48 | `max_num_batched_tokens` | 每步 token 预算；默认值由 arg_utils 按显卡决定 |
| arg_utils.py:2976–3011 | 按显存分档 | ≥160 GB（B200/B300）：16384 / 16384；≥70 GB 且非 A100（H100/H200）：离线 `LLM` 16384、`vllm serve` 8192；其他：8192 / 2048。注释引用 PR #17885：A100 上调大反而掉吞吐 |
| scheduler.py:79 | `long_prefill_token_threshold = 0` | 0 表示不限；只有一条请求时不生效 |
| scheduler.py:86 | `long_prefill_token_threshold_adaptive = False` | 打开后阈值下限为 `max_num_batched_tokens / 请求数` |
| scheduler.py:134 | `enable_chunked_prefill = True` | V1 默认开启 |
| scheduler.py:203 | `prefill_schedule_interval = 1` | 每 N 步才放一次 prefill；被推迟的步连已收请求的后续 chunk 也不跑 |
| scheduler.py:211 | `async_scheduling = None` | None 时由 vllm.py:1701–1735 判断：pooling 模型、不支持的 executor 等情况关闭，其余默认开启 |
| scheduler.py:309–313 | 启动日志 | `Chunked prefill is enabled with max_num_batched_tokens=%d.` |
| scheduler.py:317–330 | 关 chunked 的校验 | 预算 < max_model_len 直接报错 |
| scheduler.py:332–337 | 预算 ≥ max_num_seqs | 否则每条 decode 分不到 1 个 token |

## 2. 调度（vllm/v1/core/sched/scheduler.py）

| 行 | 代码 | 说明 |
|---|---|---|
| 747–751 | `defer_prefills = throttle_prefills and not prefill_capacity_bound ...` | DP 均衡：非节拍步不跑 prefill，除非已经积压 |
| 753–769 | 阈值生效条件 | 合格请求 > 1 才截断；adaptive 时阈值下限为 `input_budget // n` |
| 771–791 | 本地 prefill 节拍 | `prefill_schedule_interval > 1` 时，上次带 prefill 的步之后 N−1 步只跑 decode；waiting 排不完时放开（#44558） |
| 795 | running 循环 | decode 和未切完的 chunk 都在这里，先于 waiting 拿预算 |
| 822 | `if defer_prefills and request.is_prefill_chunk` | 推迟步里，已经在 running 的 chunk 也跳过 |
| 839–848 | `num_new_tokens` 统一公式 + 截断 | 长 prompt 的下一段在这里被预算切开 |
| 1306 | `num_new_tokens = request.num_tokens − num_computed_tokens` | waiting 侧的公式 |
| 1332–1333 | 阈值截断 | 同 running |
| 1337–1344 | 关 chunked 且放不下 → break | 整条 prompt 必须一步装下 |
| 1863–1865 | `is_prefill_chunk = num_computed_tokens < num_tokens + num_output_placeholders` | 本步排完还没追上，说明下一步还要接着切 |

## 3. 引擎循环（vllm/v1/engine/core.py、vllm/config/vllm.py）

| 位置 | 说明 |
|---|---|
| core.py:674 `step()` | 同步：schedule → execute_model(non_block) → `future.result()` 阻塞 → update_from_output。CPU 和 GPU 串行 |
| core.py:715 `step_with_batch_queue()` | 异步：队列没满就继续 schedule 下一步并立刻返回（742–776）；满了才阻塞在最老的那一批（785） |
| vllm.py:603 `max_concurrent_batches` | 异步时 V1 runner 为 2（pp ≤ 1），MRV2 为 pp + 1；不开异步时为 pp_size |
| async_scheduler.py:24–55 | 非 prefill chunk 的请求 `num_output_placeholders += 1(+spec)`，让下一步在输出没回来时就能排它 |

## 4. Worker 侧（vllm/v1/worker、vllm/v1/attention）

| 位置 | 说明 |
|---|---|
| gpu_model_runner.py:2062 | `discard_request_mask = optimistic_seq_lens < num_tokens`：本步只算了 prompt 中间一段的请求照样走采样，但采出来的 token 丢掉 |
| attention/backends/utils.py:955 `reorder_batch_to_split_decodes_and_prefills` | 把一个 batch 重排成 decode → short_extend → long_extend → prefill 四段，注意力后端分别走 decode kernel 和 prefill kernel |
| attention/backends/utils.py:853 `split_decodes_and_prefills` | 按 `reorder_batch_threshold` 找分界点 |

## 5. SGLang 对照（python/sglang/srt/managers/）

| 位置 | 说明 |
|---|---|
| scheduler.py:1379–1381 | `is_mixed_chunk = chunked_prefill_size is not None and enable_mixed_chunk`，默认关闭 |
| scheduler.py:1403 `_should_defer_prefill` | `prefill_decode_interval`，默认 None |
| scheduler.py:2029 `event_loop_normal` | 不重叠 |
| scheduler.py:2064 `event_loop_overlap` | 默认：先 launch 本批，再处理上一批的结果（result_queue） |
| scheduler.py:2139 `is_disable_overlap_for_batch` | 连续两个 prefill batch 时可关闭重叠（`SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`，默认 False） |
| scheduler.py:4053 | `# Run prefill first if possible`：有新 prefill batch 就跑它，否则跑 decode |
| scheduler.py:4441–4470 | mixed chunk：`check_decode_mem` 通过才 `mix_with_running`；内存不够时 decode 本步不参加，也不 retract |
| schedule_policy.py:617 `PrefillAdder` | `rem_chunk_tokens = chunked_prefill_size − num_mixed_decode_tokens` |
| schedule_policy.py:1150 `add_chunked_req` | 同一时刻只有一条 `chunked_req` |
| schedule_batch.py:3274 `mix_with_running` | 拼接后 `forward_mode = MIXED` |
| arg_groups/memory_hook.py:93–170 | `chunked_prefill_size` 默认按显存分档：<20 GB 2048，<60 GB 4096，<160 GB 8192（H100 80GB 落在这一档），≥160 GB 16384 |
