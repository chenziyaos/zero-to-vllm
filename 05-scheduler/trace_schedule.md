# 一次 schedule() 调用：逐行对照

基线 vllm 主干 0d6f92a。文件默认 `vllm/v1/core/sched/scheduler.py`。

## 0. 入口与预算

| 行 | 代码 | 说明 |
|---|---|---|
| 699 | `def schedule(self, throttle_prefills=False)` | EngineCore 每步调一次 |
| 701–710 | NOTE(woosuk) | 没有 prefill/decode 阶段，只有 `num_computed_tokens` 追 `num_tokens_with_spec` |
| 719 | `token_budget = self.max_num_scheduled_tokens` | 默认等于 `max_num_batched_tokens`（139–143 行） |
| 722 | `input_budget = max_num_batched_tokens` | 与 draft slot 一起算，投机解码时比 token_budget 先耗尽 |
| 757–769 | `long_prefill_token_threshold` | 只在合格请求 > 1 时生效；adaptive 模式下限是 `input_budget // n` |
| 780–791 | `defer_prefills` | `prefill_schedule_interval > 1` 时，非整数倍步不跑任何 prefill |

## 1. running 循环（795–1036）

| 行 | 分支 | 说明 |
|---|---|---|
| 795 | `while req_index < len(self.running) and token_budget > 0` | running 永远先于 waiting |
| 800–814 | 异步调度到达 max_tokens | 占位符已经顶到上限，跳过 |
| 839–843 | `num_new_tokens = num_tokens_with_spec + num_output_placeholders − num_computed_tokens` | 统一公式：decode 是 1（+spec），未完成的 chunk 是剩余 prompt |
| 844–848 | threshold、`token_budget`、`input_budget − draft_slots` | 三重截断 |
| 852–857 | `max_model_len − num_computed_tokens − num_sampled_tokens_per_step` | 给采样留位置 |
| 891–909 | `num_new_tokens == 0 → continue` | 注释：“we do not strictly follow the FCFS” |
| 913–922 | `allocate_slots(request, num_new_tokens, num_lookahead_tokens)` | 成功就跳出 while |
| 932–938 | 选受害者 | PRIORITY：`max(running, key=(priority, arrival_time))`；FCFS：`running[-1]` |
| 950–968 | PRIORITY 归还预算 | 受害者本步已排过，退回它的 token、块和 encoder 预算 |
| 970 | `self.running.pop()` | FCFS 弹队尾 |
| 972–977 | `_preempt_request(...)` | 见第 3 节 |
| 978–980 | 受害者就是自己 → break | 自己也放不下了 |
| 982–994 | `new_blocks is None` | FCFS 直接 break，后面的 running 本步都不排；PRIORITY 且自己被抢就 continue |
| 997–1004 | 记账 | `token_budget -= num_new_tokens` |

## 2. waiting 循环（1050–）

| 行 | 分支 | 说明 |
|---|---|---|
| 1050 | `if not preempted_reqs and ...` | 本步抢过人，就不收新请求 |
| 1062–1068 | 队列选择 | 先排持有 KV 的 `kv_holding_waiting`（异步 KV 加载完成的），再排 `waiting` |
| 1073 | `num_running >= max_num_active_reqs → break` | 默认等于 `max_num_seqs`（134–138 行） |
| 1306 | `num_new_tokens = request.num_tokens − num_computed_tokens` | 用 `num_tokens` 而不是 prompt，是为了覆盖被抢占后恢复的请求 |
| 1311–1330 | spec decode padding | 让新 decode 请求凑满 1+k，保住 full CUDA graph |
| 1332–1333 | threshold | 同 running |
| 1337–1344 | 关了 chunked prefill 且放不下 → break | |
| 1435–1452 | `allocate_slots(..., full_sequence_must_fit=scheduler_reserve_full_isl, has_scheduled_reqs=bool(self.running))` | 整条 prompt 放不下就不收 |

## 3. 抢占（1799–1843）

```
_free_request_blocks(request)          # 块还给 BlockPool（有前缀缓存时哈希还在，可能被命中）
request.status = PREEMPTED
request.num_computed_tokens = 0        # V1 只有重算，没有 swap
request.num_stale_output_tokens = request.num_in_flight_tokens   # 异步：已发出那一步的输出照样交付
request.num_output_placeholders = 0
request.num_preemptions += 1
record_event(PREEMPTED)
self.waiting.prepend_request(request)  # FCFS 队列 appendleft（request_queue.py:92–94）
self.reset_preempted_req_ids.add(...)  # 告诉 worker 丢掉这条请求的持久状态
```

## 4. 调度之后（1845–）

`_update_after_schedule`：`num_computed_tokens += num_scheduled_tokens`。推进发生在 forward 之前，所以下一步可以立刻接着排同一个请求的下一段 chunk。异步调度（`async_scheduler.py:24–55`）再给非 prefill-chunk 请求加 `num_output_placeholders += 1(+spec)`。

## 5. KV 侧：allocate_slots（vllm/v1/core/kv_cache_manager.py）

| 行 | 说明 |
|---|---|
| 210 | `watermark_blocks = int(watermark × num_blocks)` |
| 517–524 | 水位只对 WAITING/PREEMPTED 且已有请求在跑时生效 |
| 530–547 | `full_sequence_must_fit`：整条序列需要的块 + 水位 > 空闲块就返回 None |

## 6. SGLang 对照（8fabf5e，`python/sglang/srt/managers/`）

| 位置 | 说明 |
|---|---|
| `schedule_policy.py:617` `PrefillAdder` | `rem_total_tokens` 先扣掉 running 请求的预留：`min(max_new − 已生成, 4096) × new_token_ratio` |
| `schedule_policy.py:891` | 放不下 → `AddReqResult.NO_TOKEN`，本步不再收 |
| `schedule_batch.py:3439` `check_decode_mem` | decode 前检查本步要的槽位 |
| `schedule_batch.py:3458` `retract_decode` | 按 `(len(output_ids), −len(origin_input_ids))` 倒序从尾部弹，至少留 1 条 |
| `scheduler.py:4524` `update_running_batch` | 打印 `KV cache pool is full. Retract requests.` 并重设 `new_token_ratio` |
