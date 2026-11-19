# 一条流式 chat 请求的调用链（vLLM 0d6f92a，单卡默认配置）

`python3 sim_processes.py --trace` 会打印同样的列表，并附带 GitHub 行号链接。

## 1. 23 步

| # | 进程 / 线程 | 位置 | 做什么 |
|---|---|---|---|
| 1 | API server（uvloop） | `vllm/entrypoints/openai/chat_completion/api_router.py:59` | `create_chat_completion`：FastAPI 路由 |
| 2 | API server | `vllm/entrypoints/openai/chat_completion/serving.py:287` | `create_chat_completion`：校验、取 lora / 模型名 |
| 3 | API server | `vllm/entrypoints/openai/chat_completion/serving.py:302` | `_create_chat_completion`：`render_chat_request` 套 chat template 并 tokenize |
| 4 | API server | `vllm/v1/engine/async_llm.py:726` | `AsyncLLM.generate`：建 `RequestOutputCollector` |
| 5 | API server | `vllm/v1/engine/async_llm.py:325` | `check_admission`：超过 `--max-num-queued-reqs` / `--max-num-queued-tokens` 时返回 503 |
| 6 | API server | `vllm/v1/engine/input_processor.py:424` | `process_inputs`：`arrival_time = time.time()`，组 `EngineCoreRequest` |
| 7 | API server | `vllm/v1/engine/async_llm.py:567` | `_add_request`：在 `OutputProcessor` 里登记，再 `engine_core.add_request_async` |
| 8 | API server | `vllm/v1/engine/core_client.py:1771` | DP>1 且内部 LB 时 `get_core_engine_for_request` 选 rank；然后 ROUTER 发出 msgpack 帧 |
| 9 | EngineCore 输入线程 | `vllm/v1/engine/core.py:1990` | `process_input_sockets`：DEALER 收帧，msgpack 解码 |
| 10 | EngineCore 输入线程 | `vllm/v1/engine/core.py:1168` | `preprocess_add_request`：建 `Request`、算块哈希、启动 grammar 编译，放进 `input_queue` |
| 11 | EngineCore 主线程 | `vllm/v1/engine/core.py:1669` | `run_busy_loop`：`_process_input_queue` → `_process_engine_step` |
| 12 | EngineCore 主线程 | `vllm/v1/core/sched/scheduler.py:2929` | `Scheduler.add_request`：进 waiting，记 `QUEUED`（2960 行） |
| 13 | EngineCore 主线程 | `vllm/v1/engine/core.py:715` | `step_with_batch_queue`：schedule → `execute_model(non_block=True)` |
| 14 | EngineCore 主线程 | `vllm/v1/core/sched/scheduler.py:699` | `Scheduler.schedule`：分 token 预算；744 行取 `scheduled_timestamp`，1555 行记 `SCHEDULED` |
| 15 | EngineCore 主线程 | `vllm/v1/executor/multiproc_executor.py:406` | `execute_model`：写入 `rpc_broadcast_mq`（uni executor 时直接在本进程调用） |
| 16 | Worker | `vllm/v1/executor/multiproc_executor.py:1154` | `worker_busy_loop`：`dequeue` 出 `(method, args, kwargs, output_rank)` |
| 17 | Worker | `vllm/v1/worker/gpu_worker.py:1337` | `Worker.execute_model`：PP 时收发中间张量 |
| 18 | Worker | `vllm/v1/worker/gpu/model_runner.py:1794` | MRV2 `execute_model`：更新请求状态 → prepare_inputs → forward |
| 19 | Worker | `vllm/v1/worker/gpu/model_runner.py:2181` | `sample_tokens`：采样，返回 `AsyncModelRunnerOutput`；`WorkerAsyncOutputCopy` 线程等 D2H 完成再回 |
| 20 | EngineCore 主线程 | `vllm/v1/core/sched/scheduler.py:2260` | `update_from_output`：追加 token、检查停止条件；`EngineCoreOutputs` 构造时打 `timestamp`（`vllm/v1/engine/__init__.py:321`） |
| 21 | EngineCore 输出线程 | `vllm/v1/engine/core.py:2099` | `process_output_sockets`：msgpack `encode_into` → PUSH |
| 22 | API server | `vllm/v1/engine/async_llm.py:876` | `output_handler`：PULL 收到后按 128 条一片调 `process_outputs`，片间 `await asyncio.sleep(0)` |
| 23 | API server | `vllm/v1/engine/output_processor.py:728` → `serving.py:472` | 算指标、增量 detokenize、放进 collector；`chat_completion_stream_generator` 拼 SSE chunk |

## 2. 五个时间戳

| 字段 | 进程 | 时钟 | 写入位置 | 用在哪 |
|---|---|---|---|---|
| `arrival_time` | API server | `time.time()`（墙钟） | `input_processor.py:424`，经 `output_processor.py:270` 存进 `RequestStateStats` | TTFT、E2E 的起点 |
| `queued_ts` | EngineCore | `time.monotonic()` | `scheduler.py:2960` 记 `QUEUED` 事件；`stats.py:574` 取出 | `request_queue_time = scheduled_ts − queued_ts` |
| `scheduled_ts` | EngineCore | `time.monotonic()` | `scheduler.py:744/1555`；`stats.py:577` 只记第一次，被抢占后重排不覆盖 | queue / prefill / inference 时间 |
| `first_token_ts` | EngineCore | `time.monotonic()` | `EngineCoreOutputs.timestamp`（`__init__.py:321`），`stats.py:553` | `prefill_time = first_token_ts − scheduled_ts` |
| `last_token_ts` | EngineCore | `time.monotonic()` | 同上，每批更新（`stats.py:558`） | ITL、`decode_time` |

TTFT 本身不是用这五个字段相减得到的：`first_token_latency = IterationStats.iteration_timestamp − arrival_time`（`stats.py:525`），两个值都是 **API server 进程里的 `time.time()`**。`iteration_timestamp` 是 `output_handler` 每收到一批输出时新建 `IterationStats` 记下的（`stats.py:478`）。所以 Prometheus 的 TTFT 包含 ZMQ 回程和 API server 排队处理输出的时间；queue / prefill / decode 三段全部是 EngineCore 进程的单调时钟，彼此可以相减，但不能和 `arrival_time` 相减（源码注释：`timestamps should not be compared with timestamps from other processes`，`__init__.py:213`）。
