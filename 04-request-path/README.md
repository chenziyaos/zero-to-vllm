# VLLM-04 请求旅程

对应文档：VLLM-04｜原理｜请求旅程：API server、EngineCore 与 GPU worker

| 文件 | 运行环境 | 做什么 |
|---|---|---|
| `sim_processes.py` | 仅 Python 3 标准库 | 输入 `vllm serve` 的并行参数（TP/PP/DP/EP、LB 模式、节点数），按 `vllm/entrypoints/cli/serve.py` 的分派逻辑打印：会起哪些进程、各叫什么名字、前端用哪种 client、EngineCore 和 worker 之间用什么通道 |
| `trace_request.md` | 阅读 | 一条流式 chat 请求经过的 23 个函数（带行号），以及五个时间戳分别在哪个进程、用哪种时钟记录 |

源码基线：vllm 主干 0d6f92a；SGLang 对照 sgl-project/sglang 主干 8fabf5e。

## 用法

```bash
python3 sim_processes.py                                    # 单卡默认
python3 sim_processes.py --tp 2
python3 sim_processes.py --dp 4                             # 内部 LB，4 个 API server
python3 sim_processes.py --dp 4 --lb external --dp-rank 0   # 外部 LB，每个 rank 一个 vllm serve
python3 sim_processes.py --dp 8 --moe --ep --lb external --dp-rank 0
python3 sim_processes.py --tp 4 --pp 2 --nnodes 2           # 跨节点 TP×PP，主节点
python3 sim_processes.py --tp 4 --pp 2 --nnodes 2 --node-rank 1   # 从节点
python3 sim_processes.py --dp 4 --lb hybrid --nnodes 2 --dp-local 2
python3 sim_processes.py --trace                            # 打印调用链和 GitHub 行号链接
```

## 期望输出

单卡默认：2 个进程。

```
== 进程树（vLLM 0d6f92a，本节点 node_rank=0）
vllm serve 主进程            uvloop.run(run_server(args))（API server 就是 vllm serve 主进程）
│  （HTTP、tokenize、detokenize 都在这个进程里）
└─ VLLM::EngineCore           调度器 + executor；busy loop + 2 个 I/O 线程
      （uni executor：模型就在 EngineCore 进程里，占 GPU 0）

== 通道
前端 → EngineCore：每个前端 1 个 ROUTER（输入）+ 1 个 PULL（输出），共 1 对，地址 ipc://
  EngineCore 侧：输入线程 DEALER 连到每个前端的 ROUTER；输出线程 PUSH 连到每个前端的 PULL
  编码：msgspec msgpack，大张量走带外帧（recv_multipart(copy=False)）
前端客户端：AsyncMPClient
executor：uni；world_size=1（TP 1 × PP 1 × PCP 1），DP=1
batch_queue_size=2 → EngineCore.step_fn = step_with_batch_queue（async scheduling + MRV2）
```

`--tp 2`：EngineCore 下多出两个 worker 进程，中间是共享内存广播队列。

```
└─ VLLM::EngineCore           调度器 + executor；busy loop + 2 个 I/O 线程
      ├─ VLLM::Worker_TP0               GPU 0  ← output_rank，只有它回结果
      └─ VLLM::Worker_TP1               GPU 1
EngineCore → workers：rpc_broadcast_mq = MessageQueue(n_reader=2, n_local_reader=2)
  本地 reader：共享内存环形缓冲，10 个 chunk × VLLM_MQ_MAX_CHUNK_BYTES_MB(16 MiB)，每个 chunk 1 个写标志 + 每个 reader 1 个读标志
workers → EngineCore：每个 worker 一个 worker_response_mq；execute_model/sample_tokens 只等 rank 0
```

`--dp 4`：4 个 API server 共享一个监听端口（SO_REUSEPORT），外加一个 DPCoordinator。

```
vllm serve 主进程            run_multi_api_server（SO_REUSEPORT 共享监听 socket）
├─ VLLM::APIServer_0 … VLLM::APIServer_3
├─ VLLM::DPCoordinator         汇总各 rank 的 waiting/running/kv 用量，每 100 ms 推给前端
├─ VLLM::EngineCore_DP0 … EngineCore_DP3（各带一个 Worker_DP{r}）
前端 → EngineCore：每个前端 1 个 ROUTER（输入）+ 1 个 PULL（输出），共 4 对，地址 ipc://
前端客户端：DPLBAsyncMPClient（前端按 score 在多个 EngineCore 间选路）
```

`--dp 4 --lb external --dp-rank 0`：只有一个 EngineCore_DP0，没有 coordinator，client 是 `DPAsyncMPClient`。

`--dp 8 --moe --ep --lb external --dp-rank 0`：外部 LB 下 MoE 仍然要 coordinator（负责 `START_DP_WAVE`），worker 名带 `_EP0`。

`--tp 4 --pp 2 --nnodes 2`：主节点 4 个 worker，`MessageQueue(n_reader=8, n_local_reader=4)`，另外 4 个远端 reader 走 ZMQ；output_rank 是 4（最后一个 PP stage 的第一个 TP rank）；`batch_queue_size=3`。

`--tp 4 --pp 2 --nnodes 2 --node-rank 1`：从节点没有 API server、没有 EngineCore。

```
vllm serve 主进程            run_headless → node_rank_within_dp > 0：只起 MultiprocExecutor，没有 API server 也没有 EngineCore
├─ VLLM::Worker_PP1_TP0           GPU 0  ← output_rank，结果经 socket 回到 node 0 的 EngineCore
├─ VLLM::Worker_PP1_TP1 … Worker_PP1_TP3
从 node 0 的 rpc_broadcast_mq 以远端 reader 身份订阅（ZMQ socket），本节点不建共享内存广播队列
```

`--dp 4 --lb hybrid --nnodes 2 --dp-local 2`：本节点 2 个 API server、2 个 EngineCore，ZMQ 地址变成 `tcp://`，前端只在本节点的两个 rank 之间选路。

`--no-multiproc`：脚本会提示 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 对 `vllm serve` 无效。`AsyncLLM` 总是调用 `make_async_mp_client`（async_llm.py:193），这个开关只影响离线 `LLM()`。

## 读法

- 数进程：单卡 2 个；TP=N 时 2+N 个；内部 LB 的 DP=N 时 1 + N（API server）+ 1（coordinator）+ N（EngineCore）+ N（worker）。`ps -ef | grep VLLM::` 应该和脚本对得上（进程名前缀可用 `VLLM_PROCESS_NAME_PREFIX` 改）。
- API server 和 EngineCore 之间永远是 ZMQ，EngineCore 和 worker 之间是共享内存环形缓冲（同节点）或 ZMQ（跨节点）。
- 只有 output_rank 回结果，其余 worker 的 response 队列只在 collective_rpc 时才读。
- 用真机核对：起服务后执行 `ps -eo pid,ppid,comm | grep -E "VLLM|python"`，再对照 `trace_request.md` 用 `py-spy dump --pid <EngineCore pid>` 看 busy loop 停在哪。
