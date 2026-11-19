"""给定 vllm serve 的并行参数，推出会起哪些进程、它们之间用什么通道连起来。

只依赖标准库，规则逐条对照 vLLM 主干 0d6f92a：
  - api_server_count 默认值与分派：vllm/entrypoints/cli/serve.py ServeSubcommand.cmd
  - 是否起 DPCoordinator：vllm/config/vllm.py needs_dp_coordinator + vllm/v1/engine/utils.py launch_core_engines
  - ZMQ 地址用 IPC 还是 TCP：vllm/v1/engine/utils.py get_engine_zmq_bind_addresses
  - 前端客户端类型：vllm/v1/engine/core_client.py EngineCoreClient.make_async_mp_client
  - executor 后端：vllm/config/parallel.py（world_size_across_dp > 1 默认 mp，单卡 uni）
  - worker 进程名：vllm/v1/executor/multiproc_executor.py WorkerProc（_DP/_PP/_PCP/_TP/_DCP/_EP 后缀）
  - output_rank：MultiprocExecutor._get_output_rank = world_size - tp * pcp
  - batch_queue_size：vllm/config/vllm.py max_concurrent_batches

用法：
    python3 sim_processes.py                              # 单卡 Qwen3-8B：2 个进程
    python3 sim_processes.py --tp 2                       # 加上 2 个 worker 和一条共享内存广播队列
    python3 sim_processes.py --dp 4                       # 内部负载均衡：4 个 API server + DPCoordinator
    python3 sim_processes.py --dp 4 --lb external         # 外部 LB：每个 rank 一个 API server，不起 coordinator
    python3 sim_processes.py --dp 8 --tp 1 --moe --lb external   # MoE 即使外部 LB 也要 coordinator
    python3 sim_processes.py --tp 4 --pp 2 --nnodes 2     # 跨节点：远端 reader 走 socket
    python3 sim_processes.py --trace                      # 一条 chat 请求的调用链（文件:行号）
"""
import argparse

SHA = "0d6f92a"


def resolve(a):
    """返回一个 dict，描述进程与通道。"""
    out = {"warnings": []}
    # ---- serve.py：LB 模式与 api_server_count
    lb = a.lb
    if a.headless:
        if a.api_server_count not in (None, 0):
            raise SystemExit("ValueError: --api-server-count 不能和 --headless 同时使用")
        api = 0
    else:
        api = a.api_server_count
    if api is None:
        if lb in ("multi-port", "external") or a.rust:
            api = 1
        elif lb == "hybrid":
            api = a.dp_local or 1
        else:
            api = a.dp
    elif a.rust and api > 1:
        out["warnings"].append(f"Rust 前端忽略 --api-server-count={api}，按 1 处理")
        api = 1
    if a.elastic_ep and api > 1:
        out["warnings"].append(f"Elastic EP 最多 1 个 API server，{api} → 1")
        api = 1
    follower = a.dp == 1 and a.nnodes > 1 and a.node_rank > 0
    out["follower"] = follower
    if follower:
        api = 0
        mode = "run_headless → node_rank_within_dp > 0：只起 MultiprocExecutor，没有 API server 也没有 EngineCore"
    elif lb == "multi-port":
        mode = "run_dp_supervisor（每个本地 DP rank 一个 APIServer_DP{r} 子进程）"
    elif api < 1:
        mode = "run_headless（只起 EngineCore，不监听 HTTP）"
    elif api > 1 or a.rust:
        mode = "run_multi_api_server（SO_REUSEPORT 共享监听 socket）"
    else:
        mode = "uvloop.run(run_server(args))（API server 就是 vllm serve 主进程）"
    out["api"] = api
    out["mode"] = mode

    # ---- 本节点有几个 EngineCore
    dp_local = a.dp_local or a.dp
    if lb in ("external",):
        dp_local = 1  # 每个 vllm serve 只管自己的 --data-parallel-rank
    out["dp_local"] = dp_local
    # ---- coordinator
    needs = a.dp > 1 and (a.moe or lb != "external")
    # coordinator 只在 DP rank 0 所在节点（外部 LB 时是 --data-parallel-rank 0 的那个 serve）
    out["coordinator"] = needs and a.dp_rank == 0 and a.dp_start == 0
    # ---- 前端客户端
    if a.dp > 1 and lb in ("external", "multi-port"):
        out["client"] = "DPAsyncMPClient（每个前端只连一个 EngineCore）"
    elif a.dp > 1:
        out["client"] = "DPLBAsyncMPClient（前端按 score 在多个 EngineCore 间选路）"
    else:
        out["client"] = "AsyncMPClient"
    if not a.multiproc:
        # AsyncLLM.__init__ 直接调 make_async_mp_client（async_llm.py:193），不看这个开关；
        # make_client 里 asyncio_mode 且非多进程会抛 NotImplementedError（core_client.py:115）。
        # VLLM_ENABLE_V1_MULTIPROCESSING=0 只影响离线 LLM / LLMEngine（llm_engine.py:170）。
        print("注意：VLLM_ENABLE_V1_MULTIPROCESSING=0 对 vllm serve 无效（AsyncLLM 总是多进程），"
              "只有离线 LLM() 会退化成 InprocClient。下面按默认多进程输出。\n")
        a.multiproc = True
    # ---- ZMQ 地址
    client_local_only = lb == "external" or dp_local == a.dp
    out["zmq"] = "ipc://" if client_local_only else "tcp://<data-parallel-address>:<随机端口>"
    # ---- executor
    world = a.tp * a.pp * a.pcp
    out["world"] = world
    if world * a.dp > 1:
        backend = "mp"
    else:
        backend = "uni"
    if a.backend:
        backend = a.backend
    out["backend"] = backend
    # DP>1 时跨节点的是 DP rank，每个 EngineCore 的 TP×PP 组仍在一个节点内；DP=1 时才是 TP×PP 组跨节点
    out["local_world"] = world // a.nnodes if (a.nnodes > 1 and a.dp == 1) else world
    out["output_rank"] = world - a.tp * a.pcp
    # ---- batch queue
    if a.async_sched:
        if not a.v1_runner:
            bq = a.pp + 1
        elif a.pp <= 1:
            bq = 2
        else:
            bq = a.pp
    else:
        bq = a.pp
    out["batch_queue"] = bq
    out["step_fn"] = "step_with_batch_queue" if bq > 1 else "step"
    return out


def worker_name(dp, dp_r, pp, pp_r, pcp, pcp_r, tp, tp_r, ep, ep_r):
    n = "Worker"
    if dp > 1:
        n += f"_DP{dp_r}"
    if pp > 1:
        n += f"_PP{pp_r}"
    if pcp > 1:
        n += f"_PCP{pcp_r}"
    if tp > 1:
        n += f"_TP{tp_r}"
    if ep:
        n += f"_EP{ep_r}"
    return n


def show(a):
    r = resolve(a)
    P = a.prefix
    print(f"== 进程树（vLLM {SHA}，本节点 node_rank={a.node_rank}）")
    print(f"vllm serve 主进程            {r['mode']}")
    if r["api"] > 1 or (a.rust and r["api"] >= 1):
        for i in range(r["api"]):
            print(f"├─ {P}::APIServer_{i}")
    elif r["api"] == 1 and a.lb == "multi-port":
        for i in range(r["dp_local"]):
            print(f"├─ {P}::APIServer_DP{a.dp_start + i}")
    elif r["api"] == 1:
        print("│  （HTTP、tokenize、detokenize 都在这个进程里）")
    if r["coordinator"]:
        print(f"├─ {P}::DPCoordinator         汇总各 rank 的 waiting/running/kv 用量，每 100 ms 推给前端"
              + ("；MoE 还负责 START_DP_WAVE" if a.moe else ""))
    if not a.multiproc:
        print("（EngineCore 在本进程内，没有 ZMQ）")
    ep = a.moe and a.ep
    if r["follower"]:
        n_local = r["local_world"]
        for k in range(n_local):
            g = a.node_rank * n_local + k
            pp_r, rem = divmod(g, a.tp * a.pcp)
            pcp_r, tp_r = divmod(rem, a.tp)
            nm = worker_name(a.dp, 0, a.pp, pp_r, a.pcp, pcp_r, a.tp, tp_r, ep, g)
            tag = "  ← output_rank，结果经 socket 回到 node 0 的 EngineCore" if g == r["output_rank"] else ""
            print(f"{'└─' if k == n_local - 1 else '├─'} {P}::{nm:<24} GPU {k}{tag}")
        print()
        print("== 通道")
        print(f"从 node 0 的 rpc_broadcast_mq 以远端 reader 身份订阅（ZMQ socket），本节点不建共享内存广播队列")
        return
    for e in range(r["dp_local"]):
        dp_r = (a.dp_rank if a.lb == "external" else a.dp_start + e)
        title = f"EngineCore_DP{dp_r}" if a.dp > 1 else "EngineCore"
        last = e == r["dp_local"] - 1
        br = "└─" if last else "├─"
        if a.multiproc:
            print(f"{br} {P}::{title:<20} 调度器 + executor；busy loop + 2 个 I/O 线程")
        pad = "   " if last else "│  "
        if r["backend"] == "uni":
            print(f"{pad}   （uni executor：模型就在 EngineCore 进程里，占 GPU {e}）")
            continue
        n_local = r["local_world"]
        base = a.node_rank * n_local
        for k in range(n_local):
            g = base + k
            pcp_tp = a.tp * a.pcp
            pp_r, rem = divmod(g, pcp_tp)
            pcp_r, tp_r = divmod(rem, a.tp)
            ep_r = dp_r * r["world"] + g
            nm = worker_name(a.dp, dp_r, a.pp, pp_r, a.pcp, pcp_r, a.tp, tp_r, ep, ep_r)
            tag = "  ← output_rank，只有它回结果" if g == r["output_rank"] else ""
            gpu = e * n_local + k
            print(f"{pad}   {'└─' if k == n_local - 1 else '├─'} {P}::{nm:<24} GPU {gpu}{tag}")
    print()
    print("== 通道")
    if a.multiproc and r["api"] >= 1:
        n_fe = r["api"] if a.lb != "multi-port" else r["dp_local"]
        print(f"前端 → EngineCore：每个前端 1 个 ROUTER（输入）+ 1 个 PULL（输出），共 {n_fe} 对，地址 {r['zmq']}")
        print("  EngineCore 侧：输入线程 DEALER 连到每个前端的 ROUTER；输出线程 PUSH 连到每个前端的 PULL")
        print("  编码：msgspec msgpack，大张量走带外帧（recv_multipart(copy=False)）")
    if r["coordinator"]:
        print("DPCoordinator：EngineCore → coordinator 上报计数；coordinator → 前端 XPUB 发布 (waiting, running, kv_usage)")
    if r["backend"] == "mp":
        print(f"EngineCore → workers：rpc_broadcast_mq = MessageQueue(n_reader={r['world']}, n_local_reader={r['local_world']})")
        print("  本地 reader：共享内存环形缓冲，10 个 chunk × VLLM_MQ_MAX_CHUNK_BYTES_MB(16 MiB)，每个 chunk 1 个写标志 + 每个 reader 1 个读标志")
        if r["world"] > r["local_world"]:
            print(f"  远端 reader：{r['world'] - r['local_world']} 个，走 ZMQ socket（每个节点的 rank 0 之外不再转发）")
        print(f"workers → EngineCore：每个 worker 一个 worker_response_mq；execute_model/sample_tokens 只等 rank {r['output_rank']}")
        if a.async_sched:
            print("  output_rank 上：WorkerAsyncOutputCopy 线程等 D2H 拷贝完成再入队，主线程继续收下一步")
    print(f"前端客户端：{r['client']}")
    print(f"executor：{r['backend']}；world_size={r['world']}（TP {a.tp} × PP {a.pp} × PCP {a.pcp}），DP={a.dp}")
    print(f"batch_queue_size={r['batch_queue']} → EngineCore.step_fn = {r['step_fn']}"
          + ("（async scheduling + MRV2）" if a.async_sched and not a.v1_runner else ""))
    for w in r["warnings"]:
        print("WARN:", w)


TRACE = [
    ("API server", "vllm/entrypoints/openai/chat_completion/api_router.py", 59, "create_chat_completion：FastAPI 路由"),
    ("API server", "vllm/entrypoints/openai/chat_completion/serving.py", 302, "_create_chat_completion：render_chat_request 套模板 + tokenize"),
    ("API server", "vllm/v1/engine/async_llm.py", 726, "AsyncLLM.generate：建 RequestOutputCollector"),
    ("API server", "vllm/v1/engine/async_llm.py", 325, "check_admission：max_num_queued_reqs / tokens 超限返回 503"),
    ("API server", "vllm/v1/engine/input_processor.py", 424, "process_inputs：arrival_time = time.time()，组 EngineCoreRequest"),
    ("API server", "vllm/v1/engine/async_llm.py", 567, "_add_request：OutputProcessor 登记 + engine_core.add_request_async"),
    ("API server", "vllm/v1/engine/core_client.py", 1771, "DPLBAsyncMPClient.get_core_engine_for_request（DP>1 时选 rank）"),
    ("EngineCore 输入线程", "vllm/v1/engine/core.py", 1990, "process_input_sockets：DEALER 收帧 → msgpack 解码"),
    ("EngineCore 输入线程", "vllm/v1/engine/core.py", 1168, "preprocess_add_request：建 Request、算块哈希、启动 grammar 编译"),
    ("EngineCore 主线程", "vllm/v1/engine/core.py", 1669, "run_busy_loop：_process_input_queue → _process_engine_step"),
    ("EngineCore 主线程", "vllm/v1/core/sched/scheduler.py", 2929, "Scheduler.add_request：进 waiting，记 QUEUED 事件"),
    ("EngineCore 主线程", "vllm/v1/engine/core.py", 715, "step_with_batch_queue：schedule → execute_model(non_block)"),
    ("EngineCore 主线程", "vllm/v1/core/sched/scheduler.py", 699, "Scheduler.schedule：分 token 预算、记 SCHEDULED"),
    ("EngineCore 主线程", "vllm/v1/executor/multiproc_executor.py", 406, "MultiprocExecutor.execute_model：写入共享内存广播队列"),
    ("Worker", "vllm/v1/executor/multiproc_executor.py", 1154, "worker_busy_loop：dequeue (method, args, output_rank)"),
    ("Worker", "vllm/v1/worker/gpu_worker.py", 1337, "Worker.execute_model：PP 收发中间张量"),
    ("Worker", "vllm/v1/worker/gpu/model_runner.py", 1794, "MRV2 execute_model：更新请求状态 → prepare_inputs → forward"),
    ("Worker", "vllm/v1/worker/gpu/model_runner.py", 2181, "sample_tokens：采样，返回 AsyncModelRunnerOutput"),
    ("EngineCore 主线程", "vllm/v1/core/sched/scheduler.py", 2260, "update_from_output：追加 token、检查停止条件"),
    ("EngineCore 输出线程", "vllm/v1/engine/core.py", 2099, "process_output_sockets：msgpack encode_into → PUSH"),
    ("API server", "vllm/v1/engine/async_llm.py", 876, "output_handler：按 128 条一片调 process_outputs"),
    ("API server", "vllm/v1/engine/output_processor.py", 728, "process_outputs：算指标、增量 detokenize、放进 collector"),
    ("API server", "vllm/entrypoints/openai/chat_completion/serving.py", 472, "chat_completion_stream_generator：拼 SSE chunk"),
]


def trace():
    print(f"== 一条流式 chat 请求经过的函数（vLLM {SHA}，单卡默认配置）")
    last = None
    for proc, f, ln, what in TRACE:
        if proc != last:
            print(f"[{proc}]")
            last = proc
        print(f"  {f}:{ln:<5} {what}")
        print(f"    https://github.com/vllm-project/vllm/blob/{SHA}/{f}#L{ln}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--pcp", type=int, default=1, help="prefill context parallel")
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--dp-local", type=int, default=None)
    p.add_argument("--dp-start", type=int, default=0)
    p.add_argument("--dp-rank", type=int, default=0, help="外部 LB 时本进程的 --data-parallel-rank")
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--node-rank", type=int, default=0)
    p.add_argument("--api-server-count", type=int, default=None)
    p.add_argument("--lb", choices=["internal", "hybrid", "external", "multi-port"], default="internal")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--moe", action="store_true")
    p.add_argument("--ep", action="store_true", help="--enable-expert-parallel")
    p.add_argument("--elastic-ep", action="store_true")
    p.add_argument("--rust", action="store_true", help="VLLM_USE_RUST_FRONTEND=1")
    p.add_argument("--backend", choices=["mp", "uni", "ray"], default=None)
    p.add_argument("--no-async", dest="async_sched", action="store_false")
    p.add_argument("--v1-runner", action="store_true", help="关掉 MRV2，用 V1 model runner")
    p.add_argument("--no-multiproc", dest="multiproc", action="store_false", help="VLLM_ENABLE_V1_MULTIPROCESSING=0")
    p.add_argument("--prefix", default="VLLM", help="VLLM_PROCESS_NAME_PREFIX")
    p.add_argument("--trace", action="store_true")
    a = p.parse_args()
    if a.trace:
        trace()
        return
    if a.lb == "hybrid" and a.dp_local is None:
        a.dp_local = max(a.dp // a.nnodes, 1)
        a.dp_start = a.node_rank * a.dp_local
    show(a)


if __name__ == "__main__":
    main()
