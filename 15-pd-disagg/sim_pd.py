"""PD 分离模拟：KV 传输账本、同机混跑 vs P/D 的 ITL 尾巴、goodput、xPyD 配比、pull 与 push 的关键路径、
P 侧 KV 租约、NIXL 连接器判定逻辑、异构 TP 的读源映射、SGLang 对照（只依赖标准库）。

对照的源码（vllm 主干 0d6f92a，sglang 主干 8fabf5e）：
  vllm   config/kv_transfer.py:64–88              engine_id / kv_buffer_device / kv_role / kv_port=14579 / kv_load_failure_policy="fail"
         envs.py:236–237                           VLLM_NIXL_SIDE_CHANNEL_HOST=localhost / _PORT=5600（实际监听 base + dp_rank）
         distributed/kv_transfer/kv_connector/v1/nixl/
           base_scheduler.py:82–88                 kv_lease_duration=30，heartbeat_interval = lease // 6
           base_scheduler.py:168–187               kv_recompute_threshold=64 / bidirectional_kv_xfer=False / decoder_kv_blocks_ttl=480
           base_scheduler.py:201–251               on_new_request：只有 do_remote_prefill 的请求（D 侧）才登记心跳
           base_scheduler.py:385–436               _prefill_backoff / _get_remote_prefill_token_count / _truncate_request_for_prefill
           base_scheduler.py:513–518               build_connector_meta 里按 heartbeat_interval 节流打包心跳
           pull_scheduler.py:34–106                get_num_new_matched_tokens（本文件 pull_matched 逐行移植）
           pull_scheduler.py:191–292               request_finished：P 延迟释放，到期 = now + kv_lease_duration；D 侧用 decoder_kv_blocks_ttl
           push_scheduler.py:96–100, 128–205       push_registration_timeout 默认 = decoder_kv_blocks_ttl；D 分配完就登记 PUSH_REG
           base_worker.py:643–647, 3412            lease_extension = lease·2//3；收到心跳 expiry = max(旧, now + extension)
           base_worker.py:1881                     num_descs = sum(region_num_blocks)：每个 region 每个块一个描述符
           tp_mapping.py:63–150                    compute_head_offset / compute_tp_mapping（本文件 tp_sources 移植注意力部分）
         v1/core/sched/scheduler.py:3465–3514      _update_waiting_for_remote_kvs：整段收齐后 num_computed = num_tokens - 1
         engine/arg_utils.py:2976–3045             max_num_batched_tokens 默认：L40S / A100 的 server 是 2048，H100/H200 是 8192
  sglang srt/disaggregation/decode.py:512–520      _required_admission_tokens（本文件 sgl_admission 移植）
         srt/disaggregation/decode.py:898          prefill DP rank = bootstrap_room % dp_size
         srt/environ.py:654, 757–767               CLIP_MAX_NEW_TOKENS 4096 / BOOTSTRAP_TIMEOUT 300 / HEARTBEAT 5.0 × 2 / WAITING_TIMEOUT 300
         sgl-model-gateway/.../mini_lb.py:103–176  random 选对；P、D 用 asyncio.gather 同时发

假设：
  - 单步耗时 = max(FLOP / (峰值·MFU), 读的字节 / HBM 带宽) + 固定开销。prefill MFU 0.5，开销 L40S 3 ms，H100 TP2 4 ms，
    都是假设值。线性层 FLOP 2·P·n，注意力 4·L·H·d·(n·p + n²/2)。
  - interference / goodput 场景按 vLLM 拆分指南的实验搭：两张 L40S、Qwen2.5-7B、约 8k prompt、256 输出、每个速率 100 个
    Poisson 请求、前缀缓存关闭；同机基线是 DP=2，P/D 是 1P1D；chunked prefill 预算取 L40S 的 server 默认 2048。
  - P/D 传输时间两档：博客实测的 1.3 s（大部分是 D 在前向步之间轮询等开销），以及"修好"后的 12 GB/s 主机中转 + 20 ms。
  - pushpull 场景的每描述符成本是用 NIXL push 博客"8 QPS、16k 输入时 pull TPOT 10.4 ms、push 约 7 ms"反推的，只用来
    说明量级；每层按 K、V 各一个 region 计描述符（实际取决于 KV 布局，以 vllm:nixl_num_descriptors 为准）。

用法：
  python3 sim_pd.py --scenario transfer      # 每个请求要搬多少 KV、多少描述符、各种链路要多久
  python3 sim_pd.py --scenario interference  # 同机混跑 vs 1P1D 的 ITL 中位数 / p99 与 TTFT（约 5 秒）
  python3 sim_pd.py --scenario goodput       # TTFT ≤ 2 s 且请求内 p99 ITL ≤ 50 ms 的达标率（约 10 秒）
  python3 sim_pd.py --scenario ratio         # 不同负载形状下要几个 P、几个 D、多少传输带宽
  python3 sim_pd.py --scenario pushpull      # pull 与 push 的关键路径、D 侧 inline 发描述符的代价
  python3 sim_pd.py --scenario lease         # P 侧 KV 租约：30 s 租约 + 心跳 vs 旧的 480 s 超时
  python3 sim_pd.py --scenario lifecycle     # 一个请求走完 proxy → P → D 的每一步（含双向模式第二轮）
  python3 sim_pd.py --scenario tpmap         # 异构 TP：D 的每个 rank 从 P 的哪些 rank 读
  python3 sim_pd.py --scenario sglang        # SGLang 的 bootstrap_room、并发派发、准入公式、超时对照
"""
import argparse
import math
import os
import random
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, MODELS, Model  # noqa: E402

KiB, MiB, GiB = 1 << 10, 1 << 20, 1 << 30
BLOCK = 16
MFU = 0.5

# 本章用到、common/specs.py 里没有的两个模型（Hugging Face config.json）
LOCAL = {
    # Qwen/Qwen3-32B：64 层，hidden 5120，64 头，8 KV 头，head_dim 128
    'qwen3-32b': Model('Qwen3-32B', 32.8, 32.8, 64, 5120, 64, 8, 128, 151936),
    # Qwen/Qwen2.5-7B-Instruct：28 层，hidden 3584，28 头，4 KV 头，head_dim 3584/28 = 128
    'qwen2.5-7b': Model('Qwen2.5-7B', 7.6, 7.6, 28, 3584, 28, 4, 128, 152064),
}
ALL = dict(MODELS, **LOCAL)

# NIXL 默认值（base_scheduler.py / base_worker.py / envs.py / config/kv_transfer.py）
LEASE = 30
HB_INTERVAL = LEASE // 6            # 5
LEASE_EXT = LEASE * 2 // 3          # 20
OLD_TIMEOUT = 480                   # 旧的 VLLM_NIXL_ABORT_REQUEST_TIMEOUT
DECODER_TTL = 480
RECOMPUTE_THRESHOLD = 64
SIDE_CHANNEL_PORT = 5600
KV_PORT = 14579

# SGLang 默认值（environ.py / disagg.py）
SGL_RESERVED_DECODE = 512
SGL_CLIP_MAX_NEW = 4096
SGL_BOOTSTRAP_TIMEOUT = 300
SGL_WAITING_TIMEOUT = 300
SGL_HB_INTERVAL = 5.0
SGL_HB_MAX_FAIL = 2
SGL_BOOTSTRAP_PORT = 8998


# --------------------------------------------------------------------------- 基础账本
def kv_per_token(m, dtype_bytes=2):
    return m.kv_bytes_per_token(dtype_bytes)


def kv_per_rank(m, tp=1, dtype_bytes=2):
    """每个 TP rank 每 token 的 KV 字节。KV 头数小于 TP 时会复制，按 min(tp, kv_heads) 切。MLA 每个 rank 都是全量。"""
    b = m.kv_bytes_per_token(dtype_bytes)
    if not m.mla_kv_rank:
        b /= min(tp, m.kv_heads)
    return b


def descriptors(m, n_tokens, regions_per_layer=2, block=BLOCK):
    """一次传输每个 rank 要发的 NIXL 描述符数：regions × 块数（base_worker.py:1881，未合并相邻块时）。"""
    return m.layers * regions_per_layer * math.ceil(n_tokens / block)


def prefill_flops(m, n_new, n_prefix=0):
    lin = 2 * m.active_b * 1e9 * n_new
    attn = 4 * m.layers * m.heads * m.head_dim * (n_new * n_prefix + n_new * n_new / 2)
    return lin + attn


def weight_bytes(m, tp=1):
    return m.params_b * 1e9 * 2 / tp


# --------------------------------------------------------------------------- 1. 传输账本
LINKS = [
    ('PCIe 4.0 主机中转（假设 12 GB/s）', 12e9),
    ('100 Gb/s RoCE（线速）', 12.5e9),
    ('400 Gb/s 单网卡（线速）', 50e9),
    ('NVLink 4（单向 450 GB/s）', 450e9),
]


def scenario_transfer():
    print('== 每 token 的 KV（BF16）：P 算完要整段搬给 D')
    print(f'  {"模型":14s} {"层":>3s} {"KV头":>4s} {"每 token":>9s} {"8k 请求":>9s} {"32k 请求":>9s}')
    for key in ('qwen2.5-7b', 'llama3.1-8b', 'qwen3-32b', 'llama3.1-70b', 'deepseek-v3'):
        m = ALL[key]
        b = kv_per_token(m)
        print(f'  {m.name:14s} {m.layers:3d} {m.kv_heads:4d} {b / KiB:7.0f}Ki {8192 * b / MiB:7.0f}Mi {32768 * b / GiB:7.2f}Gi')
    print('  核对 vLLM 拆分指南：Qwen2.5-7B 56 KiB/token，8k prompt 约 470 MB；Llama-3.1-70B 320 KiB/token，'
          '10k token 约 3 GB')
    m = ALL['llama3.1-70b']
    b10k = 10_000 * kv_per_token(m)
    print(f'  本模型：10k × 320 KiB = {b10k / 1e9:.2f} GB，400 Gb/s 线速 {b10k / 50e9 * 1e3:.0f} ms（指南写约 65 ms）')
    print()
    print('== 一个请求在各种链路上要搬多久（整段、单流、线速，不含握手和排队）')
    hdr = ''.join(f'{n[:14]:>16s}' for n, _ in LINKS)
    print(f'  {"模型 / prompt":24s}{hdr}')
    for key, n in (('qwen2.5-7b', 8192), ('qwen3-32b', 16384), ('llama3.1-70b', 10_000), ('llama3.1-70b', 32768),
                   ('deepseek-v3', 32768)):
        mm = ALL[key]
        tot = n * kv_per_token(mm)
        cells = ''.join(f'{tot / bw * 1e3:14.1f}ms' for _, bw in LINKS)
        print(f'  {mm.name + " " + str(n):24s}{cells}')
    print('  链路名：' + '；'.join(n for n, _ in LINKS))
    print()
    print('== 按 TP 切开以后：每个 rank 搬自己那份，描述符按"层 × K/V × 块"算')
    print(f'  {"模型":14s} {"TP":>2s} {"prompt":>7s} {"每 rank":>9s} {"每 rank 描述符":>14s} {"8 rank 总量":>11s}')
    for key, tp, n in (('qwen2.5-7b', 1, 8192), ('qwen3-32b', 8, 16384), ('qwen3-32b', 2, 16384),
                       ('llama3.1-70b', 4, 32768), ('deepseek-v3', 8, 32768)):
        mm = ALL[key]
        per = n * kv_per_rank(mm, tp)
        regions = 1 if mm.mla_kv_rank else 2
        print(f'  {mm.name:14s} {tp:2d} {n:7,d} {per / MiB:7.0f}Mi {descriptors(mm, n, regions):14,d} '
              f'{per * tp / GiB:9.2f}Gi')
    print('  Qwen3-32B TP8：8 个 KV 头正好一 rank 一个，16k prompt 每个 rank 512 MiB、131,072 个描述符。')
    print('  push 连接器在后台线程里把相邻块合并成大区域再发 WRITE；pull 在 D 的 worker 上逐个准备描述符。')
    print('  MLA（DeepSeek-V3）每个 rank 存全量 latent，D 只需从一个 P rank 读（tp_mapping.py:103–108）。')


# --------------------------------------------------------------------------- 2/3. 单卡步进模型
class Box:
    """一张卡上的单步耗时模型。"""

    def __init__(self, m, gpu, tp=1, ovh=0.003, mfu=MFU):
        self.m, self.g, self.tp, self.ovh, self.mfu = m, gpu, tp, ovh, mfu
        self.W = weight_bytes(m, tp)
        self.kvb = kv_per_rank(m, tp)

    def step(self, decode_ctx, chunks):
        """decode_ctx：本步每个 decode 请求的上下文长度；chunks：[(新算 token 数, 已有前缀)]。"""
        fl = sum(prefill_flops(self.m, 1, c) for c in decode_ctx)
        fl += sum(prefill_flops(self.m, n, p) for n, p in chunks)
        by = self.W + self.kvb * (sum(decode_ctx) + sum(n + p for n, p in chunks))
        t_c = fl / self.tp / (self.g.bf16_tflops * 1e12 * self.mfu)
        t_m = by / (self.g.bw_tbs * 1e12)
        return max(t_c, t_m) + self.ovh


def engine(box, reqs, role, budget=2048):
    """一个 vLLM 实例的离散步进：role = mixed（同机）/ prefill（P）/ decode（D）。
    reqs：[(到达时间, rid, prompt, 输出数)]，按到达时间排序。
    返回 {rid: dict(first=首 token 时间, itl=[...], done=prefill 完成时间)}。"""
    out = {}
    t, i = 0.0, 0
    waiting = deque()           # [rid, prompt, out_len, total, done]
    running = []                # [rid, ctx, remaining, last]
    while i < len(reqs) or waiting or running:
        if not waiting and not running:
            t = max(t, reqs[i][0])
        while i < len(reqs) and reqs[i][0] <= t:
            a, rid, n, o = reqs[i]
            # D 侧：KV 已经搬来，只剩最后 1 个 token 重算（scheduler.py:3514）
            waiting.append([rid, n, o, n, n - 1 if role == 'decode' else 0])
            i += 1
        if not waiting and not running:
            continue
        dec = running if role != 'prefill' else []
        bud = budget - len(dec)
        chunks = []
        for w in waiting:
            if bud <= 0:
                break
            take = min(bud, w[3] - w[4])
            chunks.append((w, take))
            bud -= take
        dt = box.step([r[1] for r in dec], [(take, w[4]) for w, take in chunks])
        t += dt
        for r in dec:
            out[r[0]]['itl'].append(t - r[3])
            r[3] = t
            r[1] += 1
            r[2] -= 1
        running[:] = [r for r in running if r[2] > 0]
        for w, take in chunks:
            w[4] += take
            if w[4] == w[3]:
                waiting.remove(w)
                rec = out.setdefault(w[0], dict(first=None, itl=[], done=None))
                rec['done'] = t
                if role != 'prefill':
                    rec['first'] = t
                    if w[2] > 1:
                        running.append([w[0], w[1], w[2] - 1, t])
    return out


def poisson(rate, n, prompt, out_len, seed=0):
    rng = random.Random(seed)
    t, reqs = 0.0, []
    for rid in range(n):
        t += rng.expovariate(rate)
        reqs.append((t, rid, prompt, out_len))
    return reqs


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def run_colocated(box, reqs, budget=2048, dp=2):
    """DP=dp 的同机混跑：round-robin 分到 dp 张卡。"""
    res = {}
    for k in range(dp):
        res.update(engine(box, reqs[k::dp], 'mixed', budget))
    return {r[1]: dict(ttft=res[r[1]]['first'] - r[0], itl=res[r[1]]['itl']) for r in reqs}


def run_pd(box, reqs, xfer_s, budget=2048):
    """1P1D：P 只做 prefill，算完 xfer_s 秒后 KV 到 D，D 下一步吐首 token。"""
    p = engine(box, reqs, 'prefill', budget)
    d_reqs = sorted((p[r[1]]['done'] + xfer_s, r[1], r[2], r[3]) for r in reqs)
    d = engine(box, d_reqs, 'decode', budget)
    return {r[1]: dict(ttft=d[r[1]]['first'] - r[0], itl=d[r[1]]['itl']) for r in reqs}


def summarize(res, ttft_slo=2.0, itl_slo=0.050):
    itl = [x for v in res.values() for x in v['itl']]
    ttft = [v['ttft'] for v in res.values()]
    ok = sum(1 for v in res.values() if v['ttft'] <= ttft_slo and (not v['itl'] or pct(v['itl'], 0.99) <= itl_slo))
    return dict(itl50=pct(itl, 0.5), itl99=pct(itl, 0.99), ttft50=pct(ttft, 0.5), ttft99=pct(ttft, 0.99),
                attain=ok / len(res))


def l40s_box():
    return Box(ALL['qwen2.5-7b'], GPUS['l40s'], tp=1, ovh=0.003)


def xfer_fixed(m, n):
    return n * kv_per_token(m) / 12e9 + 0.020


def scenario_interference():
    box = l40s_box()
    m = box.m
    n, o = 8192, 256
    xf = xfer_fixed(m, n)
    print('== 两张 L40S，Qwen2.5-7B，8k prompt，256 输出，每个速率 100 个 Poisson 请求，chunked prefill 预算 2048')
    print(f'  单步：纯 decode {box.step([n], []) * 1e3:.1f} ms；带一个 2048 token 的 prefill chunk '
          f'{box.step([n], [(2048, 6144)]) * 1e3:.1f} ms；一个 8k prompt 独占 prefill '
          f'{sum(box.step([], [(2048, k * 2048)]) for k in range(4)):.2f} s')
    print(f'  P/D 传输两档：博客实测 1.30 s；修好以后 {n * kv_per_token(m) / 1e6:.0f} MB / 12 GB/s + 20 ms = {xf * 1e3:.0f} ms')
    print()
    print(f'  {"req/s":>5s} | {"同机 DP=2":^26s} | {"1P1D 传输 1.3 s":^26s} | {"1P1D 传输修好":^26s}')
    print(f'  {"":5s} | {"ITL50":>7s}{"ITL99":>8s}{"TTFT50":>9s}  | {"ITL50":>7s}{"ITL99":>8s}{"TTFT50":>9s}  |'
          f' {"ITL50":>7s}{"ITL99":>8s}{"TTFT50":>9s}')
    for rate in (0.2, 0.4, 0.6, 1.0, 2.0):
        reqs = poisson(rate, 100, n, o, seed=1)
        rows = [summarize(run_colocated(box, reqs)), summarize(run_pd(box, reqs, 1.3)), summarize(run_pd(box, reqs, xf))]
        cells = ' | '.join(f'{s["itl50"] * 1e3:6.1f}ms{s["itl99"] * 1e3:6.0f}ms{s["ttft50"]:7.2f}s ' for s in rows)
        print(f'  {rate:5.1f} | {cells}')
    print()
    print('  博客实测：同机 p99 ITL 23 → 169 ms（0.4 req/s）→ 263 ms（2 req/s）；P/D 25–52 ms；两边 ITL 中位数 21–24 ms；')
    print('           0.2 req/s 时 P/D TTFT 中位数 2.2 s，同机 0.7 s。')
    print('  2 req/s 时单个 P 每秒要算 2 × 8k token，超过一张 L40S 的 prefill 能力，P 排队，TTFT 一路涨：1P1D 在这里该改成 2P1D'
          '或回到同机。')


def scenario_goodput():
    box = l40s_box()
    m = box.m
    n, o = 8192, 256
    xf = xfer_fixed(m, n)
    print('== SLO：TTFT ≤ 2 s，且请求内 p99 ITL ≤ 50 ms；达标率 = 两条都满足的请求占比（DistServe 的 goodput 口径）')
    print(f'  {"req/s":>5s} {"同机 DP=2":>10s} {"1P1D 1.3 s":>11s} {"1P1D 修好":>10s}')
    best = {'co': 0, 'pd13': 0, 'pdfix': 0}
    for rate in (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0):
        reqs = poisson(rate, 100, n, o, seed=2)
        a = {'co': summarize(run_colocated(box, reqs))['attain'],
             'pd13': summarize(run_pd(box, reqs, 1.3))['attain'],
             'pdfix': summarize(run_pd(box, reqs, xf))['attain']}
        for k, v in a.items():
            if v >= 0.9:
                best[k] = max(best[k], rate)
        print(f'  {rate:5.1f} {a["co"] * 100:9.0f}% {a["pd13"] * 100:10.0f}% {a["pdfix"] * 100:9.0f}%')
    print(f'  达标率 ≥ 90% 的最高速率：同机 {best["co"]} req/s，1P1D（1.3 s）{best["pd13"]} req/s，'
          f'1P1D（修好）{best["pdfix"]} req/s')
    print('  读法：ITL 尾巴让同机最先掉出 SLO；传输 1.3 s 让 P/D 在 TTFT 上一个都不达标；传输修好后 P/D 的 goodput 才拉开。')
    print('  对照：MoRI-IO 在 MI300X 上跑 Qwen3-235B-FP8、8 req/s，1 s TTFT + 50 ms ITL 达标 73/100，同机 30/100。')


# --------------------------------------------------------------------------- 4. xPyD 配比
def p_throughput(m, gpu, tp, n, hit=0.0, util=0.7):
    """一个 P 实例每秒能吃下多少 prompt token（按 util 留排队余量）。命中的前缀只算注意力不算线性层，这里简化成直接扣掉。"""
    new = n * (1 - hit)
    fl = prefill_flops(m, new, n - new)
    sec = fl / (tp * gpu.bf16_tflops * 1e12 * MFU)
    return util / sec          # 每秒请求数


def d_capacity(m, gpu, tp, ctx, itl_slo=0.050, ovh=0.004, mem_util=0.9):
    """一个 D 实例的 decode 并发与吞吐：受 ITL SLO 和 KV 池两边约束。"""
    W = weight_bytes(m, tp)
    kv = kv_per_rank(m, tp)
    bw = gpu.bw_tbs * 1e12
    pool = gpu.mem_gb * 1e9 * mem_util - W
    b_mem = int(pool / (ctx * kv))
    b_slo = int(((itl_slo - ovh) * bw - W) / (ctx * kv))
    b = max(1, min(b_mem, b_slo))
    step = (W + b * ctx * kv) / bw + ovh
    return b, step, b / step, ('KV 池' if b_mem <= b_slo else 'ITL')


WORKLOADS = [
    ('对话', 1_500, 350, 0.0),
    ('RAG', 8_000, 256, 0.0),
    ('长文摘要', 16_000, 512, 0.0),
    ('agent，无前缀命中', 30_000, 150, 0.0),
    ('agent，80% 前缀命中', 30_000, 150, 0.8),
]


def scenario_ratio(rate=10.0):
    m, gpu, tp = ALL['qwen3-32b'], GPUS['h100-sxm'], 2
    print(f'== Qwen3-32B，P、D 都是 H100 TP2，到达 {rate:g} req/s，ITL SLO 50 ms，P 利用率按 70% 留排队余量')
    print(f'  {"负载":18s} {"输入":>6s} {"输出":>5s} {"每P req/s":>9s} {"D并发":>6s} {"约束":>5s} {"每D tok/s":>9s} '
          f'{"P 数":>5s} {"D 数":>5s} {"P:D":>6s} {"P→D 带宽":>9s}')
    for name, n, o, hit in WORKLOADS:
        pr = p_throughput(m, gpu, tp, n, hit)
        b, step, thr, why = d_capacity(m, gpu, tp, n + o // 2)
        need_p = rate / pr
        need_d = rate * o / thr
        bw = rate * n * (1 - hit) * kv_per_token(m)
        print(f'  {name:18s} {n:6,d} {o:5d} {pr:9.2f} {b:6d} {why:>5s} {thr:9,.0f} {need_p:5.1f} {need_d:5.1f} '
              f'{need_p / need_d:6.2f} {bw / 1e9:7.1f}GB/s')
    print('  前缀命中的部分不用重新传（P 侧本地命中）；双向模式下 D 上留着的 KV 还能回传给 P，见 lifecycle 场景。')
    print('  输入越长、输出越短，P 越多：DeepSeek-V4.1-Flash 博客在 GB300 上用到 8P1D、7P1D。')
    print('  D 的并发常常先撞 KV 池而不是 ITL：长上下文 D 要的是显存，不是算力。')


# --------------------------------------------------------------------------- 5. pull vs push
# 用 NIXL push 博客 8 QPS / 16k 的 TPOT 反推：pull 10.4 ms、push ≈ 7 ms → D 有 1 - 7/10.4 的时间花在 inline 准备描述符
BLOG_PULL_TPOT, BLOG_PUSH_TPOT, BLOG_QPS, BLOG_N = 10.4e-3, 7.0e-3, 8, 16384


def calibrated_desc_cost():
    m = ALL['qwen3-32b']
    f = 1 - BLOG_PUSH_TPOT / BLOG_PULL_TPOT
    post = f / BLOG_QPS
    return post / descriptors(m, BLOG_N), post, f


def pull_tpot(push_tpot, qps, n, desc_cost):
    f = qps * descriptors(ALL['qwen3-32b'], n) * desc_cost
    return push_tpot / (1 - f) if f < 1 else float('inf')


def scenario_pushpull():
    m = ALL['qwen3-32b']
    n = 16384
    per_rank = n * kv_per_rank(m, 8)
    print('== 关键路径：Qwen3-32B，P、D 各一台 8×H200（TP8），16k prompt，每 rank 两张 200 Gb/s EFA')
    print(f'  每 rank 要搬 {per_rank / MiB:.0f} MiB，线速 50 GB/s 需 {per_rank / 50e9 * 1e3:.1f} ms；'
          f'描述符 {descriptors(m, n):,} 个（64 层 × K/V × 1,024 块）')
    print()
    print('  pull（NixlConnector）                                   push（NixlPushConnector）')
    steps = [
        ('proxy 把请求发给 P（do_remote_decode, max_tokens=1）', 'proxy 先选好 D，P、D 同时收到请求'),
        ('P prefill', 'D 分配块，同一步里把 PUSH_REG 发给 P（与 prefill 重叠）'),
        ('P 把 remote_block_ids 等放进响应，回到 proxy', 'P prefill'),
        ('proxy 再把请求连同参数发给 D', 'P 后台线程 nixl-push-writer：合并相邻块、发 WRITE'),
        ('D 下一个调度步分配块', '写完发完成通知，D 下一步吐首 token'),
        ('D worker 在 engine 线程上准备描述符、发 READ', ''),
        ('读完，D 下一步吐首 token', ''),
    ]
    for a, b in steps:
        print(f'  · {a:52s} · {b}')
    print()
    c, post, f = calibrated_desc_cost()
    print(f'== 反推：8 QPS × 16k 时 D 有 {f * 100:.0f}% 的时间在 inline 准备描述符 → 每请求 {post * 1e3:.0f} ms，'
          f'每描述符 {c * 1e9:.0f} ns')
    print('  博客没有给每描述符成本，这个数只是让模型与 8 QPS / 16k 那个点对上，用来看量级与随 QPS、输入长度的走向。')
    print(f'  {"输入":>6s} ' + ''.join(f'{"QPS " + str(q):>12s}' for q in (1, 2, 4, 8)))
    for nn in (1024, 4096, 8192, 16384):
        cells = ''.join(f'{pull_tpot(BLOG_PUSH_TPOT, q, nn, c) * 1e3:10.1f}ms' for q in (1, 2, 4, 8))
        print(f'  {nn:6,d} {cells}')
    print('  （表中是 pull 的 TPOT，push 取 7 ms 不变）博客实测：轻载两者都在 5.6–5.8 ms；16k 时 pull 在 4–8 QPS 约 10.4 ms。')
    print('  模型在 4 QPS 给出约 8.4 ms，低于实测：D 侧还有排队和批次变大的效应，线性模型抓不到。')
    print()
    print('  TTFT 实测：8 QPS、16k 时 pull 约 1,744 ms，push 约 540 ms（博客，vllm 0.29.0 + nixl 1.4.1）。')
    print('  代价：push 要在 prefill 之前定下 D；pull 可以等 prefill 完再按负载挑 D，并且支持双向传输。')


# --------------------------------------------------------------------------- 6. 租约
def handle_heartbeat(expiry, now, ext=LEASE_EXT):
    """base_worker.py:3412：每个请求 expiry = max(旧, now + extension)。"""
    return max(expiry, now + ext)


def lease_release_time(d_dead_at=None, read_at=None, lease=LEASE, hb=HB_INTERVAL, ext=LEASE_EXT, horizon=3600):
    """P 在 t=0 算完、开始持有块。D 在 read_at 读完（None 表示一直没读），或在 d_dead_at 死掉（不再发心跳）。
    返回 P 释放块的时间。D 活着且请求在等时，每 hb 秒发一次心跳。"""
    expiry = lease
    t = 0.0
    while t <= horizon:
        if read_at is not None and read_at <= t:
            return read_at
        if expiry <= t:
            return expiry
        t_next = t + hb
        if read_at is not None and read_at < t_next:
            if read_at < expiry:
                return read_at
            return expiry
        if d_dead_at is None or t_next < d_dead_at:
            if t_next > expiry:
                return expiry
            expiry = handle_heartbeat(expiry, t_next, ext)
        t = t_next
    return expiry


def scenario_lease():
    print(f'== 默认值：kv_lease_duration {LEASE} s，心跳间隔 lease//6 = {HB_INTERVAL} s，每次心跳续到 now + lease·2//3 = '
          f'{LEASE_EXT} s')
    print('  旧设计：VLLM_NIXL_ABORT_REQUEST_TIMEOUT 一个 480 s 的硬超时，D 挂了 P 要白占 8 分钟')
    print(f'  {"情形":34s} {"新：释放于":>10s} {"旧：释放于":>10s}')
    cases = [
        ('D 0.3 s 后读完', dict(read_at=0.3)),
        ('D 排队 25 s 后读完（心跳续着）', dict(read_at=25)),
        ('D 排队 200 s 后读完（心跳续着）', dict(read_at=200)),
        ('D 在 3 s 时挂掉', dict(d_dead_at=3)),
        ('D 在 47 s 时挂掉', dict(d_dead_at=47)),
        ('请求一开始就没到 D（无心跳）', dict(d_dead_at=0)),
    ]
    for name, kw in cases:
        new = lease_release_time(**kw)
        old = kw['read_at'] if kw.get('read_at') is not None and kw['read_at'] <= OLD_TIMEOUT else OLD_TIMEOUT
        print(f'  {name:34s} {new:9.1f}s {old:9.1f}s')
    print()
    m = ALL['qwen3-32b']
    per_req = 16384 * kv_per_token(m)
    inflight = 64
    print(f'== D 挂掉时 P 白占多少：Qwen3-32B 16k 请求每个 {per_req / GiB:.0f} GiB，D 上同时在等的有 {inflight} 个')
    pinned = inflight * per_req
    print(f'  一共 {pinned / GiB:.0f} GiB 的块被钉在 P 上：新设计最多 {LEASE} s 后释放，旧设计 {OLD_TIMEOUT} s；'
          f'按"GiB·秒"算差 {OLD_TIMEOUT / LEASE:.0f} 倍')
    print('  租约到期但 D 其实还在读：P 计 vllm:nixl_num_kv_expired_reqs；D 晚到的完成通知计 '
          'vllm:nixl_num_notifications_after_expiry，并打 "Potentially invalid KV blocks"。')
    print('  心跳只对 D 侧（do_remote_prefill）的请求登记：双向模式里 D 留着给第二轮回读的块不走租约，'
          f'而是固定 decoder_kv_blocks_ttl = {DECODER_TTL} s，不续期。')
    print(f'  push 模式 D 侧登记等待 push_registration_timeout，默认也取 decoder_kv_blocks_ttl（{DECODER_TTL} s）。')


# --------------------------------------------------------------------------- 7. 生命周期（逐行移植判定）
def prefill_backoff(has_mamba=False, bounded_replay=False, lookahead=0):
    """base_scheduler.py:385：max(1 if mamba/bounded_replay else 0, num_prefill_lookahead_tokens - 1)。"""
    return max(1 if has_mamba or bounded_replay else 0, lookahead - 1)


def remote_prefill_count(n_prompt, backoff):
    """base_scheduler.py:403：D 只从 P 读 n - backoff 个 token，尾巴由 D 自己重算。"""
    if backoff and n_prompt > backoff:
        return n_prompt - backoff
    return n_prompt


def pull_matched(params, n_prompt, num_computed, threshold=RECOMPUTE_THRESHOLD, backoff=0):
    """pull_scheduler.py:34–106 get_num_new_matched_tokens。返回 (外部 token 数, 是否异步)。"""
    if params and params.get('do_remote_prefill'):
        count = remote_prefill_count(n_prompt, backoff) - num_computed
        if count > 0:
            return count, True
    if (params and params.get('do_remote_decode') and params.get('remote_block_ids')
            and all(k in params for k in ('remote_engine_id', 'remote_request_id', 'remote_host', 'remote_port'))):
        remote = params.get('remote_num_tokens') or 0
        count = min(remote, n_prompt) - num_computed
        if count > 0:
            if threshold > 0 and count < threshold:
                return 0, False
            return count, True
    return 0, False


def after_recv(num_computed, num_tokens):
    """scheduler.py:3514：整段收齐后回退 1 个 token，让 D 算出下一个 token 的 logits。"""
    return num_tokens - 1 if num_computed == num_tokens else num_computed


def request_finished(params, finished_ok, has_blocks, bidirectional=False, now=0.0):
    """pull_scheduler.py:191–292 的分支。返回 (是否延迟释放, 到期时间或 None)。"""
    if not params:
        return False, None
    is_p = bool(params.get('do_remote_decode'))
    if params.get('do_remote_prefill'):
        return False, None              # 还没分配就被 abort：通知 P 释放
    if not is_p and not bidirectional:
        return False, None
    if not finished_ok:
        return False, None
    if not has_blocks:
        return False, None
    ttl = LEASE if is_p else DECODER_TTL
    return True, now + ttl


def scenario_lifecycle():
    n = 8192
    print(f'== 一个 {n:,} token 的请求走 pull 模式（toy_proxy_server.py 的串行流程）')
    p_params = dict(do_remote_decode=True, do_remote_prefill=False)
    print('  1 proxy → P：kv_transfer_params = {do_remote_decode: True}，stream=False，max_tokens=1')
    print(f'    P 的 get_num_new_matched_tokens → {pull_matched(p_params, n, 0)}：P 本地照常 prefill')
    delay, exp = request_finished(p_params, True, True)
    print(f'  2 P 结束：request_finished → 延迟释放={delay}，到期 t+{exp:.0f} s（租约），响应里带回 remote_block_ids / '
          f'engine_id / host / port / tp_size / remote_num_tokens')
    d_params = dict(do_remote_prefill=True, remote_block_ids=[1, 2, 3], remote_engine_id='P0',
                    remote_request_id='r', remote_host='10.0.0.1', remote_port=SIDE_CHANNEL_PORT)
    c, asyn = pull_matched(d_params, n, 0)
    print(f'  3 proxy → D：do_remote_prefill=True；D 的 get_num_new_matched_tokens → ({c}, {asyn})：'
          f'整段 {c} token 异步读，请求进 WAITING_FOR_REMOTE_KVS')
    print(f'  4 D 读完：num_computed {n} → after_recv → {after_recv(n, n)}，下一步只算最后 1 个 token 并吐首 token')
    print(f'  5 D 有本地前缀命中 4,096 时：({pull_matched(d_params, n, 4096)[0]}, True) —— 只读差的那段')
    print()
    print('== backoff：哪些模型 P 要少算尾巴')
    for name, kw in (('普通 Transformer', {}), ('Mamba / 混合模型', dict(has_mamba=True)),
                     ('DeepSeek-V4.1 SWA bounded replay', dict(bounded_replay=True)),
                     ('多模块 MTP，lookahead 3', dict(lookahead=3))):
        b = prefill_backoff(**kw)
        print(f'  {name:32s} backoff {b}：P 算 {n - b:,}，D 读 {remote_prefill_count(n, b):,}，D 自己重算 {b} 个')
    print()
    print(f'== 双向模式第二轮（bidirectional_kv_xfer=True，kv_recompute_threshold={RECOMPUTE_THRESHOLD}）')
    d_fin = dict(do_remote_prefill=False, do_remote_decode=False)
    print(f'  第一轮 D 结束：request_finished → {request_finished(d_fin, True, True, bidirectional=True)}：'
          f'块留 {DECODER_TTL} s，元数据经流式响应交给有状态 proxy')
    back = dict(do_remote_decode=True, remote_block_ids=[7, 8], remote_engine_id='D0', remote_request_id='r1',
                remote_host='10.0.0.2', remote_port=SIDE_CHANNEL_PORT + 1)
    for remote, computed, prompt in ((9000, 0, 9500), (9000, 8960, 9500), (9000, 8900, 9500)):
        back['remote_num_tokens'] = remote
        print(f'  第二轮 prompt {prompt:,}，D 上留了 {remote:,}，P 本地已有 {computed:,} → '
              f'{pull_matched(back, prompt, computed)}')
    print('  差额小于 64 个 token 时 P 直接重算，不发 RDMA（双向博客：短前缀的互联与协调开销会超过重算）。')


# --------------------------------------------------------------------------- 8. 异构 TP
def tp_sources(d_rank, d_tp, p_tp, kv_heads, mla=False):
    """tp_mapping.py:compute_tp_mapping 的注意力部分：D 的 d_rank 要读 P 的哪些 rank。"""
    if mla or d_tp >= p_tp:
        return [d_rank * p_tp // d_tp]
    abs_tp = p_tp // d_tp
    start = d_rank * abs_tp
    seen, ranks = set(), []
    for k in range(abs_tp):
        head = (start + k) * kv_heads // p_tp
        if head not in seen:              # np.unique(return_index) 保留每个头的第一个 rank
            seen.add(head)
            ranks.append(start + k)
    return ranks


def head_offset(d_rank, d_tp, p_tp, kv_heads):
    """tp_mapping.py:compute_head_offset：D 比 P 切得细时，本 rank 的头在远端块里的偏移。"""
    if d_tp <= p_tp:
        return 0
    if d_tp > kv_heads:
        remote_rank = d_rank * p_tp // d_tp
        return d_rank * kv_heads // d_tp - remote_rank * kv_heads // p_tp
    return d_rank % (d_tp // p_tp)


def scenario_tpmap():
    print('== D 的每个 rank 从 P 的哪些 rank 读（Qwen3-32B：8 个 KV 头；Qwen2.5-7B：4 个；MLA 全复制）')
    cases = [('Qwen3-32B', 8, 8, 8, False), ('Qwen3-32B', 8, 4, 8, False), ('Qwen3-32B', 4, 8, 8, False),
             ('Qwen3-32B', 8, 2, 8, False), ('Qwen2.5-7B', 8, 2, 4, False), ('Qwen2.5-7B', 2, 8, 4, False),
             ('DeepSeek-V3', 8, 4, 1, True)]
    for name, p_tp, d_tp, kh, mla in cases:
        srcs = [tp_sources(r, d_tp, p_tp, kh, mla) for r in range(d_tp)]
        offs = [0 if mla else head_offset(r, d_tp, p_tp, kh) for r in range(d_tp)]
        desc = '  '.join(f'D{r}←P{",".join(map(str, s))}' + (f'(+{o})' if o else '') for r, (s, o) in
                         enumerate(zip(srcs, offs)))
        print(f'  {name:11s} P TP{p_tp} → D TP{d_tp}: {desc}')
    print('  P TP > D TP：一个 D rank 读多个 P rank；KV 头少于 P TP 时几个 P rank 存的是同一个头，np.unique 去重，只读一次')
    print('  P TP < D TP：多个 D rank 读同一个 P rank 的不同头切片，(+k) 是头偏移；要求 LBHNC（头在前）布局，'
          'LBNHC 不支持异构 TP 切头')
    print('  MLA：每个 rank 都是全量 latent，D 的每个 rank 只读一个 P rank，P 的块要等所有消费者都读完才释放')
    print('  SGLang：异构 TP 推荐开 staging buffer，文档给的提升是 2–5 倍')


# --------------------------------------------------------------------------- 9. SGLang 对照
def sgl_admission(prompt, prefix, max_new, allocated, retractable=0, reserved=SGL_RESERVED_DECODE, clip=SGL_CLIP_MAX_NEW):
    """decode.py:512–520 _required_admission_tokens。"""
    return max(allocated + reserved, prompt - prefix + min(max_new, clip) - retractable)


def sgl_dp_rank(bootstrap_room, dp_size):
    """decode.py:898：D 用 bootstrap_room 对 P 的 DP 数取模，决定去哪个 DP rank 取 KV。"""
    return 0 if dp_size == 1 else bootstrap_room % dp_size


def scenario_sglang():
    print('== 谁先动：vLLM toy proxy 串行（P 返回后再发 D）；SGLang mini_lb 用 asyncio.gather 同时发 P、D')
    rng = random.Random(0)
    rooms = [rng.randint(0, 2 ** 63 - 1) for _ in range(6)]
    print(f'  bootstrap_room 是 proxy 随机生成的 64 位整数，D 用它找 P 上的同一个请求；P 开 DP=4 时 D 去 room % 4：'
          f'{[sgl_dp_rank(r, 4) for r in rooms]}')
    print(f'  bootstrap 端口默认 {SGL_BOOTSTRAP_PORT}，传输后端默认 mooncake；vLLM 侧是 NIXL side channel {SIDE_CHANNEL_PORT} + dp_rank')
    print('  方向：SGLang 的 Mooncake / NIXL 后端都是 P 往 D 写（D 先把自己的 kv_indices 发给 P）；vLLM pull 是 D 读，push 是 P 写')
    print()
    print('== D 侧准入：SGLang 预留未来的 decode token，vLLM 只分配已有 token')
    pool = 500_000
    print(f'  D 的 KV 池 {pool:,} token；SGLang 公式 max(已分配 + 512, prompt - 前缀 + min(max_new, 4096) - 可回收)')
    print(f'  {"prompt":>7s} {"max_new":>8s} {"SGLang 每请求":>12s} {"能进几个":>8s} {"vLLM 每请求":>11s} {"能进几个":>8s}')
    for n, mx in ((2_000, 512), (8_000, 256), (16_000, 8_192), (32_000, 32_768)):
        need = sgl_admission(n, 0, mx, n)
        vneed = math.ceil(n / BLOCK) * BLOCK
        print(f'  {n:7,d} {mx:8,d} {need:12,d} {pool // need:8d} {vneed:11,d} {pool // vneed:8d}')
    print('  vLLM 准入更松，代价是之后 KV 不够时要抢占；SGLang 准入更紧，换来 decode 中途少抢占。')
    print('  SGLang 默认 decode 侧不开 radix cache（disaggregation_decode_enable_radix_cache=False）。')
    print()
    print('== 超时与健康检查')
    rows = [
        ('P 侧等 D 来取', f'租约 {LEASE} s + 心跳每 {HB_INTERVAL} s 续 {LEASE_EXT} s', f'BOOTSTRAP_TIMEOUT {SGL_BOOTSTRAP_TIMEOUT} s'),
        ('D 侧等 KV', '无超时，停在 WAITING_FOR_REMOTE_KVS', f'WAITING_TIMEOUT {SGL_WAITING_TIMEOUT} s'),
        ('探活', 'NIXL 通知里的 HB:req_id', f'每 {SGL_HB_INTERVAL:g} s 探 P 的 bootstrap，连续 {SGL_HB_MAX_FAIL} 次失败判下线'),
        ('传输失败', 'kv_load_failure_policy fail（默认）/ recompute', '请求 abort'),
        ('D 选择时机', 'pull 晚绑定；push 早绑定', 'early（默认）/ prefill_complete（只支持 mooncake 等限制）'),
    ]
    for a, b, c in rows:
        print(f'  {a:10s} | vLLM: {b:38s} | SGLang: {c}')
    print(f'  SGLang 判 P 下线最快约 {SGL_HB_INTERVAL * SGL_HB_MAX_FAIL:g} s；vLLM 的 P 发现 D 没了最晚 {LEASE} s。')


SCENARIOS = dict(transfer=scenario_transfer, interference=scenario_interference, goodput=scenario_goodput,
                 ratio=scenario_ratio, pushpull=scenario_pushpull, lease=scenario_lease,
                 lifecycle=scenario_lifecycle, tpmap=scenario_tpmap, sglang=scenario_sglang)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=list(SCENARIOS), default='transfer')
    SCENARIOS[ap.parse_args().scenario]()
