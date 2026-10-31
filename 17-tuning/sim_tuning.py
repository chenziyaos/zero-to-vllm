"""调参前先把"不调时引擎替你选了什么"算清楚（只依赖标准库，不需要 GPU）。

对照的源码（vllm 主干 0d6f92a，sglang 主干 8fabf5e）：
  vllm   engine/arg_utils.py:2946      get_batch_defaults：按显存 / 卡名 / usage context 选 max_num_batched_tokens、max_num_seqs
         engine/arg_utils.py:3157      _set_default_max_num_seqs_and_batched_tokens_args：throughput 翻倍、按 max_model_len 截断
         config/vllm.py:2416-2620      _set_cudagraph_sizes：捕获上限 min(max_num_seqs × q_len × 2, 512/1024)，网格 1,2,4 + 8 步长 + 16 步长
         config/vllm.py:147-359        OptimizationLevel O0–O3 与各档默认开关
         config/cache.py:215           gpu_memory_utilization = 0.92
         v1/worker/gpu_worker.py:648   determine_available_memory：requested − non_kv − cudagraph 估计
         config/scheduler.py:69-105    max_num_active_seqs / max_num_queued_reqs / max_num_queued_tokens
         docs/configuration/optimization.md:463-507  CPU 核数公式
  sglang srt/arg_groups/memory_hook.py:90-171   按显存分档的 chunked_prefill_size / decode cuda graph max_bs
         srt/arg_groups/memory_hook.py:281-325  mem_fraction_static = (gpu − reserved) / gpu
         srt/arg_groups/memory_hook.py:357      reserve_for_graph_mb
         srt/arg_groups/cuda_graph_hook.py:480-532  decode / prefill 捕获网格

场景：
  defaults   各卡不调参时两边选出来的预算、并发上限、捕获网格
  mem        gpu_memory_utilization 每加 0.01 换来多少 KV；SGLang mem_fraction_static 怎么来的
  cudagraph  不同 max_num_seqs / performance_mode 下的捕获尺寸、padding 浪费、eager 比例
  budget     max_num_batched_tokens 扫描：TTFT、ITL、goodput（复用 06 的时间模型）
  seqs       max_num_seqs 扫描：KV 才是真正的并发上限
  queue      max_num_queued_tokens 怎么从 TTFT 目标反推
  opt        O0–O3 各档开了什么
  cpu        需要多少物理核

用法：
  python3 sim_tuning.py --scenario defaults
  python3 sim_tuning.py --scenario mem --gpu h100-sxm
  python3 sim_tuning.py --scenario cudagraph
  python3 sim_tuning.py --scenario budget --rate 10 --slo-ttft 2000 --slo-tpot 30
  python3 sim_tuning.py --scenario seqs
  python3 sim_tuning.py --scenario queue --ttft-ms 2000
  python3 sim_tuning.py --scenario opt
  python3 sim_tuning.py --scenario cpu --tp 8 --dp 1 --api-servers 4
  python3 sim_tuning.py --scenario all

时间模型、KV 容量都沿用 02 / 06 两章（H100 SXM + Qwen3-8B BF16），只比较相对快慢，绝对值不要拿去报数。
"""
import argparse
import math
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '06-chunked-prefill'))
from common.specs import MODELS  # noqa: E402
import sim_chunked as sc  # noqa: E402

GiB = 1 << 30

# --------------------------------------------------------------------------- 卡
# torch 报的总显存（GiB，决定 vLLM 档位）、nvidia-smi memory.total（MiB，决定 SGLang 档位）、cc 主版本、卡名
# H100 的两个数取自 ch01 示意日志 / ch02；其余按常见 nvidia-smi 读数取整，只用来判档，不参与精确计算
CARDS = {
    'a100-80g': dict(name='NVIDIA A100-SXM4-80GB', torch_gib=79.1, smi_mib=81920, cc=80),
    'h100-sxm': dict(name='NVIDIA H100 80GB HBM3', torch_gib=79.2, smi_mib=81559, cc=90),
    'h200':     dict(name='NVIDIA H200', torch_gib=139.8, smi_mib=143771, cc=90),
    'b200':     dict(name='NVIDIA B200', torch_gib=178.4, smi_mib=183359, cc=100),
    'l40s':     dict(name='NVIDIA L40S', torch_gib=44.4, smi_mib=46068, cc=89),
    'rtx4090':  dict(name='NVIDIA GeForce RTX 4090', torch_gib=23.6, smi_mib=24564, cc=89),
}


# --------------------------------------------------------------------------- vLLM 默认值
def vllm_batch_defaults(card, usage='api'):
    """arg_utils.py get_batch_defaults，只看 CUDA 分支；返回 (max_num_batched_tokens, max_num_seqs)。"""
    c = CARDS[card]
    if c['torch_gib'] >= 160:
        tok, seqs = {'llm': 16384, 'api': 16384}, {'llm': 1024, 'api': 1024}
    elif c['torch_gib'] >= 70 and 'a100' not in c['name'].lower() and c['cc'] != 121:
        tok, seqs = {'llm': 16384, 'api': 8192}, {'llm': 1024, 'api': 1024}
    else:
        tok, seqs = {'llm': 8192, 'api': 2048}, {'llm': 256, 'api': 256}
    return tok[usage], seqs[usage]


def vllm_resolve(card, usage='api', mode='balanced', max_model_len=40960, chunked=True,
                 user_tokens=None, user_seqs=None):
    """_set_default_max_num_seqs_and_batched_tokens_args 的四步后处理。"""
    tok, seqs = vllm_batch_defaults(card, usage)
    tok = user_tokens if user_tokens is not None else tok
    seqs = user_seqs if user_seqs is not None else seqs
    if mode == 'throughput':                      # 3190：只翻倍用户没设的那个
        if user_tokens is None:
            tok *= 2
        if user_seqs is None:
            seqs *= 2
    if user_tokens is None:
        if not chunked:
            tok = max(max_model_len, tok)
        tok = min(seqs * max_model_len, tok)
    if user_seqs is None:
        seqs = min(seqs, tok)
    return tok, seqs


def vllm_capture_sizes(max_num_seqs, max_num_batched_tokens, cc=90, mode='balanced', q_len=1):
    """config/vllm.py _set_cudagraph_sizes 默认分支。q_len>1（投机解码）时只补 uniform decode 尺寸的单层网格。"""
    default_max = 1024 if cc // 10 == 10 else 512
    cap = min(max_num_seqs * q_len * 2, default_max)
    uniform = []
    if q_len > 1:
        r = min(max_num_seqs, default_max)
        counts = [n for n in (1, 2, 4) if n <= r] + list(range(8, min(r + 1, 256), 8)) + list(range(256, r + 1, 16))
        uniform = sorted({n * q_len for n in set(counts + [r]) if n * q_len <= cap})
    elif max_num_seqs <= cap:
        uniform = [max_num_seqs]
    cap = min(max_num_batched_tokens, cap)
    if mode == 'interactivity':
        sizes = list(range(1, min(cap, 32) + 1))
    else:
        sizes = [i for i in (1, 2, 4) if i <= cap]
    if cap >= 8:
        sizes += list(range(8, min(cap + 1, 256), 8))
    if cap >= 256:
        sizes += list(range(256, cap + 1, 16))
    if max_num_batched_tokens <= cap and max_num_batched_tokens not in sizes:
        sizes.append(max_num_batched_tokens)
    sizes += [s for s in uniform if s <= max_num_batched_tokens]
    return sorted(set(sizes))


# --------------------------------------------------------------------------- SGLang 默认值
def sglang_tier(card, tp=1):
    """memory_hook.py handle_gpu_memory_settings 的显存分档；返回 (chunked_prefill_size, decode max_bs)。"""
    g = CARDS[card]['smi_mib']
    if g < 20 * 1024:
        return 2048, 8
    if g < 35 * 1024:
        return 4096, 48 if tp < 4 else 160
    if g < 60 * 1024:
        return 4096, 32 if tp < 4 else 160
    if g < 90 * 1024:
        return 8192, 256 if tp < 4 else 512
    if g < 160 * 1024:
        return 8192, 256 if tp < 4 else 512
    return 16384, 512


def sglang_decode_sizes(max_bs, spec=False):
    if spec:
        bs = (list(range(1, 9)) + list(range(10, 33, 2)) + list(range(40, 65, 4))
              + list(range(72, 257, 8)) + list(range(272, max_bs + 1, 16)))
    else:
        bs = [1, 2, 4, 8, 12] + list(range(16, 257, 8)) + list(range(272, 512, 16)) + list(range(512, max_bs + 1, 32))
    bs = [b for b in bs if b <= max_bs]
    if max_bs not in bs:
        bs.append(max_bs)
    return bs


def sglang_prefill_sizes(max_bs):
    bs = (list(range(4, 33, 4)) + list(range(48, 257, 16)) + list(range(288, 513, 32))
          + list(range(576, 1025, 64)) + list(range(1280, 4097, 256)) + list(range(4608, max_bs + 1, 512)))
    return [b for b in bs if b <= max_bs]


def sglang_mem_fraction(card, tp=1, pp=1, chunked=None, max_bs=None, mla=False):
    """memory_hook.py 281-325 + reserve_for_graph_mb；单位 MiB。返回 (reserved, mem_fraction_static)。"""
    g = CARDS[card]['smi_mib']
    c0, b0 = sglang_tier(card, tp)
    chunked = chunked or c0
    max_bs = max_bs or b0
    prefill_max = min(8192, chunked) if mla else chunked
    graph = max_bs * 2 + (1.5 * 1024 if mla else len(sglang_prefill_sizes(prefill_max)) * 8)
    reserved = 512 + max(chunked, 2048) * 1.5 + tp * pp / 8 * 1024 + graph
    if g > 60 * 1024:
        reserved = max(reserved, 10 * 1024)
    return reserved, round((g - reserved) / g, 3)



# --------------------------------------------------------------------------- 显存账（沿用 ch02 的 H100 + Qwen3-8B 数字）
LEDGER = dict(total_bytes=85_045_231_616, weights=15.27, nontorch=0.62, transient=1.43, cg_est=0.71)
BLOCK = 16


def vllm_kv(util, kv_bytes=None, model='qwen3-8b'):
    """gpu_worker.py determine_available_memory → (KV GiB, 块数, token 数)。kv_bytes 给定时模拟 --kv-cache-memory-bytes。"""
    L = LEDGER
    if kv_bytes is not None:
        avail = kv_bytes
    else:
        requested = math.ceil(L['total_bytes'] * util)
        avail = requested - (L['weights'] + L['nontorch'] + L['transient'] + L['cg_est']) * GiB
    page = MODELS[model].kv_bytes_per_token(2) * BLOCK
    blocks = int(avail // page)
    return avail / GiB, blocks, blocks * BLOCK


def sim_args(**kw):
    a = dict(scenario='mixed', mode='vllm', budget=8192, threshold=0, max_num_seqs=1024, max_model_len=40960,
             max_prefill_tokens=16384, rate=6.0, duration=120.0, long_at=2000.0, n=2000, bs=256, sync=False,
             sweep=False, seed=0)
    a.update(kw)
    return SimpleNamespace(**a)


def per_req(res):
    """每条请求的 TTFT、TPOT（首 token 之后的平均间隔），ms。"""
    out = []
    for r in res['reqs']:
        if not r.emits:
            continue
        ttft = r.emits[0] - r.arrival
        tpot = (r.emits[-1] - r.emits[0]) / (len(r.emits) - 1) if len(r.emits) > 1 else 0.0
        out.append((ttft, tpot))
    return out


def goodput(res, slo_ttft, slo_tpot):
    """DistServe 的口径：同时满足 TTFT 与 TPOT SLO 的请求数 / 墙钟时间。"""
    ok = sum(1 for t, p in per_req(res) if t <= slo_ttft and p <= slo_tpot)
    return ok / (res['makespan'] / 1e3), ok / max(1, res['n'])


def run_admission(a, mode, limit):
    """sim_chunked.run 的复刻，多一道 API server 准入：prefill 中的 prompt token 总和超过 limit 就 503。"""
    reqs = sc.workload(a)
    S = {'vllm': lambda: sc.Vllm(a), 'sglang': lambda: sc.Sglang(a)}[mode]()
    pending = sorted(reqs, key=lambda r: r.arrival)
    clock, accepted, rejected = 0.0, [], 0
    while pending or S.running or S.waiting:
        while pending and pending[0].arrival <= clock:
            r = pending.pop(0)
            backlog = sum(x.prompt for x in S.waiting) + sum(x.prompt for x in S.running if x.prefilling)
            if limit and backlog + r.prompt > limit:   # async_llm.py:371 按整条 prompt 记账，直到离开 prefill
                rejected += 1
                continue
            S.waiting.append(r)
            accepted.append(r)
        batch = S.schedule()
        if not batch:
            if not pending:
                break
            clock = pending[0].arrival
            continue
        g = sc.gpu_ms([(n, r.computed) for r, n in batch])
        c = sc.cpu_ms(len(batch))
        clock += max(c, g)
        emit_at = clock + 0.4 * c
        for r, n in batch:
            r.computed += n
            if not r.prefilling:
                r.gen += 1
                r.emits.append(emit_at)
                if r.gen >= r.out:
                    S.finish(r)
    ttft = [r.emits[0] - r.arrival for r in accepted]
    return dict(n=len(reqs), accepted=len(accepted), rejected=rejected,
                ttft50=sc.pct(ttft, 0.5), ttft99=sc.pct(ttft, 0.99), makespan=clock)


def prefill_tps(budget, decodes=64):
    """一步里 decodes 条 decode 先占位，剩下的预算全给一条从 0 开始的 prompt：每秒能吃掉多少 prompt token。"""
    chunk = budget - decodes
    g = sc.gpu_ms([(chunk, 0)] + [(1, 600)] * decodes)
    step = max(g, sc.cpu_ms(decodes + 1))
    return chunk / step * 1e3



# --------------------------------------------------------------------------- 场景
def show_defaults(a):
    print('== 不调参时各卡的默认值（Qwen3-8B，max_model_len 40960，TP=1）')
    print(f"{'卡':<9} {'vLLM API':>14} {'vLLM LLM 类':>14} {'API+throughput':>15} {'vLLM 图上限':>10} {'图个数':>6}"
          f" | {'SGL chunked':>11} {'SGL dec max_bs':>14} {'dec 图':>6} {'pre 图':>6} {'mem_frac':>8}")
    for card in CARDS:
        t, s = vllm_resolve(card, 'api')
        tl, sl = vllm_resolve(card, 'llm')
        tt, st = vllm_resolve(card, 'api', 'throughput')
        cg = vllm_capture_sizes(s, t, CARDS[card]['cc'])
        ch, mb = sglang_tier(card, a.tp)
        _, mf = sglang_mem_fraction(card, a.tp)
        print(f"{card:<9} {t:>6}/{s:<7} {tl:>6}/{sl:<7} {tt:>7}/{st:<7} {cg[-1]:>10} {len(cg):>6}"
              f" | {ch:>11} {mb:>14} {len(sglang_decode_sizes(mb)):>6} {len(sglang_prefill_sizes(ch)):>6} {mf:>8.3f}")
    print('  vLLM 列格式：max_num_batched_tokens / max_num_seqs；A100 80G 显存够 70 GiB 但卡名含 a100，落到小档（PR #17885）')
    print('  SGLang 只按 nvidia-smi 显存分档，不看卡名，A100 80G 与 H100 同档')
    print('  max_model_len 很短时 tokens 会被 max_num_seqs × max_model_len 截断，例：')
    for mml in (512, 2048):
        t, s = vllm_resolve('h100-sxm', 'api', max_model_len=mml)
        print(f'    H100 API max_model_len {mml}: {t}/{s}')
    t, s = vllm_resolve('h100-sxm', 'api', chunked=False)
    print(f'  关掉 chunked prefill：tokens 抬到 max(max_model_len, 默认) = {t}')
    t, s = vllm_resolve('h100-sxm', 'api', user_seqs=64)
    print(f'  只设 --max-num-seqs 64：tokens 仍走默认 {t}，图上限变成 {vllm_capture_sizes(64, t)[-1]}')


def show_mem(a):
    print('== gpu_memory_utilization 与 KV（H100 80GB + Qwen3-8B，账目沿用 ch02）')
    L = LEDGER
    print(f"   non_kv = 权重 {L['weights']} + 非 torch {L['nontorch']} + transient {L['transient']} + 图估计 {L['cg_est']} GiB")
    print(f"{'util':>6} {'KV GiB':>8} {'块':>8} {'token':>9} {'较 0.92':>9} {'满 40k 请求数':>12}")
    base = vllm_kv(0.92)[2]
    for u in (0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.97):
        g, b, tok = vllm_kv(u)
        print(f'{u:>6.2f} {g:>8.2f} {b:>8,} {tok:>9,} {tok - base:>+9,} {tok / 40960:>12.2f}')
    d = vllm_kv(0.93)[2] - vllm_kv(0.92)[2]
    print(f'   每 0.01 util ≈ {0.01 * L["total_bytes"] / GiB:.2f} GiB ≈ {d:,} token ≈ {d / 1024:.1f} 条 1k 上下文')
    g, b, tok = vllm_kv(None, kv_bytes=50 * GiB)
    print(f'   --kv-cache-memory-bytes 50GiB：{b:,} 块 {tok:,} token，util 不再参与，profiling 跳过')
    print('== SGLang mem_fraction_static 默认值（memory_hook.py，TP=1）')
    print(f"{'卡':<9} {'smi MiB':>8} {'chunked':>7} {'max_bs':>6} {'激活':>6} {'图':>6} {'reserved':>9} {'mem_frac':>8}")
    for card in CARDS:
        ch, mb = sglang_tier(card, a.tp)
        reserved, mf = sglang_mem_fraction(card, a.tp)
        graph = mb * 2 + len(sglang_prefill_sizes(ch)) * 8
        print(f"{card:<9} {CARDS[card]['smi_mib']:>8} {ch:>7} {mb:>6} {max(ch, 2048) * 1.5:>6.0f} {graph:>6} {reserved:>9,.0f} {mf:>8.3f}")
    print('   >60G 的卡 reserved 至少 10240 MiB；H100 算出 13,904 MiB → 0.830，与 ch02 一致')
    for tp in (1, 2, 4, 8):
        r, mf = sglang_mem_fraction('h100-sxm', tp)
        print(f'   H100 TP={tp}: max_bs {sglang_tier("h100-sxm", tp)[1]}, reserved {r:,.0f} MiB, mem_frac {mf:.3f}')


def pad_stats(sizes, max_running):
    """decode batch 从 1 到 max_running 均匀出现时：平均 padding 浪费、超出最大捕获尺寸走 eager 的比例。"""
    waste, eager = [], 0
    for b in range(1, max_running + 1):
        hit = next((s for s in sizes if s >= b), None)
        if hit is None:
            eager += 1
        else:
            waste.append((hit - b) / hit)
    return (sum(waste) / len(waste) if waste else 0.0), eager / max_running


def show_cudagraph(a):
    print('== vLLM 捕获网格（H100，max_num_batched_tokens 8192）')
    print(f"{'max_num_seqs':>12} {'模式':<13} {'q_len':>5} {'上限':>5} {'个数':>5} {'平均 padding':>12} {'eager 比例':>10}  头尾")
    for seqs, mode, q in ((64, 'balanced', 1), (128, 'balanced', 1), (256, 'balanced', 1), (1024, 'balanced', 1),
                          (1024, 'interactivity', 1), (256, 'balanced', 4), (1024, 'throughput', 1)):
        sz = vllm_capture_sizes(seqs, 8192, 90, mode, q)
        w, e = pad_stats(sz, seqs)
        print(f'{seqs:>12} {mode:<13} {q:>5} {sz[-1]:>5} {len(sz):>5} {w:>12.1%} {e:>10.1%}  {sz[:6]} … {sz[-3:]}')
    print('   max_num_seqs 1024 时上限被 512 卡住：batch 513–1024 全走 eager（Blackwell cc 10.x 上限 1024）')
    sz = vllm_capture_sizes(1024, 8192, 100)
    w, e = pad_stats(sz, 1024)
    print(f'   B200 同配置：上限 {sz[-1]}，{len(sz)} 个，eager 比例 {e:.1%}')
    print('== SGLang decode 网格（cuda_graph_hook.py）')
    for mb, spec in ((256, False), (512, False), (256, True), (48, False)):
        sz = sglang_decode_sizes(mb, spec)
        w, e = pad_stats(sz, mb)
        print(f'   max_bs {mb:>4} spec={str(spec):<5} {len(sz):>3} 个，平均 padding {w:.1%}  {sz[:6]} … {sz[-3:]}')
    w, e = pad_stats(sglang_decode_sizes(256), 1024)
    print(f'   H100 默认 max_bs 256 而 max_running 1024 时：batch>256 走 eager 的比例 {e:.1%}')


def show_budget(a):
    slo_t, slo_p = a.slo_ttft, a.slo_tpot
    print(f'== max_num_batched_tokens 扫描（06 的混合负载，{a.rate:g} req/s，{a.duration:g} s；SLO TTFT≤{slo_t:g} ms 且 TPOT≤{slo_p:g} ms）')
    print(f"{'预算':>6} {'模式':<7} {'输出 tok/s':>10} {'TTFT p50':>9} {'TTFT p99':>9} {'ITL p50':>8} {'ITL p99':>8} {'达标率':>6} {'goodput':>8}")
    best = {}
    for b in (512, 1024, 2048, 4096, 8192, 16384):
        for m in ('vllm', 'sglang'):
            r = sc.run(sim_args(budget=b, rate=a.rate, duration=a.duration), m)
            gp, frac = goodput(r, slo_t, slo_p)
            best.setdefault(m, []).append((gp, b))
            print(f"{b:>6} {m:<7} {r['out_tps']:>10,.0f} {sc.fmt(r['ttft50']):>9} {sc.fmt(r['ttft99']):>9} "
                  f"{sc.fmt(r['itl50']):>8} {sc.fmt(r['itl99']):>8} {frac:>6.1%} {gp:>8.2f}")
    for m, xs in best.items():
        gp, b = max(xs)
        print(f'   {m}: goodput 最高在预算 {b}（{gp:.2f} req/s）')


def show_seqs(a):
    print(f'== max_num_seqs 扫描（离线 burst：{a.n} 条同时到达，预算 8192）')
    g, blocks, tok = vllm_kv(0.92)
    print(f'   KV {tok:,} token；平均序列 ~{(0.85 * 1100 + 0.15 * 9160):.0f} token → KV 能同时装下约 {tok / (0.85 * 1100 + 0.15 * 9160):.0f} 条')
    print(f"{'max_num_seqs':>12} {'输出 tok/s':>10} {'耗时':>8} {'峰值 KV 占用':>12} {'ITL p50':>8} {'ITL p99':>8}")
    for s in (32, 64, 128, 256, 512, 1024):
        r = sc.run(sim_args(scenario='burst', n=a.n, max_num_seqs=s), 'vllm')
        print(f"{s:>12} {r['out_tps']:>10,.0f} {r['makespan'] / 1e3:>7.1f}s {r['peak_kv'] / sc.KV_CAP:>12.1%} "
              f"{sc.fmt(r['itl50']):>8} {sc.fmt(r['itl99']):>8}")
    print('   max_num_seqs 超过 KV 能装的条数之后，吞吐不再涨：真正的并发上限是 KV（占用略超 100% 是因为模拟器只按 prompt 准入、decode 增长不再检查）')


def show_queue(a):
    print(f'== max_num_queued_tokens = 目标 TTFT × prefill 吞吐（预算 8192，64 条 decode 占位）')
    tps = prefill_tps(8192)
    limit = int(a.ttft_ms / 1e3 * tps)
    print(f'   单卡 prefill 吞吐 ≈ {tps:,.0f} token/s；目标 TTFT {a.ttft_ms:g} ms → 上限 ≈ {limit:,} token')
    print(f"{'rate':>5} {'上限':>9} {'接收':>6} {'503':>5} {'TTFT p50':>9} {'TTFT p99':>9}")
    for rate in (10, 13, 16, 20):
        for lim in (None, limit, int(limit * 0.6)):
            r = run_admission(sim_args(rate=rate, duration=a.duration), 'vllm', lim)
            print(f"{rate:>5} {str(lim or '—'):>9} {r['accepted']:>6} {r['rejected']:>5} {sc.fmt(r['ttft50']):>9} {sc.fmt(r['ttft99']):>9}")
    print('   上限只管 prefill 积压，排队后还要和 decode 抢步，p99 会超过目标：线上按 0.6–0.8 倍起步再按实测收紧')
    print('   上限按整条 prompt 记账，部分 prefill 的请求直到离开 prefill 前都算满额（scheduler.py 注释），所以偏保守')


OPT = [
    ('O0', 'NONE', '全关', '关', '不编译、不捕获图：启动最快，调试用'),
    ('O1', 'PIECEWISE', 'norm/act + quant fusion', '开', 'Dynamo + Inductor 编译，只捕获 attention 之外的分段图'),
    ('O2', 'FULL_AND_PIECEWISE', 'O1 + allreduce_rms、attn_quant（量化模型）、rope_kvcache、qk_norm_rope_kvcache', '开', '默认档'),
    ('O3', 'FULL_AND_PIECEWISE', '同 O2', '开', '当前与 O2 完全相同，预留给更激进的优化'),
]


def show_opt(a):
    print('== OptimizationLevel（config/vllm.py:147-359，默认 O2）')
    print(f"{'档':<3} {'cudagraph_mode':<19} {'flashinfer autotune':<19} fusion / 说明")
    for lv, cg, fu, at, note in OPT:
        print(f'{lv:<3} {cg:<19} {at:<19} {fu}；{note}')
    print('   --enforce-eager 等价于把 cudagraph_mode 设成 NONE；编译缓存在 VLLM_CACHE_ROOT（默认 ~/.cache/vllm）')


def show_cpu(a):
    n = a.tp * a.dp
    api = a.api_servers or a.dp
    cores = api + a.dp + n + (1 if a.dp > 1 else 0)
    print(f'== CPU 核数（docs/configuration/optimization.md）TP={a.tp} DP={a.dp} API server={api}')
    print(f'   物理核下限 = A + DP + N + (DP>1) = {api} + {a.dp} + {n} + {1 if a.dp > 1 else 0} = {cores}；开超线程按 vCPU 算要 {2 * cores}')
    for tp, dp in ((1, 1), (8, 1), (2, 4), (1, 8)):
        n2 = tp * dp
        c = dp + dp + n2 + (1 if dp > 1 else 0)
        print(f'   TP={tp} DP={dp}（A 取默认 = DP）：{c} 物理核 / {2 * c} vCPU')
    print('   06 的时间模型里 CPU 每步 0.8 ms + 8 µs/请求：engine core 被抢核时这部分直接拉长 ITL')


SCEN = dict(defaults=show_defaults, mem=show_mem, cudagraph=show_cudagraph, budget=show_budget, seqs=show_seqs,
            queue=show_queue, opt=show_opt, cpu=show_cpu)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', choices=list(SCEN) + ['all'], default='defaults')
    ap.add_argument('--gpu', default='h100-sxm', choices=CARDS)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--dp', type=int, default=1)
    ap.add_argument('--api-servers', dest='api_servers', type=int, default=0)
    ap.add_argument('--rate', type=float, default=10.0)
    ap.add_argument('--duration', type=float, default=120.0)
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--slo-ttft', dest='slo_ttft', type=float, default=2000.0)
    ap.add_argument('--slo-tpot', dest='slo_tpot', type=float, default=30.0)
    ap.add_argument('--ttft-ms', dest='ttft_ms', type=float, default=2000.0)
    a = ap.parse_args()
    for name in (SCEN if a.scenario == 'all' else [a.scenario]):
        SCEN[name](a)
        print()


if __name__ == '__main__':
    main()
