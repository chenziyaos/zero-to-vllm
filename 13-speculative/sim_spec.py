"""投机解码模拟：期望加速比、batch 越大越不划算的 roofline、CUDA Graph 尺寸表、n-gram KMP、
Dynamic SD 查表与 SGLang adaptive 状态机、两边的指标口径（只依赖标准库）。

对照的源码（vllm 主干 0d6f92a，sglang 主干 8fabf5e）：
  vllm   config/vllm.py:641–710            num_lookahead_tokens / num_prefill_lookahead_tokens / uniform_decode_query_len
         config/vllm.py:2416–2650          _set_cudagraph_sizes：request_counts 网格、动态 SD 分档（本文件 capture_sizes 逐行移植）
         config/compilation.py:1508, 1537  MRV1 + FULL decode 时把尺寸向上取整到 1+k 的倍数（adjust_cudagraph_sizes_for_spec_decode）
         v1/worker/gpu/cudagraph_utils.py:259–350   MRV2 _init_candidates：每个 decode query len 一组 uniform decode 图
         v1/spec_decode/dynamic/utils.py   validate_and_normalize_dynamic_sd_schedule / build_dynamic_sd_schedule_lookup（逐行移植）
         v1/spec_decode/ngram_proposer.py:210–295   _find_longest_matched_ngram_and_propose_tokens（KMP，逐行移植）
         v1/core/sched/scheduler.py:372–380, 1308–1330, 2354–2366   每步预留块数、padding 到 1+k、拒绝后回退 num_computed_tokens
         v1/spec_decode/metrics.py:52–135  Mean acceptance length / Per-position acceptance rate
         config/speculative.py:618         _acceptance_length_to_rates；v1/spec_decode/utils.py:850 unconditional_to_conditional_rates
  sglang srt/speculative/adaptive_spec_params.py   DEFAULT_ADAPTIVE_CONFIG / AdaptiveStepSlot / _route（逐行移植）
         srt/server_args_hooks/speculative_hook.py:1471   _auto_choose_speculative_params；max_running_requests 重置为 48
         srt/arg_groups/fields/spec.py      speculative_* 参数与默认值
         srt/metrics/collector.py           sglang:spec_accept_length / sglang:spec_accept_rate

假设：theory / batch 场景用 Leviathan et al. 的 i.i.d. 接受模型（每个草稿 token 独立以 α 被接受）；
batch 场景是理想 roofline，不打折扣：一步时间 = max(读权重 + 读 KV 的字节 / 带宽, 线性层 + 注意力 FLOP / 峰值算力)，
草稿模型按 EAGLE-3 的形状估算（1 层 decoder + 3h→h 融合层 + 32k 草稿词表的 lm_head，约 0.40B 参数），
忽略 CPU 调度、kernel launch、采样和拒绝采样本身的开销，所以绝对值偏乐观，拐点位置可信。
n-gram 树形对照是简化模型（把所有历史匹配的后续合并成一棵 trie），不是 SGLang C++ 实现的移植。

用法：
  python3 sim_spec.py --scenario theory     # α、k、草稿成本 c → 期望接受长度、加速比、最优 k
  python3 sim_spec.py --scenario batch      # H100 + Llama-3.1-8B：batch 变大后最优 k 怎么往下掉
  python3 sim_spec.py --scenario cudagraph  # 开投机后 CUDA Graph 抓哪些尺寸、能覆盖多少并发、每步预留几个块
  python3 sim_spec.py --scenario ngram      # KMP 找最长后缀匹配；链式 vs 树形草稿
  python3 sim_spec.py --scenario dynamic    # vLLM 按 batch 查表 vs SGLang 按 EMA 自适应
  python3 sim_spec.py --scenario metrics    # 同一批验证结果，两边各报出什么数
  python3 sim_spec.py --scenario sglang     # 同一个需求两边怎么配
"""
import argparse
import bisect
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, MODELS  # noqa: E402


def cdiv(a, b):
    return -(-a // b)


def round_up(x, m):
    return cdiv(x, m) * m


# --------------------------------------------------------------------------- 1. 理论：期望接受长度与加速比
def expected_tokens(alpha, k):
    """一次验证平均产出多少 token（含 bonus）：1 + α + α² + ... + α^k（Leviathan et al. 2023 式 (1)）。"""
    if k == 0:
        return 1.0
    if alpha >= 1.0:
        return k + 1.0
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def speedup(alpha, k, c):
    """c = 草稿一步 / 目标一步 的时间比。一轮 = k 次草稿 + 1 次验证（验证 1+k 个 token 当作和 decode 1 个一样贵）。"""
    return expected_tokens(alpha, k) / (k * c + 1)


def best_k(alpha, c, kmax=16):
    return max(range(kmax + 1), key=lambda k: speedup(alpha, k, c))


def scenario_theory():
    print('== 期望接受长度 E = (1 - α^(k+1)) / (1 - α)，含 bonus token')
    ks = [1, 2, 3, 4, 5, 7, 10]
    print('  α     ' + ''.join(f'k={k:<6}' for k in ks))
    for a in (0.5, 0.6, 0.7, 0.8, 0.9):
        print(f'  {a:.1f}   ' + ''.join(f'{expected_tokens(a, k):<8.2f}' for k in ks))
    print('  上限 1/(1-α)：α=0.6 → 2.5，α=0.8 → 5，α=0.9 → 10。k 再大也越不过这条线。')

    print('\n== 加速比 = E / (k·c + 1)，c = 草稿一步相对目标一步的耗时')
    cases = [('n-gram（CPU 查表）', 0.0), ('EAGLE / MTP 头', 0.05), ('小草稿模型（1B 配 8B）', 0.15),
             ('大草稿模型（8B 配 70B）', 0.25)]
    for a in (0.6, 0.8):
        print(f'  α = {a}')
        for name, c in cases:
            kb = best_k(a, c)
            row = '  '.join(f'k{k}:{speedup(a, k, c):.2f}' for k in (1, 3, 5, 7))
            print(f'    {name:<22} c={c:<5} {row}   最优 k={kb:<2} → {speedup(a, kb, c):.2f}x')
    print('  c = 0 时最优 k 永远是上限（再多也不亏），所以 n-gram 的 k 只受 "匹配后面还有几个 token" 限制；')
    print('  c > 0 时有内点最优：α 每降 0.1，最优 k 掉一截。论文里的 2~3x 都是 α ≥ 0.7、batch = 1 的数。')


# --------------------------------------------------------------------------- 2. batch roofline
EAGLE3_DRAFT_PARAMS = None


def draft_params(m):
    """EAGLE-3 草稿头：1 层 decoder（GQA 注意力 + SwiGLU）+ 3h→h 融合 + 32000 草稿词表 lm_head。"""
    h, kvh, hd = m.hidden, m.kv_heads, m.head_dim
    inter = 14336 if h == 4096 else int(h * 3.5)
    attn = h * h * 2 + h * kvh * hd * 2
    mlp = 3 * h * inter
    fuse = 3 * h * h
    lm_head = h * 32000
    return attn + mlp + fuse + lm_head


def step_time(gpu, params, layers, kv_bytes_tok, heads_hd, batch, q_tokens, ctx, eff_bw=1.0, eff_fl=1.0):
    """一次前向的耗时（秒）：内存项 = 权重 + 所有请求的 KV；计算项 = 线性层 2PT + 注意力 4·ctx·heads·hd·layers·T。
    eff_bw / eff_fl = 实际能拿到的带宽 / 算力比例（理想 roofline 取 1）。"""
    bw = gpu.bw_tbs * 1e12 * eff_bw
    fl = gpu.bf16_tflops * 1e12 * eff_fl
    mem = (params * 2 + batch * ctx * kv_bytes_tok) / bw
    tokens = batch * q_tokens
    flops = 2 * params * tokens + 4 * ctx * heads_hd * layers * tokens
    return max(mem, flops / fl), mem, flops / fl


def spec_throughput(gpu, m, batch, k, alpha, ctx, eff_bw=1.0, eff_fl=1.0, draft_launch=0.0):
    """draft_launch = 每个草稿步额外的固定开销（秒）：kernel launch、采样、组 input，草稿头太小，这部分不可忽略。"""
    p_t = m.params_b * 1e9
    kv_t = m.kv_bytes_per_token()
    t_target, _, _ = step_time(gpu, p_t, m.layers, kv_t, m.heads * m.head_dim, batch, 1 + k, ctx, eff_bw, eff_fl)
    p_d = draft_params(m)
    kv_d = 2 * m.kv_heads * m.head_dim * 2
    t_draft, _, _ = step_time(gpu, p_d, 1, kv_d, m.heads * m.head_dim, batch, 1, ctx, eff_bw, eff_fl)
    t_draft += draft_launch
    t = t_target + k * t_draft
    return batch * expected_tokens(alpha, k) / t, t_target, t_draft


def scenario_batch():
    gpu, m = GPUS['h100-sxm'], MODELS['llama3.1-8b']
    ctx, alpha = 2048, 0.75
    p_d = draft_params(m)
    print(f'== {m.name} @ {gpu.name}，平均上下文 {ctx}，α = {alpha}（EAGLE-3 量级），理想 roofline')
    print(f'  目标模型 {m.params_b:.1f}B 参数，权重 {m.params_b * 2:.1f} GB；KV {m.kv_bytes_per_token() / 1024:.0f} KB/token')
    print(f'  草稿头 {p_d / 1e9:.2f}B 参数（1 层 decoder + 融合层 + 32k 词表 lm_head），KV 4 KB/token')
    ridge = gpu.bf16_tflops / gpu.bw_tbs
    print(f'  H100 拐点 {ridge:.0f} FLOP/B；只读权重时，一次前向要 ~{ridge / 2 * 2:.0f} 个 token 才吃满算力'
          f'（2 FLOP/参数 vs 2 B/参数）')
    kv_cap = (gpu.mem_gb * 0.9 - m.params_b * 2) * 1e9 / (m.kv_bytes_per_token() * ctx)
    print(f'  显存（util 0.9）最多放 {kv_cap:.0f} 个这样的请求，超过的行标 *')

    ks = list(range(0, 8))
    print('\n  batch  ' + ''.join(f'k={k:<6}' for k in ks) + '最优 k   t_target(k=0)  草稿/目标')
    for b in (1, 4, 16, 32, 64, 128, 192, 256):
        base, t0, td = spec_throughput(gpu, m, b, 0, alpha, ctx)
        row = []
        best = (0, 1.0)
        for k in ks:
            th, _, _ = spec_throughput(gpu, m, b, k, alpha, ctx)
            s = th / base
            row.append(f'{s:<8.2f}')
            if s > best[1] + 1e-9:
                best = (k, s)
        flag = '*' if b > kv_cap else ' '
        print(f'  {b:>4}{flag}  ' + ''.join(row) + f'{best[0]:<8} {t0 * 1e3:6.2f} ms      {td / t0:.3f}')
    print('  （表内是相对 k=0 的吞吐倍数；同一 batch 下的 ITL 倍数与之相同）')

    print('\n== 为什么会掉：把 k=3 拆开看内存项和计算项')
    for b in (1, 32, 64, 128, 256):
        _, mem, comp = step_time(gpu, m.params_b * 1e9, m.layers, m.kv_bytes_per_token(), m.heads * m.head_dim,
                                 b, 4, ctx)
        _, mem0, comp0 = step_time(gpu, m.params_b * 1e9, m.layers, m.kv_bytes_per_token(),
                                   m.heads * m.head_dim, b, 1, ctx)
        bound = '算力' if comp > mem else '带宽'
        print(f'  batch {b:>3}: 验证 {b * 4:>4} token  内存 {mem * 1e3:6.2f} ms  计算 {comp * 1e3:6.2f} ms  → {bound}瓶颈；'
              f'k=0 时计算 {comp0 * 1e3:5.2f} ms')
    print('  理想 roofline 下 8B 模型在 256 并发时仍是带宽瓶颈，所以最优 k 只从 7 掉到 5——这是上界，不是现实。')

    print('\n== 打折扣：带宽拿到 80%、算力只拿到 35%（大 batch decode 的典型 MFU）、每个草稿步固定 0.3 ms')
    for a in (0.75, 0.6):
        print(f'  α = {a}')
        print('  batch  ' + ''.join(f'k={k:<6}' for k in ks) + '最优 k')
        for b in (1, 16, 32, 64, 128, 256):
            kw = dict(eff_bw=0.8, eff_fl=0.35, draft_launch=3e-4)
            base, _, _ = spec_throughput(gpu, m, b, 0, a, ctx, **kw)
            sp = [spec_throughput(gpu, m, b, k, a, ctx, **kw)[0] / base for k in ks]
            kb = max(ks, key=lambda k: sp[k])
            print(f'  {b:>4}   ' + ''.join(f'{x:<8.2f}' for x in sp) + f'{kb}')
    print('  α 掉到 0.6、batch 256 时最优 k=1，k ≥ 5 直接比不开投机还慢——和 vLLM 博客 "高 QPS 下 1.4~1.8x 变慢" 同一个方向。')
    print('  验证 1+k 个 token 的 "免费" 只在带宽瓶颈区成立；一旦 batch·(1+k) 越过拐点，被拒的草稿 token 都是真金白银的 FLOP。')
    print('  这就是 vLLM 文档里 "effective BS becomes BS*K" 的意思，也是 Dynamic SD 和 SGLang adaptive 存在的理由。')


# --------------------------------------------------------------------------- 3. CUDA Graph 尺寸
def request_counts(max_reqs, default_max_graph_size):
    max_reqs = min(max_reqs, default_max_graph_size)
    counts = [n for n in (1, 2, 4) if n <= max_reqs]
    counts += list(range(8, min(max_reqs + 1, 256), 8))
    counts += list(range(256, max_reqs + 1, 16))
    return sorted(set(counts + [max_reqs]))


def capture_sizes(max_num_seqs, max_num_batched_tokens, num_spec, schedule=None, sm100=False,
                  interactivity=False):
    """config/vllm.py _set_cudagraph_sizes 的默认分支（max_cudagraph_capture_size 和 capture_sizes 都没指定）。"""
    default_max_graph_size = 1024 if sm100 else 512
    decode_query_len = 1 + num_spec
    max_cap = min(max_num_seqs * decode_query_len * 2, default_max_graph_size)
    uniform_decode_sizes = []
    if decode_query_len > 1:
        decode_tiers = [(decode_query_len, max_num_seqs)]
        if schedule is not None:
            try:
                dense = build_dynamic_sd_schedule_lookup(schedule, max_num_seqs, num_spec)
            except ValueError:
                pass
            else:
                widest_batch = {}
                for batch_size, k in enumerate(dense[1:], start=1):
                    widest_batch[k + 1] = batch_size
                decode_tiers = list(widest_batch.items())
        uniform_decode_sizes = sorted({
            n * q for q, tmax in decode_tiers for n in request_counts(tmax, default_max_graph_size)
            if n * q <= max_cap
        })
    elif max_num_seqs <= max_cap:
        uniform_decode_sizes = [max_num_seqs]
    max_num_tokens = max_num_batched_tokens
    max_cap = min(max_num_tokens, max_cap)
    if interactivity:
        sizes = list(range(1, min(max_cap, 32) + 1))
    else:
        sizes = [i for i in (1, 2, 4) if i <= max_cap]
    if max_cap >= 8:
        sizes += list(range(8, min(max_cap + 1, 256), 8))
    if max_cap >= 256:
        sizes += list(range(256, max_cap + 1, 16))
    if max_num_tokens <= max_cap and max_num_tokens not in sizes:
        sizes.append(max_num_tokens)
    sizes += [s for s in uniform_decode_sizes if s <= max_num_tokens]
    return sorted(set(sizes))


def mrv1_round_up(sizes, qlen):
    """config/compilation.py adjust_cudagraph_sizes_for_spec_decode（只在 MRV1 + FULL decode + qlen > 1 时调用）。"""
    if not sizes or qlen <= 1:
        return list(sizes)
    max_cap = sizes[-1]
    rounded = sorted({round_up(s, qlen) for s in sizes if round_up(s, qlen) <= max_cap})
    if not rounded and qlen <= max_cap:
        rounded = [qlen]
    if not rounded:
        raise ValueError(f'No valid cudagraph sizes after rounding to multiple of {qlen} (num_speculative_tokens + 1)')
    return rounded


def mrv2_uniform_decode(sizes, max_num_reqs, num_spec, schedule=None):
    """v1/worker/gpu/cudagraph_utils.py _init_candidates 的 uniform decode 分支：返回 {qlen: [(tokens, reqs), ...]}。"""
    dql = 1 + num_spec
    max_decode_tokens = max_num_reqs * dql
    max_cg = sizes[-1]
    if schedule is not None and dql > num_spec:
        n_new = dql - num_spec
        dense = build_dynamic_sd_schedule_lookup(schedule, max_num_reqs, num_spec)
        qlens = sorted({k + n_new for k in dense[1:]})
    else:
        qlens = [dql]
    out = {q: [] for q in qlens}
    for s in sorted(sizes):
        for q in qlens:
            t = round_up(s, q)
            r = t // q
            if t > max_decode_tokens or t > max_cg or r > max_num_reqs:
                continue
            if (t, r) not in out[q]:
                out[q].append((t, r))
    return out


def step_blocks(num_spec, lookahead, block_size=16):
    """v1/core/sched/scheduler.py:372：每个请求每步要预留的 KV 块数（异步 KV 加载准入时用）。"""
    n = 1 + num_spec + lookahead if num_spec else 0
    return cdiv(n, block_size)


def lookahead_of(method, k):
    """config/vllm.py num_lookahead_tokens：DFlash 要 k+1，EAGLE 家族和 draft_model 要 k，ngram / suffix 是 0。"""
    if method == 'dflash':
        return k + 1
    if method in ('eagle', 'eagle3', 'mtp', 'dspark', 'draft_model'):
        return k
    return 0


def scenario_cudagraph():
    seqs, mnbt = 1024, 8192   # H100 80GB 上 vllm serve 的默认（engine/arg_utils.py:2979）
    print(f'== H100 默认 max_num_seqs={seqs}、max_num_batched_tokens={mnbt}（API server），capture 上限 512')
    print('  k   qlen  尺寸个数  最大尺寸  MRV1 取整后个数  MRV1 FULL decode 最多覆盖并发   MRV2 uniform decode 图个数  最多覆盖并发')
    for k in (0, 1, 3, 5, 7, 15):
        s = capture_sizes(seqs, mnbt, k)
        q = 1 + k
        if q > 1:
            r = mrv1_round_up(s, q)
            v1_cover = r[-1] // q
            u = mrv2_uniform_decode(s, seqs, k)[q]
            v2_n, v2_cover = len(u), max(x[1] for x in u)
            print(f'  {k:<3} {q:<5} {len(s):<9} {s[-1]:<9} {len(r):<16} {v1_cover:<30} {v2_n:<27} {v2_cover}')
        else:
            print(f'  {k:<3} {q:<5} {len(s):<9} {s[-1]:<9} {"-":<16} {s[-1]:<30} {"-":<27} -')
    print('  读法：capture 上限按 token 算（512），开投机后每个请求一步占 1+k 个 token，FULL decode 图能覆盖的并发直接除以 1+k。')
    print('  超过的并发不会报错，只是这一步退回 PIECEWISE / eager；k 越大，越早掉出 FULL 图。')

    s3 = capture_sizes(seqs, mnbt, 3)
    print('\n== k=3 时实际抓的尺寸（前 40 个）')
    print('  ', s3[:40])
    print('  MRV1 取整到 4 的倍数：', mrv1_round_up(s3, 4)[:24], '...')
    s0 = set(capture_sizes(seqs, mnbt, 0))
    print(f'  k=3 时 uniform_decode_sizes（4 × request_counts）全部落在默认网格上，额外尺寸 {sorted(set(s3) - s0)}；'
          f'k=5 时 6 × n 补进来 {sorted(set(capture_sizes(seqs, mnbt, 5)) - s0)}，所以是 53 个。')

    print('\n== 源码注释里的两个数（config/vllm.py _set_cudagraph_sizes）')
    k = 16
    grid_ext = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 512 * (k + 1) + 1, 16))
    rc17 = {n * (k + 1) for n in request_counts(512, 512)}
    grid512 = [s for s in grid_ext if s <= 512]
    print(f'  "581 sizes"：token 网格延长到 512×17={512 * 17} 有 {len(set(grid_ext))} 个，再并上 17×request_counts '
          f'→ {len(set(grid_ext) | rc17)} 个；其中是 17 倍数、真能被 decode 用上的只有 '
          f'{sum(1 for s in set(grid_ext) if s % 17 == 0)} 个网格点。')
    print(f'  "100 with the request-count grid"：默认 512 网格 {len(grid512)} 个 ∪ 17×request_counts {len(rc17)} 个 '
          f'→ {len(set(grid512) | rc17)} 个（两者重合 {len(set(grid512) & rc17)} 个）。')
    print(f'  默认上限 512 下真正抓的是 {len(capture_sizes(512, 16384, 16))} 个：17×n 只补到 n=24（408 token），'
          f'更宽的 decode 走 eager，除非手动放大 capture 范围。')
    print(f'  "at query length 17 a captured 560 rounds to 561 and is rejected"：round_up(560, 17) = {round_up(560, 17)} > 560。')

    print('\n== 动态 SD（文档示例 [[1,64,3],[65,128,1],[129,512,0]]，max_num_seqs=512，k=3）')
    sched = [[1, 64, 3], [65, 128, 1], [129, 512, 0]]
    sd = capture_sizes(512, mnbt, 3, schedule=sched)
    single = capture_sizes(512, mnbt, 3)
    print(f'  按档补尺寸：{len(sd)} 个；只按最宽档 qlen=4 补：{len(single)} 个')
    u = mrv2_uniform_decode(sd, 512, 3, schedule=sched)
    u1 = mrv2_uniform_decode(single, 512, 3, schedule=sched)
    for q in sorted(u):
        print(f'  MRV2 qlen={q}: 按档 {len(u[q]):>3} 个图，最多覆盖 {max(x[1] for x in u[q]):>3} 个请求；'
              f'只按最宽档补 {len(u1[q]):>3} 个图，最多覆盖 {max(x[1] for x in u1[q]):>3} 个请求')
    print('  MRV1 下动态 SD 直接把 FULL 改成 PIECEWISE（config/vllm.py:1084 的 WARNING），只有 MRV2 才按档抓 FULL 图。')

    print('\n== 每步每请求预留的 KV 块（block_size 16）：1 + k + lookahead')
    for method, k in (('ngram', 5), ('eagle3', 3), ('eagle3', 7), ('mtp', 3), ('dflash', 15)):
        la = lookahead_of(method, k)
        print(f'  {method:<7} k={k:<3} lookahead={la:<3} → {1 + k + la:>3} 个 slot = {step_blocks(k, la)} 块')
    print('  EAGLE 的草稿头要给 k 个草稿位置写自己的 KV，所以 lookahead = k；DFlash 多一个 bonus query，lookahead = k+1。')
    print('  uniform_decode_query_len 永远是 1+k（不是 1+lookahead）：DFlash 预留 k+1 个 slot，但验证只有 1+k 个 token。')


# --------------------------------------------------------------------------- 4. n-gram：KMP
def find_longest_ngram_propose(origin_tokens, min_ngram, max_ngram, max_model_len, k):
    """v1/spec_decode/ngram_proposer.py _find_longest_matched_ngram_and_propose_tokens，逐行移植（numba 换成纯 Python）。"""
    total_token = len(origin_tokens)
    if total_token < min_ngram:
        return []
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return []
    tokens = origin_tokens[::-1]
    lps = [0] * max_ngram
    longest_ngram = 0
    position = 0
    prev_lps = 0
    i = 1
    while i < total_token:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1
    if longest_ngram < min_ngram:
        return []
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return list(origin_tokens[start_position:start_position + k])


def brute_force_propose(tokens, min_n, max_n, max_model_len, k):
    """对照实现：从长到短找后缀在前文最早出现的位置。和 KMP 版语义一致（最长优先，同长取最早）。"""
    total = len(tokens)
    if total < min_n:
        return []
    k = min(k, max_model_len - total)
    if k <= 0:
        return []
    for n in range(min(max_n, total - 1), min_n - 1, -1):
        suffix = tokens[total - n:]
        for start in range(0, total - n):
            if tokens[start:start + n] == suffix:
                s = start + n
                return list(tokens[s:s + min(k, total - s)])
    return []


def trie_drafts(tokens, min_n, max_n, budget):
    """简化的树形草稿：所有和后缀匹配（长度 ≥ min_n）的历史位置，把后续 token 合并成一棵 trie，按出现次数 BFS 取 budget 个节点。"""
    total = len(tokens)
    children = {}
    for n in range(min(max_n, total - 1), min_n - 1, -1):
        suffix = tokens[total - n:]
        hits = [s for s in range(0, total - n) if tokens[s:s + n] == suffix]
        if hits:
            for s in hits:
                node = ()
                for t in tokens[s + n:s + n + budget]:
                    node = node + (t,)
                    children[node] = children.get(node, 0) + 1
            break
    ranked = sorted(children.items(), key=lambda kv: (len(kv[0]), -kv[1]))
    return {path for path, _ in ranked[:budget]}


def make_copy_stream(seed, vocab=50, length=600, p_copy=0.6, seg=(4, 20)):
    """生成一段 "会抄前文" 的 token 流：以 p_copy 概率从历史里抄一段，否则随机写一段。模拟 RAG / 代码修改 / 多轮对话。"""
    rng = random.Random(seed)
    out = [rng.randrange(vocab) for _ in range(30)]
    while len(out) < length:
        n = rng.randint(*seg)
        if rng.random() < p_copy and len(out) > n + 1:
            s = rng.randrange(0, len(out) - n)
            out.extend(out[s:s + n])
        else:
            out.extend(rng.randrange(vocab) for _ in range(n))
    return out[:length]


def replay_ngram(stream, prompt_len, min_n, max_n, k, tree_budget=0):
    """在已知 "真实后续" 上回放：每步提草稿，贪心验证（草稿 token 等于真实下一个就接受），统计每步产出。"""
    pos = prompt_len
    steps = 0
    produced = 0
    while pos < len(stream) - 1:
        ctx = stream[:pos]
        if tree_budget:
            paths = trie_drafts(ctx, min_n, max_n, tree_budget)
            acc = 0
            while tuple(stream[pos:pos + acc + 1]) in paths and pos + acc + 1 < len(stream):
                acc += 1
        else:
            d = find_longest_ngram_propose(ctx, min_n, max_n, 1 << 30, k)
            acc = 0
            for t in d:
                if pos + acc < len(stream) and stream[pos + acc] == t:
                    acc += 1
                else:
                    break
        step = acc + 1
        pos += step
        produced += step
        steps += 1
    return produced / steps


def scenario_ngram():
    words = 'the cache hit rate of the prefix cache depends on the prefix cache size and the prefix'.split()
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words))}
    toks = [vocab[w] for w in words]
    inv = {i: w for w, i in vocab.items()}
    print('== KMP：在反转后的序列上算前缀函数，找 "和结尾相同的最长 n-gram" 在前文的最早位置')
    print('  上下文：', ' '.join(words))
    for mn, mx in ((1, 3), (2, 2), (3, 5), (5, 5)):
        d = find_longest_ngram_propose(toks, mn, mx, 1 << 30, 4)
        print(f'  prompt_lookup_min={mn} max={mx}, k=4 → 草稿 {[inv[t] for t in d] or "（无匹配，这步不投机）"}')
    print('  默认 min = max = 5（config/speculative.py，"TODO(woosuk): Tune these values"）：只有 5-gram 完全重合才出草稿。')
    print('  只写 prompt_lookup_max=4（文档示例）时，1393 行把 min 也设成 4：docstring 写 "Defaults to 1"，代码是 min = max。')
    print('  想要 "找不到 4-gram 就退到更短的"，必须显式写 prompt_lookup_min，例如 min=2、max=4。')

    print('\n== 复杂度：每步 O(上下文长度)，numba 并行；8192 token 以上才开多线程（num_tokens_threshold）')
    print('  注意源码里 num_numba_thread_available = min(1, cpu_count // 2)，再除以 TP，实际上限 1 线程（TODO 写着要提到 8）。')

    print('\n== "会抄前文" 的流（600 token，词表 50，60% 概率抄一段 4~20 token 的历史）上贪心回放，每步平均产出 token')
    print('  p_copy  链式 k=3   链式 k=5   链式 k=8   min=max=5   树 12 节点 min=2')
    for pc in (0.0, 0.3, 0.6, 0.9):
        st = make_copy_stream(7, p_copy=pc)
        a = replay_ngram(st, 100, 2, 4, 3)
        b = replay_ngram(st, 100, 2, 4, 5)
        c = replay_ngram(st, 100, 2, 4, 8)
        d = replay_ngram(st, 100, 5, 5, 5)
        e = replay_ngram(st, 100, 2, 4, 0, tree_budget=12)
        print(f'  {pc:<7} {a:<10.2f} {b:<10.2f} {c:<10.2f} {d:<11.2f} {e:.2f}')
    print('  p_copy = 0 时几乎没有收益（随机流里偶然匹配），这时验证 1+k 个 token 的计算全浪费——高 QPS 下就是负收益。')
    print('  树形草稿在 "同一个前缀后面跟过不同内容" 时更稳：链式只押最早那次出现，树把所有出现过的后续都摆上去。')
    print('  SGLang 的 NGRAM 就是树：默认 12 个草稿 token、BFS 宽度 1~10、trie 深度 18，只看输入和输出最后 18 个 token；')
    print('  vLLM 的 ngram 是链：没有 tree attention，一条草稿 k 个 token。')


# --------------------------------------------------------------------------- 5. Dynamic SD vs SGLang adaptive
def validate_and_normalize_dynamic_sd_schedule(schedule):
    if schedule is None:
        raise ValueError('num_speculative_tokens_per_batch_size is required for dynamic speculative decoding.')
    if not isinstance(schedule, list):
        raise ValueError('num_speculative_tokens_per_batch_size must be a non-empty list of '
                         '(range_start, range_end, num_speculative_tokens) entries.')
    if not schedule:
        raise ValueError('num_speculative_tokens_per_batch_size must not be empty.')
    parsed = []
    for entry in schedule:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError('Each num_speculative_tokens_per_batch_size entry must be a 3-item sequence: '
                             '(range_start, range_end, num_speculative_tokens).')
        a, b, k = int(entry[0]), int(entry[1]), int(entry[2])
        if a <= 0 or b <= 0:
            raise ValueError(f'Batch-size range ({a}, {b}) must be positive.')
        if a > b:
            raise ValueError(f'Batch-size range start must be <= end for ({a}, {b}, {k}).')
        if k < 0:
            raise ValueError('num_speculative_tokens_per_batch_size values must be >= 0.')
        parsed.append((a, b, k))
    parsed.sort(key=lambda e: e[0])
    prev_end = 0
    for a, b, _ in parsed:
        if a <= prev_end:
            raise ValueError('Batch-size ranges must be non-overlapping and sorted.')
        prev_end = b
    if parsed[0][0] != 1:
        raise ValueError('The first batch-size range must start at 1 so every runtime batch size has a defined schedule.')
    return parsed


def build_dynamic_sd_schedule_lookup(schedule, vllm_max_batch_size, vllm_num_speculative_tokens):
    if vllm_max_batch_size <= 0:
        raise ValueError('vllm_max_batch_size must be > 0.')
    if vllm_num_speculative_tokens <= 0:
        raise ValueError('vllm_num_speculative_tokens must be > 0.')
    parsed = validate_and_normalize_dynamic_sd_schedule(schedule)
    dense = [0] * (vllm_max_batch_size + 1)
    nxt = 1
    last = None
    for a, b, k in parsed:
        if a > nxt and last is not None:
            for bs in range(nxt, min(a, vllm_max_batch_size + 1)):
                dense[bs] = min(vllm_num_speculative_tokens, last)
        for bs in range(max(a, nxt), min(b, vllm_max_batch_size) + 1):
            dense[bs] = min(vllm_num_speculative_tokens, k)
        nxt = max(nxt, b + 1)
        last = k
        if nxt > vllm_max_batch_size:
            break
    if last is None:
        raise ValueError('num_speculative_tokens_per_batch_size must contain at least one valid batch-size range.')
    for bs in range(nxt, vllm_max_batch_size + 1):
        dense[bs] = min(vllm_num_speculative_tokens, last)
    return dense


SGL_DEFAULT_ADAPTIVE_CONFIG = {
    '1': {'candidate_steps': [1, 3, 5, 7], 'up_hysteresis': 0.0, 'down_hysteresis': -0.25, 'ceiling_coeff': 0},
    '8': {'candidate_steps': [0, 1, 3], 'up_hysteresis': 0.0, 'down_hysteresis': 0.0, 'ceiling_coeff': 0},
    '32': {'candidate_steps': [0, 1], 'up_hysteresis': 0.0, 'down_hysteresis': 0.0, 'ceiling_coeff': 0},
    '64': {'candidate_steps': [0], 'up_hysteresis': 0.0, 'down_hysteresis': 0.0, 'ceiling_coeff': 0},
}


class AdaptiveStepSlot:
    """sglang srt/speculative/adaptive_spec_params.py AdaptiveStepSlot，逐行移植（日志换成 self.log）。"""

    def __init__(self, initial_steps, cfg):
        self.candidate_steps = sorted(set(cfg['candidate_steps']))
        self.ema_alpha = cfg.get('ema_alpha', 0.2)
        self.update_interval = cfg.get('update_interval', 5)
        self.warmup_batches = cfg.get('warmup_batches', 10)
        self.down_hysteresis = cfg.get('down_hysteresis', -0.25)
        self.up_hysteresis = cfg.get('up_hysteresis', 0.0)
        self.ceiling_coeff = cfg.get('ceiling_coeff', 0)
        if initial_steps in self.candidate_steps:
            self.current_steps = initial_steps
        else:
            self.current_steps = self.candidate_steps[len(self.candidate_steps) // 2]
        self.ema_accept_len = float(self.current_steps - 1)
        self._batch_count = 0
        self.log = []

    def update(self, num_correct_drafts_per_req):
        if not num_correct_drafts_per_req:
            return False
        if self.current_steps > 0:
            batch_avg = sum(num_correct_drafts_per_req) / len(num_correct_drafts_per_req)
            self.ema_accept_len = (1 - self.ema_alpha) * self.ema_accept_len + self.ema_alpha * batch_avg
        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            return False
        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False
        return self._recompute_params()

    def _recompute_params(self):
        old_steps = self.current_steps
        current_idx = self.candidate_steps.index(old_steps)
        old_idx = current_idx
        if old_steps == 0:
            current_idx = min(current_idx + 1, len(self.candidate_steps) - 1)
            target = self.candidate_steps[current_idx]
            if target > 0 and self.ema_accept_len < 0:
                self.ema_accept_len = float(target - 1)
            return self._apply(old_steps, target)
        while current_idx > 0:
            prev_step = self.candidate_steps[current_idx - 1]
            drop = (0.5 if prev_step == 0 else prev_step - 0.5) + self.down_hysteresis
            if self.ema_accept_len <= drop:
                current_idx -= 1
            else:
                break
        if not current_idx < old_idx:
            while current_idx < len(self.candidate_steps) - 1:
                cur = self.candidate_steps[current_idx]
                if self.ema_accept_len > cur - 0.5 + self.up_hysteresis:
                    current_idx += 1
                else:
                    break
        target = self.candidate_steps[current_idx]
        if self.ceiling_coeff > 0:
            ceiling = max(1, math.ceil(self.ema_accept_len * self.ceiling_coeff))
            if target > ceiling and target <= old_steps:
                while current_idx > 0 and self.candidate_steps[current_idx] > ceiling:
                    current_idx -= 1
                target = self.candidate_steps[current_idx]
        return self._apply(old_steps, target)

    def _apply(self, old, target):
        if target != old:
            self.current_steps = target
            self.log.append((self._batch_count, old, target, round(self.ema_accept_len, 2)))
            return True
        return False


class AdaptiveSpeculativeParams:
    """按 batch size 路由到 slot：先 pad 到 CUDA Graph 尺寸（bisect_left），再向下取最近的 slot（bisect_right - 1）。"""

    def __init__(self, initial_steps, cfg=None, cuda_graph_bs=None):
        cfg = cfg or SGL_DEFAULT_ADAPTIVE_CONFIG
        entries = {int(k): v for k, v in cfg.items() if k.isdigit()}
        self.bs_list = sorted(entries)
        self.slots = {bs: AdaptiveStepSlot(initial_steps, {**cfg, **entries[bs]}) for bs in self.bs_list}
        self.cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs else None

    def pad(self, bs):
        if self.cuda_graph_bs is None:
            return bs
        i = bisect.bisect_left(self.cuda_graph_bs, bs)
        return self.cuda_graph_bs[i] if i < len(self.cuda_graph_bs) else bs

    def closest(self, t):
        i = bisect.bisect_right(self.bs_list, t) - 1
        return self.bs_list[max(0, i)]

    def route(self, bs):
        return self.slots[self.closest(self.pad(bs))]


def draw_accepts(rng, alpha, steps, n):
    out = []
    for _ in range(n):
        a = 0
        while a < steps and rng.random() < alpha:
            a += 1
        out.append(a)
    return out


def scenario_dynamic():
    print('== vLLM：按 batch size 查一张静态表（启动时展开成稠密数组，调度时 dense[len(num_scheduled_tokens)]）')
    for sched, mx, k in (([[1, 64, 3], [65, 128, 1], [129, 512, 0]], 512, 3),
                         ([[1, 16, 3], [32, 128, 2]], 160, 3),
                         ([[1, 8, 7], [9, 32, 4]], 64, 5)):
        d = build_dynamic_sd_schedule_lookup(sched, mx, k)
        segs = []
        start = 1
        for bs in range(2, mx + 2):
            if bs == mx + 1 or d[bs] != d[start]:
                segs.append(f'{start}-{bs - 1}:k={d[start]}')
                start = bs
        print(f'  {str(sched):<38} max_num_seqs={mx:<4} k={k} → {"  ".join(segs)}')
    print('  空档（17~31）沿用前一档；尾部沿用最后一档；每档的 k 被 num_speculative_tokens 截顶（第三行 7 → 5）。')
    for bad in ([[2, 8, 3]], [[1, 8, 3], [8, 16, 1]], [[1, 8, -1]]):
        try:
            validate_and_normalize_dynamic_sd_schedule(bad)
        except ValueError as e:
            print(f'  {str(bad):<22} → ValueError: {e}')

    print('\n== SGLang：每个 BS 档一个 EMA 状态机，按观测到的接受长度调 speculative_num_steps')
    print('  默认四档 DEFAULT_ADAPTIVE_CONFIG：', {k: v['candidate_steps'] for k, v in SGL_DEFAULT_ADAPTIVE_CONFIG.items()})
    print('  ema_alpha 0.2、warmup 10 批、每 5 批重算一次；升档门槛 ema > 当前步数 - 0.5，降档门槛 ema <= 下一档 - 0.5 + 滞回')
    p = AdaptiveSpeculativeParams(initial_steps=3, cuda_graph_bs=[1, 2, 4, 8, 16, 24, 32, 40, 48])
    for bs in (1, 3, 5, 9, 31, 33, 48, 60, 100):
        slot_bs = p.closest(p.pad(bs))
        print(f'  batch {bs:>3} → pad {p.pad(bs):>3} → slot "{slot_bs}" candidate_steps={p.slots[slot_bs].candidate_steps}')
    print('  batch 5 被 pad 到 8 → 落 slot "8"；batch 3 pad 到 4 → slot "1"。≥64 只有 [0]：大 batch 直接关掉投机。')

    print('\n== slot "1" 的轨迹：先 120 批 α=0.9，再 120 批 α=0.4，再 120 批 α=0.75（初始 steps=3，SGLang 自动参数的默认）')
    rng = random.Random(0)
    slot = AdaptiveStepSlot(3, {**SGL_DEFAULT_ADAPTIVE_CONFIG, **SGL_DEFAULT_ADAPTIVE_CONFIG['1']})
    for phase, alpha in ((0, 0.9), (1, 0.4), (2, 0.75)):
        for _ in range(120):
            slot.update(draw_accepts(rng, alpha, slot.current_steps, 1))
    for batch, old, new, ema in slot.log:
        print(f'  第 {batch:>3} 批：steps {old} -> {new}（ema_accept_len={ema}）')
    print('  对照：同样的流量 vLLM 的查表只看 batch size，batch=1 时永远是表里那个 k，不管 α 是 0.9 还是 0.4。')

    print('\n== slot "8"（候选 [0, 1, 3]）在 α=0.3 下会关到 0，然后每 5 批探测一次 1 步')
    rng = random.Random(1)
    slot = AdaptiveStepSlot(3, {**SGL_DEFAULT_ADAPTIVE_CONFIG, **SGL_DEFAULT_ADAPTIVE_CONFIG['8']})
    for _ in range(60):
        slot.update(draw_accepts(rng, 0.3, slot.current_steps, 8))
    for batch, old, new, ema in slot.log:
        print(f'  第 {batch:>3} 批：steps {old} -> {new}（ema_accept_len={ema}）')
    print('  steps=0 时 EMA 不更新（没有草稿可数），所以每次重算都会 +1 去探测，再被低接受率打回 0——这就是 "关了也会周期性试一下"。')

    print('\n== 两种思路的边界')
    print('  vLLM Dynamic SD：离线 sweep 出 (batch 区间, k) 表；确定性、每个 DP rank 一致；不跟踪 α 漂移；DP > 1 时禁用；')
    print('                   MRV1 下把 FULL CUDA Graph 降成 PIECEWISE，MRV2 按档抓图。')
    print('  SGLang adaptive：在线 EMA；只支持 EAGLE/EAGLE3 topk=1；不支持 attention DP、TBO、pdmux、multi-layer EAGLE；')
    print('                   候选步数的并集决定要预分配多少 CUDA Graph / buffer。')


# --------------------------------------------------------------------------- 6. 指标口径
def acceptance_length_to_rates(length, n):
    num_drafts = length - 1
    num_full = int(num_drafts)
    return ([1.0] * num_full + [num_drafts - num_full] + [0.0] * (n - num_full - 1))[:n]


def unconditional_to_conditional_rates(rates):
    return [p / q if q > 0.0 else 0.0 for p, q in zip(rates, [1.0, *rates[:-1]])]


def vllm_metrics(accepts, k):
    num_drafts = len(accepts)
    num_draft_tokens = num_drafts * k
    num_accepted = sum(accepts)
    per_pos = [sum(1 for a in accepts if a > i) / num_drafts for i in range(k)]
    return {
        'mean_acceptance_length': 1 + num_accepted / num_drafts,
        'per_position': per_pos,
        'avg_draft_acceptance_rate': num_accepted / num_draft_tokens * 100,
    }


def sglang_metrics(accepts, k):
    return {
        'spec_accept_length': sum(a + 1 for a in accepts) / len(accepts),
        'spec_accept_rate': sum(accepts) / (len(accepts) * k),
    }


def scenario_metrics():
    rng = random.Random(3)
    k, alpha = 5, 0.7
    accepts = draw_accepts(rng, alpha, k, 20000)
    v = vllm_metrics(accepts, k)
    s = sglang_metrics(accepts, k)
    print(f'== 同一批验证结果（2 万次，k={k}，i.i.d. α={alpha}）')
    print('  vLLM 日志 "SpecDecoding metrics:"（v1/spec_decode/metrics.py）')
    print(f'    Mean acceptance length: {v["mean_acceptance_length"]:.2f}    ← 1 + accepted / num_drafts，含 bonus')
    print(f'    Per-position acceptance rate: {", ".join(f"{x:.3f}" for x in v["per_position"])}   ← 无条件：第 i 位被接受的次数 / 草稿次数')
    print(f'    Avg Draft acceptance rate: {v["avg_draft_acceptance_rate"]:.1f}%   ← accepted / drafted')
    print('  SGLang（srt/metrics/collector.py）')
    print(f'    sglang:spec_accept_length = {s["spec_accept_length"]:.2f}   ← "accepted drafts + bonus token per forward"')
    print(f'    sglang:spec_accept_rate   = {s["spec_accept_rate"]:.3f}  ← "accepted drafts / proposed drafts"')
    print(f'  理论：E = {expected_tokens(alpha, k):.2f}；第 i 位无条件接受率 α^i = {", ".join(f"{alpha ** (i + 1):.3f}" for i in range(k))}')
    cond = unconditional_to_conditional_rates(v['per_position'])
    print(f'  换成条件接受率（c_i = p_i / p_(i-1)）：{", ".join(f"{x:.3f}" for x in cond)}  ← i.i.d. 时每位都 ≈ α')
    print('  读法：两边的 "接受长度" 是同一个量；"接受率" 也是同一个量。真正要看的是 Per-position：')
    print('  它从第几位开始掉到 0.3 以下，k 就该砍到第几位。')

    print('\n== Prometheus 计数器（算任意窗口的接受长度：1 + Δaccepted / Δdrafts）')
    print('  vLLM  vllm:spec_decode_num_drafts / _num_draft_tokens / _num_accepted_tokens / _num_accepted_tokens_per_pos')
    print('  SGLang sglang:spec_accept_length / sglang:spec_accept_rate（gauge，已经是比值）')

    print('\n== synthetic 模式：不用真草稿，按给定接受长度伪造接受（压测 / 容量规划用，输出分布不对）')
    for L, n in ((2.5, 5), (3.0, 3), (4.2, 7)):
        r = acceptance_length_to_rates(L, n)
        print(f'  synthetic_acceptance_length={L}, k={n} → 无条件 {r} → 条件 {[round(x, 3) for x in unconditional_to_conditional_rates(r)]}')
    print('  "minimum-variance schedule"：前 floor(L-1) 位必中，下一位按小数部分，后面全拒——方差最小，但和真实接受曲线形状不同。')

    print('\n== 拒绝后的账（v1/core/sched/scheduler.py:2354）：k=5、接受 2 个')
    num_computed, k, gen = 1000, 5, 3
    accepted = max(gen - 1, 0)
    rejected = k - accepted
    print(f'  调度时 num_computed_tokens 先按 1+k 推进到 {num_computed + 1 + k}；验证产出 {gen} 个 token（2 草稿 + 1 bonus），')
    print(f'  rejected = {k} - {accepted} = {rejected}，num_computed_tokens 回退到 {num_computed + 1 + k - rejected}；被拒位置的 KV 槽下一步直接覆盖。')


# --------------------------------------------------------------------------- 7. SGLang 对照
def sglang_auto_params(algorithm, arch):
    if algorithm == 'STANDALONE':
        return (3, 1, 4)
    if arch in ('LlamaForCausalLM', 'Grok1ForCausalLM', 'Grok1VForCausalLM'):
        return (5, 4, 8)
    return (3, 1, 4)


def scenario_sglang():
    print('== 同一个需求两边怎么配')
    rows = [
        ('EAGLE-3，3 个草稿 token',
         """vllm serve M -sc '{"method":"eagle3","model":"D","num_speculative_tokens":3}'""",
         '--speculative-algorithm EAGLE3 --speculative-draft-model-path D --speculative-num-steps 3 '
         '--speculative-eagle-topk 1 --speculative-num-draft-tokens 4'),
        ('DeepSeek MTP',
         """-sc '{"method":"mtp","num_speculative_tokens":1}'（不写 model，用 checkpoint 自带的 MTP 层）""",
         '--speculative-algorithm EAGLE（NEXTN 是别名）+ steps/topk/draft-tokens，不写会自动选 (3,1,4)'),
        ('n-gram',
         """-sc '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_min":2,"prompt_lookup_max":4}'""",
         '--speculative-algorithm NGRAM（树：draft tokens 默认 12，BFS 1~10，trie 深度 18）'),
        ('小草稿模型',
         """-sc '{"method":"draft_model","model":"D","num_speculative_tokens":4}'""",
         '--speculative-algorithm STANDALONE --speculative-draft-model-path D'),
        ('按 batch 调 k',
         """-sc '{...,"num_speculative_tokens_per_batch_size":[[1,64,3],[65,128,1],[129,512,0]]}'""",
         '--speculative-adaptive [--speculative-adaptive-config cfg.json]'),
        ('有损加速',
         """-sc '{...,"rejection_sample_method":"fly"}'（FLy，窗口 + 熵门限）""",
         '--speculative-accept-threshold-single / -acc < 1'),
        ('块验证',
         """-sc '{...,"rejection_sample_method":"block"}'（只有 MRV2 的 kernel 认；V1 采样器只分支 fly/synthetic）""",
         '--speculative-use-block-verification（topk=1 的 EAGLE/EAGLE3/NEXTN）'),
    ]
    for need, v, s in rows:
        print(f'  {need}\n    vLLM:   {v}\n    SGLang: {s}')

    print('\n== SGLang 不写 steps/topk/draft-tokens 时自动选（speculative_hook.py:1471）')
    for alg, arch in (('EAGLE', 'LlamaForCausalLM'), ('EAGLE3', 'LlamaForCausalLM'), ('EAGLE', 'DeepseekV3ForCausalLM'),
                      ('EAGLE', 'Qwen3ForCausalLM'), ('STANDALONE', 'LlamaForCausalLM')):
        st, tk, dt = sglang_auto_params(alg, arch)
        print(f'  {alg:<10} {arch:<24} → steps={st} topk={tk} draft_tokens={dt}')
    print('  Llama 默认是 topk=4 的树（8 个节点），这时 adaptive 和 rejection sampling 都用不了（只支持 topk=1）。')
    print('  另一个副作用：开投机且没写 --max-running-requests 时，SGLang 把它重置为 48。vLLM 不改 max_num_seqs，')
    print('  所以同一台机器上 "默认配置对比" 时，SGLang 的并发上限已经被压低了。')

    print('\n== 验证路径（sampling 请求，temperature > 0）')
    print('  vLLM V1   standard：p(x)/q(x) 比值检验；草稿默认 greedy → q 是 one-hot，接受概率 = p(草稿 token)')
    print('            拒绝后从 max(p - q, 0) 里采：每个请求一组指数噪声，argmax((p-q)+ / E) 等价于按比例采样')
    print('  vLLM MRV2 standard / block / synthetic / fly 都在一个 Triton kernel 里，greedy 请求直接比 argmax')
    print('  SGLang    默认 tree_speculative_sampling_target_only：只看目标概率，兄弟节点累加 prob_acc 共用一枚硬币；')
    print('            阈值都是 1 时等价于 one-hot 草稿的标准拒绝采样；拒绝的 token 在残差里置零（"FIXME: leverage draft probs"）')
    print('            --speculative-use-rejection-sampling（topk=1）走 chain_speculative_sampling_triton，才真正用草稿概率')
    print('            全 greedy batch、CPU、XPU、或 HIP 且没开 rejection sampling → 直接比 argmax')


SCENARIOS = {
    'theory': scenario_theory,
    'batch': scenario_batch,
    'cudagraph': scenario_cudagraph,
    'ngram': scenario_ngram,
    'dynamic': scenario_dynamic,
    'metrics': scenario_metrics,
    'sglang': scenario_sglang,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=list(SCENARIOS), default='theory')
    SCENARIOS[ap.parse_args().scenario]()


if __name__ == '__main__':
    main()
