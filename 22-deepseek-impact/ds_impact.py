#!/usr/bin/env python3
"""VLLM-22：DeepSeek 开源对 vLLM 的影响，配套计算器，只用 Python 标准库。

场景：
  kvbytes     V2/V3 MLA、V3.2 DSA、V4 压缩注意力、V4.1 全局 KV + SWA 的每 token 字节账
  eplb        vllm/distributed/eplb/policy/default.py 的纯 Python 移植：
              balanced_packing → replicate_experts → 分层重排，外加 balancedness
  replay      V4.1 SWA 有界重放：prefill 的 token×层 计数，和不重放时缓存 SWA 的代价
  adaptive    DSpark 自适应验证：生存概率累乘、取全局 top-B、按 期望 token/耗时 取 argmax
  components  vLLM 0d6f92a 与 SGLang 8fabf5e 引入 DeepSeek 组件的方式
  grep        30 天窗口里两边标题命中 DeepSeek 组件关键词的提交数
  lines       vLLM 0d6f92a 里 DeepSeek 专用代码的行数

用法：
  python3 ds_impact.py --scenario all
  python3 ds_impact.py --scenario eplb --experts 64 --gpus 16 --redundant 16
  python3 ds_impact.py --scenario replay --hit 32768 --new 8192
  python3 ds_impact.py --scenario adaptive --reqs 32 --budget-knee 96
  python3 ds_impact.py --grep "deepgemm|mega ?moe"
"""
import argparse
import gzip
import math
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(HERE, "..", "21-frontier-month", "data")
GiB = 1024 ** 3


# ------------------------------------------------------------------ kvbytes
# 每层每 token 的 KV 字节。MLA 存 kv_lora_rank=512 的潜向量 + 64 维 RoPE key。
MLA_BF16 = (512 + 64) * 2              # 1152 B
MLA_FP8_VLLM = 512 + 16 + 128          # 656 B：512 NoPE fp8 + 4 个 fp32 scale + 64 维 RoPE bf16
V3_MHA_EQUIV = 128 * (192 + 128) * 2   # 128 头，K 192 维（128 NoPE + 64 RoPE），V 128 维，BF16
INDEXER_FP8 = 128 + 128 // 128 * 4     # 132 B：V3.2 indexer K，128 fp8 + 1 个 fp32 scale
INDEXER_FP4 = 128 // 2 + 128 // 32     # 68 B：MXFP4 两值一字节 + 每 32 值一个 UE8M0
V3_LAYERS = 61

V41_GLOBAL = 890        # B/token，全部 KV-source 层的全局（压缩）KV 合计，FP4
V41_SWA = {"fp8": 584, "mxfp8": 528}   # attention.py:522
V41_WINDOW = 128
V41_SWA_LAYERS = 40


def kv_rows():
    rows = [
        ("V3 等价 MHA（BF16）", V3_MHA_EQUIV, V3_MHA_EQUIV * V3_LAYERS),
        ("V2/V3 MLA（BF16）", MLA_BF16, MLA_BF16 * V3_LAYERS),
        ("V3 MLA（vLLM FP8 布局）", MLA_FP8_VLLM, MLA_FP8_VLLM * V3_LAYERS),
        ("V3.2 MLA FP8 + indexer FP8", MLA_FP8_VLLM + INDEXER_FP8, (MLA_FP8_VLLM + INDEXER_FP8) * V3_LAYERS),
    ]
    return rows


def kvbytes(args):
    print("-- 每 token 的 KV 字节（V3 系列 61 层）")
    print(f"   {'布局':28s}{'B/层':>8s}{'KiB/token':>12s}{'1M 上下文 GiB':>16s}")
    for name, per_layer, per_tok in kv_rows():
        print(f"   {name:28s}{per_layer:8d}{per_tok / 1024:12.1f}{per_tok * 2 ** 20 / GiB:16.1f}")
    print(f"   MLA BF16 / 等价 MHA = 1/{V3_MHA_EQUIV / MLA_BF16:.1f}；indexer FP8 {INDEXER_FP8} B，FP4 {INDEXER_FP4} B")
    print("-- V4（vLLM 博客 2026-04-24 的口径，BF16，1M 上下文）")
    v32_bf16 = (MLA_BF16 + 128 * 2) * V3_LAYERS * 2 ** 20 / GiB
    print(f"   V3.2 风格：MLA BF16 1152 B + indexer K BF16 256 B，61 层，1M 上下文 = {v32_bf16:.1f} GiB（博客 83.9）")
    print(f"   V4 混合压缩 9.62 GiB，{83.9 / 9.62:.1f}x；FP8 注意力 + FP4 indexer 再省约 2x")
    print("   c4a：每 4 个位置压 1 条，top-512；c128a：每 128 个位置压 1 条，k=8192；SWA 窗口 128")
    print("   256 个原生位置为一个逻辑块：c4a 块里 64 条，c128a 块里 2 条")
    print("-- V4.1：全局 KV 随长度增长，SWA 每请求固定")
    swa = V41_SWA[args.swa_dtype]
    swa_req = swa * V41_WINDOW * V41_SWA_LAYERS
    print(f"   全局 KV {V41_GLOBAL} B/token；SWA {swa} B × {V41_WINDOW} 位置 × {V41_SWA_LAYERS} 层 = {swa_req / 2 ** 20:.2f} MiB/请求（{args.swa_dtype}）")
    for L in (4096, 32768, 131072, 1048576):
        g = V41_GLOBAL * L
        print(f"   上下文 {L:>8d}：全局 {g / 2 ** 20:9.1f} MiB，SWA 占 {swa_req / (g + swa_req) * 100:5.1f}%")
    blk = args.block
    per_boundary = swa_req
    per_block_global = V41_GLOBAL * blk
    print(f"-- 不做有界重放：前缀缓存要在每个 {blk} token 块边界存一份 SWA 快照")
    print(f"   每块全局 KV {per_block_global / 1024:.0f} KiB，SWA 快照 {per_boundary / 1024:.0f} KiB，后者是前者的 {per_boundary / per_block_global:.1f}x")


# ------------------------------------------------------------------ eplb
def argmin(xs):
    return min(range(len(xs)), key=lambda i: (xs[i], i))


def argmax(xs):
    return max(range(len(xs)), key=lambda i: (xs[i], -i))


def balanced_packing(weight, num_packs):
    """weight: 一层的 n 个物品，装进 num_packs 个包，每包恰好 n/num_packs 个。"""
    n = len(weight)
    assert n % num_packs == 0
    per = n // num_packs
    if per == 1:
        return list(range(n)), [0] * n
    order = sorted(range(n), key=lambda i: -weight[i])
    pack_w = [0.0] * num_packs
    pack_n = [0] * num_packs
    pack_index, rank = [-1] * n, [-1] * n
    for g in order:
        p = argmin(pack_w)
        pack_index[g], rank[g] = p, pack_n[p]
        pack_w[p] += weight[g]
        pack_n[p] += 1
        if pack_n[p] == per:
            pack_w[p] = math.inf
    return pack_index, rank


def replicate_experts(weight, num_phy):
    num_log = len(weight)
    phy2log = list(range(num_log)) + [0] * (num_phy - num_log)
    cnt = [1] * num_log
    for i in range(num_log, num_phy):
        j = argmax([w / c for w, c in zip(weight, cnt)])
        phy2log[i] = j
        cnt[j] += 1
    return phy2log, cnt


def inverse(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def rebalance_hierarchical(weight, num_phy, num_groups, num_nodes, num_gpus):
    num_log = len(weight)
    gsize = num_log // num_groups
    gpn = num_groups // num_nodes
    per_gpu = num_phy // num_gpus
    # 1. 专家组装到节点
    tpg = [sum(weight[g * gsize:(g + 1) * gsize]) for g in range(num_groups)]
    gpack, grank = balanced_packing(tpg, num_nodes)
    log2mlog = [0] * num_log
    for e in range(num_log):
        g = e // gsize
        log2mlog[e] = (gpack[g] * gpn + grank[g]) * gsize + e % gsize
    mlog2log = inverse(log2mlog)
    # 2. 节点内复制
    w_mlog = [weight[mlog2log[i]] for i in range(num_log)]
    lpn, ppn = num_log // num_nodes, num_phy // num_nodes
    phy2mlog, mlogcnt = [], []
    for nd in range(num_nodes):
        p2m, cnt = replicate_experts(w_mlog[nd * lpn:(nd + 1) * lpn], ppn)
        phy2mlog.append(p2m)
        mlogcnt.append(cnt)
    # 3. 物理专家装到节点内的 GPU
    out = []
    for nd in range(num_nodes):
        local = w_mlog[nd * lpn:(nd + 1) * lpn]
        tpp = [local[m] / mlogcnt[nd][m] for m in phy2mlog[nd]]
        pidx, prank = balanced_packing(tpp, num_gpus // num_nodes)
        phy2pphy = [pidx[i] * per_gpu + prank[i] for i in range(ppn)]
        pphy2phy = inverse(phy2pphy)
        for pp in range(ppn):
            out.append(mlog2log[phy2mlog[nd][pphy2phy[pp]] + nd * lpn])
    return out


def rebalance(weight, num_phy, num_groups, num_nodes, num_gpus):
    if num_groups % num_nodes == 0:
        return rebalance_hierarchical(weight, num_phy, num_groups, num_nodes, num_gpus)
    return rebalance_hierarchical(weight, num_phy, 1, 1, num_gpus)


def gpu_loads(weight, phy2log, num_gpus):
    cnt = [0] * len(weight)
    for e in phy2log:
        cnt[e] += 1
    per = len(phy2log) // num_gpus
    return [sum(weight[e] / cnt[e] for e in phy2log[g * per:(g + 1) * per]) for g in range(num_gpus)]


def balancedness(layers_loads):
    """eplb_state._compute_eplb_load_stats：各层 mean 求和 / 各层 max 求和。"""
    avg = sum(sum(l) / len(l) for l in layers_loads)
    mx = sum(max(l) for l in layers_loads)
    return avg / mx


def skewed_weights(n, layers, seed, alpha):
    rng = random.Random(seed)
    out = []
    for _ in range(layers):
        base = [1000.0 / (r + 1) ** alpha for r in range(n)]
        rng.shuffle(base)
        out.append([round(b * rng.uniform(0.8, 1.2)) for b in base])
    return out


VLLM_TEST_W = [[90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86],
               [20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27]]
VLLM_TEST_PHY2LOG = [[5, 6, 5, 7, 8, 4, 3, 4, 10, 9, 10, 2, 0, 1, 11, 1],
                     [7, 10, 6, 8, 6, 11, 8, 9, 2, 4, 5, 1, 5, 0, 3, 1]]


def eplb(args):
    print("-- vLLM 单测 tests/distributed/test_eplb_algo.py 的例子：12 专家、4 组、2 节点、8 GPU、16 个物理槽")
    for li, w in enumerate(VLLM_TEST_W):
        p = rebalance(w, 16, 4, 2, 8)
        ok = "一致" if p == VLLM_TEST_PHY2LOG[li] else "不一致"
        before = [sum(w[i * 3:(i + 1) * 3]) for i in range(4)]  # 参照：12 个专家按编号摊在 4 张卡，不复制
        after = gpu_loads(w, p, 8)
        print(f"   层 {li}：phy2log {p}（与单测期望{ok}）")
        print(f"         每卡负载 {[round(x, 1) for x in after]}，max {max(after):.1f}，mean {sum(after) / 8:.1f}，"
              f"balancedness {sum(after) / 8 / max(after):.2f}（参照：按编号摊 4 卡 {sum(before) / 4 / max(before):.2f}）")
    n, gpus, red = args.experts, args.gpus, args.redundant
    W = skewed_weights(n, args.layers, args.seed, args.alpha)
    print(f"-- 合成负载：{n} 专家、{args.groups} 组、{args.nodes} 节点、{gpus} GPU、{args.layers} 层，Zipf α={args.alpha}")
    base = [gpu_loads(w, list(range(n)), gpus) for w in W]
    print(f"   不做 EPLB（专家按编号平铺）       balancedness {balancedness(base):.2f}")
    for r in sorted({0, red // 2, red, red * 2}):
        if (n + r) % gpus:
            continue
        loads = [gpu_loads(w, rebalance(w, n + r, args.groups, args.nodes, gpus), gpus) for w in W]
        print(f"   EPLB 冗余 {r:3d}（每卡 {(n + r) // gpus} 槽）            balancedness {balancedness(loads):.2f}")
    print("   balancedness = Σ层 mean(rank 负载) / Σ层 max(rank 负载)，与 eplb_state.py:631 日志同口径")


# ------------------------------------------------------------------ replay
def replay(args):
    L, cut, win = args.layers, args.cut, args.window
    H, N = args.hit, args.new
    full = N * L
    dec = N * (cut + 1) + min(N, win) * (L - cut - 1)
    enc = min(H, win) * (cut + 1) if H else 0
    print(f"-- V4.1 prefill 的 token×层 计数：{L} 层，最后一个 KV-source 层 = {cut}，窗口 {win}")
    print(f"   请求：前缀命中 {H}，新 token {N}")
    print(f"   不重放（新 token 走全部层）              {full:>12d}")
    print(f"   decoder 侧重放（层 0–{cut} 全部 token，层 {cut + 1}–{L - 1} 只算最后 {win} 个）{dec:>12d}  {dec / full * 100:5.1f}%")
    print(f"   encoder 侧重放补算命中末尾 {min(H, win)} 个位置（层 0–{cut}）   {enc:>12d}  {enc / full * 100:5.1f}%")
    print(f"   合计 {dec + enc}，比不重放少 {(1 - (dec + enc) / full) * 100:.1f}%")
    print("   这是忽略层间成本差异的上限；博客实测 prefill 计算省 30–40%")
    print("-- 不同新 token 数下 decoder 侧重放省下的比例")
    for n in (128, 512, 2048, 8192, 32768):
        d = n * (cut + 1) + min(n, win) * (L - cut - 1)
        print(f"   新 token {n:6d}：省 {(1 - d / (n * L)) * 100:5.1f}%")


# ------------------------------------------------------------------ adaptive
def adaptive_budget(conf, n_drafts, n_non_draft, n_sampling, draft_cost, verify_cost, max_budget=None):
    """adaptive_verification.py get_num_tokens 的移植。conf[i][k] 是请求 i 第 k 个草稿位置的置信度。"""
    scores = []
    for c, nd in zip(conf, n_drafts):
        s = 1.0
        for k in range(nd):
            s *= c[k]
            scores.append(s)
    scores.sort(reverse=True)
    mb = sum(n_drafts) if max_budget is None else min(max_budget, sum(n_drafts))
    scores = scores[:mb]
    T = sum(n_non_draft)
    exp_tokens = [float(n_sampling)]
    for s in scores:
        exp_tokens.append(exp_tokens[-1] + s)
    costs = [draft_cost + verify_cost(T + b) for b in range(mb + 1)]
    ratio = [e / c for e, c in zip(exp_tokens, costs)]
    B = argmax(ratio)
    return B, exp_tokens, costs


def adaptive(args):
    rng = random.Random(args.seed)
    R, K = args.reqs, args.k
    conf = []
    for _ in range(R):
        q = rng.uniform(0.55, 0.95)
        conf.append([max(0.02, min(0.99, q - 0.06 * k + rng.uniform(-0.05, 0.05))) for k in range(K)])
    knee, slope, base = args.budget_knee, args.slope, args.verify_base

    def verify_cost(t):
        return base + max(0, t - knee) * slope

    B, exp, cost = adaptive_budget(conf, [K] * R, [1] * R, R, args.draft_cost, verify_cost)
    full = R * K
    surv = [sum(math.prod(c[:k + 1]) for c in conf) / R for k in range(K)]
    print(f"-- DSpark 自适应验证：{R} 个请求，每个 {K} 个草稿位置，验证成本在 {knee} token 后每 token +{slope} ms")
    print("   平均生存概率（第 k 个草稿被接受的概率）：" + "  ".join(f"k{k + 1} {s:.2f}" for k, s in enumerate(surv)))
    print(f"   全部验证：B={full}，期望产出 {exp[full]:.1f} token，耗时 {cost[full]:.2f} ms，{exp[full] / cost[full]:.2f} token/ms")
    print(f"   只验 top-B：B={B}，期望产出 {exp[B]:.1f} token，耗时 {cost[B]:.2f} ms，{exp[B] / cost[B]:.2f} token/ms")
    print(f"   不投机：B=0，产出 {exp[0]:.0f} token，耗时 {cost[0]:.2f} ms，{exp[0] / cost[0]:.2f} token/ms")
    print("-- 换一条验证成本曲线：拐点越早（越接近算力瓶颈），B 越小")
    for kn in (32, 64, 96, 160, 256):
        b, e, c = adaptive_budget(conf, [K] * R, [1] * R, R, args.draft_cost,
                                  lambda t, kn=kn: base + max(0, t - kn) * slope)
        print(f"   拐点 {kn:3d} token：B={b:3d}/{full}，{e[b] / c[b]:.2f} token/ms（全验 {e[full] / c[full]:.2f}）")


# ------------------------------------------------------------------ components
COMPONENTS = [
    # 组件, vLLM 0d6f92a 的来源, 钉在哪, SGLang 8fabf5e 的来源
    ("DeepGEMM", "vllm-project/DeepGEMM 分叉 5cdb145", "cmake/external_projects/deepgemm.cmake:30-41",
     "sgl-deep-gemm==0.2.3.post2 wheel"),
    ("FlashMLA", "vllm-project/FlashMLA 分叉 e2afa5f", "cmake/external_projects/flashmla.cmake:21-22",
     "sgl-project/FlashMLA 分叉 3620b63"),
    ("DeepSelect", "vllm-project/DeepSelect 分叉 d96d33a", "cmake/external_projects/deepselect.cmake:21-22",
     "（内置于 sglang kernels）"),
    ("DeepEP", "deepseek-ai/DeepEP 上游 d4f41e4", "tools/ep_kernels/install_python_libraries.sh:13,244",
     "sgl-deep-ep==0.1.2.post1 wheel"),
    ("EPLB", "改写进 policy/default.py，加 preserve_intragpu_slots", "vllm/distributed/eplb/policy/default.py",
     "eplb_algorithms/deepseek.py 原样复制，另加 vec / elasticity_aware"),
    ("DualPipe 微批重叠", "DBO（enable_dbo，阈值 32/512）", "vllm/config/parallel.py:258-268",
     "TBO（two_batch_overlap.py）"),
    ("TileLang", "mHC 等少数 kernel 走 JIT", "#62745、#58685", "tilelang==0.1.12"),
]


def components(args):
    print("-- DeepSeek 组件进入两个引擎的方式")
    for name, v, where, s in COMPONENTS:
        print(f"   {name:16s} vLLM: {v}")
        print(f"   {'':16s}       {where}")
        print(f"   {'':16s} SGLang: {s}")


# ------------------------------------------------------------------ grep
TOPICS = [
    ("deepgemm", r"deep_?gemm"),
    ("megamoe", r"mega[ _-]?moe"),
    ("deepep", r"deep_?ep"),
    ("flashmla", r"flash_?mla"),
    ("deepselect", r"deep_?select"),
    ("eplb", r"\beplb"),
    ("dbo/tbo", r"\bdbo\b|\btbo\b|two[- ]batch overlap|dual[- ]batch overlap"),
    ("tilelang", r"tile_?lang"),
    ("mla", r"\bmla\b"),
    ("dsa/indexer", r"\bdsa\b|indexer"),
    ("mhc", r"\bmhc\b|hyper[- ]connection"),
    ("engram", r"engram"),
    ("replay", r"replay"),
    ("dspark", r"dspark"),
    ("mtp", r"\bmtp\b"),
    ("dsv4.1", r"deepseek[- ]?v4\.1|dsv4\.1|ds-?v4\.1"),
    ("dsv4", r"deepseek[- ]?v4(?!\.1)|dsv4(?!\.1)|ds-?v4(?!\.1)"),
    ("deepinfer", r"deepinfer"),
]


def load_titles(name):
    out = []
    with gzip.open(os.path.join(SNAP, name), "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            out.append((p[1][:10], p[-1]))
    return out


def topic_counts(v, s):
    rows = []
    for key, pat in TOPICS:
        rx = re.compile(pat, re.I)
        rows.append((key, sum(1 for _, t in v if rx.search(t)), sum(1 for _, t in s if rx.search(t))))
    return rows


def grep_scn(args):
    v, s = load_titles("vllm_commits.tsv.gz"), load_titles("sglang_commits.tsv.gz")
    print(f"-- 2026-09-29 → 2026-10-29，vLLM {len(v)} 个提交 / SGLang {len(s)} 个提交，按标题正则计数（一个提交可命中多项）")
    print(f"   {'主题':14s}{'vLLM':>6s}{'SGLang':>8s}")
    for key, a, b in topic_counts(v, s):
        print(f"   {key:14s}{a:6d}{b:8d}")


def grep_pat(pat, k=40):
    rx = re.compile(pat, re.I)
    for name in ("vllm_commits.tsv.gz", "sglang_commits.tsv.gz"):
        hit = [(d, t) for d, t in load_titles(name) if rx.search(t)]
        print(f"-- {name.split('_')[0]}：{len(hit)} 条匹配 /{pat}/")
        for d, t in hit[:k]:
            print(f"   {d} {t[:110]}")


# ------------------------------------------------------------------ lines
LINES = [  # 0d6f92a 上 find … -name '*.py' | xargs cat | wc -l
    ("vllm/models/deepseek_v32/", 4895, "amd 1158 / nvidia 1849 / common 1193"),
    ("vllm/models/deepseek_v4/", 34831, "amd 13166 / nvidia 7793 / common 6422 / cpu 1844 / xpu 3073"),
    ("vllm/models/deepseek_v41/", 38686, "amd 15519 / nvidia 15009 / common 5340"),
    ("model_executor/models/deepseek_v2.py", 2010, "V2/V3/R1 共用"),
    ("model_executor/models/deepseek_mtp.py", 572, "MTP 草稿头"),
]


def lines(args):
    tot = sum(n for _, n, _ in LINES)
    print("-- vLLM 0d6f92a 里按模型分目录的 DeepSeek 代码（wc -l）")
    for p, n, note in LINES:
        print(f"   {p:40s}{n:7d}  {note}")
    print(f"   合计 {tot}；三个按代际分的目录 {sum(n for _, n, _ in LINES[:3])}")


SCEN = {"kvbytes": kvbytes, "eplb": eplb, "replay": replay, "adaptive": adaptive,
        "components": components, "grep": grep_scn, "lines": lines}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", choices=list(SCEN) + ["all"])
    ap.add_argument("--grep", help="按标题正则列出两边窗口内的提交")
    ap.add_argument("--swa-dtype", default="fp8", choices=list(V41_SWA))
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--nodes", type=int, default=2)
    ap.add_argument("--gpus", type=int, default=16)
    ap.add_argument("--redundant", type=int, default=16)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cut", type=int, default=20)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--hit", type=int, default=32768)
    ap.add_argument("--new", type=int, default=8192)
    ap.add_argument("--reqs", type=int, default=32)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--budget-knee", type=int, default=96)
    ap.add_argument("--slope", type=float, default=0.02)
    ap.add_argument("--verify-base", type=float, default=6.0)
    ap.add_argument("--draft-cost", type=float, default=1.5)
    a = ap.parse_args(argv)
    if a.grep:
        grep_pat(a.grep)
        return
    for name, fn in SCEN.items():
        if a.scenario in (name, "all"):
            if a.scenario == "all":
                print(f"== {name}")
            fn(a)


if __name__ == "__main__":
    main()
