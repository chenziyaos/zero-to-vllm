#!/usr/bin/env python3
"""VLLM-23：DeepSeek-V4.1-Flash 部署规划器，只用 Python 标准库，不联网、不需要 GPU。

场景：
  ckpt     checkpoint 的 476 GiB 拆成哪几块，哪些能切、哪些能搬到主机内存
  mem      给定卡型 / 卡数 / TP / Engram 位置 / 专家卸载量，算每卡常驻、预留和 KV 池
  kv       每 token 的 KV 字节：博客的 890 B 是 SM100 的格式，SM90 / SM120 是 1790 B
  indexer  sparse indexer 启动时的 logits 预留，三个版本三种形状
  host     主机内存：Engram 的 pinned 内存按 2 的幂取整，188.83 GiB 的表实际占多少
  decode   带宽估算：每步读多少字节、PCIe 卸载拖多少（全部是估算）
  rank     十几种卡配置放在一起排
  all      全部

用法：
  python3 deploy_planner.py --scenario all
  python3 deploy_planner.py --scenario mem --gpu rtxpro6000 --n 4 --tp 4 --engram host
  python3 deploy_planner.py --scenario mem --gpu h20 --n 8 --tp 8 --engram gpu
  python3 deploy_planner.py --scenario indexer --gpu h20 --max-model-len 1048576 --max-num-seqs 256
  python3 deploy_planner.py --scenario host --gpu rtxpro6000 --n 1 --tp 1 --offload-gib 214

源码行号对应 vllm 主干 0d6f92a；recipe 指 recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml。
"""
import argparse
import math

GiB = 1024 ** 3
MiB = 1024 ** 2

# ------------------------------------------------------------------ 模型常量
# recipe 第 1247–1262 行的 Memory 表（GiB，磁盘上的存储字节）
CKPT = [
    ("路由专家 + DSpark 专家（MXFP4）", 557.2, 259.5),
    ("Engram 表（FP8）", 196.6, 183.1),
    ("注意力、稠密投影、router（FP8）", 7.4, 6.9),
    ("Embedding + LM head（BF16）、norm（FP32）", 2.0, 3.9),
    ("UE8M0 block scale", 23.6, 21.9),
]
CKPT_GB = 511                 # recipe：roughly 511 GB on disk
VRAM_MIN_GB = 614             # recipe 第 125 行：511 × 1.2
HIDDEN, MOE_INTER, LAYERS = 5120, 2304, 40
N_EXPERTS, TOPK, N_SHARED = 384, 6, 1
VOCAB = 129280
# Engram：第 1、14 层两张表，合计 768,022,850 行 × 256 维 FP8，每 32 维一个 UE8M0 scale
ENGRAM_ROWS, ENGRAM_DIM, ENGRAM_BLOCK, ENGRAM_TABLES = 768_022_850, 256, 32, 2
ENGRAM_VALUES_GIB = ENGRAM_ROWS * ENGRAM_DIM / GiB                                   # 183.11
ENGRAM_GIB = ENGRAM_ROWS * (ENGRAM_DIM + ENGRAM_DIM // ENGRAM_BLOCK) / GiB           # 188.83，engram.py:1067 的口径
TOTAL_GIB = sum(x[2] for x in CKPT)                                                   # 475.3
NON_ENGRAM_GIB = TOTAL_GIB - ENGRAM_GIB        # 286.5：Engram 的 scale 记在 UE8M0 那一行里，这里一并扣掉

# 一个专家 = w13（2×2304×5120）+ w2（5120×2304），MXFP4 每值 0.5 B，每 32 值一个 UE8M0
EXPERT_PARAMS = 3 * HIDDEN * MOE_INTER
EXPERT_BYTES = EXPERT_PARAMS * (0.5 + 1 / 32)
# 每步必读的稠密部分：注意力 / 投影 6.9 GiB + LM head BF16
DENSE_STEP_BYTES = 6.9 * GiB + VOCAB * HIDDEN * 2

# KV：只有 kv-source 层（2、8、14、20）分配全局 KV 和 indexer K（attention.py:1206、:446）
# 2、8、14 层压缩比 2，20 层压缩比 1：每 token 2.5 条记录
KV_SOURCE_CR = {2: 2, 8: 2, 14: 2, 20: 1}
ENTRIES_PER_TOKEN = sum(1 / cr for cr in KV_SOURCE_CR.values())                     # 2.5
SWA_BYTES = {"sm100": 528, "other": 584}       # attention.py:522 swa_bytes_per_token
COMP_BYTES = {"nvfp4_ds_mla": 288}             # attention.py:525，其余格式等于 SWA 字节
INDEXER_K = {"fp8": 132, "mxfp4": 68}          # attention.py:101 _indexer_k_cache_head_dim
SWA_WINDOW, SWA_LAYERS = 128, 40
BLOG_GLOBAL = 890                              # vLLM 博客 2026-10-23：890 bytes per token

# ------------------------------------------------------------------ 卡
# mem 取 nvidia-smi 显示的总量（GiB，估算）；bw 是标称显存带宽 TB/s；link 决定 all-reduce 延迟
GPUS = {
    "rtx4090":     dict(arch="sm89", mem=24.0, bw=1.008, link="pcie4", label="RTX 4090 24G"),
    "rtx4090_48g": dict(arch="sm89", mem=48.0, bw=1.008, link="pcie4", label="RTX 4090 48G（改装）"),
    "rtx5090":     dict(arch="sm120", mem=31.8, bw=1.792, link="pcie5", label="RTX 5090 32G"),
    "rtxpro6000":  dict(arch="sm120", mem=95.6, bw=1.792, link="pcie5", label="RTX PRO 6000 96G"),
    "h20":         dict(arch="sm90", mem=95.6, bw=4.0, link="nvlink", label="H20 96G"),
    "h20_141g":    dict(arch="sm90", mem=140.4, bw=4.8, link="nvlink", label="H20-3e 141G"),
    "h100":        dict(arch="sm90", mem=79.6, bw=3.35, link="nvlink", label="H100 80G"),
    "h200":        dict(arch="sm90", mem=140.4, bw=4.8, link="nvlink", label="H200 141G"),
    "b200":        dict(arch="sm100", mem=178.4, bw=8.0, link="nvlink", label="B200 180G"),
}
# 常驻系数：实测常驻 / 静态账（NON_ENGRAM_GIB / TP）。
#   sm90：recipe 第 362 行 H200 TP4 实测 81.2 GiB，静态账 71.6 → 1.134
#   sm120：commdata2338 4×PRO 6000 TP4 实测 74.29 GiB → 1.037
#   sm89 没有公开实测，按 sm90 取（估算）；sm100 取 1.0（估算）
RESIDENT_FACTOR = {"sm90": 81.2 / (NON_ENGRAM_GIB / 4), "sm120": 74.29 / (NON_ENGRAM_GIB / 4),
                   "sm89": 81.2 / (NON_ENGRAM_GIB / 4), "sm100": 1.0}
# 其他运行时占用（CUDA graph、profile 激活峰值、NCCL 缓冲），按架构各用一条实测反推。
# gather workspace 在 profile run 里分配，算进激活峰值，不再单独扣。
#   sm90：H200 TP4，0.92 × 140.4 − 81.2（常驻）− 38.5（KV）− 0.5（v0.31 indexer）≈ 9.0
#         recipe 的 H200 跑 batched 8192、seqs 128 档，CUDA graph 抓得多
#   sm120：commdata2338 4×PRO 6000，0.93 × 95.6 − 74.29 − 8,050,969 × 1790 B − 16 × SWA − 0.5 ≈ 0.6
#         seqs 16、batched 4096，图小
#   sm89 没有公开实测，按 sm120 取（同样是小 seqs 的消费级配置，估算）；sm100 按 sm90 取（估算）
_SM120_KV_GIB = 8_050_969 * 1790 / GiB + 16 * 584 * 128 * 40 / GiB
OTHER_BY_ARCH = {"sm90": 0.92 * 140.4 - 81.2 - 38.5 - 0.5,
                 "sm120": 0.93 * 95.6 - 74.29 - _SM120_KV_GIB - 0.5}
OTHER_BY_ARCH["sm89"] = OTHER_BY_ARCH["sm120"]
OTHER_BY_ARCH["sm100"] = OTHER_BY_ARCH["sm90"]
# 一次小消息 all-reduce 的延迟（微秒，估算）和 PCIe 主机读带宽（GB/s，估算）
AR_US = {"nvlink": 10, "pcie5": 25, "pcie4": 35}
H2D_GBS = {"nvlink": 50, "pcie5": 50, "pcie4": 25}


def kv_format(arch):
    """非 SM100 上 auto / fp8 都解析成 fp8_ds_mla（attention.py:141），indexer 只能 FP8（mla/indexer.py:96）。"""
    if arch == "sm100":
        return dict(comp=COMP_BYTES["nvfp4_ds_mla"], idx=INDEXER_K["mxfp4"], swa=SWA_BYTES["sm100"],
                    name="nvfp4_ds_mla + MXFP4 indexer")
    return dict(comp=SWA_BYTES["other"], idx=INDEXER_K["fp8"], swa=SWA_BYTES["other"],
                name="fp8_ds_mla + FP8 indexer")


def kv_bytes_per_token(arch):
    f = kv_format(arch)
    return (f["comp"] + f["idx"]) * ENTRIES_PER_TOKEN


def swa_bytes_per_req(arch):
    return kv_format(arch)["swa"] * SWA_WINDOW * SWA_LAYERS


def round_up(x, a):
    return (x + a - 1) // a * a


def prefill_buffer_entries(max_len):
    """mla/indexer.py:916 get_max_prefill_buffer_size：len × (576*2//132) × factor，
    factor 在 len ≤ 256K 时是 5，否则 2（config/vllm.py:1888）。"""
    factor = 5 if max_len <= 256 * 1024 else 2
    return max_len * (576 * 2 // 132) * factor


def indexer_reserve(arch, max_len, seqs, batched, spec, version, deep_select=False):
    """sparse indexer 启动时的 logits 预留（字节）。
    v031：v0.31.0 / v0.32.0 固定 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB = 512 MiB。
    recipe：recipe 注释里的 [max-num-batched-tokens, max-model-len] × 2 B。
    main：sparse_attn_indexer.py:486–543（#62860 之后），取 prefill 预算与 decode 行数两者大者。"""
    if version == "v031":
        return 512 * MiB
    if version == "recipe":
        return batched * max_len * 2
    sm100 = arch == "sm100"
    itemsize, split_kv = (2, 384) if sm100 else (4, 256)
    row_align = math.lcm(split_kv, 1024 // itemsize)
    copy_bytes = 0 if itemsize == 4 else 4
    fp32_decode = not deep_select
    decode_copy = copy_bytes if fp32_decode else 0
    rows = min(seqs, batched) * (1 + spec)
    prefill = 512 * MiB // 4 * (itemsize + copy_bytes)
    decode = rows * round_up(max_len, row_align) * (itemsize + decode_copy)
    return max(prefill, decode)


def gather_workspace(arch, max_len):
    """sparse_attn_indexer.py:332 _gather_workspace_shapes，最坏是压缩比 1 的层（20 层以后）。"""
    t = prefill_buffer_entries(max_len)
    per = 64 + 4 if arch == "sm100" else 128 + 4
    return t * per


def engram_shards(n, tp, engram="host"):
    """engram.py:1075 _get_shard_info：DP=1 时按 TP 切。
    单机 DP>1 且表在主机时，dp_shared_memory 默认打开（config/engram.py resolve_dp_shared_memory），
    层里把 dp_size 置 1，仍按 TP 切，整机只存一份放 /dev/shm；表在显存时按 TP×DP 切。"""
    if n // tp == 1 or engram == "host":
        return tp
    return n


def plan(gpu, n, tp, engram="host", offload=0.0, gmu=0.92, max_len=1048576, seqs=16, batched=8192,
         spec=5, version="main", thp=False):
    g = GPUS[gpu]
    arch = g["arch"]
    static = NON_ENGRAM_GIB / tp
    resident = static * RESIDENT_FACTOR[arch] - offload
    shards = engram_shards(n, tp, engram)
    eng_gpu = ENGRAM_GIB / shards if engram == "gpu" else 0.0
    idx = indexer_reserve(arch, max_len, seqs, batched, spec, version) / GiB
    ws = gather_workspace(arch, max_len) / GiB
    other = OTHER_BY_ARCH[arch]
    reserve = idx + other
    usable = g["mem"] * gmu
    kv = usable - resident - eng_gpu - reserve
    bpt = kv_bytes_per_token(arch)
    swa = swa_bytes_per_req(arch)
    tokens = max(kv, 0) * GiB / bpt
    per_req = max_len * bpt + swa
    conc = max(kv, 0) * GiB / per_req
    return dict(gpu=gpu, arch=arch, n=n, tp=tp, dp=n // tp, engram=engram, offload=offload, static=static,
                resident=resident, eng_gpu=eng_gpu, shards=shards, idx=idx, ws=ws, other=other, reserve=reserve,
                usable=usable, kv=kv, bpt=bpt, tokens=tokens, conc=conc, max_len=max_len,
                fits=kv > 0 and conc >= 1, host=host_need(n, tp, engram, offload, thp))


# ------------------------------------------------------------------ host
def pow2(x):
    return 1 << (int(x) - 1).bit_length() if x > 1 else 1


def engram_host_bytes(n, tp, thp=False, shared=False):
    """每张表每个 rank 两个 pinned 张量：rows×256 的值和 rows×8 的 scale。
    CUDA 的 pin_memory 按 2 的幂取整（engram.py:1122 的注释），THP / /dev/shm 共享按实际大小。
    表按哈希头切，各头大小不同；这里按平均行数算（估算）。"""
    shards = engram_shards(n, tp)
    shared = shared or n // tp > 1
    rows = ENGRAM_ROWS / ENGRAM_TABLES / shards
    vals, scales = rows * ENGRAM_DIM, rows * (ENGRAM_DIM // ENGRAM_BLOCK)
    exact = (vals + scales) * ENGRAM_TABLES * shards
    if thp or shared:          # 共享时一份 /dev/shm，按实际大小
        return exact, exact
    pinned = (pow2(math.ceil(vals)) + pow2(math.ceil(scales))) * ENGRAM_TABLES * shards
    return exact, pinned


def host_need(n, tp, engram, offload, thp=False, margin_gib=48):
    """主机内存需求（GiB）：Engram + UVA 专家卸载（每卡 offload GiB）+ 系统 / 加载余量（估算 48 GiB）。"""
    eng = engram_host_bytes(n, tp, thp)[1] / GiB if engram == "host" else 0.0
    return eng + offload * n + margin_gib


def dimm(x):
    for s in (128, 256, 384, 512, 768, 1024, 1536, 2048):
        if s >= x:
            return s
    return x


# ------------------------------------------------------------------ decode（估算）
def experts_touched(tokens):
    """一步里 tokens 个 token 各选 top-6，期望命中多少个不同的路由专家（均匀路由假设）。"""
    return N_EXPERTS * (1 - (1 - TOPK / N_EXPERTS) ** tokens) + N_SHARED


def decode_bound(gpu, n, tp, bs=1, spec=5, accept=4.2, offload=0.0, eta=0.6, fixed_ms=2.0):
    """带宽估算：每卡每步要读的字节 /（带宽 × η）+ 每层两次 all-reduce + 固定开销。
    专家卸载到主机时，被卸载的那部分走 PCIe。返回 (每步 ms, 单路 tok/s, 总 tok/s)。全部是估算。"""
    g = GPUS[gpu]
    t = bs * (1 + spec)
    expert_bytes = experts_touched(t) * EXPERT_BYTES * LAYERS / tp
    resident_expert = 259.5 / tp
    off_frac = min(offload / resident_expert, 1.0) if resident_expert else 0
    hbm = expert_bytes * (1 - off_frac) + DENSE_STEP_BYTES / tp
    pcie = expert_bytes * off_frac
    ms = hbm / (g["bw"] * 1e12 * eta) * 1e3 + pcie / (H2D_GBS[g["link"]] * 1e9) * 1e3
    if tp > 1:
        ms += LAYERS * 2 * AR_US[g["link"]] / 1e3
    ms += fixed_ms
    per_user = (1 + accept) / ms * 1e3
    return ms, per_user, per_user * bs


# ------------------------------------------------------------------ 场景
def s_ckpt(a):
    print("-- checkpoint 构成（recipe Memory 表）")
    for name, p, s in CKPT:
        print(f"   {name:40s}{p:8.1f}B{s:9.1f} GiB")
    print(f"   合计 {TOTAL_GIB:.1f} GiB（recipe 写约 {CKPT_GB} GB / 476 GiB）；vram_minimum_gb = {CKPT_GB} × 1.2 = {CKPT_GB * 1.2:.0f} GB")
    print(f"   Engram：{ENGRAM_ROWS:,} 行 × {ENGRAM_DIM} = {ENGRAM_VALUES_GIB:.2f} GiB 值，加 scale = {ENGRAM_GIB:.2f} GiB")
    print(f"   除 Engram 外 {NON_ENGRAM_GIB:.1f} GiB 要进显存；Engram 可以整张搬到主机内存")
    print(f"   一个专家 {EXPERT_PARAMS / 1e6:.1f}M 参数，MXFP4 {EXPERT_BYTES / 1e6:.1f} MB；"
          f"{N_EXPERTS} 路由 × {LAYERS} 层 = {N_EXPERTS * LAYERS * EXPERT_PARAMS / 1e9:.1f}B 参数")
    print(f"   每步必读稠密部分 {DENSE_STEP_BYTES / 1e9:.1f} GB；bs=1 不投机时读 7 个专家 × 40 层 = "
          f"{7 * EXPERT_BYTES * LAYERS / 1e9:.1f} GB")
    for tp in (1, 2, 4, 8):
        print(f"   TP{tp}：静态常驻 {NON_ENGRAM_GIB / tp:6.1f} GiB/卡，Engram 留在显存再加 {ENGRAM_GIB / tp:6.1f} GiB/卡（DP=1）")


def show_plan(p):
    g = GPUS[p["gpu"]]
    print(f"   {g['label']} × {p['n']}，TP{p['tp']} DP{p['dp']}，Engram 在{'显存' if p['engram'] == 'gpu' else '主机'}"
          f"{'，专家卸载 %.0f GiB/卡' % p['offload'] if p['offload'] else ''}")
    print(f"     可用 {p['usable']:.1f} GiB = {g['mem']} × gmu")
    print(f"     常驻权重 {p['resident']:.1f} GiB（静态 {p['static']:.1f} × 系数 {RESIDENT_FACTOR[p['arch']]:.3f}"
          f"{' − 卸载 %.0f' % p['offload'] if p['offload'] else ''}）")
    if p["eng_gpu"]:
        print(f"     Engram {p['eng_gpu']:.1f} GiB（{ENGRAM_GIB:.2f} / {p['shards']} 份）")
    print(f"     预留 {p['reserve']:.1f} GiB = indexer logits {p['idx']:.2f} + 其他 {p['other']:.2f}（其他是两条实测校准出的残差；gather workspace {p['ws']:.2f} 不单列）")
    print(f"     KV 池 {p['kv']:.1f} GiB → {p['tokens'] / 1e6:.2f}M token（{p['bpt']:.0f} B/token），"
          f"{p['max_len']} 上下文并发 {p['conc']:.2f}{'' if p['fits'] else '  ✗ 起不来：装不下一条满长请求'}")
    print(f"     主机内存 ≥ {p['host']:.0f} GiB，建议配 {dimm(p['host'] * 1.1)} GB")


def s_mem(a):
    print("-- 显存账（每卡）")
    p = plan(a.gpu, a.n, a.tp, a.engram, a.offload_gib, a.gmu, a.max_model_len, a.max_num_seqs, a.batched,
             a.spec, a.version, a.thp)
    show_plan(p)
    print("-- 和 recipe 对账")
    h = plan("h200", 8, 4, "host", 0, 0.92, 1048576, 16, 8192, 5, "v031")
    print(f"   H200 TP4 Engram 卸载：常驻 {h['resident']:.1f}（recipe 实测 81.2），KV {h['kv']:.1f} GiB（recipe 38.5），"
          f"{h['tokens'] / 1e6:.1f}M token（recipe 约 20M）")
    print(f"   H100 TP8 静态常驻 {NON_ENGRAM_GIB / 8:.1f} GiB（recipe 写约 36，是静态账）")
    print(f"   8×H200 TP4×2：按 TP×DP 摊是每 rank {ENGRAM_GIB / 8:.1f} GiB、每副本 {ENGRAM_GIB / 2:.1f} GiB（recipe 注释 23.6 / 94）；"
          f"主干默认 dp_shared_memory，整机一份 {ENGRAM_GIB:.1f} GiB 在 /dev/shm")


def s_kv(a):
    print("-- 每 token 全局 KV（只有 2、8、14、20 层分配，压缩比 2/2/2/1，每 token 2.5 条）")
    sm100 = (COMP_BYTES["nvfp4_ds_mla"] + INDEXER_K["mxfp4"]) * ENTRIES_PER_TOKEN
    print(f"   SM100 nvfp4_ds_mla：(288 + 68) × 2.5 = {sm100:.0f} B（博客 {BLOG_GLOBAL} B）")
    print(f"   SM100 fp8_ds_mla MXFP8 记录：(528 + 68) × 2.5 = {(528 + 68) * 2.5:.0f} B")
    print(f"   SM90 / SM120 / SM89 fp8_ds_mla：(584 + 132) × 2.5 = {kv_bytes_per_token('sm90'):.0f} B，是博客的 "
          f"{kv_bytes_per_token('sm90') / BLOG_GLOBAL:.2f} 倍")
    for arch in ("sm100", "sm90"):
        swa = swa_bytes_per_req(arch)
        print(f"   SWA（{arch}）：{kv_format(arch)['swa']} B × 128 × 40 层 = {swa / MiB:.2f} MiB/请求")
    print("-- 单请求 KV（SM90 / SM120 格式）")
    bpt, swa = kv_bytes_per_token("sm90"), swa_bytes_per_req("sm90")
    for L in (4096, 32768, 131072, 262144, 1048576):
        g = L * bpt
        print(f"   {L:>8d} token：全局 {g / MiB:8.1f} MiB + SWA {swa / MiB:.2f} MiB，SWA 占 {swa / (g + swa) * 100:5.1f}%")


def s_indexer(a):
    arch = GPUS[a.gpu]["arch"]
    L, seqs, b, k = a.max_model_len, a.max_num_seqs, a.batched, a.spec
    print(f"-- indexer logits 预留（{GPUS[a.gpu]['label']}，len={L}，seqs={seqs}，batched={b}，spec={k}）")
    print(f"   v0.31.0 / v0.32.0：固定 {indexer_reserve(arch, L, seqs, b, k, 'v031') / MiB:.0f} MiB")
    print(f"   recipe 注释的形状 batched × len × 2 B：{indexer_reserve(arch, L, seqs, b, k, 'recipe') / GiB:.2f} GiB")
    print(f"     8192 × 1M = {8192 * 1048576 * 2 / GiB:.0f} GiB，4096 × 1M = {4096 * 1048576 * 2 / GiB:.0f} GiB")
    itemsize, split = (2, 384) if arch == "sm100" else (4, 256)
    print(f"   主干 #62860 之后：min(seqs, batched) × (1+spec) × round_up(len, {math.lcm(split, 1024 // itemsize)}) × "
          f"{itemsize + (4 if arch == 'sm100' else 0)} B = {indexer_reserve(arch, L, seqs, b, k, 'main') / GiB:.2f} GiB")
    print("   主干形状随 max-num-seqs 变化（1M 上下文，非 SM100）：")
    for s in (8, 16, 64, 256, 1024):
        print(f"     seqs={s:<5d}{indexer_reserve('sm90', 1048576, s, 8192, 5, 'main') / GiB:8.2f} GiB")
    print(f"   gather workspace：len × 8 × factor 条 × {132 if arch != 'sm100' else 68} B = {gather_workspace(arch, L) / GiB:.2f} GiB")
    for x in (131072, 262144, 524288, 1048576):
        print(f"     len={x:<8d}factor={5 if x <= 262144 else 2}，{prefill_buffer_entries(x):>10d} 条，"
              f"{gather_workspace('sm90', x) / GiB:.2f} GiB")


def s_host(a):
    print("-- Engram 放主机：日志写的大小和实际 pinned 的大小")
    for n, tp in ((1, 1), (4, 4), (8, 8), (8, 4)):
        exact, pinned = engram_host_bytes(n, tp)
        sh = engram_shards(n, tp)
        how = "DP 副本共享 /dev/shm" if n // tp > 1 else "每 rank 私有 pinned"
        print(f"   {n} 卡 TP{tp}：{sh} 份，{how}，每份日志 {exact / sh / GiB:6.2f} GiB，实占 {pinned / sh / GiB:6.1f} GiB；"
              f"合计 {exact / GiB:.2f} → {pinned / GiB:.1f} GiB")
    print("   私有 pinned 按 2 的幂取整（engram.py:1122 注释）；加 use_thp 走大页，按实际大小")
    print("-- 当前配置的主机内存")
    p = plan(a.gpu, a.n, a.tp, a.engram, a.offload_gib, a.gmu, a.max_model_len, a.max_num_seqs, a.batched,
             a.spec, a.version, a.thp)
    print(f"   {GPUS[a.gpu]['label']} × {a.n} TP{a.tp}，专家卸载 {a.offload_gib:.0f} GiB/卡，use_thp={a.thp}："
          f"≥ {p['host']:.0f} GiB，建议 {dimm(p['host'] * 1.1)} GB")
    if p["engram"] == "host" and not a.thp and a.n // a.tp == 1:
        t = host_need(a.n, a.tp, "host", a.offload_gib, True)
        print(f"   加 use_thp：≥ {t:.0f} GiB，建议 {dimm(t * 1.1)} GB（大页分配失败会退回普通 pinned，按上一行准备更稳）")


def s_decode(a):
    print("-- decode 带宽估算（η=0.6，均匀路由，接受 4.2 个草稿 → 每步 5.2 token）")
    rows = [("rtxpro6000", 4, 4, 0), ("rtxpro6000", 1, 1, 214), ("rtx5090", 8, 8, 9), ("rtx4090_48g", 8, 8, 0),
            ("h20", 8, 8, 0), ("h20_141g", 8, 8, 0)]
    for gpu, n, tp, off in rows:
        out = []
        for bs in (1, 8, 16):
            ms, pu, tot = decode_bound(gpu, n, tp, bs, 5, 4.2, off)
            out.append(f"bs{bs} {ms:6.1f} ms/步 {tot:7.0f} tok/s")
        print(f"   {GPUS[gpu]['label']:22s}×{n} 卸载{off:>4.0f}  " + "；".join(out))
    print("   单卡把 214 GiB 专家放主机，每步要从 PCIe 读一大半命中的专家，PCIe 成了瓶颈")
    print("-- 公开实测 / 估算（估算按各自的接受长度重算）")
    for name, gpu, n, tp, off, bs, acc, measured, src in MEASURED:
        _, pu, tot = decode_bound(gpu, n, tp, bs, 5, acc - 1, off)
        bound = pu if bs == 1 else tot
        print(f"   {name:38s}bs{bs:<3d}接受长度 {acc:.2f}  实测 {measured:7.1f}  估算 {bound:7.0f}  比值 {measured / bound:.2f}  {src}")
    print("   比值 > 1：同一条序列的草稿 token 选的专家高度重合，均匀路由高估了要读的专家数")
    print("   比值 < 0.6：卡在带宽以外的地方，常见是 PCIe all-reduce、专家卸载、kernel 没调好（Marlin 回退、分叉旧版）")
    print("   H20 上唯一的公开点是 SGLang，比值 0.39：H20 的 c1 / c16 估算按上限读，vLLM 在 H20 上还没有公开实测")


# 公开实测点：(名字, gpu, 卡数, TP, 每卡卸载, bs, 每步接受长度, 实测 tok/s, 来源)
# 接受长度：yhfgyyf 报 5.206；sglang#46310 报每次验证 4.4–4.8，取 4.6；其余没报，按 commdata2338 的 3.23–3.95 取中 3.6（估算）
# sglang#46310 只报 TPOT（8K 输入 c8 是 5.56 ms），换算成 8 路合计 decode 速度 8 × 1000 / 5.56
MEASURED = [
    ("yhfgyyf 4×PRO 6000 分叉 8K c1", "rtxpro6000", 4, 4, 0, 1, 5.206, 267.0, "sm89 分叉 README"),
    ("commdata2338 4×PRO 6000 prose bs1", "rtxpro6000", 4, 4, 0, 1, 3.6, 252.0, "Kraken + B12X"),
    ("commdata2338 4×PRO 6000 prose bs8", "rtxpro6000", 4, 4, 0, 8, 3.6, 822.0, "Kraken + B12X"),
    ("commdata2338 4×PRO 6000 prose bs16", "rtxpro6000", 4, 4, 0, 16, 3.6, 1150.0, "Kraken + B12X"),
    ("devin-lai 8×5090 c1", "rtx5090", 8, 8, 9, 1, 3.6, 74.75, "主线 vLLM + Marlin"),
    ("devin-lai 8×5090 c8", "rtx5090", 8, 8, 9, 8, 3.6, 168.49, "主线 vLLM + Marlin"),
    ("sglang#46310 8×H20-3e c8（TPOT 换算）", "h20_141g", 8, 8, 0, 8, 4.6, 8 * 1000 / 5.56, "SGLang，非 vLLM"),
]

CONFIGS = [
    # (名字, gpu, n, tp, engram, offload, gmu, max_len, seqs, batched, 档, 路径)
    ("1×PRO 6000 + 768G 内存", "rtxpro6000", 1, 1, "host", 214, 0.92, 131072, 2, 4096, "入门", "vLLM UVA 卸载；更快的是 KTransformers"),
    ("1×5090 + 768G 内存", "rtx5090", 1, 1, "host", 270, 0.90, 32768, 1, 2048, "不推荐", "能起但 decode 约 10 tok/s，用 llama.cpp / Maya"),
    ("8×4090 24G", "rtx4090", 8, 8, "host", 0, 0.98, 32768, 4, 4096, "不可行", "常驻 40 GiB/卡 > 24"),
    ("8×4090 48G（改装）", "rtx4090_48g", 8, 8, "host", 0, 0.98, 262144, 4, 4096, "可用", "只有 SM89 分叉"),
    ("8×5090", "rtx5090", 8, 8, "host", 9, 0.95, 131072, 32, 8192, "可用", "每卡卸 9 GiB 专家"),
    ("4×PRO 6000", "rtxpro6000", 4, 4, "host", 0, 0.93, 1048576, 16, 4096, "消费级首选", "主线 vLLM + FlashInfer #4380"),
    ("8×PRO 6000", "rtxpro6000", 8, 8, "gpu", 0, 0.92, 1048576, 64, 8192, "消费级高配", "Engram 可留显存"),
    ("8×H20 96G Engram 主机", "h20", 8, 8, "host", 0, 0.92, 1048576, 64, 4096, "数据中心可用", "套 H100 模板"),
    ("8×H20 96G Engram 显存", "h20", 8, 8, "gpu", 0, 0.92, 262144, 64, 4096, "数据中心可用", "不依赖主机内存"),
    ("8×H20-3e TP8 Engram 显存", "h20_141g", 8, 8, "gpu", 0, 0.92, 1048576, 128, 8192, "数据中心首选", "一台机一个副本"),
    ("8×H20-3e TP4×2 Engram 主机", "h20_141g", 8, 4, "host", 0, 0.92, 1048576, 64, 8192, "数据中心高吞吐", "套 H200 模板"),
    ("8×H100 80G（recipe）", "h100", 8, 8, "host", 0, 0.92, 1048576, 64, 4096, "参照", "recipe 的 H100 模板"),
    ("8×H200（recipe）", "h200", 8, 4, "host", 0, 0.92, 1048576, 64, 8192, "参照", "recipe 的 H200 模板"),
]


def rank_rows(version="main"):
    out = []
    for name, gpu, n, tp, eng, off, gmu, L, seqs, b, tier, path in CONFIGS:
        p = plan(gpu, n, tp, eng, off, gmu, L, seqs, b, 5, version)
        _, pu1, _ = decode_bound(gpu, n, tp, 1, 5, 4.2, off)
        _, _, tot16 = decode_bound(gpu, n, tp, 16, 5, 4.2, off)
        out.append(dict(name=name, tier=tier, path=path, p=p, c1=pu1, c16=tot16 * (p["dp"])))
    return out


def s_rank(a):
    print(f"-- 卡配置排行（indexer 预留按 {a.version}；c1 / c16 是 decode 带宽估算，全部是估算）")
    print(f"   {'配置':28s}{'常驻':>7s}{'KV':>7s}{'KV token':>10s}{'上下文':>9s}{'并发':>7s}{'主机内存':>8s}"
          f"{'c1 估算':>9s}{'c16 估算':>10s}  档位 / 路径")
    for r in rank_rows(a.version):
        p = r["p"]
        kv = f"{p['kv']:.1f}" if p["kv"] > 0 else "✗"
        print(f"   {r['name']:28s}{p['resident'] + p['eng_gpu']:7.1f}{kv:>7s}{p['tokens'] / 1e6:9.2f}M"
              f"{p['max_len'] // 1024:>8d}K{p['conc']:7.1f}{dimm(p['host'] * 1.1):>7d}G"
              + (f"{r['c1']:9.0f}{r['c16']:10.0f}" if p["fits"] else f"{'—':>9s}{'—':>10s}")
              + f"  {r['tier']}：{r['path']}")


SCEN = {"ckpt": s_ckpt, "mem": s_mem, "kv": s_kv, "indexer": s_indexer, "host": s_host, "decode": s_decode,
        "rank": s_rank}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="all", choices=list(SCEN) + ["all"])
    ap.add_argument("--gpu", default="rtxpro6000", choices=list(GPUS))
    ap.add_argument("--n", type=int, default=4, help="卡数")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--engram", default="host", choices=["host", "gpu"])
    ap.add_argument("--offload-gib", type=float, default=0.0, help="每卡卸载到主机的专家权重（--cpu-offload-gb）")
    ap.add_argument("--gmu", type=float, default=0.93)
    ap.add_argument("--max-model-len", type=int, default=1048576)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--batched", type=int, default=4096, help="--max-num-batched-tokens")
    ap.add_argument("--spec", type=int, default=5, help="DSpark 草稿 token 数")
    ap.add_argument("--version", default="main", choices=["v031", "recipe", "main"], help="indexer 预留按哪个版本算")
    ap.add_argument("--thp", action="store_true", help='--engram-config \'{"cpu_offload":true,"use_thp":true}\'')
    a = ap.parse_args(argv)
    for name, fn in SCEN.items():
        if a.scenario in (name, "all"):
            if a.scenario == "all":
                print(f"== {name}")
            fn(a)


if __name__ == "__main__":
    main()
