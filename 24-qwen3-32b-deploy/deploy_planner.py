#!/usr/bin/env python3
"""VLLM-24：Qwen3-32B 部署规划器，只用 Python 标准库，不联网、不需要 GPU。

场景：
  weights  四种 checkpoint 的字节数，和 recipe 的 vram_minimum_gb 对账
  mem      给定卡型 / 卡数 / TP / 格式 / KV dtype，算每卡常驻、其他占用和 KV 池
  kv       每 token 的 KV 字节，40K / 128K 上下文一条请求要多少显存
  tp       同样两张卡：TP2 跑 BF16，还是两份单卡量化
  spec     EAGLE3 多占多少显存、k 取几；ngram 不占显存
  profile  批大小默认值、CUDA graph 抓哪些尺寸、显存提示里的 gmu 怎么算
  decode   单流 decode 带宽估算，和公开实测对照（全部是估算）
  rank     十几种卡配置放在一起排
  qwen35   Qwen3.5-27B：GDN 混合层的块大小、每条序列的状态和 GPTQ 的坑
  all      全部

用法：
  python3 deploy_planner.py --scenario all
  python3 deploy_planner.py --scenario mem --gpu h100 --fmt fp8
  python3 deploy_planner.py --scenario mem --gpu rtx4090 --fmt awq --kv-dtype fp8 --max-model-len 16384 --max-num-seqs 8 --gmu 0.95
  python3 deploy_planner.py --scenario mem --gpu rtx3090 --n 2 --tp 2 --fmt awq --capture 8
  python3 deploy_planner.py --scenario profile --gpu a100 --max-num-seqs 64

源码行号对应 vllm 主干 0d6f92a；recipe 指 recipes/models/Qwen/Qwen3-32B.yaml 和 Qwen3.5-27B.yaml。
"""
import argparse
import math

GiB = 1024 ** 3
MiB = 1024 ** 2

# ------------------------------------------------------------------ 模型常量
# Qwen/Qwen3-32B config.json
LAYERS, HIDDEN, INTER = 64, 5120, 25600
Q_HEADS, KV_HEADS, HEAD_DIM = 64, 8, 128
VOCAB, MAX_POS = 151936, 40960
PARAMS_TOTAL = 32_762_123_264
PARAMS_LINEAR = 31_205_621_760        # 64 层里能量化的线性层
PARAMS_BF16 = 1_556_501_504           # 不量化：embed 和 lm_head（不共享）各 151936 × 5120，加 norm
# safetensors 合计字节（HF 仓库文件大小相加）
CKPT_BYTES = {
    "bf16": 65_524_328_560,           # Qwen/Qwen3-32B
    "fp8": 34_322_567_640,            # Qwen/Qwen3-32B-FP8，block 128×128
    "awq": 19_325_481_744,            # Qwen/Qwen3-32B-AWQ，int4 g128 带零点
    "nvfp4": 20_666_400_824,          # RedHatAI/Qwen3-32B-NVFP4
}
# recipe vram_minimum_gb：bf16 第 62 行，fp8 第 67 行，awq 第 72 行，nvfp4 第 101 行
RECIPE_MIN_GB = {"bf16": 77, "fp8": 39, "awq": 20, "nvfp4": 22}
# 加载后常驻（GiB，TP1）。awq 和 nvfp4 用 issue 日志里的 "Model loading took"：
#   awq   #22344  RTX 5090 D，awq_marlin，18.1412 GiB（Marlin 重排后比 checkpoint 多 0.14）
#   nvfp4 #22783  RTX 5090，compressed-tensors，21.2691 GiB
#   bf16 / fp8 没有干净的单卡日志，取 checkpoint 大小（估算）
RESIDENT_TP1 = {"bf16": CKPT_BYTES["bf16"] / GiB, "fp8": CKPT_BYTES["fp8"] / GiB, "awq": 18.1412, "nvfp4": 21.2691}
KV_ELEM = {"bf16": 2, "fp8": 1}

# EAGLE3 草稿 RedHatAI/Qwen3-32B-speculator.eagle3：文件 2.907 GiB，1 层，草稿词表 32000
# #28919 带 eagle3 加载 62.3970 GiB，比 BF16 checkpoint 多 1.373 GiB（embed 和目标模型共享，没再算一份）
EAGLE3_FILE_GIB = 2.907
EAGLE3_EXTRA_GIB = 62.3970 - CKPT_BYTES["bf16"] / GiB
EAGLE3_LAYERS = 1
# 模型卡上的接受长度（每次验证平均产出 token 数），k = 1..7
EAGLE3_ACCEPT = {
    "coding":        [1.67, 2.06, 2.29, 2.39, 2.47, 2.50, 2.53],
    "math":          [1.73, 2.21, 2.49, 2.69, 2.80, 2.83, 3.08],
    "summarization": [1.62, 1.95, 2.15, 2.23, 2.27, 2.32, 2.33],
}

# ------------------------------------------------------------------ 卡
# mem 是 torch 看到的总显存（GiB）：H100 79.11 取自 #29049，其余按 nvidia-smi 换算（估算）
# bw 是标称显存带宽 TB/s；link 决定 all-reduce 延迟
GPUS = {
    "rtx3090":    dict(arch="sm86", mem=23.57, bw=0.936, link="pcie4", label="RTX 3090 24G"),
    "rtx4090":    dict(arch="sm89", mem=23.52, bw=1.008, link="pcie4", label="RTX 4090 24G"),
    "l40s":       dict(arch="sm89", mem=44.4, bw=0.864, link="pcie4", label="L40S 48G"),
    "rtx5090":    dict(arch="sm120", mem=31.6, bw=1.792, link="pcie5", label="RTX 5090 32G"),
    "rtxpro6000": dict(arch="sm120", mem=95.0, bw=1.792, link="pcie5", label="RTX PRO 6000 96G"),
    "a100":       dict(arch="sm80", mem=79.15, bw=2.039, link="nvlink", label="A100 80G"),
    "h100":       dict(arch="sm90", mem=79.11, bw=3.35, link="nvlink", label="H100 80G"),
    "h20":        dict(arch="sm90", mem=95.1, bw=4.0, link="nvlink", label="H20 96G"),
    "h20_141g":   dict(arch="sm90", mem=139.7, bw=4.8, link="nvlink", label="H20-3e 141G"),
    "h200":       dict(arch="sm90", mem=140.4, bw=4.8, link="nvlink", label="H200 141G"),
    "b200":       dict(arch="sm100", mem=178.35, bw=8.0, link="nvlink", label="B200 180G"),
}
AR_US = {"nvlink": 10, "pcie5": 25, "pcie4": 35}   # 一次小消息 all-reduce（微秒，估算）

# ------------------------------------------------------------------ 其他占用（KV 之外、权重之外）
# other = 非 torch 常量 + profile 激活峰值 + CUDA graph
#   激活峰值按 max_num_batched_tokens 线性涨，系数用 #22344 反推：
#     RTX 5090 D，eager（没有 graph），gmu 0.95，batched 32768，常驻 18.1412，KV 5.68 GiB
#     0.95 × 31.6 − 18.1412 − 5.68 − 0.4 = 5.80 GiB → 每 token 0.181 MiB（TP1）
#   MLP 那部分随 TP 切，按 80% 能切估（估算）
#   CUDA graph：#28919 抓到 512（51 个尺寸）用了 0.85 GiB，按尺寸数线性摊（估算）
#   采样器预热：num_seqs × vocab × fp32，按 3 份算（gpu_model_runner.py:5866，估算），和激活峰值取大
#   LMCache FAQ 的 TP2 H100（33.66 GiB/卡）只拿来对 token 数，不拿来校准：它多出的 3–4 GiB 里有 LMCache 自己的缓冲
NONTORCH_GIB = 0.4
NCCL_GIB = 0.3
ACT_FIT_GIB = 0.95 * 31.6 - 18.1412 - 5.68 - NONTORCH_GIB
ACT_PER_TOKEN = ACT_FIT_GIB * GiB / 32768
GRAPH_REF_GIB, GRAPH_REF_SIZES = 0.85, 51

# ------------------------------------------------------------------ 源码里的默认值


def default_batch(gpu):
    """arg_utils.py 2950–3012（#17885）：按显存分三档，A100 按名字排除。返回 (server batched, max_num_seqs)。"""
    g = GPUS[gpu]
    if g["mem"] >= 160:
        return 16384, 1024
    if g["mem"] >= 70 and "a100" not in gpu:
        return 8192, 1024
    return 2048, 256


def capture_sizes(seqs, batched, q=1, family100=False, mode="balanced", max_capture=None):
    """config/vllm.py 2465–2490 定上限，2588–2610 列尺寸。"""
    if max_capture is None:
        max_capture = min(seqs * q * 2, 1024 if family100 else 512)
    if mode == "interactivity":
        sizes = list(range(1, min(max_capture, 32) + 1))
    else:
        sizes = [i for i in (1, 2, 4) if i <= max_capture]
    if max_capture >= 8:
        sizes += list(range(8, min(max_capture + 1, 256), 8))
    if max_capture >= 256:
        sizes += list(range(256, max_capture + 1, 16))
    if batched <= max_capture and batched not in sizes:
        sizes.append(batched)
    return sorted(set(sizes))


def kv_per_token(kv_dtype="bf16", tp=1, extra_layers=0):
    """每 token 每卡 KV 字节：2（K、V）× 层 × KV 头 / TP × head_dim × 元素字节。"""
    heads = max(KV_HEADS // tp, 1)
    return 2 * (LAYERS + extra_layers) * heads * HEAD_DIM * KV_ELEM[kv_dtype]


def kernel_path(arch, fmt):
    """线性层走哪个 kernel（kernels/linear/__init__.py 493–506；scaled_mm_entry.cu 164–173）。"""
    if fmt == "bf16":
        return "BF16 cuBLAS"
    if fmt == "awq":
        return "AWQ Marlin（--quantization awq 也会落到这里，auto_awq.py:263）"
    if fmt == "fp8":
        if arch in ("sm80", "sm86", "sm89"):
            return "FP8 Marlin（W8A16，SM89 没有 block FP8 的 CUTLASS kernel）"
        if arch == "sm90":
            return "DeepGEMM（需 nvcc）或 CUTLASS block FP8"
        if arch == "sm100":
            return "FlashInfer / DeepGEMM block FP8"
        return "B12x / CUTLASS block FP8"
    if fmt == "nvfp4":
        return "原生 FP4" if arch in ("sm100", "sm120") else "NVFP4 Marlin 回退（marlin.py:35 会打警告）"
    return "?"


def other_gib(gpu, tp, batched, seqs, eager=False, capture=None, spec=0):
    act = batched * ACT_PER_TOKEN * (0.8 / tp + 0.2) / GiB
    sampler = seqs * (1 + spec) * VOCAB * 4 * 3 / GiB
    if eager:
        graph, n = 0.0, 0
    else:
        sizes = capture_sizes(seqs, batched, 1 + spec, GPUS[gpu]["arch"] == "sm100", max_capture=capture)
        n = len(sizes)
        graph = GRAPH_REF_GIB * n / GRAPH_REF_SIZES
    nontorch = NONTORCH_GIB + (NCCL_GIB if tp > 1 else 0.0)
    return dict(act=act, sampler=sampler, graph=graph, n_sizes=n, nontorch=nontorch,
                total=nontorch + max(act, sampler) + graph)


def plan(gpu, n=1, tp=1, fmt="bf16", kv_dtype="bf16", gmu=0.92, max_len=MAX_POS, seqs=None, batched=None,
         eager=False, capture=None, eagle=False, spec=0):
    g = GPUS[gpu]
    db, ds = default_batch(gpu)
    seqs = seqs or ds
    batched = batched or db
    resident = RESIDENT_TP1[fmt] / tp + (EAGLE3_EXTRA_GIB if eagle else 0.0)
    o = other_gib(gpu, tp, batched, seqs, eager, capture, spec)
    usable = g["mem"] * gmu
    kv = usable - resident - o["total"]
    bpt = kv_per_token(kv_dtype, tp, EAGLE3_LAYERS if eagle else 0)
    tokens = max(kv, 0) * GiB / bpt
    conc = tokens / max_len
    return dict(gpu=gpu, n=n, tp=tp, dp=n // tp, fmt=fmt, kv_dtype=kv_dtype, gmu=gmu, usable=usable,
                resident=resident, other=o, kv=kv, bpt=bpt, tokens=tokens, conc=conc, max_len=max_len,
                seqs=seqs, batched=batched, fits=conc >= 1, path=kernel_path(g["arch"], fmt))


def decode_bound(gpu, tp=1, fmt="bf16", bs=1, ctx=1024, kv_dtype="bf16", eta=0.75, fixed_ms=0.5):
    """一步 decode 读的字节 / (η × 带宽) + all-reduce + 固定开销。返回 (ms/步, 单流 tok/s, 合计 tok/s)。"""
    g = GPUS[gpu]
    w = RESIDENT_TP1[fmt] * GiB / tp
    kvb = bs * ctx * kv_per_token(kv_dtype, tp)
    ms = (w + kvb) / (eta * g["bw"] * 1e12) * 1e3 + fixed_ms
    if tp > 1:
        ms += 2 * LAYERS * AR_US[g["link"]] / 1e3
    return ms, 1e3 / ms, bs * 1e3 / ms


# ------------------------------------------------------------------ Qwen3.5-27B
# config.json：64 层 = 16 层全注意力 + 48 层 GDN；全注意力 24 Q 头 / 4 KV 头，head_dim 256
# GDN：k 头 16、v 头 48、头维 128，conv 核 4，时间状态 fp32
Q35_FULL, Q35_GDN, Q35_KV_HEADS, Q35_HEAD_DIM = 16, 48, 4, 256
Q35_CONV_DIM, Q35_V_HEADS, Q35_GDN_DIM, Q35_CONV_K = 10240, 48, 128, 4
# HF 仓库按模块拆的 GiB（BF16 / FP8 / GPTQ-Int4）；GPTQ 只量化了 MLP
Q35_PARTS = [("mlp", 31.88, 15.94, 8.29), ("linear_attn（GDN）", 10.36, 5.20, 10.36),
             ("self_attn", 3.13, 1.56, 3.13), ("lm_head", 2.37, 2.37, 2.37), ("embed", 2.37, 2.37, 2.37),
             ("visual", 0.86, 0.86, 0.86), ("mtp", 0.79, 0.44, 0.79)]
Q35_CKPT = {"bf16": 51.75, "fp8": 28.75, "gptq": 28.16}
Q35_RECIPE_MIN_GB = {"bf16": 65, "fp8": 33, "gptq": 17}   # Qwen3.5-27B.yaml 第 76 / 81 / 86 行


def q35_kv_per_token(kv_dtype="bf16", tp=1):
    return 2 * Q35_FULL * max(Q35_KV_HEADS // tp, 1) * Q35_HEAD_DIM * KV_ELEM[kv_dtype]


def q35_mamba_page(tp=1, spec=0, dtype_bytes=2):
    """一层 GDN 一条序列的状态字节：conv (10240/TP, 3+spec) 按模型 dtype + 时间状态 (48/TP, 128, 128) fp32。"""
    conv = (Q35_CONV_DIM // tp) * (Q35_CONV_K - 1 + spec) * dtype_bytes
    temporal = (Q35_V_HEADS // tp) * Q35_GDN_DIM * Q35_GDN_DIM * 4
    return conv + temporal


def q35_block_size(tp=1, kv_dtype="bf16", spec=0, align=16):
    """platforms/interface.py 1050–1125：注意力块放大到一页不小于 mamba 页。
    v0.19 对齐 16；主干在 flashinfer GDN prefill 后端下和 64 取最小公倍数（gdn_attn.py:34）。"""
    attn1 = 2 * max(Q35_KV_HEADS // tp, 1) * Q35_HEAD_DIM * KV_ELEM[kv_dtype]   # 一层一个 token
    return align * math.ceil(q35_mamba_page(tp, spec) / (align * attn1))


def q35_state_per_seq(tp=1, spec=0, mode="none", ckpt=0):
    """kv_cache_interface.py 1135–1141：align 模式每条序列占 2 + spec + ckpt 页，否则 1 + spec 页。"""
    pages = (2 + spec + ckpt) if mode == "align" else (1 + spec)
    return Q35_GDN * q35_mamba_page(tp, spec) * pages


# ------------------------------------------------------------------ 场景
def s_weights(a):
    print("-- checkpoint 字节（HF 仓库 safetensors 合计）")
    print(f"   参数 {PARAMS_TOTAL / 1e9:.2f}B = 可量化线性层 {PARAMS_LINEAR / 1e9:.2f}B + 保持 BF16 的 {PARAMS_BF16 / 1e9:.2f}B（embed、lm_head 各 {VOCAB * HIDDEN / 1e9:.3f}B，不共享）")
    for fmt, b in CKPT_BYTES.items():
        print(f"   {fmt:6s}{b / GiB:7.2f} GiB  {b / 1e9:6.1f} GB  每参数 {b * 8 / PARAMS_TOTAL:5.2f} bit   recipe vram_minimum_gb {RECIPE_MIN_GB[fmt]:3d}"
              f"（≈ {b / 1e9:.1f} × {RECIPE_MIN_GB[fmt] / (b / 1e9):.2f}）")
    print(f"   加载后常驻：awq {RESIDENT_TP1['awq']:.2f} GiB（#22344），nvfp4 {RESIDENT_TP1['nvfp4']:.2f} GiB（#22783），bf16 / fp8 按 checkpoint 算")
    print("   recipe 的 vram_minimum_gb：bf16 / fp8 是 checkpoint GB 的 1.14–1.18 倍，awq / nvfp4 只多 3–6%，都只够权重加一点 KV，开不满 40K")


def show_plan(p):
    g = GPUS[p["gpu"]]
    o = p["other"]
    print(f"   {g['label']} × {p['n']}，TP{p['tp']}{' DP%d' % p['dp'] if p['dp'] > 1 else ''}，{p['fmt']} 权重，KV {p['kv_dtype']}，"
          f"max-num-seqs {p['seqs']}，batched {p['batched']}")
    print(f"     可用 {p['usable']:.1f} GiB = {g['mem']} × {p['gmu']}")
    print(f"     常驻权重 {p['resident']:.2f} GiB（{p['path']}）")
    print(f"     其他 {o['total']:.2f} GiB = 非 torch {o['nontorch']:.2f} + max(激活峰值 {o['act']:.2f}, 采样器预热 {o['sampler']:.2f})"
          f" + CUDA graph {o['graph']:.2f}（{o['n_sizes']} 个尺寸）")
    if p["kv"] <= 0:
        print(f"     KV 池 {p['kv']:.2f} GiB  ✗ 起不来：权重和其他已经超过可用显存")
        return
    print(f"     KV 池 {p['kv']:.2f} GiB → {p['tokens']:,.0f} token（{p['bpt'] // 1024} KiB/token），"
          f"{p['max_len']} 上下文并发 {p['conc']:.2f}{'' if p['fits'] else '  ✗ 装不下一条满长请求'}")


def s_mem(a):
    print("-- 显存账（每卡）")
    show_plan(plan(a.gpu, a.n, a.tp, a.fmt, a.kv_dtype, a.gmu, a.max_model_len, a.max_num_seqs, a.batched,
                   a.eager, a.capture))
    print("-- 和公开日志对账")
    p = plan("rtx5090", 1, 1, "awq", "bf16", 0.95, 16384, 100, 32768, eager=True)
    print(f"   #22344 RTX 5090 D AWQ eager gmu 0.95：KV {p['kv']:.2f} GiB（日志 5.68），{p['tokens']:,.0f} token（日志 23,264），"
          f"并发 {p['conc']:.2f}（日志 1.42）。这一条是校准点")
    p = plan("h100", 2, 2, "bf16", "bf16", 0.9, MAX_POS)
    print(f"   LMCache FAQ TP2 H100：{p['tokens']:,.0f} token（FAQ 275,760），并发 {p['conc']:.2f}（FAQ 6.73）；"
          f"FAQ 每卡 KV 33.66 GiB，比估算少 {p['kv'] - 33.66:.1f}，差在 LMCache 自己的缓冲（估算）")
    t = 4096 * 16.91
    print(f"   #28919 eagle3 k=5、max-model-len 4096：日志 69,264 token、16.91x，{t:,.0f} / 4096 = 16.91，"
          f"按 64 层是 {69264 * kv_per_token() / GiB:.2f} GiB，加草稿 1 层是 {69264 * kv_per_token('bf16', 1, 1) / GiB:.2f} GiB，日志 17.08 在两者之间")
    print(f"   recipe 第 226 行说默认用 90% 显存，源码默认是 0.92（config/cache.py:215）")


def s_kv(a):
    print("-- 每 token KV（64 层 × 8 KV 头 × 128 × K/V）")
    for dt in ("bf16", "fp8"):
        row = "；".join(f"TP{tp} {kv_per_token(dt, tp) // 1024:3d} KiB" for tp in (1, 2, 4, 8))
        print(f"   {dt:5s}{row}")
    for L in (4096, 32768, 40960, 131072):
        print(f"   一条 {L:6d} token 的请求：bf16 {L * kv_per_token('bf16') / GiB:5.2f} GiB，fp8 {L * kv_per_token('fp8') / GiB:5.2f} GiB")
    print("   131072 要开 YaRN（recipe 41–47，factor 4、original 32768），recipe 注释说 32K 以下会掉点")
    print("   fp8 KV 的 scale：checkpoint 不带 q scale 时打 kv_cache.py:141 的警告，k/v 用 1.0（kv_cache.py:158）")
    print(f"   Qwen3.5-27B 只有 16 层全注意力、4 个 KV 头：bf16 {q35_kv_per_token() // 1024} KiB/token，"
          f"是 Qwen3-32B 的 1/{kv_per_token() // q35_kv_per_token()}")


def s_tp(a):
    print("-- 两张卡怎么用：TP2 一份，还是两份单卡（KV 是每个副本的，c1 / c32 是 decode 带宽估算）")
    rows = [("2×H100 TP2 BF16", "h100", 2, "bf16", "bf16", 0.92), ("2×H100 各跑 FP8", "h100", 1, "fp8", "bf16", 0.92),
            ("2×H100 TP2 FP8", "h100", 2, "fp8", "bf16", 0.92),
            ("2×L40S TP2 BF16", "l40s", 2, "bf16", "bf16", 0.92), ("2×L40S 各跑 FP8 + fp8 KV", "l40s", 1, "fp8", "fp8", 0.92),
            ("2×4090 TP2 AWQ", "rtx4090", 2, "awq", "bf16", 0.92), ("2×4090 各跑 AWQ + fp8 KV", "rtx4090", 1, "awq", "fp8", 0.95)]
    print(f"   {'配置':26s}{'副本':>4s}{'常驻/卡':>8s}{'KV/副本':>9s}{'token/副本':>12s}{'40K 并发':>9s}{'c1':>7s}{'c32 合计':>10s}")
    for name, gpu, tp, fmt, kvd, gmu in rows:
        p = plan(gpu, 2, tp, fmt, kvd, gmu)
        reps = 2 // tp
        _, c1, _ = decode_bound(gpu, tp, fmt, 1, 1024, kvd)
        _, _, c32 = decode_bound(gpu, tp, fmt, 32 // reps, 1024, kvd)
        kv = f"{p['kv']:.1f}" if p["kv"] > 0 else "✗"
        print(f"   {name:26s}{reps:4d}{p['resident']:8.1f}{kv:>9s}{p['tokens']:12,.0f}{p['conc']:9.2f}{c1:7.0f}{c32 * reps:10.0f}")
    print("   TP2 每层两次 all-reduce：NVLink 上每步多 1.3 ms，PCIe 4.0 上多 4.5 ms（估算），4090 / L40S 没有 NVLink")
    print("   KV 头 8 个，TP 最多切到 8；TP2 每卡 KV 字节减半，所以 TP2 的 token 数是单卡的两倍多")


def s_spec(a):
    print("-- EAGLE3（recipe 49–53：RedHatAI/Qwen3-32B-speculator.eagle3，k=3）")
    print(f"   草稿文件 {EAGLE3_FILE_GIB:.2f} GiB，加载后多 {EAGLE3_EXTRA_GIB:.2f} GiB（#28919：62.3970 − {CKPT_BYTES['bf16'] / GiB:.2f}），"
          f"KV 多 1 层：{kv_per_token('bf16', 1, 1) // 1024} KiB/token")
    print("   模型卡接受长度（每次验证平均产出 token），后面是比 k−1 多出来的部分：")
    for task, acc in EAGLE3_ACCEPT.items():
        cells = [f"k{k + 1} {v:.2f}" + (f"(+{v - acc[k - 1]:.2f})" if k else "") for k, v in enumerate(acc)]
        print(f"   {task:14s}" + " ".join(cells))
    print("   k=3 之后每多一个草稿只多 0.1 左右，草稿和验证的开销却按 k 涨：recipe 选 k=3 是对的，高并发再降到 1–2")
    for k in (1, 3, 5):
        sz = capture_sizes(32, 2048, 1 + k)
        print(f"   k={k}：CUDA graph 上限 min(seqs × {1 + k} × 2, 512)，seqs 32 时最大 {sz[-1]}，抓 {len(sz)} 个尺寸")
    print("   ngram（prompt_lookup_max / min，speculative.py:531 / 534）：不加载模型、不占显存，适合改写、代码补全这类输入里有原文的负载")


def s_profile(a):
    print("-- 源码默认值（arg_utils.py 2950–3012，#17885）")
    for gpu in ("rtx4090", "rtx5090", "l40s", "a100", "h100", "h20", "h200", "b200"):
        b, s = default_batch(gpu)
        print(f"   {GPUS[gpu]['label']:20s}{GPUS[gpu]['mem']:7.1f} GiB → batched {b:5d}，max-num-seqs {s:4d}"
              + ("   名字里有 a100，被排除在 70 GiB 档外" if gpu == "a100" else ""))
    print("-- CUDA graph 尺寸（config/vllm.py 2465–2490、2588–2610）")
    seqs = a.max_num_seqs or default_batch(a.gpu)[1]
    for mode in ("balanced", "interactivity"):
        sz = capture_sizes(seqs, a.batched or default_batch(a.gpu)[0], 1, GPUS[a.gpu]["arch"] == "sm100", mode)
        print(f"   {mode:13s} seqs {seqs}：{len(sz)} 个尺寸，最大 {sz[-1]}，前几个 {sz[:6]}")
    print(f"   #40121 的绕法只抓 [1, 2, 4, 8]：{len(capture_sizes(32, 8192, max_capture=8))} 个尺寸")
    print("-- 显存提示（gpu_worker.py 780–814）")
    total, util, cg = GPUS[a.gpu]["mem"], a.gmu, GRAPH_REF_GIB
    print(f"   {GPUS[a.gpu]['label']} gmu {util}，graph {cg} GiB：等价旧口径 gmu = {util} − {cg}/{total} = {util - cg / total:.4f}；"
          f"想保住 KV 池就调到 {min(util + cg / total, 1):.4f}")


# 公开实测点：(名字, gpu, TP, 格式, bs, 平均上下文, 实测单流 tok/s, 来源)
# 单流 = 1000 / TPOT；没报 TPOT 的取原文的单流速度。平均上下文按数据集估（估算）
#   Spheron H100 FP8 单流 65–80 tok/s，取中 72.5
#   jarvislabs 2×H100 TP2 BF16 ShareGPT c60 TPOT 20.80 ms
#   #40121 2×3090 AWQ TP2，32 并发每流 22 tok/s，CUDA graph 只抓到 8
#   #27649 RTX PRO 6000 BF16 TP1，单流 15 tok/s
MEASURED = [
    ("Spheron H100 FP8 单流", "h100", 1, "fp8", 1, 1024, 72.5, "Spheron 部署博客"),
    ("jarvislabs 2×H100 TP2 BF16 c60", "h100", 2, "bf16", 60, 600, 1000 / 20.80, "ShareGPT，vLLM"),
    ("#40121 2×3090 TP2 AWQ c32", "rtx3090", 2, "awq", 32, 1024, 22.0, "graph 只抓到 8"),
    ("#27649 PRO 6000 BF16 单流", "rtxpro6000", 1, "bf16", 1, 1024, 15.0, "CUDA 13.0，v0.11.0"),
]


def s_decode(a):
    print("-- decode 带宽估算（η=0.75，上下文 1K；每步读一遍权重加 KV）")
    for gpu, tp, fmt in (("rtx4090", 1, "awq"), ("rtx5090", 1, "awq"), ("l40s", 1, "fp8"), ("a100", 1, "bf16"),
                         ("h100", 1, "fp8"), ("h100", 2, "bf16"), ("h20", 1, "fp8"), ("h20", 1, "bf16"), ("b200", 1, "bf16")):
        out = []
        for bs in (1, 8, 32):
            ms, pu, tot = decode_bound(gpu, tp, fmt, bs)
            out.append(f"bs{bs} {ms:5.1f} ms {tot:6.0f} tok/s")
        print(f"   {GPUS[gpu]['label']:18s}TP{tp} {fmt:5s}" + "；".join(out))
    print("-- 公开实测 / 估算")
    for name, gpu, tp, fmt, bs, ctx, measured, src in MEASURED:
        _, pu, _ = decode_bound(gpu, tp, fmt, bs, ctx)
        print(f"   {name:32s}bs{bs:<3d}实测单流 {measured:5.1f}  估算 {pu:5.1f}  比值 {measured / pu:.2f}  {src}")
    print("   H100 单流对得上（1.03）。#40121 只有 0.55：图只抓到 8，32 并发的批走不了 CUDA graph")
    print("   #27649 是 v0.11.0 配 CUDA 13.0 跑 SM120，kernel 还没调好；jarvislabs 的 0.80 里有 ShareGPT 长短不一的 prefill 插队")


CONFIGS = [
    # (名字, gpu, n, tp, fmt, kv, gmu, max_len, seqs, batched, eager, capture, 档, 路径)
    ("1×4090 AWQ 默认参数", "rtx4090", 1, 1, "awq", "bf16", 0.92, 40960, None, None, False, None, "不可行", "40K 一条都装不下"),
    ("1×4090 AWQ fp8 KV 16K", "rtx4090", 1, 1, "awq", "fp8", 0.95, 16384, 8, 2048, False, None, "入门", "砍上下文和并发"),
    ("2×3090 TP2 AWQ", "rtx3090", 2, 2, "awq", "bf16", 0.75, 40960, 32, 8192, False, 8, "入门", "#40121 的绕法"),
    ("1×5090 AWQ fp8 KV", "rtx5090", 1, 1, "awq", "fp8", 0.92, 40960, 32, 4096, False, None, "消费级首选", "AWQ Marlin"),
    ("1×5090 NVFP4 fp8 KV", "rtx5090", 1, 1, "nvfp4", "fp8", 0.92, 40960, 32, 4096, False, None, "消费级可选", "原生 FP4"),
    ("1×L40S FP8", "l40s", 1, 1, "fp8", "bf16", 0.92, 40960, None, None, False, None, "不推荐", "FP8 Marlin，KV 不够"),
    ("1×L40S FP8 fp8 KV", "l40s", 1, 1, "fp8", "fp8", 0.92, 40960, 64, 4096, False, None, "可用", "FP8 Marlin"),
    ("2×L40S TP2 BF16", "l40s", 2, 2, "bf16", "bf16", 0.92, 40960, None, None, False, None, "可用", "PCIe all-reduce"),
    ("1×A100 BF16", "a100", 1, 1, "bf16", "bf16", 0.92, 40960, None, None, False, None, "勉强", "BF16 刚够一条 40K"),
    ("1×H100 BF16", "h100", 1, 1, "bf16", "bf16", 0.92, 40960, None, None, False, None, "不推荐", "KV 不够一条 40K"),
    ("1×H100 FP8", "h100", 1, 1, "fp8", "bf16", 0.92, 40960, None, None, False, None, "数据中心首选", "DeepGEMM / CUTLASS"),
    ("2×H100 TP2 BF16", "h100", 2, 2, "bf16", "bf16", 0.92, 40960, None, None, False, None, "参照", "recipe 第 198 行"),
    ("1×H20 FP8", "h20", 1, 1, "fp8", "bf16", 0.92, 40960, None, None, False, None, "H20 首选", "带宽 4.0 TB/s"),
    ("1×H20 BF16", "h20", 1, 1, "bf16", "bf16", 0.92, 40960, None, None, False, None, "H20 可用", "精度不让步时"),
    ("1×PRO 6000 BF16", "rtxpro6000", 1, 1, "bf16", "bf16", 0.92, 40960, None, None, False, None, "可用", "带宽只有 1.8 TB/s"),
    ("1×H200 BF16", "h200", 1, 1, "bf16", "bf16", 0.92, 40960, None, None, False, None, "高配", "单卡 BF16 开满 40K 六条"),
]


def rank_rows():
    out = []
    for name, gpu, n, tp, fmt, kvd, gmu, L, seqs, b, eager, cap, tier, path in CONFIGS:
        p = plan(gpu, n, tp, fmt, kvd, gmu, L, seqs, b, eager, cap)
        _, c1, _ = decode_bound(gpu, tp, fmt, 1, 1024, kvd)
        _, _, c32 = decode_bound(gpu, tp, fmt, 32, 1024, kvd)
        out.append(dict(name=name, tier=tier, path=path, p=p, c1=c1, c32=c32))
    return out


def s_rank(a):
    print("-- 卡配置排行（c1 / c32 是 decode 带宽估算，全部是估算）")
    print(f"   {'配置':24s}{'常驻':>7s}{'KV':>7s}{'KV token':>10s}{'上下文':>8s}{'并发':>7s}{'c1':>6s}{'c32':>7s}  档位 / 路径")
    for r in rank_rows():
        p = r["p"]
        kv = f"{p['kv']:.1f}" if p["kv"] > 0 else "✗"
        print(f"   {r['name']:24s}{p['resident']:7.1f}{kv:>7s}{p['tokens'] / 1e3:9.0f}K{p['max_len'] // 1024:>7d}K{p['conc']:7.2f}"
              + (f"{r['c1']:6.0f}{r['c32']:7.0f}" if p["fits"] else f"{'—':>6s}{'—':>7s}")
              + f"  {r['tier']}：{r['path']}")


def s_qwen35(a):
    print("-- Qwen3.5-27B checkpoint 按模块拆（GiB：BF16 / FP8 / GPTQ-Int4）")
    for name, b, f, g in Q35_PARTS:
        print(f"   {name:18s}{b:7.2f}{f:7.2f}{g:7.2f}")
    for fmt in ("bf16", "fp8", "gptq"):
        print(f"   {fmt:5s}合计 {Q35_CKPT[fmt]:6.2f} GiB = {Q35_CKPT[fmt] * GiB / 1e9:5.1f} GB，recipe vram_minimum_gb {Q35_RECIPE_MIN_GB[fmt]}")
    print("   GPTQ-Int4 只量化了 MLP，GDN 和注意力还是 BF16：28.16 GiB 比 FP8 只小 0.6，recipe 写的 17 GB / 单张 24 GB 装不下")
    print(f"   加 --language-model-only 省掉 visual 0.86，GPTQ 还剩 {Q35_CKPT['gptq'] - 0.86:.2f} GiB，24 GB 卡照样装不下")
    print("-- 混合层的块大小（注意力页要 ≥ GDN 状态页）")
    for tp, kvd, spec, al, note in ((1, "bf16", 0, 16, "v0.19，LMCache 文档写 784"), (1, "bf16", 0, 64, "主干 + flashinfer GDN prefill"),
                                    (2, "fp8", 1, 16, "v0.19，TP2 fp8 KV + MTP k=1"), (2, "fp8", 1, 64, "主干，同上")):
        print(f"   TP{tp} KV {kvd:4s} spec {spec} 对齐 {al:2d}：GDN 页 {q35_mamba_page(tp, spec):,} B → 块 {q35_block_size(tp, kvd, spec, al)} token（{note}）")
    print("   derek 在 v0.19 上跑 Qwen3.6-27B（TP2 fp8 KV、MTP k=1）日志写 1600，和这里同一量级；3.6 的 config 我们没逐项核对")
    print("-- 每条序列的 GDN 状态（48 层，和上下文长度无关）")
    for mode, ckpt in (("none", 0), ("align", 0)):
        print(f"   mamba_cache_mode={mode:5s}：TP1 {q35_state_per_seq(1, 0, mode, ckpt) / MiB:6.0f} MiB，TP2 MTP1 {q35_state_per_seq(2, 1, mode, ckpt) / MiB:6.0f} MiB")
    print("   默认 none（config/cache.py:304），开 --enable-prefix-caching 才走 align，每条多一页")
    print(f"-- KV：16 层全注意力 × 4 KV 头 × 256，bf16 {q35_kv_per_token() // 1024} KiB/token，fp8 {q35_kv_per_token('fp8') // 1024} KiB/token")
    g = GPUS["h100"]
    pool = g["mem"] * 0.92 - Q35_CKPT["fp8"] - other_gib("h100", 1, 8192, 1024)["total"]
    for L in (32768, 262144):
        per = L * q35_kv_per_token() + q35_state_per_seq()
        print(f"   1×H100 FP8：KV 池约 {pool:.1f} GiB，{L} 上下文每条 {per / GiB:5.2f} GiB，并发 {pool * GiB / per:5.1f}（估算）")
    print("   同样 1 GiB KV，Qwen3.5-27B 装的 token 是 Qwen3-32B 的 4 倍；262K 上下文单卡 H100 FP8 能开两三条")


SCEN = {"weights": s_weights, "mem": s_mem, "kv": s_kv, "tp": s_tp, "spec": s_spec, "profile": s_profile,
        "decode": s_decode, "rank": s_rank, "qwen35": s_qwen35}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="all", choices=list(SCEN) + ["all"])
    ap.add_argument("--gpu", default="h100", choices=list(GPUS))
    ap.add_argument("--n", type=int, default=1, help="卡数")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--fmt", default="fp8", choices=list(CKPT_BYTES))
    ap.add_argument("--kv-dtype", default="bf16", choices=list(KV_ELEM))
    ap.add_argument("--gmu", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=MAX_POS)
    ap.add_argument("--max-num-seqs", type=int, default=None, help="不给就用源码按卡型的默认值")
    ap.add_argument("--batched", type=int, default=None, help="--max-num-batched-tokens，不给就用默认值")
    ap.add_argument("--eager", action="store_true", help="--enforce-eager")
    ap.add_argument("--capture", type=int, default=None, help="max_cudagraph_capture_size")
    a = ap.parse_args(argv)
    for name, fn in SCEN.items():
        if a.scenario in (name, "all"):
            if a.scenario == "all":
                print(f"== {name}")
            fn(a)


if __name__ == "__main__":
    main()
