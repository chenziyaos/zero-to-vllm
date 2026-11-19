#!/usr/bin/env python3
"""VLLM-25：Qwen3.8-27B 部署规划器，只用 Python 标准库，不联网、不需要 GPU。

模型：Qwen/Qwen3.8-27B，和 2.4T 旗舰同一个 Gated DeltaNet（GDN）混合注意力骨干，
64 层 = 48 层 GDN + 16 层全注意力（full_attention_interval=4），原生 262144 上下文，
可 YaRN 到 1M，多模态（Qwen3_5ForConditionalGeneration，带视觉塔），checkpoint 自带
MTP 草稿头（text_config.mtp_num_hidden_layers=1，顶层 config 没有这个字段）。

骨干常量和 Qwen3.5-27B 逐项一致（config.json 核对），3.8 的量化 checkpoint 更多，官方在
5090 / GB300 / B200 / Ascend 950 上做过端到端验证。这个规划器只算字节和区间，实测数字
标明出处，估算数字标明「估算」。

场景：
  ckpt     六种 checkpoint 的字节，和 recipe 的 vram_minimum_gb 对账
  state    每条序列的 GDN 状态字节（conv 状态 + fp32 时间状态），和上下文长度无关
  block    混合层块大小：注意力页必须不小于 GDN 页，TP1 / TP2 各是多少
  kv       全注意力层每 token 的 KV 字节，262K / 1M 上下文一条要多少显存
  mem      给定卡型 / 格式，算权重、其他占用、KV 池、能开多少并发
  fit      recipe 实测档位复算：1×5090 NVFP4、2×5090 TP2，估算和实测摆在一起
  rank     从 1×4090 到 H200 / B200 的配置排在一起，标出哪些根本起不来
  mtp      自带 MTP（recipe 示例 k=3）多占多少状态；DFlash2 / DSpark 是外挂草稿
  decode   单流 decode 带宽估算，和 Ascend 950 的 64 tok/s 实测对照（估算）
  all      全部

用法：
  python3 deploy_planner.py --scenario all
  python3 deploy_planner.py --scenario mem --gpu h100 --fmt fp8
  python3 deploy_planner.py --scenario rank
  python3 deploy_planner.py --scenario fit

源码行号对应 vllm 主干 0d6f92a；recipe 指 recipes/models/Qwen/Qwen3.8-27B.yaml。
"""
import argparse
import math

GiB = 1024 ** 3
MiB = 1024 ** 2

# ------------------------------------------------------------------ 模型常量
# Qwen/Qwen3.8-27B config.json（text_config）
LAYERS = 64
FULL_ATTN = 16                 # full_attention_interval=4：每 4 层一层全注意力
GDN_LAYERS = 48
Q_HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
HIDDEN, INTER = 5120, 17408
VOCAB = 248320
MAX_POS = 262144
LONG_POS = 1_000_000
# GDN 层：linear_num_key_heads 16、linear_num_value_heads 48，头维 128，conv 核 4
GDN_K_HEADS, GDN_V_HEADS, GDN_HEAD_DIM, CONV_K = 16, 48, 128, 4
# conv 状态通道数（mamba_utils.py:282 gated_delta_net_state_shape）：
#   128*16*2 + 128*48 = 10240，是 hidden 5120 的两倍
GDN_CONV_DIM = GDN_HEAD_DIM * GDN_K_HEADS * 2 + GDN_HEAD_DIM * GDN_V_HEADS
SSM_DTYPE_BYTES = 4           # mamba_ssm_dtype: float32
VISUAL_GIB = 0.86             # --language-model-only 省下来的视觉塔
PARAMS = 27e9

# ------------------------------------------------------------------ checkpoint
# 字节数是 HF 树上全部 .safetensors 相加（2026-11-03 核对）；w8a8 用 recipe 注释数字
CKPT_BYTES = {
    "bf16":  55_563_006_776,   # Qwen/Qwen3.8-27B
    "fp8":   30_866_866_928,   # Qwen/Qwen3.8-27B-FP8，block 128×128
    "nvfp4": 26_381_249_312,   # Inferact/Qwen3.8-27B-NVFP4，uniform W4A4
    "nvnv":  21_921_697_280,   # nvidia/Qwen3.8-27B-NVFP4，NVFP4-FP8 混合
    "int4":  19_452_788_048,   # RedHatAI/Qwen3.8-27B-INT4，W4A16
    "mxfp4": 19_832_841_464,   # RedHatAI/Qwen3.8-27B-MXFP4，OCP MXFP4 W4A4
}
CKPT_GIB = {k: v / GiB for k, v in CKPT_BYTES.items()}
W8A8_BYTES = 32_128_768_176    # Eco-Tech/Qwen3.8-27B-w8a8（ModelScope，recipe 注释）
# recipe vram_minimum_gb（Qwen3.8-27B.yaml variants 段）
RECIPE_MIN_GB = {"bf16": 67, "fp8": 38, "nvfp4": 32, "nvnv": 32, "int4": 24, "mxfp4": 24}
CKPT_LABEL = {
    "bf16": "BF16 官方原版", "fp8": "FP8 官方 block 缩放",
    "nvfp4": "NVFP4 Inferact W4A4", "nvnv": "NVFP4 nvidia 混合精度",
    "int4": "INT4 RedHatAI W4A16", "mxfp4": "MXFP4 RedHatAI W4A4",
}
# recipe supported_hardware + guide 里写明的 kernel 支持范围
FMT_SUPPORT = {
    "bf16":  "1 GPU ≥80GB 或 TP2；GB300 TP4 verified，1M 长上下文也验过",
    "fp8":   "Hopper / Blackwell / MI300 / 5090；950 经 vllm-ascend#14852 加载时重量化到 MXFP8",
    "nvfp4": "仅 Blackwell（SM100+，真 W4A4，FlashInfer CUTLASS FP4 GEMM）；1×5090 要 --enforce-eager",
    "nvnv":  "仅 Blackwell；RTX PRO 6000 / DGX Spark verified",
    "int4":  "Hopper（Marlin W4A16）和 Blackwell；recipe 验证名单含 5090",
    "mxfp4": "Blackwell 真 W4A4（CompressedTensorsW4A4Mxfp4）；Hopper 退化为 Marlin W4A16",
}
LEGACY_GPTQ_GIB = 28.16        # Qwen3.5-27B-GPTQ-Int4 只量化 MLP 的旧坑，留作对照
# plan 是原始字节估算，未复刻混合 KV 页池的 padding、分组与共享块分配。
# checkpoint 文件大小不等于加载后权重显存，fit 仅保留上游记录，不拟合。

KV_ELEM = {"bf16": 2, "fp8": 1}

# ------------------------------------------------------------------ 卡
# mem 是 torch 看到的总显存（GiB）；5090 按 recipe 的 31.4 GiB 可用口径
# bw 是标称显存带宽 TB/s；link 决定 all-reduce 延迟
GPUS = {
    "rtx3090":    dict(arch="sm86", mem=23.57, bw=0.936, link="pcie4", label="RTX 3090 24G"),
    "rtx4090":    dict(arch="sm89", mem=23.52, bw=1.008, link="pcie4", label="RTX 4090 24G"),
    "l40s":       dict(arch="sm89", mem=44.4,  bw=0.864, link="pcie4", label="L40S 48G"),
    "rtx5090":    dict(arch="sm120", mem=31.4, bw=1.792, link="pcie5", label="RTX 5090 32G"),
    "rtxpro6000": dict(arch="sm120", mem=95.0, bw=1.792, link="pcie5", label="RTX PRO 6000 96G"),
    "a100":       dict(arch="sm80", mem=79.15, bw=2.039, link="nvlink", label="A100 80G"),
    "h100":       dict(arch="sm90", mem=79.11, bw=3.35,  link="nvlink", label="H100 80G"),
    "h20":        dict(arch="sm90", mem=95.1,  bw=4.0,   link="nvlink", label="H20 96G"),
    "h200":       dict(arch="sm90", mem=140.4, bw=4.8,   link="nvlink", label="H200 141G"),
    "b200":       dict(arch="sm100", mem=178.35, bw=8.0, link="nvlink", label="B200 180G"),
}
AR_US = {"nvlink": 10, "pcie5": 25, "pcie4": 35}   # 一次小消息 all-reduce（微秒，估算）

# ------------------------------------------------------------------ 其他占用
# 激活和 graph 沿用 VLLM-24 的系数（#22344 反推、#28919 参考，估算口径）
NONTORCH_GIB = 0.4
NCCL_GIB = 0.3
ACT_PER_TOKEN = 0.181 * MiB     # TP1 每 token 激活峰值
GRAPH_GIB = 0.85                # 抓满 51 个尺寸的参考开销
GRAPH_SIZES = 51


def default_batch(gpu):
    return (16384, 1024) if GPUS[gpu]["mem"] >= 160 else (8192, 1024) if GPUS[gpu]["mem"] >= 70 else (2048, 256)


def capture_sizes(seqs, batched, q=1, max_capture=512):
    max_capture = min(seqs * q * 2, max_capture)
    sizes = [i for i in (1, 2, 4) if i <= max_capture]
    if max_capture >= 8:
        sizes += list(range(8, min(max_capture + 1, 256), 8))
    if max_capture >= 256:
        sizes += list(range(256, max_capture + 1, 16))
    if batched <= max_capture and batched not in sizes:
        sizes.append(batched)
    return sorted(set(sizes))


def other_gib(gpu, tp, batched, seqs, eager=False, spec=0):
    act = batched * ACT_PER_TOKEN * (0.8 / tp + 0.2) / GiB
    sampler = seqs * (1 + spec) * VOCAB * 4 * 3 / GiB
    graph = 0.0 if eager else GRAPH_GIB * len(capture_sizes(seqs, batched, 1 + spec)) / GRAPH_SIZES
    nontorch = NONTORCH_GIB + (NCCL_GIB if tp > 1 else 0.0)
    return dict(act=act, sampler=sampler, graph=graph, nontorch=nontorch,
                total=nontorch + max(act, sampler) + graph)


# ------------------------------------------------------------------ GDN 状态
def gdn_state_one_layer(tp=1, spec=0, dtype_bytes=2):
    """一层 GDN 一条序列的状态字节。

    conv 状态：(10240/TP) × (3 + spec) 个模型 dtype 元素。10240 = 128·16·2 + 128·48
    （mamba_utils.py:282）；卷积核 4 留 3 个历史位，MTP 每个草稿位多留一个
    （qwen3_5_mtp.py，recipe 示例 k=3 就是 +3）。
    时间状态（temporal / ssm）：(48/TP) 个 v 头，每个头一个 128×128 的 fp32 矩阵。
    """
    conv = (GDN_CONV_DIM // tp) * (CONV_K - 1 + spec) * dtype_bytes
    temporal = (GDN_V_HEADS // tp) * GDN_HEAD_DIM * GDN_HEAD_DIM * SSM_DTYPE_BYTES
    return conv + temporal


def gdn_state_per_seq(tp=1, spec=0, mode="none", ckpt=0, dtype_bytes=2):
    """整条序列 48 层 GDN 的状态。

    kv_cache_interface.py 1135–1141：prefix caching 开了走 align 模式，每条序列占
    2 + spec + ckpt 页；不开是 1 + spec 页。ckpt 是 GDN 共享前缀检查点，默认关。
    """
    pages = (2 + spec + ckpt) if mode == "align" else (1 + spec)
    return GDN_LAYERS * gdn_state_one_layer(tp, spec, dtype_bytes) * pages


def kv_per_token(kv_dtype="bf16", tp=1):
    """全注意力层每 token 每卡 KV 字节：2 × 16 层 × (4/TP) KV 头 × 256 头维。"""
    heads = max(KV_HEADS // tp, 1)
    return 2 * FULL_ATTN * heads * HEAD_DIM * KV_ELEM[kv_dtype]


def block_size(tp=1, kv_dtype="bf16", spec=0, align=16, dtype_bytes=2):
    """混合模型的注意力块大小：一块注意力 KV 必须装下一整页 GDN 状态。

    platforms/interface.py 的 _align_hybrid_block_size 把块大小向上抬到
    「一块注意力 KV 字节 ≥ 一页 GDN 状态字节」，再按对齐粒度取整。默认对齐 16；
    FlashInfer GDN prefill 后端下按 64 对齐（gdn_attn.py:34
    FLASHINFER_GDN_CHECKPOINT_ALIGNMENT = 64）。
    """
    attn_one_token = 2 * max(KV_HEADS // tp, 1) * HEAD_DIM * KV_ELEM[kv_dtype]
    return align * math.ceil(gdn_state_one_layer(tp, spec, dtype_bytes) / (align * attn_one_token))


# ------------------------------------------------------------------ 显存账
def plan(gpu, n=1, tp=1, fmt="fp8", kv_dtype="bf16", gmu=0.92, max_len=MAX_POS,
         seqs=None, batched=None, eager=False, spec=0, text_only=False,
         prefix=False, dtype_bytes=2):
    g = GPUS[gpu]
    if tp not in (1, 2, 4) or n < tp or n % tp:
        raise ValueError("本估算器仅支持 TP1/2/4，GPU 数必须为 TP 的整数倍")
    if not 0 < gmu <= 1 or max_len <= 0 or spec < 0:
        raise ValueError("gmu 必须在 (0,1]，上下文为正数，草稿数非负")
    if seqs is not None and seqs <= 0 or batched is not None and batched <= 0:
        raise ValueError("批次参数必须为正数")
    db, ds = default_batch(gpu)
    seqs = seqs if seqs is not None else ds
    batched = batched or db
    resident = CKPT_GIB[fmt] / tp - (VISUAL_GIB / tp if text_only else 0.0)
    mode = "align" if prefix else "none"
    state_each = gdn_state_per_seq(tp, spec, mode, dtype_bytes=dtype_bytes) / GiB
    o = other_gib(gpu, tp, batched, seqs, eager, spec)
    usable = g["mem"] * gmu
    pool = usable - resident - o["total"]
    per_token = kv_per_token(kv_dtype, tp)

    def per_req(ctx):
        return ctx * per_token + state_each * GiB

    tokens_full = pool * GiB / per_token if pool > 0 else 0
    fits262 = pool * GiB >= per_req(max_len)
    conc262 = pool * GiB / per_req(max_len) if pool > 0 else 0
    return dict(gpu=gpu, n=n, tp=tp, fmt=fmt, kv_dtype=kv_dtype, usable=usable,
                resident=resident, other=o["total"], pool=pool, state_each=state_each,
                per_token=per_token, tokens=max(tokens_full, 0), fits=fits262,
                conc=conc262, max_len=max_len, seqs=seqs, batched=batched,
                mode=mode)


def decode_bound(gpu, tp=1, fmt="fp8", bs=1, ctx=32768, kv_dtype="bf16",
                 eta=0.75, fixed_ms=0.5, spec=0):
    """单流 decode 估算：读权重 + 全注意力 KV + GDN 状态，除以 η × 带宽（估算）。

    GDN 状态只有约 147 MiB / 序列，长上下文下和 KV 相比很小；权重每步都读。
    开了 MTP 之后一步出多个 token，这个估算只覆盖无投机的基线。
    TP 的 all-reduce 按 64 层每层两次小消息估（一次 reduce、一次 post-weight update）。
    """
    g = GPUS[gpu]
    w = CKPT_GIB[fmt] * GiB / tp
    kvb = bs * ctx * kv_per_token(kv_dtype, tp)
    state = bs * gdn_state_per_seq(tp, spec)
    ms = (w + kvb + state) / (eta * g["bw"] * 1e12) * 1e3 + fixed_ms
    if tp > 1:
        ms += 2 * LAYERS * AR_US[g["link"]] / 1e3
    return ms, 1e3 / ms, bs * 1e3 / ms


# ------------------------------------------------------------------ 场景输出
def s_ckpt(a):
    print("-- 六种 checkpoint 的磁盘字节（HF 树上 safetensors 相加，2026-11-03 核对）")
    print(f"   {'格式':24s}{'GB':>7s}{'GiB':>8s}{'bit/参数':>10s}{'recipe 最小显存 GB':>18s}")
    for fmt in ("bf16", "fp8", "nvfp4", "nvnv", "int4", "mxfp4"):
        gb = CKPT_BYTES[fmt] / 1e9
        gib = CKPT_GIB[fmt]
        bits = CKPT_BYTES[fmt] * 8 / PARAMS
        print(f"   {CKPT_LABEL[fmt]:24s}{gb:7.2f}{gib:8.2f}{bits:10.2f}{RECIPE_MIN_GB[fmt]:18d}")
    print(f"   Ascend w8a8（Eco-Tech，ModelScope）{W8A8_BYTES / 1e9:.1f} GB，"
          f"recipe vram_minimum_gb 39，视觉塔留 BF16、带 per-channel fp32 scale")
    print("   各仓库量化覆盖范围不同；文件平均 bit/参数不能代替逐层 dtype 清单")
    print(f"   对照 Qwen3.5-27B GPTQ-Int4：{LEGACY_GPTQ_GIB:.2f} GiB，"
          f"小于当前 FP8 {CKPT_GIB['fp8']:.2f} GiB，大于表内四种 4 bit 档")
    print("   recipe 最小显存是上游声明值，不是启动保证；其注释公式 ceil(磁盘 GB ×1.2) 与部分条目不一致")


def s_state(a):
    print("-- 一层 GDN 一条序列的状态（conv 历史 + fp32 时间状态矩阵）")
    one = gdn_state_one_layer()
    print(f"   TP1：conv {GDN_CONV_DIM * (CONV_K - 1) * 2 / 1024:6.1f} KiB + 时间状态 "
          f"{GDN_V_HEADS * GDN_HEAD_DIM ** 2 * 4 / 1024:8.1f} KiB = {one / 1024:7.1f} KiB/层")
    print("-- 整条序列 48 层的 GDN 状态，和上下文长度无关")
    print(f"   {'模式':6s}{'TP1 k=0':>12s}{'TP1 k=3':>12s}{'TP2 k=3':>12s}")
    for mode in ("none", "align"):
        v10 = gdn_state_per_seq(1, 0, mode) / MiB
        v13 = gdn_state_per_seq(1, 3, mode) / MiB
        v23 = gdn_state_per_seq(2, 3, mode) / MiB
        print(f"   {mode:6s}{v10:12.0f}{v13:12.0f}{v23:12.0f}  MiB/序列")
    print("   none 占 1+k 页，align（--enable-prefix-caching）占 2+k 页，每页一整层状态")
    print("   262K 长上下文也只多这么点，这是 GDN 相对全注意力模型省显存的地方")


def s_block(a):
    print("-- 混合层块大小：一块注意力 KV 必须装下一整页 GDN 状态（2×5090 recipe 示例 k=3）")
    for tp, kvd, spec, al, note in (
            (1, "bf16", 0, 16, "无 MTP，默认对齐 16"),
            (1, "bf16", 3, 16, "默认 MTP k=3"),
            (1, "bf16", 3, 64, "FlashInfer GDN prefill，gdn_attn.py:34"),
            (2, "fp8", 3, 16, "2×5090 TP2，fp8 KV"),
            (2, "fp8", 3, 64, "2×5090 TP2，64 对齐"),
            (1, "fp8", 3, 16, "1×5090 单卡，fp8 KV")):
        print(f"   TP{tp} KV {kvd:4s} k={spec} 对齐 {al:2d}：块 {block_size(tp, kvd, spec, al):5d} token（{note}）")
    print("   块这么大是因为 fp32 时间状态一页就 3.0 MiB（TP1），普通稠密模型的块通常只有 16 token")


def s_kv(a):
    print("-- 全注意力 KV：只有 16 层，每层 4 KV 头 × 256 头维")
    bf16, fp8 = kv_per_token("bf16"), kv_per_token("fp8")
    print(f"   bf16 {bf16 // 1024} KiB/token，fp8 {fp8 // 1024} KiB/token")
    for ctx in (32768, MAX_POS, LONG_POS):
        print(f"   一条 {ctx:>7d} 上下文：bf16 {ctx * bf16 / GiB:6.2f} GiB，"
              f"fp8 {ctx * fp8 / GiB:6.2f} GiB")
    print(f"   再加 GDN 固定状态 {gdn_state_per_seq() / MiB:.0f} MiB，和长度无关")
    print("   对照 Qwen3-32B（64 层全注意力、8 KV 头、头维 128）：每 token 256 KiB，4 倍")
    print("   1M 走 YaRN：factor 4.0，override 要嵌在 text_config 下面，"
          "524K 的活把 factor 减半到 2.0")


def s_mem(a):
    p = plan(a.gpu, a.n, a.tp, a.fmt, a.kv_dtype, a.gmu, max_len=a.max_model_len,
             seqs=a.max_num_seqs, batched=a.max_num_batched, text_only=a.text_only,
             prefix=a.prefix, spec=a.spec, eager=a.eager)
    g = GPUS[p["gpu"]]
    print(f"-- {g['label']} ×{p['n']} TP{p['tp']}，{p['fmt']} 权重，KV {p['kv_dtype']}，"
          f"gmu {p['usable'] / g['mem']:.2f}（估算）")
    print(f"   可用 {p['usable']:6.1f} GiB，权重 {p['resident']:6.1f}，其他 {p['other']:5.1f}，"
          f"KV 池 {p['pool']:6.1f}")
    print(f"   每条序列 GDN 固定状态 {p['state_each'] * 1024:6.0f} MiB（{p['mode']} 模式，k={a.spec}）")
    if p["pool"] <= 0:
        print("   权重加其他已经超了，这档起不来")
        return
    print(f"   KV 池折算 {p['tokens'] / 1e6:6.2f} M token（{p['kv_dtype']}）")
    for ctx in (32768, 131072, 262144):
        per = ctx * p["per_token"] / GiB + p["state_each"]
        print(f"   {ctx // 1024:4d}K 上下文：每条 {per:5.2f} GiB，并发 {p['pool'] / per:5.2f}")


def s_fit(a):
    print("-- 上游 recipe 记录，非本次 GPU 实测；不使用反推折减系数作验证")
    print("   来源：https://github.com/vllm-project/recipes/blob/0511d96d97551a8b0db969b9b179e6f6d198071d/models/Qwen/Qwen3.8-27B.yaml")
    print("   2×5090 TP2，262144 上下文；版本 0.26.1rc1.dev608+g99a10304d")
    print("   格式                  权重/卡 GiB   KV token   接受率")
    for name, weight, tokens, acceptance in (
        ("FP8 官方", 14.28, 377456, .771),
        ("NVFP4 Inferact", 12.02, 445875, .897),
        ("NVFP4 unsloth", 10.64, 920517, .788),
    ):
        print(f"   {name:22s}{weight:8.2f}{tokens:12d}{acceptance:9.3f}")
    print("   1×5090 Inferact NVFP4，32768 上下文，eager；recipe 的逐项记录：")
    for name, tokens in (("eager + fp8 KV",91022), ("加 text-only",135926),
                         ("再限 max-num-seqs=8",152917), ("bf16 KV 对照",76458),
                         ("MTP 对照",90112)):
        print(f"   {name:28s}{tokens:10d}")
    print("   这些行不是严格单变量试验；没有完整环境日志，不推导差值的单一原因。")
    print("   独立显存估算见 mem/rank；运行态权重、状态页池、padding 与分配器差额需启动日志校正。")


def s_rank(a):
    print("-- 各档配置（上下文按档不同，见每行标注；并发 < 1 = 一条都装不下；原始字节估算）")
    print(f"   {'配置':30s}{'权重/卡':>8s}{'KV 池':>7s}{'每条@ctx':>10s}{'并发':>6s}  结论")
    # (名称, gpu, n, tp, fmt, kvd, 额外 plan 参数, kernel 结论)
    rows = [
        ("1×4090 INT4",      "rtx4090", 1, 1, "int4",  "bf16", {}, "权重塞得进，Ada 的 W4A16 不在 recipe 验证名单"),
        ("1×4090 NVFP4",     "rtx4090", 1, 1, "nvfp4", "fp8",  {}, "sm89 没有 Blackwell FP4 W4A4 kernel"),
        ("1×5090 FP8",       "rtx5090", 1, 1, "fp8",   "fp8",  {}, "gmu 0.90 下权重 28.7 GiB 已超可用，无 KV 池"),
        ("1×5090 NVFP4",     "rtx5090", 1, 1, "nvfp4", "fp8",  {"eager": True, "max_len": 32768},
         "上游记录：eager、32K 可起；非唯一可行配置"),
        ("1×L40S INT4",      "l40s",    1, 1, "int4",  "bf16", {}, "显存放得下，sm89 kernel 未验证"),
        ("2×5090 TP2 FP8",   "rtx5090", 2, 2, "fp8",   "fp8",  {"spec": 3}, "262K verified，接受率 0.771"),
        ("2×5090 TP2 NVFP4", "rtx5090", 2, 2, "nvfp4", "fp8",  {"spec": 3}, "262K verified，接受率 0.897"),
        ("1×PRO 6000 BF16",  "rtxpro6000", 1, 1, "bf16", "fp8", {"seqs": 8}, "96G 单卡，recipe verified"),
        ("1×A100 BF16",      "a100",    1, 1, "bf16",  "bf16", {}, "262K 一条，剩一点池"),
        ("1×H100 FP8",       "h100",    1, 1, "fp8",   "fp8",  {}, "主力档"),
        ("1×H100 INT4",      "h100",    1, 1, "int4",  "fp8",  {}, "Marlin W4A16"),
        ("1×H100 MXFP4",     "h100",    1, 1, "mxfp4", "fp8",  {}, "Hopper 退化为 Marlin W4A16"),
        ("1×H20 FP8",        "h20",     1, 1, "fp8",   "fp8",  {}, "带宽大，并发吃 KV 池"),
        ("1×H200 BF16",      "h200",    1, 1, "bf16",  "fp8",  {}, "141G，长上下文主力"),
        ("1×B200 BF16",      "b200",    1, 1, "bf16",  "fp8",  {}, "180G，1M YaRN 也有余量"),
        ("1×B200 MXFP4",     "b200",    1, 1, "mxfp4", "fp8",  {}, "真 W4A4，2026-09-20 verified"),
    ]
    for name, gpu, n, tp, fmt, kvd, extra, note in rows:
        max_len = extra.pop("max_len", MAX_POS)
        # rank 口径固定 gmu 0.90、max-num-seqs 8，和 serve.sh / commands.py 一致
        p = plan(gpu, n, tp, fmt, kvd, gmu=0.90, seqs=extra.pop("seqs", 8),
                 eager=extra.pop("eager", False), spec=extra.pop("spec", 0),
                 max_len=max_len, **extra)
        per = max_len * p["per_token"] / GiB + p["state_each"]
        if note.startswith("sm89") or note.startswith("gmu"):
            verdict = f"起不来：{note}" if p["pool"] <= 0 else note
        elif p["pool"] <= 0:
            verdict = f"起不来：{note}"
        else:
            label = f"{max_len // 1024}K"
            # 容量给原始字节预算的向下取整；serve.sh 的 max-num-seqs 8 是运行档位，
            # 超过 8 条的档位需要显式调大该参数，正文会说明，不在估算器里截断。
            capacity = max(0, math.floor(p["conc"]))
            if capacity:
                tail = f"{label} 原始字节预算 {capacity} 条"
            else:
                n32 = max(0, min(p["seqs"], int(p["pool"] * GiB / (32768 * p["per_token"] + p["state_each"] * GiB))))
                tail = f"{label} 不足一条；32K 原始预算 {n32} 条"
            verdict = f"{tail}；{note}"
        print(f"   {name:30s}{p['resident']:8.1f}{p['pool']:7.1f}{per:10.2f}{p['conc']:6.2f}  {verdict}")
    print("   TP 必须满足 Q/KV/GDN 头数约束；本估算器只覆盖 TP1/2/4。2机×8卡的 TP16 不适用，不能推广为所有多机 TP 不可用。")


def s_mtp(a):
    print("-- 自带 MTP 草稿头：recipe 示例 k=3，PRO 6000 / DGX Spark 用 k=5")
    print(f"   {'k':>3s}{'TP1 状态 MiB':>14s}{'TP2 状态 MiB':>14s}")
    for k in (0, 1, 3, 5):
        v1 = gdn_state_per_seq(1, k) / MiB
        v2 = gdn_state_per_seq(2, k) / MiB
        print(f"   {k:3d}{v1:14.0f}{v2:14.0f}")
    print("   每个草稿位的 conv 增量为 TP1 20 KiB/层；普通 MTP 状态页数也增加，ReplaySSM 不在本模型范围。")
    print("   草稿头和主干共享 embed，顶层 config 没有 mtp_num_hidden_layers；"
          "要开草稿必须显式写 num_speculative_tokens，不写就是普通解码")
    print("   实测短 prompt 接受率：BF16 92.2%，FP8 84.8%，NVFP4 81.5–84.8%；"
          "2×5090 FP8 实测 0.771、Inferact 0.897、unsloth 0.788")
    print("   接受率要从 vllm:spec_decode_num_accepted_tokens_total 和 "
          "vllm:spec_decode_num_draft_tokens_total 两个指标算，吞吐看不出草稿有没有生效")
    print("   recipe 留有 GDN 投机修复 #51812 / #51674 的历史提示；必须按实际镜像 commit 核验，不据此断言当前发行版均不支持。")
    print("   外挂草稿：DFlash2（incoai/Qwen3.8-27B-DFlash2）k=7，需要 vllm#52816（≥0.28.0）；"
          "DSpark（RedHatAI speculator.dspark）k=8")


def s_decode(a):
    print("-- 单流 decode 带宽估算（η=0.75，32K 上下文，无投机基线，估算）")
    for name, kw in (
            ("1×5090 NVFP4",        dict(gpu="rtx5090", fmt="nvfp4", kv_dtype="fp8")),
            ("1×H100 FP8",          dict(gpu="h100", fmt="fp8", kv_dtype="fp8")),
            ("1×H100 BF16",         dict(gpu="h100", fmt="bf16")),
            ("1×H20 FP8",           dict(gpu="h20", fmt="fp8", kv_dtype="fp8")),
            ("1×B200 NVFP4",        dict(gpu="b200", fmt="nvfp4", kv_dtype="fp8")),
            ("2×5090 TP2 NVFP4",    dict(gpu="rtx5090", tp=2, fmt="nvfp4", kv_dtype="fp8")),
    ):
        ms, tps, _ = decode_bound(**kw)
        print(f"   {name:22s}{ms:6.1f} ms/步，{tps:6.1f} tok/s")
    print("   Ascend 950PR 实测单流 64 tok/s（2048 token 输出、单请求、MTP 开）；"
          "上面的无投机估算只给量级，开 MTP 后一步出多个 token")
    print("   147 MiB 的 GDN 状态和 32K 上下文约 1 GiB 的 KV，在权重读取面前都是零头")


SCEN = {"ckpt": s_ckpt, "state": s_state, "block": s_block, "kv": s_kv,
        "mem": s_mem, "fit": s_fit, "rank": s_rank, "mtp": s_mtp, "decode": s_decode}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", choices=list(SCEN) + ["all"])
    ap.add_argument("--gpu", default="h100", choices=list(GPUS))
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--fmt", default="fp8", choices=list(CKPT_GIB))
    ap.add_argument("--kv-dtype", default="bf16", choices=list(KV_ELEM))
    ap.add_argument("--gmu", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=MAX_POS)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--max-num-batched", type=int, default=None)
    ap.add_argument("--spec", type=int, default=0)
    ap.add_argument("--prefix", action="store_true")
    ap.add_argument("--text-only", action="store_true",
                    help="等价于 --language-model-only，不加载视觉塔")
    ap.add_argument("--eager", action="store_true")
    a = ap.parse_args(argv)
    if a.scenario == "all":
        for f in SCEN.values():
            f(a)
    else:
        SCEN[a.scenario](a)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
