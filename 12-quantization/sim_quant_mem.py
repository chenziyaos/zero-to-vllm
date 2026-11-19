"""量化模拟：权重字节账、KV cache 量化账、线性层 kernel 选择回放、TP 切分对齐、FP8 KV 的盈亏点（只依赖标准库）。

对照的源码（vllm 主干 0d6f92a，sglang 主干 8fabf5e）：
  vllm   model_executor/layers/quantization/__init__.py:15–112   QuantizationMethods、废弃名单、register / get_quantization_config
         config/quantization.py:195–241       在线量化简写 fp8_per_tensor / fp8_per_block / fp8_per_channel / mxfp8 / mxfp4 ...
         model_executor/layers/quantization/fp8.py:92–111, 240–356   Fp8Config 不再做在线量化；Fp8LinearMethod 选激活 key
         config/vllm.py:945–949               get_min_capability：AWQ 75、GPTQ 60、FP8 75、ModelOpt FP8 80 / NVFP4 75
         model_executor/kernels/linear/__init__.py:256–430   _get_linear_backend / _LINEAR_BACKEND_KERNEL_MAP / 过滤与回退
         model_executor/kernels/linear/__init__.py:446–640   各量化格式的 CUDA kernel 候选列表（顺序即优先级）
         model_executor/kernels/linear/__init__.py:648–1310  is_supported → can_implement，第一个过关的 kernel 胜出
         model_executor/layers/quantization/utils/humming/priority.py:11–49   SM90 上 Humming 插到 Marlin 前面
         model_executor/kernels/linear/mixed_precision/marlin.py:58–67        "A group straddling TP ranks cannot be fixed by padding"
         model_executor/layers/quantization/utils/fp8_utils.py:1296–1372      FP8 block 量化的 TP 切分对齐
         config/cache.py:44、v1/kv_cache_interface.py:40–138                 KV cache dtype 与 KVQuantMode
         model_executor/layers/quantization/turboquant/config.py:20–215      TurboQuant 预设、槽位字节、首尾层保护
         v1/attention/backends/mla/flashmla_sparse.py:79–132                 fp8_ds_mla 656 B / nvfp4_ds_mla 352 B
  sglang srt/configs/model_config.py:545, 2018   REQUANTIZATION_METHODS、_verify_quantization（checkpoint 与 --quantization 的兼容表）
         srt/arg_groups/fields/exec_.py:276–285   --fp8-gemm-backend / --fp4-gemm-backend
         srt/layers/quantization/fp8_utils.py:704, 1108–1207   block FP8 分发、auto 后端顺序

假设：权重字节只算线性层的数据和 scale，embedding / lm_head / MoE gate 保持 BF16，忽略 norm 和 bias；
GPTQ / AWQ 的 scale 是 FP16，AWQ 的 4-bit zero point 按 4 bit 打包；NVFP4 的每张量 FP32 global scale 忽略不计。
kernel 回放默认环境：CUDA 13 编译的官方 wheel，flashinfer / cutlass-dsl / humming-kernels 按 requirements/cuda.txt 装好，
DeepGEMM 可用（有 nvcc），b12x / conch / fbgemm-gpu / amd-quark 没装。
FP8 KV 的 ITL 系数取自 vLLM 2026-04-22 博客的拟合值，盈亏点按两条直线求交。

用法：
  python3 sim_quant_mem.py --scenario weights   # 每种格式每参数几字节，三种模型在 H100 / H200 上放不放得下、剩多少 KV
  python3 sim_quant_mem.py --scenario kv        # 每种 KV cache dtype 每 token 几字节，同一张卡能多放多少 token
  python3 sim_quant_mem.py --scenario kernel    # 回放 kernel 候选列表：SM75 ~ SM120 上每种量化落到哪个 kernel
  python3 sim_quant_mem.py --scenario tp        # TP 切分后 group / block 对不对得齐：Qwen2.5-72B 为什么把 29568 补成 29696
  python3 sim_quant_mem.py --scenario backend   # --linear-backend 过滤、无匹配时回退、linear_backend_per_quant
  python3 sim_quant_mem.py --scenario fp8kv     # FP8 KV 的 ITL 直线模型和盈亏点
  python3 sim_quant_mem.py --scenario sglang    # 同一份 checkpoint 在 vLLM 和 SGLang 上怎么起、默认值差在哪
"""
import argparse
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, MODELS  # noqa: E402

GiB = 1 << 30
OVERHEAD_GIB = 6.0


# --------------------------------------------------------------------------- 1. 权重格式：每参数字节
# (名字, 每参数字节, 说明)。scale 按 "每多少个权重一个 scale、scale 几字节" 摊进去。
FORMATS = {
    'bf16': (2.0, 'BF16 原样'),
    'fp8-tensor': (1.0, 'FP8 E4M3，每张量 1 个 FP32 scale'),
    'fp8-block': (1.0 + 4 / (128 * 128), 'FP8 E4M3，128x128 块 1 个 FP32 scale（DeepSeek-V3）'),
    'mxfp8': (1.0 + 1 / 32, 'MXFP8，每 32 个 1 字节 E8M0 scale'),
    'gptq-int4': (0.5 + 2 / 128, 'INT4 g128 对称，FP16 scale'),
    'awq-int4': (0.5 + 2 / 128 + 0.5 / 128, 'INT4 g128 非对称，FP16 scale + 4-bit zero point'),
    'nvfp4': (0.5 + 1 / 16, 'NVFP4 E2M1，每 16 个 1 字节 E4M3 scale + 每张量 FP32'),
    'mxfp4': (0.5 + 1 / 32, 'MXFP4 E2M1，每 32 个 1 字节 E8M0 scale'),
}


@dataclass(frozen=True)
class Arch:
    name: str
    layers: int
    hidden: int
    vocab: int
    attn_rep: int          # TP 下每卡完整复制的注意力参数（MLA 的 q_a_proj、kv_a_proj_with_mqa）
    attn_shard: int        # 按 TP 切的注意力参数（不含 GQA 的 K/V 投影）
    kv_proj: int           # GQA 的 K+V 投影参数
    kv_heads: int
    head_dim: int
    dense_layers: int
    dense_mlp: int
    experts: int
    expert: int
    shared: int
    mla_kv: int = 0

    @property
    def moe_layers(self):
        return self.layers - self.dense_layers if self.experts else 0

    def linear_params(self):
        return (self.layers * (self.attn_rep + self.attn_shard + self.kv_proj) + self.dense_layers * self.dense_mlp
                + self.moe_layers * (self.experts * self.expert + self.shared))

    def bf16_params(self):
        """不量化的部分：embedding + lm_head + MoE gate。"""
        return 2 * self.vocab * self.hidden + self.moe_layers * self.hidden * self.experts

    def total_params(self):
        return self.linear_params() + self.bf16_params()


def _deepseek_v3():
    h, heads = 7168, 128
    q_a, q_b = h * 1536, 1536 * heads * (128 + 64)
    kv_a, kv_b = h * (512 + 64), 512 * heads * (128 + 128)
    o = heads * 128 * h
    e = 3 * h * 2048
    return Arch('DeepSeek-V3', 61, h, 129280, q_a + kv_a, q_b + kv_b + o, 0, 128, 128,
                3, 3 * h * 18432, 256, e, e, mla_kv=512 + 64)


def _qwen3_235b():
    h = 4096
    return Arch('Qwen3-235B-A22B', 94, h, 151936, 0, h * 64 * 128 * 2, h * 4 * 128 * 2, 4, 128,
                0, 0, 128, 3 * h * 1536, 0)


def _dense(key, inter):
    m = MODELS[key]
    return Arch(m.name, m.layers, m.hidden, m.vocab, 0, m.hidden * m.heads * m.head_dim * 2,
                m.hidden * m.kv_heads * m.head_dim * 2, m.kv_heads, m.head_dim,
                m.layers, 3 * m.hidden * inter, 0, 0, 0)


ARCHS = {'llama8b': _dense('llama3.1-8b', 14336), 'llama70b': _dense('llama3.1-70b', 28672),
         'qwen3-235b': _qwen3_235b(), 'dsv3': _deepseek_v3()}


def kv_heads_per_rank(a: Arch, tp):
    return max(1, a.kv_heads // tp)


def weight_bytes_per_gpu(a: Arch, fmt, tp):
    """TP 切分后每卡权重字节。MoE 专家按 tp 摊（开 EP 时按 ep 摊，每卡字节一样）。"""
    b = FORMATS[fmt][0]
    kv_w = a.kv_proj * kv_heads_per_rank(a, tp) / a.kv_heads if a.kv_proj else 0
    w = 2 * a.vocab * a.hidden * 2 / tp
    w += a.layers * (a.attn_rep + a.attn_shard / tp + kv_w) * b
    w += a.dense_layers * a.dense_mlp / tp * b
    w += a.moe_layers * ((a.experts * a.expert + a.shared) / tp * b + a.hidden * a.experts * 2)
    return w


def kv_bytes_per_token_rank(a: Arch, tp, per_head_layer_bytes=None):
    """每卡每 token 的 KV 字节。per_head_layer_bytes：一个 KV 头一层 K+V 的字节，None 表示 BF16。"""
    if a.mla_kv:
        return a.layers * a.mla_kv * 2
    per = per_head_layer_bytes if per_head_layer_bytes is not None else 2 * a.head_dim * 2
    return a.layers * kv_heads_per_rank(a, tp) * per


def fit(a: Arch, fmt, gpu_key, tp, util=0.9):
    gpu = GPUS[gpu_key]
    requested = gpu.mem_gb * 0.99 * GiB * util
    w = weight_bytes_per_gpu(a, fmt, tp)
    free = requested - w - OVERHEAD_GIB * GiB
    kvb = kv_bytes_per_token_rank(a, tp)
    return dict(weights=w, free=free, kv_per_tok=kvb, tokens=max(0.0, free / kvb))


# --------------------------------------------------------------------------- 2. KV cache dtype 的字节账
TQ_PRESETS = {   # turboquant/config.py:20：key bits、value bits、norm correction
    'turboquant_k8v4': (8, 4, False),
    'turboquant_4bit_nc': (4, 4, True),
    'turboquant_k3v4_nc': (3, 4, True),
    'turboquant_3bit_nc': (3, 3, True),
}
TQ_DOC = {'turboquant_k8v4': '2.6x, +1.17% PPL', 'turboquant_4bit_nc': '3.8x, +2.71%',
          'turboquant_k3v4_nc': '~3.5x, +10.63%', 'turboquant_3bit_nc': '4.9x, +20.59%'}


def tq_slot_bytes(head_dim, kbits, vbits):
    """TurboQuantConfig.key_packed_size + value_packed_size，再对齐到偶数。"""
    key = head_dim if kbits == 8 else math.ceil(head_dim * kbits / 8) + 2      # MSE key：码字 + FP16 范数
    val = math.ceil(head_dim * vbits / 8) + 4                                   # 均匀量化：码字 + FP16 scale/zero
    s = key + val
    return s + (s & 1)


def kv_head_bytes(dtype, head_dim=128):
    """一个 KV 头一层（K+V）的字节。"""
    if dtype in ('auto', 'bfloat16', 'float16'):
        return 2 * head_dim * 2
    if dtype in ('fp8', 'fp8_e4m3', 'fp8_e5m2'):
        return 2 * head_dim
    if dtype in ('int8_per_token_head', 'fp8_per_token_head'):
        return 2 * (head_dim + 4)            # 每头每 token 一个 FP32 scale
    if dtype == 'int4_per_token_head':
        return 2 * (head_dim // 2 + 4)
    if dtype == 'nvfp4':
        return 2 * (head_dim // 2 + head_dim // 16)
    if dtype in TQ_PRESETS:
        k, v, _ = TQ_PRESETS[dtype]
        return tq_slot_bytes(head_dim, k, v)
    raise ValueError(dtype)


def boundary_skip_layers(num_layers, n=2, hybrid=False):
    """TurboQuantConfig.get_boundary_skip_layers：稠密模型首尾各 n 层不压缩，混合模型不跳。"""
    if hybrid or n <= 0 or num_layers <= 0:
        return []
    n = min(n, num_layers // 2)
    return sorted(set(range(n)) | set(range(num_layers - n, num_layers)))


def kv_bytes_per_token_dtype(a: Arch, dtype, tp=1):
    hb = kv_head_bytes(dtype, a.head_dim)
    kvh = kv_heads_per_rank(a, tp)
    skip = boundary_skip_layers(a.layers) if dtype in TQ_PRESETS else []
    bf = kv_head_bytes('auto', a.head_dim)
    return kvh * (len(skip) * bf + (a.layers - len(skip)) * hb)


MLA_KV = {'auto': 576 * 2, 'fp8': 576, 'fp8_ds_mla': 656, 'nvfp4_ds_mla': 352}


# --------------------------------------------------------------------------- 3. kernel 选择回放
@dataclass
class Env:
    cc: int
    cuda: int = 130                 # 编译 / 运行时 CUDA 版本 x10
    flashinfer: bool = True
    cutedsl: bool = True
    humming: bool = True
    deep_gemm: bool = True          # VLLM_USE_DEEP_GEMM=1 且 has_deep_gemm
    nvcc: bool = True
    b12x: bool = False
    conch: bool = False
    fbgemm: bool = False
    quark: bool = False
    batch_invariant: bool = False
    nvfp4_sm100_built: bool = True
    nvfp4_sm120_built: bool = True
    disabled: tuple = ()
    model_type: str = 'llama'

    def family(self, f):
        return self.cc // 10 == f // 10


@dataclass
class Layer:
    """一个线性层要什么：权重格式、激活格式、group、形状、dtype。"""
    quant: str                      # fp8 / fp8_block / mp / nvfp4 / mxfp4
    w_scale: str = 'tensor'         # fp8：tensor / channel
    act: str = 'dyn_token'          # fp8：static_tensor / dyn_token / dyn_tensor；mxfp4：none / mxfp4；nvfp4：a4 / a16
    weight_type: str = 'uint4b8'    # mp：uint4 / uint4b8 / uint8b128
    zero_points: bool = False
    group: int = 128
    act_dtype: str = 'bf16'
    k: int = 8192                   # 每卡 in_features
    n: int = 8192                   # 每卡 out_features
    full_k: int = 8192


def _ok(cond, reason):
    return (True, None) if cond else (False, reason)


def _cutlass_fp8(e):
    if e.cc >= 90:
        return e.cuda >= 120
    if e.cc >= 89:
        return e.cuda >= 124
    return False


def _cutlass_block_fp8(e):
    if e.cc >= 100:
        return e.cuda >= 128
    if e.cc >= 90:
        return e.cuda >= 120
    return False


def _cutlass_fp4(e):
    if e.cuda < 128:
        return False
    return (e.nvfp4_sm100_built and 100 <= e.cc < 120) or (e.nvfp4_sm120_built and 120 <= e.cc < 130)


def _deep_gemm_supported(e):
    arch = e.cc == 90 or e.family(100) or e.family(120)
    return arch and e.deep_gemm and e.nvcc


DEEPGEMM_EXCLUDED = {'qwen3_5_text': (100, 120), 'qwen3_5_moe_text': (100, 120), 'kolibri1': (120,)}

# 每个 kernel：(名字, is_supported(env), can_implement(env, layer))
K = {}


def _k(name, sup, imp=lambda e, l: (True, None)):
    K[name] = (sup, imp)


# FP8（per-tensor / per-channel 权重）
_k('FlashInferFP8ScaledMM', lambda e: _ok(e.flashinfer and e.cc >= 100, 'FlashInfer + >=sm_100'),
   lambda e, l: _ok(l.w_scale == 'tensor' and l.act in ('static_tensor', 'dyn_tensor'), 'needs per-tensor scales'))
_k('CutlassFP8ScaledMM', lambda e: _ok(_cutlass_fp8(e), 'CUTLASS FP8 needs sm89+CUDA12.4 / sm90+CUDA12.0'))
_k('B12xTensorFP8ScaledMM', lambda e: _ok(e.family(120) and e.b12x, 'Blackwell 12x + pip install vllm[b12x]'))
_k('PerTensorTorchFP8ScaledMM', lambda e: _ok(e.cc >= 89, 'torch._scaled_mm FP8 needs sm89'),
   lambda e, l: _ok(l.w_scale == 'tensor' and l.act in ('static_tensor', 'dyn_tensor'), 'needs per-tensor scales'))
_k('ChannelWiseTorchFP8ScaledMM', lambda e: _ok(e.cc >= 89, 'torch._scaled_mm FP8 needs sm89'),
   lambda e, l: _ok(not (l.w_scale == 'tensor' and l.act in ('static_tensor', 'dyn_tensor')), 'rejects per-tensor'))
_k('MarlinFP8ScaledMM', lambda e: _ok(e.cc >= 75 and not e.batch_invariant, 'FP8 Marlin needs sm75'))
_k('HummingFP8ScaledMM', lambda e: _ok(e.humming and e.cc >= 75, 'Humming not installed / < sm75'))


# FP8 block 128（激活 1x128 动态，权重 128x128）
def _block_base(l):
    return _ok(l.act != 'static_tensor', 'block FP8 needs dynamic activation')


def _fi_bs_sup(e):
    return e.flashinfer and e.cc == 90


def _dg_imp(e, l):
    if l.act_dtype != 'bf16':
        return False, 'DeepGEMM out bf16 only'
    if e.model_type in DEEPGEMM_EXCLUDED and any(e.family(f) for f in DEEPGEMM_EXCLUDED[e.model_type]):
        return False, f'DeepGEMM auto-disabled for {e.model_type}'
    return _ok(l.n % 64 == 0 and l.k % 128 == 0, 'N%64 / K%128')


_k('FlashInferFp8DeepGEMMDynamicBlockScaled',
   lambda e: _ok(_fi_bs_sup(e) and _deep_gemm_supported(e), 'needs FlashInfer sm90 blockscale AND DeepGEMM'),
   lambda e, l: _dg_imp(e, l) if _block_base(l)[0] else _block_base(l))
_k('DeepGemmFp8BlockScaledMM', lambda e: _ok(_deep_gemm_supported(e), 'DeepGEMM: sm90/sm10x/sm12x + nvcc'),
   lambda e, l: _dg_imp(e, l) if _block_base(l)[0] else _block_base(l))
_k('CutlassFp8BlockScaledMM', lambda e: _ok(_cutlass_block_fp8(e), 'CUTLASS block FP8 needs sm90+'),
   lambda e, l: _block_base(l))
_k('B12xFp8BlockScaledMM', lambda e: _ok(e.family(120) and e.b12x, 'Blackwell 12x + b12x'), lambda e, l: _block_base(l))
_k('TritonFp8BlockScaledMM', lambda e: (True, None), lambda e, l: _block_base(l))
_k('BlockWiseTorchFP8ScaledMM', lambda e: _ok(e.cc == 90, 'sm90 only'), lambda e, l: _block_base(l))

# WnA16（GPTQ / AWQ）：第三个元素是 get_min_capability
MP_MIN_CC = {'CutlassW4A8': 90, 'Machete': 90, 'Marlin': 75, 'Conch': 80, 'Exllama': 60, 'TritonW4A16': 0, 'Humming': 75}


def _machete_imp(e, l):
    if e.cc != 90:
        return False, 'Machete requires compute capability of 90 (Hopper)'
    types = ['uint4', 'uint8'] if l.zero_points else ['uint4b8', 'uint8b128']
    if l.weight_type not in types:
        return False, f'quant type {l.weight_type}'
    if l.group not in ([-1, 64, 128] if l.act_dtype in ('fp16', 'bf16') else [-1, 128]):
        return False, f'group {l.group}'
    if l.k % 64:
        return False, 'Input features size must be divisible by 64'
    return _ok(l.n % 128 == 0, 'Output features size must be divisible by 128')


def _marlin_imp(e, l):
    types = ['uint4'] if l.zero_points else ['uint4b8', 'uint8b128']
    if l.weight_type not in types:
        return False, f'quant type {l.weight_type}'
    if l.group not in (-1, 32, 64, 128):
        return False, f'group {l.group}'
    if l.group != -1 and l.group < l.full_k and l.k % l.group:
        return False, f'in_features per partition {l.k} is not divisible by group_size = {l.group}.'
    return True, None


def _exllama_imp(e, l):
    if l.act_dtype != 'fp16':
        return False, 'Exllama only supports float16 activations'
    if l.weight_type not in ('uint4b8', 'uint8b128'):
        return False, f'quant type {l.weight_type}'
    if l.group <= 0:
        return False, 'channelwise not supported'
    return _ok(l.full_k % l.group == 0, 'group does not divide full K')


def _triton_imp(e, l):
    if l.weight_type not in ('uint4b8', 'uint4'):
        return False, f'quant type {l.weight_type}'
    if l.n % 8:
        return False, 'N % 8'
    if l.group not in (-1, 32, 64, 128, 256) and l.group != l.full_k:
        return False, f'group {l.group}'
    g = l.group if l.group != -1 else l.k
    return _ok(l.k % g == 0, f'Input features {l.k} not divisible by group size {g}')


_k('CutlassW4A8', lambda e: (True, None),
   lambda e, l: _ok(e.cc == 90 and l.act_dtype == 'fp8' and not l.zero_points, 'needs sm90 + FP8 activation'))
_k('Machete', lambda e: (True, None), _machete_imp)
_k('Marlin', lambda e: (True, None), _marlin_imp)
_k('Conch', lambda e: (True, None), lambda e, l: _ok(e.conch, 'conch-triton-kernels is not installed'))
_k('Exllama', lambda e: (True, None), _exllama_imp)
_k('TritonW4A16', lambda e: (True, None), _triton_imp)
_k('Humming', lambda e: (True, None), lambda e, l: _ok(e.humming, 'Humming is not installed'))

# NVFP4
_k('FlashInferCuteDslNvFp4', lambda e: _ok(e.family(100) and e.flashinfer and e.cutedsl, 'sm10x + FlashInfer CuTe-DSL'))
_k('FlashInferCutlassNvFp4', lambda e: _ok(_cutlass_fp4(e) and e.cc >= 100 and e.flashinfer, 'cutlass fp4 + sm100+'))
_k('FlashInferB12xNvFp4', lambda e: _ok(e.cc >= 120 and False, 'FlashInfer b12x gemm not in this FlashInfer build'))
_k('CutlassNvFp4', lambda e: _ok(_cutlass_fp4(e), 'cutlass_scaled_mm_supports_fp4 False'))
_k('FlashInferCuteDslNvFp4W4A16',
   lambda e: _ok((e.cc in (100, 103) or 120 <= e.cc < 130) and e.flashinfer and e.cutedsl, 'sm100/103/12x'))
_k('MarlinNvFp4', lambda e: _ok(e.cc >= 75, 'Marlin FP4 not available'))
_k('FlashInferTrtllmNvFp4', lambda e: _ok(e.flashinfer and e.cc >= 100, 'FlashInfer TRT-LLM FP4 (Blackwell)'))
_k('FlashInferCudnnNvFp4', lambda e: _ok(e.flashinfer and e.cc >= 100, 'FlashInfer cuDNN FP4 (Blackwell)'))
_k('FbgemmNvFp4', lambda e: _ok(e.fbgemm, 'fbgemm-gpu not installed'))
_k('B12xNvFp4', lambda e: _ok(e.family(120) and e.b12x, 'Blackwell 12x + b12x'))
_k('B12xNvFp4W4A16', lambda e: _ok(e.family(120) and e.b12x, 'Blackwell 12x + b12x'))
_k('TorchNvFp4', lambda e: _ok(e.family(100), 'sm10x + torch fp4'))
_k('EmulationNvFp4', lambda e: (True, None))
_k('HummingNvFp4', lambda e: _ok(e.humming and e.cc >= 75, 'Humming not installed / < sm75'))

# MXFP4
_k('FlashInferMxFp4', lambda e: _ok(e.cc >= 100 and e.cutedsl, 'FlashInfer + >=sm_100'),
   lambda e, l: _ok(l.act == 'mxfp4', 'only supports MXFP4 dynamic activation'))
_k('MarlinMxFp4', lambda e: _ok(e.cc >= 75, 'Marlin FP4 not available'))
_k('HummingMxFp4', lambda e: _ok(e.humming and e.cc >= 75, 'Humming not installed / < sm75'))
_k('B12xMxFp4', lambda e: _ok(e.family(120) and e.b12x, 'Blackwell 12x + b12x'),
   lambda e, l: _ok(l.act == 'mxfp4', 'requires dynamic MXFP4 activations'))
_k('EmulationMxfp4', lambda e: (True, None), lambda e, l: _ok(e.quark, 'needs amd-quark'))

LISTS = {   # kernels/linear/__init__.py 的 CUDA 候选列表
    'fp8': ['FlashInferFP8ScaledMM', 'CutlassFP8ScaledMM', 'B12xTensorFP8ScaledMM', 'PerTensorTorchFP8ScaledMM',
            'ChannelWiseTorchFP8ScaledMM', 'MarlinFP8ScaledMM', 'HummingFP8ScaledMM'],
    'fp8_block': ['FlashInferFp8DeepGEMMDynamicBlockScaled', 'DeepGemmFp8BlockScaledMM', 'CutlassFp8BlockScaledMM',
                  'B12xFp8BlockScaledMM', 'MarlinFP8ScaledMM', 'HummingFP8ScaledMM', 'TritonFp8BlockScaledMM',
                  'BlockWiseTorchFP8ScaledMM'],
    'mp': ['CutlassW4A8', 'Machete', 'Marlin', 'Conch', 'Exllama', 'TritonW4A16', 'Humming'],
    'nvfp4': ['FlashInferCuteDslNvFp4', 'FlashInferCutlassNvFp4', 'FlashInferB12xNvFp4', 'CutlassNvFp4',
              'FlashInferCuteDslNvFp4W4A16', 'MarlinNvFp4', 'FlashInferTrtllmNvFp4', 'FlashInferCudnnNvFp4',
              'FbgemmNvFp4', 'B12xNvFp4', 'B12xNvFp4W4A16', 'TorchNvFp4', 'EmulationNvFp4', 'HummingNvFp4'],
    'mxfp4': ['FlashInferMxFp4', 'MarlinMxFp4', 'HummingMxFp4', 'B12xMxFp4', 'EmulationMxfp4'],
}
QUANT_KEY = {'fp8': 'fp8_w8a8', 'fp8_block': 'fp8_block_w8a8', 'mp': 'mixed_precision', 'mxfp4': 'mxfp4'}
NVFP4_A16 = ['B12xNvFp4W4A16', 'FlashInferCuteDslNvFp4W4A16', 'MarlinNvFp4', 'HummingNvFp4']

BACKEND_MAP = {   # _LINEAR_BACKEND_KERNEL_MAP 里与 CUDA 相关的部分
    'b12x': {'B12xFp8BlockScaledMM', 'B12xMxFp4', 'B12xNvFp4', 'B12xNvFp4W4A16', 'B12xTensorFP8ScaledMM'},
    'cutlass': {'CutlassFP8ScaledMM', 'CutlassFp8BlockScaledMM', 'CutlassW4A8', 'CutlassNvFp4'},
    'flashinfer_cutlass': {'FlashInferFP8ScaledMM', 'FlashInferFp8DeepGEMMDynamicBlockScaled',
                           'FlashInferCutlassNvFp4', 'FlashInferMxFp4'},
    'flashinfer_cutedsl': {'FlashInferCuteDslNvFp4', 'FlashInferCuteDslNvFp4W4A16'},
    'flashinfer_trtllm': {'FlashInferTrtllmNvFp4'},
    'flashinfer_cudnn': {'FlashInferCudnnNvFp4'},
    'humming': {'HummingFP8ScaledMM', 'Humming', 'HummingMxFp4', 'HummingNvFp4'},
    'marlin': {'MarlinFP8ScaledMM', 'Marlin', 'MarlinNvFp4', 'MarlinMxFp4'},
    'triton': {'TritonFp8BlockScaledMM', 'TritonW4A16'},
    'deep_gemm': {'DeepGemmFp8BlockScaledMM'},
    'torch': {'PerTensorTorchFP8ScaledMM', 'ChannelWiseTorchFP8ScaledMM', 'BlockWiseTorchFP8ScaledMM', 'TorchNvFp4'},
    'machete': {'Machete'},
    'emulation': {'EmulationNvFp4'},
}


def prioritize_humming(names, cc):
    """priority.py：只在 SM90 上把第一个 humming 挪到第一个 marlin 前面。"""
    if cc != 90:
        return list(names)
    low = [n.lower() for n in names]
    h = next((i for i, n in enumerate(low) if 'humming' in n), None)
    m = next((i for i, n in enumerate(low) if 'marlin' in n), None)
    if h is None or m is None or h < m:
        return list(names)
    out = list(names)
    out.insert(m, out.pop(h))
    return out


def resolve_backend(names, cc, backend='auto', log=None):
    names = prioritize_humming(names, cc)
    if backend == 'auto':
        return names
    keep = [n for n in names if n in BACKEND_MAP.get(backend, set())]
    if not keep:
        if log is not None:
            log.append(f'--linear-backend={backend} has no kernel for this linear layer type; '
                       'using automatic selection for that layer type.')
        return names
    return keep


def _try(name, e, l):
    if name in e.disabled:
        return False, f'{name} disabled by environment variable'
    if l.quant == 'mp' and MP_MIN_CC[name] > e.cc:
        return False, f'{name} min capability {MP_MIN_CC[name]}'
    sup, imp = K[name]
    ok, why = sup(e)
    if not ok:
        return False, f'{name}: {why}'
    ok, why = imp(e, l)
    return (True, None) if ok else (False, f'{name}: {why}')


def backend_for(quant_key, backend='auto', per_quant=None):
    """_get_linear_backend：linear_backend_per_quant 优先。"""
    if per_quant and per_quant.get(quant_key):
        return per_quant[quant_key]
    return backend


def select(e: Env, l: Layer, backend='auto', per_quant=None):
    """返回 (kernel 名, 失败原因列表, 日志列表)。找不到时 kernel 为 None。"""
    log, reasons = [], []
    if l.quant == 'nvfp4':
        a16 = l.act == 'a16'
        qk = 'nvfp4_w4a16' if a16 else 'nvfp4_w4a4'
        be = backend_for(qk, backend, per_quant)
        force = None
        if e.batch_invariant:
            force = 'CutlassNvFp4' if K['CutlassNvFp4'][0](e)[0] else 'EmulationNvFp4'
        elif be == 'auto' and a16:
            if e.cc in (100, 103) and K['FlashInferCuteDslNvFp4W4A16'][0](e)[0]:
                force = 'FlashInferCuteDslNvFp4W4A16'
            elif e.cc != 90:
                force = 'MarlinNvFp4'
        if force:
            if a16 and force not in NVFP4_A16:
                return None, [f'{force} does not support W4A16'], log
            ok, why = K[force][0](e)
            if not ok:
                return None, [f'Forced NVFP4 kernel {force} is not supported: {why}'], log
            return force, reasons, log
        names = [n for n in LISTS['nvfp4'] if not a16 or n in NVFP4_A16]
        names = resolve_backend(names, e.cc, be, log)
        for n in names:
            ok, why = _try(n, e, l)
            if ok:
                if n == 'EmulationNvFp4' and reasons:
                    log.append('NVFP4 linear falling back to the slow and unoptimized emulation backend')
                return n, reasons, log
            reasons.append(why)
        return None, reasons, log
    key = l.quant
    be = backend_for(QUANT_KEY[key], backend, per_quant)
    names = resolve_backend(LISTS[key], e.cc, be, log)
    for n in names:
        ok, why = _try(n, e, l)
        if ok:
            return n, reasons, log
        reasons.append(why)
    return None, reasons, log


ARCH_ENVS = [('T4 / sm75', 75), ('A100 / sm80', 80), ('L40S / sm89', 89), ('H100 / sm90', 90),
             ('B200 / sm100', 100), ('RTX PRO 6000 / sm120', 120)]
CASES = [
    ('FP8 静态 per-tensor（ModelOpt FP8）', Layer('fp8', 'tensor', 'static_tensor')),
    ('FP8 per-channel + 动态 per-token（FP8_DYNAMIC）', Layer('fp8', 'channel', 'dyn_token')),
    ('FP8 block 128（DeepSeek-V3）', Layer('fp8_block', act='dyn_group')),
    ('AWQ INT4 g128 + zp（BF16 激活）', Layer('mp', weight_type='uint4', zero_points=True)),
    ('GPTQ INT4 g128（FP16 激活）', Layer('mp', weight_type='uint4b8', act_dtype='fp16')),
    ('NVFP4 W4A4（ModelOpt NVFP4）', Layer('nvfp4', act='a4')),
    ('NVFP4 W4A16', Layer('nvfp4', act='a16')),
    ('MXFP4 权重 + BF16 激活', Layer('mxfp4', act='none')),
    ('MXFP4 权重 + MXFP4 激活', Layer('mxfp4', act='mxfp4')),
]
QUANT_MIN_CC = {'FP8': 75, 'ModelOpt FP8': 80, 'ModelOpt NVFP4': 75, 'AWQ': 75, 'GPTQ': 60, 'MXFP4': 80,
                'compressed-tensors': 70}


# --------------------------------------------------------------------------- 4. FP8 KV 的 ITL 直线
FP8KV = [   # (模型, GPU, BF16 slope, BF16 intercept, FP8 slope, FP8 intercept)，ms/token 与 ms
    ('Llama-3.1-8B', 'H100', 4.37e-05, 6.44, 2.37e-05, 6.58),
    ('Llama-3.1-8B', 'B200', 1.80e-05, 3.93, 9.72e-06, 3.96),
    ('gpt-oss-20b', 'B200', 3.56e-06, 3.15, 2.06e-06, 3.17),
]
FP8KV_TABLE = [   # 博客 Table 3
    ('Llama-3.1-8B', 'v0.10.2', 'FP8', 24889, 63), ('Llama-3.1-8B', 'v0.19.1', 'FP8', 7010, 54),
    ('gpt-oss-20b', 'v0.10.2', 'FP8', 741565, 96), ('gpt-oss-20b', 'v0.19.1', 'FP8', 22109, 80),
    ('gpt-oss-20b', 'v0.19.1', 'FP8 skip-SW', 7659, 71),
]


def break_even(s_bf, i_bf, s_fp, i_fp):
    return (i_fp - i_bf) / (s_bf - s_fp)


# --------------------------------------------------------------------------- 5. TP 对齐
def tp_alignment(full_k, tp, group=128):
    k = full_k // tp
    exact = full_k % tp == 0
    marlin = exact and not (group < full_k and k % group)
    return k, k % group, marlin


# --------------------------------------------------------------------------- scenarios
def scenario_weights():
    print('== 每参数字节（线性层；embedding / lm_head / gate 保持 BF16）')
    for f, (b, d) in FORMATS.items():
        print(f'  {f:<11} {b:>8.5f} B   {d}')
    print()
    print(f'{"模型":<18} {"总参数 B":>9} ' + ' '.join(f'{f:>10}' for f in FORMATS))
    for key in ('llama8b', 'llama70b', 'qwen3-235b', 'dsv3'):
        a = ARCHS[key]
        row = ' '.join(f'{weight_bytes_per_gpu(a, f, 1) / GiB:>10.1f}' for f in FORMATS)
        print(f'{a.name:<18} {a.total_params() / 1e9:>9.1f} {row}')
    print('（单位 GiB，整模型）')
    print()
    cases = [('llama70b', 'h100-sxm', ['bf16', 'fp8-tensor', 'awq-int4', 'nvfp4'], (1, 2, 4, 8)),
             ('qwen3-235b', 'h200', ['bf16', 'fp8-block', 'awq-int4', 'nvfp4'], (2, 4, 8)),
             ('dsv3', 'h200', ['bf16', 'fp8-block', 'nvfp4', 'mxfp4'], (8,))]
    for key, g, fmts, tps in cases:
        a = ARCHS[key]
        print(f'== {a.name} @ {GPUS[g].name}，util 0.9，每卡杂项 {OVERHEAD_GIB:.0f} GiB，KV BF16')
        print(f'{"格式":<11} {"TP":>3} {"每卡权重 GiB":>12} {"每卡 KV 余量 GiB":>16} {"每卡可放 token":>14}')
        for f in fmts:
            for tp in tps:
                r = fit(a, f, g, tp)
                tok = f'{r["tokens"]:>14,.0f}' if r['free'] > 0 else f'{"放不下":>12}'
                print(f'{f:<11} {tp:>3} {r["weights"] / GiB:>12.1f} {r["free"] / GiB:>16.1f} {tok}')
        print()
    print('读法：同一张卡上，权重省下来的每个 GiB 都变成 KV；KV 余量翻倍 ≈ 能同时跑的长请求翻倍。')


def scenario_kv():
    a = ARCHS['llama8b']
    r = fit(a, 'bf16', 'h100-sxm', 1)
    free = r['free']
    print(f'== {a.name}（32 层，8 个 KV 头，head_dim 128）@ H100 TP1，BF16 权重，KV 余量 {free / GiB:.1f} GiB')
    print(f'{"--kv-cache-dtype":<22} {"每头每层 B":>10} {"每 token KB":>11} {"压缩比":>7} {"可放 token":>12}  说明')
    base = kv_bytes_per_token_dtype(a, 'auto')
    rows = ['auto', 'fp8', 'int8_per_token_head', 'int4_per_token_head', 'nvfp4',
            'turboquant_k8v4', 'turboquant_4bit_nc', 'turboquant_k3v4_nc', 'turboquant_3bit_nc']
    notes = {'auto': '跟随模型 dtype', 'fp8': 'E4M3，scale 默认 1.0，校准后才准',
             'int8_per_token_head': '每头每 token 一个 FP32 scale', 'int4_per_token_head': '同上，4 bit',
             'nvfp4': 'FlashInfer 后端，每 16 个一个 E4M3 scale'}
    for d in rows:
        b = kv_bytes_per_token_dtype(a, d)
        note = notes.get(d, f'首尾各 2 层保持 BF16；docstring：{TQ_DOC.get(d, "")}')
        print(f'{d:<22} {kv_head_bytes(d):>10} {b / 1024:>11.2f} {base / b:>6.2f}x {free / b:>12,.0f}  {note}')
    print()
    print('== TurboQuant 槽位字节（head_dim 128）：key = FP8 时 hd，否则 ceil(hd*b/8)+2；value = ceil(hd*b/8)+4；对齐到偶数')
    for d, (kb, vb, nc) in TQ_PRESETS.items():
        s = tq_slot_bytes(128, kb, vb)
        print(f'  {d:<20} K{kb} V{vb} NC={str(nc):<5} 槽位 {s:>3} B / BF16 512 B = {512 / s:.2f}x')
    print(f'  首尾保护：32 层的稠密模型跳过 {boundary_skip_layers(32)}；混合模型（Mamba / 线性注意力）返回 []')
    print()
    d = ARCHS['dsv3']
    print('== MLA（DeepSeek 系）每层每 token 字节：latent 512 + rope 64')
    for k, v in MLA_KV.items():
        print(f'  {k:<14} {v:>5} B  x61 层 = {v * 61 / 1024:>6.1f} KB/token  {d.layers * MLA_KV["auto"] / (v * 61):.2f}x')
    print('  fp8_ds_mla / nvfp4_ds_mla 是 DeepSeek-V3.2 稀疏 MLA（FlashMLA sparse）的专用布局，nvfp4_ds_mla 只在 SM100。')


def scenario_kernel():
    print('== 每种量化在不同 GPU 上落到哪个线性层 kernel（默认环境：官方 CUDA 13 wheel，flashinfer + humming 已装）')
    for title, layer in CASES:
        print(f'-- {title}')
        for label, cc in ARCH_ENVS:
            k, reasons, log = select(Env(cc), layer)
            first = reasons[0] if reasons else ''
            skipped = len(reasons)
            extra = f'（前面跳过 {skipped} 个，第一个原因：{first}）' if skipped else ''
            print(f'  {label:<22} {k or "无 kernel → 启动报错"}{extra}')
    print()
    print('== H100 上的三个开关')
    blk = Layer('fp8_block', act='dyn_group')
    for desc, env in (('默认', Env(90)), ('没有 nvcc（DeepGEMM JIT 不可用）', Env(90, nvcc=False)),
                      ('VLLM_USE_DEEP_GEMM=0', Env(90, deep_gemm=False))):
        print(f'  FP8 block，{desc:<30} → {select(env, blk)[0]}')
    nv = Layer('nvfp4', act='a4')
    for desc, env in (('默认（humming 已装）', Env(90)), ('卸掉 humming-kernels', Env(90, humming=False))):
        print(f'  NVFP4 W4A4，{desc:<28} → {select(env, nv)[0]}')
    awq = Layer('mp', weight_type='uint4', zero_points=True, n=8200)
    print(f'  AWQ 但 N=8200（Machete 要求 N%128）            → {select(Env(90), awq)[0]}')
    print(f'  同上，VLLM_DISABLED_KERNELS=Machete（N=8192）   → '
          f'{select(Env(90, disabled=("Machete",)), Layer("mp", weight_type="uint4", zero_points=True))[0]}')
    print()
    print('== Blackwell 上 DeepGEMM 的模型黑名单（utils/deep_gemm.py:29）')
    for mt in ('llama', 'qwen3_5_moe_text', 'kolibri1'):
        print(f'  B200 model_type={mt:<17} FP8 block → {select(Env(100, model_type=mt), blk)[0]}')
    print()
    print('== 量化方法自己的最低算力（config/vllm.py:945，不满足直接启动报错）')
    print('  ' + '，'.join(f'{k} {v}' for k, v in QUANT_MIN_CC.items()))
    print('  V100（sm70）上 AWQ 直接拒绝；GPTQ 能起：Marlin 要 75 跳过，FP16 激活落到 Exllama，BF16 激活落到 TritonW4A16。')
    for dt in ('fp16', 'bf16'):
        print(f'  sm70 GPTQ {dt} → {select(Env(70), Layer("mp", weight_type="uint4b8", act_dtype=dt))[0]}')


def scenario_tp():
    print('== Qwen2.5-72B：down_proj 的 in_features = intermediate_size，按 TP 切 K（RowParallelLinear）')
    print(f'{"intermediate":>12} {"TP":>3} {"每卡 K":>7} {"K%128":>6} {"Marlin INT4 g128":>18} {"FP8 block 128":>14}')
    for inter, src in ((29568, 'Qwen2.5-72B-Instruct（BF16）'), (29696, 'Qwen2.5-72B-Instruct-GPTQ-Int4 / AWQ')):
        for tp in (1, 2, 4, 8):
            k, r, ok = tp_alignment(inter, tp)
            fp8_ok = tp == 1 or r == 0
            print(f'{inter:>12} {tp:>3} {k:>7} {r:>6} {("通过" if ok else "拒绝"):>16} {("通过" if fp8_ok else "拒绝"):>12}')
        print(f'  ↑ {src}')
    print()
    print('  Marlin 的拒绝原因：in_features per partition 3696 is not divisible by group_size = 128.')
    print('  FP8 block 的拒绝原因：Weight input_size_per_partition = 3696 is not divisible by weight quantization block_k = 128.')
    print('  量化发布版把 29568 补零到 29696 = 232 x 128，TP 2/4/8 每卡都是整数个 group。')
    print()
    print('== DeepSeek-V3 FP8 block 128：MoE 专家 intermediate 2048、稠密 MLP 18432（validate_fp8_block_shape_moe）')
    for tp in (8, 16, 32):
        moe, dense = 2048 // tp, 18432 // tp
        print(f'  TP{tp:<3} 专家每卡 {moe:>4}（{"通过" if moe % 128 == 0 else "拒绝"}）  稠密每卡 {dense:>5}'
              f'（{"通过" if dense % 128 == 0 else "拒绝"}）')
    print('  所以 TP32 跑不了 DeepSeek-V3 的 FP8 block 权重；超过 16 卡要用 DP + EP（专家整块放，不切 intermediate）。')
    print()
    print('== Llama-3.1-70B（28672）')
    for tp in (2, 4, 8, 16):
        k, r, ok = tp_alignment(28672, tp)
        print(f'  TP{tp:<3} 每卡 K {k:>6}  K%128 = {r:>3}  {"通过" if ok else "拒绝"}')


def scenario_backend():
    print('== --linear-backend 只是过滤候选列表，没有匹配就回退到 auto（不会启动失败）')
    nv = Layer('nvfp4', act='a4')
    blk = Layer('fp8_block', act='dyn_group')
    fp8 = Layer('fp8', 'channel', 'dyn_token')
    mx = Layer('mxfp4', act='none')
    rows = [
        (100, 'auto', None), (100, 'flashinfer_cutedsl', None), (100, 'cutlass', None),
        (100, 'marlin', None), (90, 'auto', None), (90, 'marlin', None), (90, 'humming', None),
        (90, 'deep_gemm', None), (90, 'triton', None),
    ]
    print(f'{"GPU":<6} {"--linear-backend":<20} {"NVFP4 W4A4":<30} {"FP8 block":<40} {"FP8 channel":<30} {"MXFP4 W4A16":<16}')
    for cc, be, pq in rows:
        out, warns = [], 0
        for l in (nv, blk, fp8, mx):
            k, _, log = select(Env(cc), l, be, pq)
            warns += len(log)
            out.append(k or '报错')
        print(f'sm{cc:<4} {be:<20} {out[0]:<30} {out[1]:<40} {out[2]:<30} {out[3]:<16}'
              + (f'  （{warns} 条回退 WARNING）' if warns else ''))
    print()
    print('== linear_backend_per_quant 优先于 linear_backend（_get_linear_backend）')
    a16 = Layer('nvfp4', act='a16')
    for cc, be, pq in ((100, 'auto', None), (100, 'auto', {'nvfp4_w4a16': 'humming'}),
                       (100, 'marlin', {'nvfp4_w4a16': 'humming'}), (120, 'auto', None)):
        k = select(Env(cc), a16, be, pq)[0]
        k2 = select(Env(cc), blk, be, pq)[0]
        print(f'  sm{cc} linear_backend={be:<7} per_quant={str(pq):<28} NVFP4 W4A16 → {k:<28} FP8 block → {k2}')
    print('  W4A16 在 auto 下是 "强制" 选：sm100/103 有 CuTe-DSL 就用它，非 sm90 一律 Marlin，只有 sm90 走列表（Humming 在 Marlin 前）。')
    print()
    print('== VLLM_BATCH_INVARIANT=1 时 NVFP4 线性层')
    for cc in (100, 90):
        print(f'  sm{cc} → {select(Env(cc, batch_invariant=True), nv)[0]}')


def scenario_fp8kv():
    print('== ITL = slope x context + intercept（单请求 decode，vLLM 博客 2026-04-22 的拟合值）')
    print(f'{"模型":<14} {"GPU":<5} {"slope 比":>8} {"盈亏点 token":>12} ' +
          ' '.join(f'{"ITL@" + str(c // 1024) + "k BF16/FP8 ms":>24}' for c in (4096, 32768, 131072)))
    for m, g, sb, ib, sf, i_f in FP8KV:
        be = break_even(sb, ib, sf, i_f)
        cols = ' '.join(f'{sb * c + ib:>11.2f} / {sf * c + i_f:<10.2f}' for c in (4096, 32768, 131072))
        print(f'{m:<14} {g:<5} {sf / sb:>7.0%} {be:>12,.0f} {cols}')
    print()
    print('== 博客 Table 3（H100）：两级累加修好精度之后，kernel 调优把盈亏点拉回来')
    for m, v, var, be, pct in FP8KV_TABLE:
        print(f'  {m:<14} {v:<9} {var:<12} 盈亏点 {be:>8,} token  slope {pct}% of BF16')
    print()
    print('  并发 8 的吞吐：Llama-3.1-8B +14.9%（450.3 → 517.5 tok/s，ITL 15.18 → 12.93 ms），gpt-oss-20b +4.8%。')
    print('  gemma-4-E2B（head_dim 256）：decode slope 68%，但 prefill TTFT 二次项 6.93e-07 → 1.12e-06（约 1.6x 更慢）。')


def scenario_sglang():
    print('== 同一份 checkpoint 两边怎么起')
    rows = [
        ('BF16 权重在线转 FP8', '--quantization fp8_per_tensor（fp8 已不做在线量化，会报错）', '--quantization fp8（仍然在线量化）'),
        ('FP8 checkpoint（DeepSeek-V3）', '自动识别 quantization_config，不用传', '同左；传了不兼容的方法会报 does not match'),
        ('AWQ / GPTQ', '自动；awq / awq_marlin / auto_awq 都映射到 AutoAWQConfig', 'awq_marlin / gptq_marlin，awq 走非优化路径会 WARNING'),
        ('ModelOpt NVFP4', '自动（modelopt_fp4）', '--quantization modelopt_fp4，或 nvfp4_online 在线转'),
        ('指定 GEMM 后端', '--linear-backend X，或 kernel_config.linear_backend_per_quant',
         '--fp8-gemm-backend / --fp4-gemm-backend'),
        ('KV cache FP8', '--kv-cache-dtype fp8（scale 读 checkpoint，没有就 1.0 + WARNING）',
         '--kv-cache-dtype fp8_e4m3 + --quantization-param-path'),
        ('KV cache 4 bit', 'nvfp4 / int4_per_token_head / turboquant_*', 'nvfp4 / fp4_mx_block16'),
    ]
    for k, v, s in rows:
        print(f'  {k:<22} vLLM:   {v}')
        print(f'  {"":<22} SGLang: {s}')
    print()
    print('== FP8 GEMM 的 auto 顺序')
    print('  vLLM  block：FlashInfer+DeepGEMM 组合（仅 sm90）→ DeepGEMM → CUTLASS → b12x → Marlin → Humming → Triton → torch')
    print('  SGLang block：DeepGEMM（ENABLE_JIT_DEEPGEMM）→ Blackwell + FlashInfer → sm120 CUTLASS → aiter → npu → xpu → Triton')
    print('  SGLang --fp4-gemm-backend auto：sm100 flashinfer_cutedsl，sm80~sm90 marlin，其它 flashinfer_cutlass')
    print()
    print('== KV cache dtype 选项个数')
    v = ['auto', 'float16', 'bfloat16', 'fp8', 'fp8_e4m3', 'fp8_e5m2', 'fp8_inc', 'fp8_ds_mla', 'nvfp4_ds_mla',
         'turboquant_k8v4', 'turboquant_4bit_nc', 'turboquant_k3v4_nc', 'turboquant_3bit_nc', 'octave_k3v4',
         'octave_k3v3', 'octave_k3v3_compact', 'ultraquant_4bit', 'int4_per_token_head', 'int8_per_token_head',
         'fp8_per_token_head', 'nvfp4', 'nvfp4_4over6']
    s = ['auto', 'fp8_e5m2', 'fp8_e4m3', 'mxfp8', 'bf16', 'bfloat16', 'nvfp4', 'fp4_mx_block16']
    print(f'  vLLM {len(v)} 个（config/cache.py:44）；SGLang {len(s)} 个（srt/arg_groups/fields/model.py:197）')


def main():
    ap = argparse.ArgumentParser()
    sc = {'weights': scenario_weights, 'kv': scenario_kv, 'kernel': scenario_kernel, 'tp': scenario_tp,
          'backend': scenario_backend, 'fp8kv': scenario_fp8kv, 'sglang': scenario_sglang}
    ap.add_argument('--scenario', default='weights', choices=list(sc))
    sc[ap.parse_args().scenario]()


if __name__ == '__main__':
    main()
