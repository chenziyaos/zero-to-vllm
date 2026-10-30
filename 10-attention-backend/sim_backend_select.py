"""注意力后端选择模拟：vLLM vs SGLang（只依赖标准库）。

对照的源码（vllm 主干 0d6f92a，sglang 主干 8fabf5e）：
  vllm   platforms/cuda.py:84–181         _get_backend_priorities：按 MLA / SM 架构 / 是否 causal 给出候选顺序
         platforms/cuda.py:393–421        get_valid_backends：逐个 validate_configuration，记下 (优先级, 原因)
         platforms/cuda.py:424–520        get_attn_backend_cls：用户指定 → 只验这一个；否则取优先级最小的合法后端
         v1/attention/backend.py:127–143  supports_block_size：block 是某个支持尺寸（或 MultipleOf.base）的倍数即可
         v1/attention/backend.py:158–167  get_preferred_block_size：默认 16 能用就 16，否则取支持列表里最小的
         v1/attention/backend.py:284–373  validate_configuration：逐项收集不合法原因
         v1/attention/selector.py:134–138 只有用户显式给了 --block-size，选择时才把 block_size 传进去
         platforms/interface.py:755       update_block_size_for_backend："Setting kv cache block size to %d for %s backend(s)."
         v1/attention/backends/flash_attn.py:286–500, :619   FLASH_ATTN 能力 + CG 支持（FA3/4 ALWAYS，FA2 UNIFORM_BATCH）
         v1/attention/backends/fa_utils.py:74–345            FA 版本选择、FP8 KV、sinks、MLA 条件
         v1/attention/backends/flashinfer.py:572–730, :1176  FLASHINFER 能力 + get_cudagraph_support
         utils/flashinfer.py:719–765                         supports_trtllm_attention / can_use_trtllm_attention
         v1/attention/backends/triton_attn.py:97, :319–395   TRITON_ATTN
         v1/attention/backends/flex_attention.py:88–160      FLEX_ATTENTION
         v1/attention/backends/mla/*.py                      FLASHINFER_MLA / TOKENSPEED_MLA / CUTLASS_MLA / FLASH_ATTN_MLA / FLASHMLA / TRITON_MLA
         model_executor/layers/attention/mla_attention.py:40–135  MLA 的两条计算路径（forward_mha / forward_mqa）
         config/compilation.py:1376       resolve_cudagraph_mode_and_sizes（规则同 09-model-runner）
  sglang srt/arg_groups/model_override_base.py:277–351  get_default_attn_backend
         srt/arg_groups/model_overrides/deepseek_v2.py:179–188  DeepSeek V3/R1 在 SM100 改用 trtllm_mla
         srt/models/deepseek_common/attention_backend_handler.py:108–141    MLA 模型每个 batch 选 MHA / MLA

假设（与真机不同时以真机日志为准）：所有内核都已编译进 wheel、FlashInfer 能下载 trtllm cubin、tokenspeed_mla 未安装
（--tokenspeed 打开）、FlashMLA 可导入。没有建模的项：mm_prefix 组合后端、sparse MLA 的细则、DCP/PCP、KV connector。

用法：
  python3 sim_backend_select.py --scenario matrix   # 典型模型 × GPU × KV dtype → 选中谁、CG 支持、block
  python3 sim_backend_select.py --scenario trace    # 几个配置的完整选择过程（每个候选的拒绝原因 + 日志）
  python3 sim_backend_select.py --scenario block    # --block-size 怎么把高优先级后端挤掉
  python3 sim_backend_select.py --scenario cg       # 后端 CG 支持 × 投机解码 → 实际 cudagraph 模式
  python3 sim_backend_select.py --scenario mla      # MLA 两条路径的 FLOPs / 字节 / 算术强度
  python3 sim_backend_select.py --scenario kv       # 每 token KV 字节：MHA / GQA / MLA
  python3 sim_backend_select.py --scenario sglang   # 同样的配置 SGLang 默认选谁
"""
import argparse
import os
import sys
from dataclasses import dataclass, field, replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, MODELS  # noqa: E402

# --------------------------------------------------------------------------- 0. 硬件与配置
# compute capability；B200 / RTX PRO 6000 不在 specs.GPUS 里，这里只需要 SM 号
CC = {'a100-80g': (8, 0), 'l40s': (8, 9), 'rtx4090': (8, 9), 'h100-sxm': (9, 0), 'h200': (9, 0),
      'b200': (10, 0), 'rtx-pro-6000': (12, 0)}
GPU_NAME = {'b200': 'B200', 'rtx-pro-6000': 'RTX PRO 6000', **{k: v.name for k, v in GPUS.items()}}

# AttentionCGSupport（v1/attention/backend.py:570）
ALWAYS, UNIFORM_BATCH, UNIFORM_SINGLE_TOKEN_DECODE, NEVER = 3, 2, 1, 0
CG_NAME = {3: 'ALWAYS', 2: 'UNIFORM_BATCH', 1: 'UNIFORM_SINGLE_TOKEN_DECODE', 0: 'NEVER'}

QUANT_KV = {'fp8', 'fp8_e4m3', 'fp8_e5m2', 'nvfp4', 'int8_per_token_head', 'int4_per_token_head', 'fp8_per_token_head'}


@dataclass(frozen=True)
class Arch:
    """选择器关心的模型结构。数字来自各模型 config.json。"""
    name: str
    heads: int
    kv_heads: int
    head_size: int
    mla: bool = False
    sink: bool = False
    sliding_window: bool = False
    qk_nope: int = 0
    qk_rope: int = 0
    v_head: int = 0


ARCHS = {
    'qwen3-8b': Arch('Qwen3-8B', 32, 8, 128),
    'llama3.1-70b': Arch('Llama-3.1-70B', 64, 8, 128),
    # openai/gpt-oss-120b config.json：64 头 / 8 KV 头 / head_dim 64，每层带 attention sinks，隔层 sliding_window 128
    'gpt-oss-120b': Arch('gpt-oss-120b', 64, 8, 64, sink=True, sliding_window=True),
    # MLA 层传给选择器的 head_size = kv_lora_rank + qk_rope_head_dim = 512 + 64
    'deepseek-v3': Arch('DeepSeek-V3', 128, 1, 576, mla=True, qk_nope=128, qk_rope=64, v_head=128),
}


@dataclass(frozen=True)
class Cfg:
    arch: str
    gpu: str
    kv: str = 'auto'
    block_size: int | None = None      # 只有用户显式 --block-size 才非 None（selector.py:134）
    backend: str | None = None         # --attention-backend
    batch_invariant: bool = False
    non_causal: bool = False
    tokenspeed: bool = False           # 是否 pip 装了 tokenspeed-mla
    artifactory: bool = True           # FlashInfer 能否下载 trtllm cubin（utils/flashinfer.py:731）

    @property
    def a(self):
        return ARCHS[self.arch]

    @property
    def cc(self):
        return CC[self.gpu]


def quant(kv):
    return kv in QUANT_KV


# --------------------------------------------------------------------------- 1. FA / trtllm 辅助
def fa_version(cfg):
    """fa_utils.py:94–107：SM9x→3，SM10x→4，其余→2（假设对应内核都编译了）。"""
    m = cfg.cc[0]
    return 3 if m == 9 else 4 if m == 10 else 2


def fa4_available(cfg):
    return cfg.cc[0] in (9, 10, 11)    # flash_attn_interface.py:78–80


def fa_supports_kv(cfg):
    """fa_utils.py:282–313 flash_attn_supports_kv_cache_dtype。"""
    if cfg.kv == 'fp8_e5m2':
        return False
    v, sm90 = fa_version(cfg), cfg.cc[0] == 9
    sm90_ok = v == 3 or (v == 4 and cfg.a.head_size == 512)
    return (sm90_ok and sm90) or (v == 4 and cfg.cc[0] == 10)


def trtllm(cfg, prefill):
    """utils/flashinfer.py:719 supports_trtllm_attention。"""
    if cfg.batch_invariant or not cfg.artifactory:
        return False
    if cfg.cc == (9, 0) or cfg.cc[0] == 12:
        return not prefill        # XQA 只有 decode
    return cfg.cc[0] == 10


# --------------------------------------------------------------------------- 2. 后端能力表
@dataclass
class Backend:
    name: str
    mla: bool = False
    sparse: bool = False
    dtypes: tuple = ('fp16', 'bf16')
    kv: tuple = ('auto', 'float16', 'bfloat16')
    blocks: tuple = (('mult', 16),)                 # ('mult', n) = MultipleOf(n)，('fix', n) = 固定尺寸
    head: object = None                             # callable(cfg) -> bool
    cc_ok: object = lambda cc: True
    sink: object = lambda cfg: False
    sliding: bool = False
    non_causal: bool = False
    batch_inv: bool = False
    combo: object = lambda cfg: None                # supports_combination，返回原因或 None
    cg: object = lambda cfg: NEVER
    kv_extra: object = None                         # 覆盖 supports_kv_cache_dtype
    blocks_fn: object = None                        # 依赖配置的 get_supported_kernel_block_sizes
    note: str = ''

    def supported_blocks(self, cfg):
        return self.blocks_fn(cfg) if self.blocks_fn else self.blocks

    def supports_block(self, cfg, bs):
        if bs is None:
            return True
        return any(bs % n == 0 for _, n in self.supported_blocks(cfg))

    def preferred_block(self, cfg, default=16):
        """backend.py:158 get_preferred_block_size。"""
        if self.supports_block(cfg, default):
            return default
        return min(n for _, n in self.supported_blocks(cfg))

    def supports_kv(self, cfg):
        if self.kv_extra:
            return self.kv_extra(cfg)
        return cfg.kv in self.kv


def _fa_blocks(cfg):
    a = cfg.a
    if cfg.cc[0] == 9 and quant(cfg.kv) and a.head_size == 512 and fa_version(cfg) == 4:
        return (('fix', 64),)                       # flash_attn.py:318–325
    if a.head_size == 256 and fa_version(cfg) == 4:
        return (('fix', 128),)                      # FA4_HD256_PAGE_SIZE（fa_utils.py:14）
    return (('mult', 16),)


def _fa_kv(cfg):
    if cfg.kv not in ('auto', 'float16', 'bfloat16', 'fp8', 'fp8_e4m3'):
        return False
    return fa_supports_kv(cfg) if quant(cfg.kv) else True


def _fa_combo(cfg):
    if cfg.a.sink and cfg.cc < (9, 0):
        return 'sink not supported on compute capability < 9.0'
    if quant(cfg.kv) and not fa_supports_kv(cfg):
        return 'FP8 KV cache requires FA3 on SM90, FA4 with head_size=512 on SM90, or FA4 on SM100'
    return None


def _fi_blocks(cfg):
    a = cfg.a
    large = a.kv_heads > 0 and a.heads // a.kv_heads > 1 and cfg.cc[0] == 10 and trtllm(cfg, False)
    return tuple(('fix', n) for n in ((16, 32, 64, 128, 256, 512, 1024) if large else (16, 32, 64)))


def _fi_sink(cfg):
    if cfg.cc == (9, 0) or cfg.cc[0] == 12:
        return trtllm(cfg, False)
    if cfg.cc[0] != 10:
        return False
    return trtllm(cfg, False) and trtllm(cfg, True)


def _fi_cg(cfg):
    """flashinfer.py:1176–1225（不含 DCP 分支）。"""
    if cfg.cc < (9, 0):                             # _use_fa2_native_decode
        return UNIFORM_BATCH if not cfg.non_causal else UNIFORM_SINGLE_TOKEN_DECODE
    xqa = cfg.cc == (9, 0) or cfg.cc[0] == 12
    hd = cfg.a.head_size
    ok = trtllm(cfg, False) and cfg.a.heads % max(cfg.a.kv_heads, 1) == 0
    if xqa and not (16 <= hd <= 256 and hd % 16 == 0):
        ok = False
    if ok and (not cfg.non_causal or xqa):
        return UNIFORM_BATCH
    return UNIFORM_SINGLE_TOKEN_DECODE


def _mla_head(cfg):
    return cfg.a.head_size in (320, 512, 576)       # mla_attention.py:1625


def _flashmla_combo(cfg):
    return None if cfg.cc[0] == 9 else 'FlashMLA Dense is only supported on Hopper devices.'


def _fimla_combo(cfg):
    q = cfg.a.qk_nope or 1
    return None if q in (64, 128, 192) else f'FlashInfer MLA kernel requires qk_nope_head_dim in [64, 128, 192], but got {q}'


def _tokenspeed_combo(cfg):
    if not cfg.tokenspeed:
        return 'tokenspeed_mla package is not installed. Install it with: `uv pip install tokenspeed-mla`'
    a = cfg.a
    if (a.qk_nope, a.qk_rope, a.v_head) != (128, 64, 128):
        return 'tokenspeed_mla requires DeepSeek R1 MLA dimensions'
    return None


def _fa_mla_combo(cfg):
    return None if (fa_version(cfg) == 3 and cfg.cc[0] == 9) else 'FlashAttention MLA not supported on this device'


MLA_KV = ('auto', 'float16', 'bfloat16', 'fp8', 'fp8_e4m3')

BACKENDS = {b.name: b for b in [
    Backend('FLASH_ATTN', kv=('auto', 'float16', 'bfloat16', 'fp8', 'fp8_e4m3'), kv_extra=_fa_kv,
            blocks_fn=_fa_blocks,
            head=lambda c: c.a.head_size % 8 == 0 and (c.a.head_size <= 256 or (fa4_available(c) and c.a.head_size <= 512)),
            cc_ok=lambda cc: cc >= (8, 0), sink=lambda c: fa_version(c) in (3, 4),
            sliding=True, non_causal=True, batch_inv=True, combo=_fa_combo,
            cg=lambda c: ALWAYS if fa_version(c) in (3, 4) else UNIFORM_BATCH),
    Backend('FLASHINFER', kv=('auto', 'float16', 'bfloat16', 'fp8', 'fp8_e4m3', 'fp8_e5m2'), blocks_fn=_fi_blocks,
            head=lambda c: c.a.head_size in (64, 128, 256, 512),
            cc_ok=lambda cc: (8, 0) <= cc <= (12, 1), sink=_fi_sink,
            sliding=True, non_causal=True, cg=_fi_cg),
    Backend('TRITON_ATTN', dtypes=('fp16', 'bf16', 'fp32'),
            kv=('auto', 'float16', 'bfloat16', 'fp8', 'fp8_e4m3', 'fp8_e5m2',
                'int4_per_token_head', 'int8_per_token_head', 'fp8_per_token_head'),
            head=lambda c: c.a.head_size >= 32, sink=lambda c: True,
            sliding=True, non_causal=True, batch_inv=True, cg=lambda c: ALWAYS),
    Backend('FLEX_ATTENTION', dtypes=('fp16', 'bf16', 'fp32'), head=lambda c: True,
            sliding=True, non_causal=True, batch_inv=True, cg=lambda c: ALWAYS),
    Backend('TURBOQUANT', kv=('turboquant_k8v4', 'turboquant_4bit_nc', 'turboquant_k3v4_nc', 'turboquant_3bit_nc'),
            head=lambda c: True, cg=lambda c: UNIFORM_SINGLE_TOKEN_DECODE),
    # ---- MLA
    Backend('FLASHINFER_MLA', mla=True, kv=MLA_KV, blocks=(('fix', 32), ('fix', 64)), head=_mla_head,
            cc_ok=lambda cc: cc[0] == 10, combo=_fimla_combo, cg=lambda c: ALWAYS),
    Backend('TOKENSPEED_MLA', mla=True, kv=('fp8', 'fp8_e4m3'), blocks=(('fix', 32), ('fix', 64)), head=_mla_head,
            cc_ok=lambda cc: cc[0] == 10, combo=_tokenspeed_combo, cg=lambda c: UNIFORM_BATCH),
    Backend('CUTLASS_MLA', mla=True, kv=MLA_KV, blocks=(('fix', 128),), head=_mla_head,
            cc_ok=lambda cc: cc[0] == 10, cg=lambda c: UNIFORM_SINGLE_TOKEN_DECODE),
    Backend('FLASH_ATTN_MLA', mla=True, kv=('auto', 'float16', 'bfloat16'), head=lambda c: c.a.head_size == 576,
            cc_ok=lambda cc: cc[0] == 9, combo=_fa_mla_combo, cg=lambda c: UNIFORM_BATCH),
    Backend('FLASHMLA', mla=True, kv=MLA_KV, blocks=(('fix', 64),), head=_mla_head,
            cc_ok=lambda cc: cc[0] in (9, 10), combo=_flashmla_combo, cg=lambda c: UNIFORM_BATCH),
    Backend('TRITON_MLA', mla=True, kv=MLA_KV, head=_mla_head, cg=lambda c: UNIFORM_BATCH),
]}
# 稀疏 MLA（DSA）后端：对稠密模型统一给 "non-sparse not supported"
for _n in ('FLASHINFER_MLA_SPARSE', 'FLASHMLA_SPARSE', 'FLASH_ATTN_MLA_SPARSE', 'FLASH_ATTN_MLA_SPARSE_FA4',
           'FLASHINFER_MLA_SPARSE_SM90', 'FLASHINFER_MLA_SPARSE_SM120'):
    BACKENDS[_n] = Backend(_n, mla=True, sparse=True, kv=MLA_KV, head=_mla_head, note='稀疏 MLA')


# --------------------------------------------------------------------------- 3. 优先级与校验
def priorities(cfg, use_mm_prefix=False):
    """platforms/cuda.py:84–181 _get_backend_priorities。"""
    m = cfg.cc[0]
    if cfg.a.mla:
        if m == 10:
            if quant(cfg.kv):
                sparse = ['FLASHINFER_MLA_SPARSE', 'FLASHMLA_SPARSE']
            elif cfg.a.heads <= 16:
                sparse = ['FLASH_ATTN_MLA_SPARSE_FA4', 'FLASHINFER_MLA_SPARSE', 'FLASHMLA_SPARSE']
            else:
                sparse = ['FLASHMLA_SPARSE', 'FLASHINFER_MLA_SPARSE']
            return ['FLASHINFER_MLA', 'TOKENSPEED_MLA', 'CUTLASS_MLA', 'FLASH_ATTN_MLA', 'FLASHMLA', 'TRITON_MLA', *sparse]
        if m == 12:
            return ['TRITON_MLA', 'FLASHINFER_MLA_SPARSE_SM120']
        tail = ['FLASH_ATTN_MLA_SPARSE', 'FLASHMLA_SPARSE']
        if cfg.a.head_size == 512:
            tail.insert(0, 'FLASHINFER_MLA_SPARSE_SM90')
        else:
            tail.append('FLASHINFER_MLA_SPARSE_SM90')
        return ['FLASH_ATTN_MLA', 'FLASHMLA', 'FLASHINFER_MLA', 'TRITON_MLA', *tail]
    if m == 10 and not cfg.non_causal:
        return [*(['TRITON_FLASHINFER'] if use_mm_prefix else []),
                'FLASHINFER', 'FLASH_ATTN', 'TRITON_ATTN', 'FLEX_ATTENTION', 'TURBOQUANT']
    return [*(['TRITON_FLASH_ATTN'] if m == 9 and use_mm_prefix else []),
            'FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION', 'TURBOQUANT']


def validate(b, cfg, dtype='bf16'):
    """v1/attention/backend.py:284–373 validate_configuration（按源码顺序收集原因）。"""
    r = []
    if not b.head(cfg):
        r.append('head_size not supported')
    if dtype not in b.dtypes:
        r.append('dtype not supported')
    if not b.supports_kv(cfg):
        r.append('kv_cache_dtype not supported')
    if not b.supports_block(cfg, cfg.block_size):
        r.append('block_size not supported')
    if cfg.a.mla != b.mla:
        r.append('MLA not supported' if cfg.a.mla else 'non-MLA not supported')
    if cfg.a.sink and not b.sink(cfg):
        r.append('attention sinks not supported')
    if b.sparse:
        r.append('non-sparse not supported')
    if not b.cc_ok(cfg.cc):
        r.append('compute capability not supported')
    if cfg.a.sliding_window and not b.sliding:
        r.append('sliding window not supported')
    if cfg.non_causal and not b.non_causal:
        r.append('non-causal attention not supported')
    if cfg.batch_invariant and not b.batch_inv:
        r.append('batch invariance not supported')
    c = b.combo(cfg)   # supports_combination 总会跑，原因附在最后
    if c:
        r.append(c)
    return r


@dataclass
class Result:
    backend: str | None
    valid: list
    invalid: dict                       # name -> (priority, reasons)
    logs: list = field(default_factory=list)
    block: int | None = None
    cg: int | None = None
    error: str | None = None


def select(cfg):
    """platforms/cuda.py:424–520 get_attn_backend_cls + interface.py:755 block size 调整。"""
    res = Result(None, [], {})
    if cfg.backend:
        b = BACKENDS[cfg.backend]
        r = validate(b, cfg)
        if r:
            res.error = f'Selected backend AttentionBackendEnum.{cfg.backend} is not valid for this configuration. Reason: {r}'
            return res
        res.backend = cfg.backend
        res.logs.append(f'INFO Using AttentionBackendEnum.{cfg.backend} backend.')
    else:
        for p, name in enumerate(priorities(cfg)):
            r = validate(BACKENDS[name], cfg)
            if r:
                res.invalid[name] = (p, r)
            else:
                res.valid.append((p, name))
        if not res.valid:
            res.error = 'No valid attention backend found'
            return res
        sp, sel = min(res.valid)
        res.backend = sel
        if cfg.block_size is not None:
            excluded = [n for n, (p, r) in res.invalid.items() if p < sp and r == ['block_size not supported']]
            if excluded:
                res.logs.append(f'WARNING --block-size {cfg.block_size} precluded higher-priority backend(s) '
                                f'{", ".join(excluded)}. Using {sel} instead, which may result in reduced performance. '
                                'Consider removing --block-size to auto-select the optimal block size.')
        res.logs.append(f'INFO Using {sel} attention backend out of potential backends: '
                        '[' + ', '.join(f"'{n}'" for _, n in res.valid) + '].')
    b = BACKENDS[res.backend]
    if cfg.block_size is None:
        res.block = b.preferred_block(cfg)
        if res.block != 16:
            res.logs.append(f'INFO Setting kv cache block size to {res.block} for {res.backend} backend(s).')
    else:
        res.block = cfg.block_size
    res.cg = b.cg(cfg)
    return res


# --------------------------------------------------------------------------- 4. CG 降级（同 09）
MODES = {'NONE': ('NONE', 'NONE'), 'PIECEWISE': ('PIECEWISE', 'PIECEWISE'), 'FULL': ('FULL', 'FULL'),
         'FULL_DECODE_ONLY': ('FULL', 'NONE'), 'FULL_AND_PIECEWISE': ('FULL', 'PIECEWISE')}


def resolve_mode(mode, support, query_len=1):
    """compilation.py:1376 的降级规则（VLLM_COMPILE + 注意力切图的默认情形）。"""
    if MODES[mode][1] == 'FULL' and support != ALWAYS:
        if support == NEVER:
            return 'PIECEWISE'
        mode = 'FULL_AND_PIECEWISE'
    if MODES[mode][0] == 'FULL' and support == NEVER:
        mode = 'PIECEWISE'
    if MODES[mode][0] == 'FULL' and query_len > 1 and support < UNIFORM_BATCH:
        mode = 'PIECEWISE'
    return mode


# --------------------------------------------------------------------------- 5. MLA 与 KV 账
@dataclass(frozen=True)
class MLADims:
    heads: int = 128
    nope: int = 128
    rope: int = 64
    v: int = 128
    lkv: int = 512


def mla_decode_cost(d, ctx, bytes_per=2):
    """单个 decode token、单层、上下文 ctx。返回 {路径: (FLOPs, 读 KV 字节)}。

    mqa（吸收）：q_nope 先乘 W_UK 得到 Lkv 维的 ql_nope，和 latent 做 MQA（head_dim = Lkv + R），输出再乘 W_UV。
    mha（解压）：把 ctx 个 latent 解压成每头 K（nope+rope）和 V，再做普通 MHA。
    mha_cached：假设解压后的 K/V 已经常驻显存（等价于不用 MLA 的 KV cache）。
    """
    H, P, R, V, L = d.heads, d.nope, d.rope, d.v, d.lkv
    absorb = 2 * H * P * L + 2 * H * L * V                          # W_UK 吸收进 q、W_UV 作用在输出上
    mqa_attn = 2 * H * ctx * (L + R) + 2 * H * ctx * L               # QK^T 在 576 维、PV 在 512 维
    mqa = (absorb + mqa_attn, ctx * (L + R) * bytes_per)
    decompress = 2 * ctx * L * H * (P + V)                           # latent → 每头 K_nope、V
    mha_attn = 2 * H * ctx * (P + R) + 2 * H * ctx * V
    mha = (decompress + mha_attn, ctx * (L + R) * bytes_per)
    mha_cached = (mha_attn, ctx * H * (P + R + V) * bytes_per)
    return {'mqa': mqa, 'mha': mha, 'mha_cached': mha_cached}


def mla_prefill_attn_flops(d, s, ctx):
    """一个 prefill chunk：s 个新 token 看 ctx 个 token（只算注意力本体，不算投影）。"""
    H, P, R, V, L = d.heads, d.nope, d.rope, d.v, d.lkv
    mha = 2 * H * s * ctx * (P + R) + 2 * H * s * ctx * V + 2 * ctx * L * H * (P + V)
    mqa = 2 * H * s * ctx * (L + R) + 2 * H * s * ctx * L
    return mha, mqa


def gqa_decode_cost(m, ctx, bytes_per=2):
    """非 MLA 模型单层 decode 一个 token：FLOPs 与读 KV 字节。"""
    flops = 2 * m.heads * ctx * m.head_dim * 2
    byts = ctx * 2 * m.kv_heads * m.head_dim * bytes_per
    return flops, byts


# --------------------------------------------------------------------------- 6. SGLang 默认
@dataclass(frozen=True)
class SglEnv:
    gpu: str
    cuda: tuple = (12, 9)
    has_flashinfer: bool = True
    spec_topk: int | None = None      # None = 不开投机；EAGLE topk
    asymmetric_kv: bool = False
    arch_name: str = ''


def sglang_default(arch, env):
    """model_override_base.py:277–351 get_default_attn_backend + deepseek_v2.py:179 覆盖。"""
    cc = CC[env.gpu]
    hopper = cc[0] == 9 and env.cuda >= (12, 3)
    sm100 = cc[0] == 10
    no_spec_or_topk1 = env.spec_topk in (None, 1)
    if env.arch_name == 'WhisperForConditionalGeneration':
        return 'flashinfer'
    if not arch.mla:
        if hopper and no_spec_or_topk1:
            return 'fa3'
        if sm100 and no_spec_or_topk1:
            return 'fa4' if env.asymmetric_kv else 'trtllm_mha'
        if env.has_flashinfer and not arch.sink:
            return 'flashinfer'
        return 'triton'
    if env.arch_name == 'DeepseekV3ForCausalLM' and sm100:
        return 'trtllm_mla'                       # deepseek_v2.py:179–188
    if hopper:
        return 'fa3'
    if sm100:
        return 'flashinfer'
    return 'triton'


# --------------------------------------------------------------------------- 7. 场景
MATRIX = [
    Cfg('qwen3-8b', 'a100-80g'), Cfg('qwen3-8b', 'l40s'), Cfg('qwen3-8b', 'h100-sxm'), Cfg('qwen3-8b', 'b200'),
    Cfg('qwen3-8b', 'a100-80g', kv='fp8'), Cfg('qwen3-8b', 'h100-sxm', kv='fp8'), Cfg('qwen3-8b', 'b200', kv='fp8'),
    Cfg('qwen3-8b', 'h100-sxm', kv='fp8_e5m2'), Cfg('qwen3-8b', 'h100-sxm', batch_invariant=True),
    Cfg('gpt-oss-120b', 'a100-80g'), Cfg('gpt-oss-120b', 'h100-sxm'), Cfg('gpt-oss-120b', 'b200'),
    Cfg('deepseek-v3', 'h100-sxm'), Cfg('deepseek-v3', 'h100-sxm', kv='fp8'),
    Cfg('deepseek-v3', 'b200'), Cfg('deepseek-v3', 'b200', kv='fp8', tokenspeed=True),
    Cfg('deepseek-v3', 'a100-80g'), Cfg('deepseek-v3', 'rtx-pro-6000'),
]


def label(c):
    s = f'{ARCHS[c.arch].name} @ {GPU_NAME[c.gpu]}'
    if c.kv != 'auto':
        s += f' kv={c.kv}'
    if c.block_size:
        s += f' --block-size {c.block_size}'
    if c.backend:
        s += f' -ab {c.backend}'
    if c.batch_invariant:
        s += ' batch-invariant'
    if c.tokenspeed:
        s += ' +tokenspeed'
    return s


def scenario_matrix():
    print(f'{"配置":<46} {"选中":<15} {"FA":>3} {"block":>5} {"CG 支持":<28} 合法候选')
    for c in MATRIX:
        r = select(c)
        fa = fa_version(c) if r.backend == 'FLASH_ATTN' else '-'
        print(f'{label(c):<46} {r.backend or "ERROR":<15} {fa!s:>3} {r.block!s:>5} {CG_NAME.get(r.cg, "-"):<28} '
              + ','.join(n for _, n in r.valid))


TRACES = [
    Cfg('qwen3-8b', 'a100-80g', kv='fp8'),
    Cfg('gpt-oss-120b', 'a100-80g'),
    Cfg('deepseek-v3', 'b200'),
    Cfg('deepseek-v3', 'a100-80g', backend='FLASHMLA'),
    Cfg('qwen3-8b', 'h100-sxm', backend='FLASHINFER', kv='fp8_e5m2'),
]


def trace(c):
    print(f'== {label(c)}')
    r = select(c)
    if not c.backend:
        for p, name in enumerate(priorities(c)):
            if name in r.invalid:
                print(f'  [{p}] {name:<28} ✗ {r.invalid[name][1]}')
            else:
                print(f'  [{p}] {name:<28} ✓')
    for line in r.logs:
        print('  ' + line)
    if r.error:
        print('  ValueError: ' + r.error)
    print()


def scenario_trace():
    for c in TRACES:
        trace(c)


def scenario_block():
    for c in [Cfg('deepseek-v3', 'b200', block_size=16), Cfg('deepseek-v3', 'b200', block_size=64),
              Cfg('deepseek-v3', 'h100-sxm', kv='fp8', block_size=16), Cfg('qwen3-8b', 'b200', block_size=128),
              Cfg('qwen3-8b', 'h100-sxm', block_size=8)]:
        trace(c)
    print('== 不指定 --block-size 时各后端自动取的 block（get_preferred_block_size，默认 16）')
    for name, c in [('FLASH_ATTN', Cfg('qwen3-8b', 'h100-sxm')), ('FLASHINFER', Cfg('qwen3-8b', 'b200')),
                    ('FLASHINFER_MLA', Cfg('deepseek-v3', 'b200')), ('CUTLASS_MLA', Cfg('deepseek-v3', 'b200')),
                    ('FLASHMLA', Cfg('deepseek-v3', 'h100-sxm')), ('TRITON_MLA', Cfg('deepseek-v3', 'h100-sxm'))]:
        b = BACKENDS[name]
        print(f'  {name:<16} 支持 {[f"{k}:{n}" for k, n in b.supported_blocks(c)]!s:<60} → {b.preferred_block(c)}')


def scenario_cg():
    rows = [('FLASH_ATTN (FA2, A100)', Cfg('qwen3-8b', 'a100-80g'), 'FLASH_ATTN'),
            ('FLASH_ATTN (FA3, H100)', Cfg('qwen3-8b', 'h100-sxm'), 'FLASH_ATTN'),
            ('FLASHINFER (A100, FA2 decode)', Cfg('qwen3-8b', 'a100-80g'), 'FLASHINFER'),
            ('FLASHINFER (H100, XQA)', Cfg('qwen3-8b', 'h100-sxm'), 'FLASHINFER'),
            ('FLASHINFER (H100, 无 cubin)', Cfg('qwen3-8b', 'h100-sxm', artifactory=False), 'FLASHINFER'),
            ('FLASHINFER (B200, trtllm)', Cfg('qwen3-8b', 'b200'), 'FLASHINFER'),
            ('TRITON_ATTN', Cfg('qwen3-8b', 'h100-sxm'), 'TRITON_ATTN'),
            ('FLASHINFER_MLA (B200)', Cfg('deepseek-v3', 'b200'), 'FLASHINFER_MLA'),
            ('FLASH_ATTN_MLA (H100)', Cfg('deepseek-v3', 'h100-sxm'), 'FLASH_ATTN_MLA'),
            ('FLASHMLA (H100)', Cfg('deepseek-v3', 'h100-sxm'), 'FLASHMLA'),
            ('CUTLASS_MLA (B200)', Cfg('deepseek-v3', 'b200'), 'CUTLASS_MLA')]
    print(f'{"后端":<32} {"CG 支持":<28} {"默认 F&P":<20} {"要 FULL":<20} {"F&P + EAGLE k=3":<20}')
    for lab, c, name in rows:
        s = BACKENDS[name].cg(c)
        print(f'{lab:<32} {CG_NAME[s]:<28} {resolve_mode("FULL_AND_PIECEWISE", s):<20} '
              f'{resolve_mode("FULL", s):<20} {resolve_mode("FULL_AND_PIECEWISE", s, query_len=4):<20}')


def scenario_mla():
    d = MLADims()
    print('== DeepSeek-V3 单层 decode 一个 token（H=128, nope=128, rope=64, v=128, Lkv=512, BF16）')
    print(f'{"ctx":>6} {"路径":<11} {"GFLOP":>9} {"读 KV MiB":>10} {"FLOP/B":>8}')
    for ctx in (1024, 8192, 32768):
        for k, (f, b) in mla_decode_cost(d, ctx).items():
            print(f'{ctx:>6} {k:<11} {f / 1e9:>9.3f} {b / 2**20:>10.2f} {f / b:>8.1f}')
    m = MODELS['qwen3-8b']
    f, b = gqa_decode_cost(m, 8192)
    print(f'  对照 Qwen3-8B（GQA 32/8）ctx 8192：{f / 1e9:.3f} GFLOP，读 {b / 2**20:.2f} MiB，{f / b:.1f} FLOP/B')
    for g in ('a100-80g', 'h100-sxm', 'h200'):
        print(f'  {GPUS[g].name} BF16 拐点 {GPUS[g].ridge:.0f} FLOP/B')
    print()
    print('== prefill chunk：s 个新 token 看 ctx 个 token 的注意力 FLOPs（GFLOP，含 MHA 路径的解压）')
    for s, ctx in ((1, 8192), (64, 8192), (2048, 2048), (8192, 8192)):
        mha, mqa = mla_prefill_attn_flops(d, s, ctx)
        print(f'  s={s:>5} ctx={ctx:>5}  MHA {mha / 1e9:>10.1f}  MQA {mqa / 1e9:>10.1f}  MQA/MHA {mqa / mha:>5.2f}')


def scenario_kv():
    print(f'{"模型":<16} {"结构":<18} {"每 token KV（BF16）":>20} {"FP8":>12} {"128K 上下文 GiB":>16}')
    for k in ('qwen3-8b', 'llama3.1-8b', 'llama3.1-70b', 'deepseek-v3'):
        m = MODELS[k]
        kind = f'MLA {m.mla_kv_rank}+{m.mla_rope_dim}' if m.mla_kv_rank else f'GQA {m.heads}/{m.kv_heads}'
        b = m.kv_bytes_per_token()
        print(f'{m.name:<16} {kind:<18} {b:>20,.0f} {m.kv_bytes_per_token(1):>12,.0f} {b * 131072 / 2**30:>16.2f}')
    d = MODELS['deepseek-v3']
    mha = d.layers * d.heads * (128 + 64 + 128) * 2
    print(f'{"DSV3 若存 MHA":<16} {"128 头 K192+V128":<18} {mha:>20,.0f} {mha / 2:>12,.0f} {mha * 131072 / 2**30:>16.2f}')
    print(f'  MLA 相对 MHA 缩小 {mha / d.kv_bytes_per_token():.1f} 倍')


def scenario_sglang():
    rows = [('qwen3-8b', SglEnv('a100-80g')), ('qwen3-8b', SglEnv('h100-sxm')), ('qwen3-8b', SglEnv('h100-sxm', spec_topk=4)),
            ('qwen3-8b', SglEnv('b200')), ('qwen3-8b', SglEnv('a100-80g', has_flashinfer=False)),
            ('gpt-oss-120b', SglEnv('a100-80g')), ('gpt-oss-120b', SglEnv('h100-sxm')),
            ('deepseek-v3', SglEnv('h100-sxm', arch_name='DeepseekV3ForCausalLM')),
            ('deepseek-v3', SglEnv('b200', arch_name='DeepseekV3ForCausalLM')),
            ('deepseek-v3', SglEnv('a100-80g', arch_name='DeepseekV3ForCausalLM'))]
    print(f'{"配置":<40} {"SGLang 默认":<14} {"vLLM 默认":<16}')
    for k, e in rows:
        lab = f'{ARCHS[k].name} @ {GPU_NAME[e.gpu]}' + (f' EAGLE topk={e.spec_topk}' if e.spec_topk else '') \
            + ('' if e.has_flashinfer else ' 无 flashinfer')
        v = select(Cfg(k, e.gpu)).backend
        print(f'{lab:<40} {sglang_default(ARCHS[k], e):<14} {v or "ERROR":<16}')


def main():
    ap = argparse.ArgumentParser()
    sc = {'matrix': scenario_matrix, 'trace': scenario_trace, 'block': scenario_block, 'cg': scenario_cg,
          'mla': scenario_mla, 'kv': scenario_kv, 'sglang': scenario_sglang}
    ap.add_argument('--scenario', default='matrix', choices=list(sc))
    sc[ap.parse_args().scenario]()


if __name__ == '__main__':
    main()
