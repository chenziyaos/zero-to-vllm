"""CUDA graph 捕获尺寸、dispatch 与 padding 模拟：vLLM vs SGLang（只依赖标准库）。

对照的源码（vllm 主干 0d6f92a，sglang 主干 8fabf5e）：
  vllm   config/compilation.py:53          CUDAGraphMode：NONE / PIECEWISE / FULL / FULL_DECODE_ONLY / FULL_AND_PIECEWISE
         config/compilation.py:1376        resolve_cudagraph_mode_and_sizes：按注意力后端的 AttentionCGSupport 降级
         config/compilation.py:1537        adjust_cudagraph_sizes_for_spec_decode：MRv1 把尺寸取成 query_len 的倍数
         config/compilation.py:1567        get_compile_ranges：endpoints → [1, e0], [e0+1, e1], …
         config/vllm.py:2470–2625          _set_cudagraph_sizes：默认上限 512（Blackwell 1024）、[1,2,4]+步长 8+步长 16
         v1/cudagraph_dispatcher.py:71     _compute_bs_to_padded_graph_size：每个 bs 映射到 ≥ bs 的最小捕获尺寸
         v1/cudagraph_dispatcher.py:130    _create_padded_batch_descriptor：FULL 的 uniform decode 键带精确 num_reqs
         v1/cudagraph_dispatcher.py:164    initialize_cudagraph_keys：mixed 键 + 单独的 FULL decode 键
         v1/cudagraph_dispatcher.py:233    dispatch：先 FULL 键，再放宽成 PIECEWISE 键，都没有就 NONE
         lora/utils.py:55                  get_captured_lora_counts：2 的幂 + max_loras+1
         v1/worker/gpu_model_runner.py:6268 profile_cudagraph_memory：首个 + (n−1) × max(第二个, 1 MiB)
         compilation/backends.py:1093      编译缓存目录 = sha256(str([env, config, code, compiler]))[:10]
  sglang srt/arg_groups/cuda_graph_hook.py:480  prefill 捕获尺寸（token）
         srt/arg_groups/cuda_graph_hook.py:500  decode 捕获尺寸（请求数）：[1,2,4,8,12]+16..256 步长 8+…
         srt/arg_groups/memory_hook.py:89       按显存档位给 decode max_bs 和 chunked_prefill_size

用法：
  python3 sim_capture_sizes.py --scenario sizes      # 两边默认的捕获尺寸列表
  python3 sim_capture_sizes.py --scenario resolve    # 后端能力 × 请求的模式 → 实际模式
  python3 sim_capture_sizes.py --scenario dispatch   # 几个典型 batch 走哪张图、补多少
  python3 sim_capture_sizes.py --scenario padding    # 不同 batch 分布下的 padding 浪费与 eager 比例
  python3 sim_capture_sizes.py --scenario keys       # 每种模式要捕获多少张图、LoRA 翻倍、显存估算下限
  python3 sim_capture_sizes.py --scenario cache      # 编译缓存目录怎么算、compile range 怎么切
"""
import argparse
import hashlib
from bisect import bisect_left
from dataclasses import dataclass, replace

# --------------------------------------------------------------------------- 1. 模式
NONE, PIECEWISE, FULL = 'NONE', 'PIECEWISE', 'FULL'
MODES = {
    'NONE': (NONE, NONE),
    'PIECEWISE': (PIECEWISE, PIECEWISE),
    'FULL': (FULL, FULL),
    'FULL_DECODE_ONLY': (FULL, NONE),          # (decode, mixed)
    'FULL_AND_PIECEWISE': (FULL, PIECEWISE),
}
SEPARATE = {'FULL_DECODE_ONLY', 'FULL_AND_PIECEWISE'}

# AttentionCGSupport（v1/attention/backend.py:570）
ALWAYS, UNIFORM_BATCH, UNIFORM_SINGLE_TOKEN_DECODE, NEVER = 3, 2, 1, 0
SUPPORT_NAME = {3: 'ALWAYS', 2: 'UNIFORM_BATCH', 1: 'UNIFORM_SINGLE_TOKEN_DECODE', 0: 'NEVER'}
# docs/design/cuda_graphs.md 175–187
BACKENDS = [
    ('FlashAttention 3', ALWAYS),
    ('Triton', ALWAYS),
    ('FlashAttention 2', UNIFORM_BATCH),
    ('FlashMLA', UNIFORM_BATCH),
    ('FlashInfer', UNIFORM_SINGLE_TOKEN_DECODE),
    ('CUTLASS MLA / Mamba', UNIFORM_SINGLE_TOKEN_DECODE),
    ('未列出的后端', NEVER),
]


def decode_mode(m):
    return MODES[m][0]


def mixed_mode(m):
    return MODES[m][1]


def has_mode(m, x):
    return x in MODES[m]


def resolve_mode(mode, support, query_len=1, vllm_compile=True, split_attention=True):
    """compilation.py:1376 resolve_cudagraph_mode_and_sizes 的降级规则。返回 (模式, 原因)。"""
    if mode == 'NONE':
        return 'NONE', ''
    why = []
    if mixed_mode(mode) == FULL and support != ALWAYS:
        if support == NEVER:
            return 'ERROR', '后端不支持任何 full graph'
        mode = 'FULL_AND_PIECEWISE' if split_attention else 'FULL_DECODE_ONLY'
        why.append('mixed 不能 FULL')
    if decode_mode(mode) == FULL and support == NEVER:
        mode = 'PIECEWISE' if (vllm_compile and split_attention) else 'NONE'
        why.append('decode 也不能 FULL')
    if decode_mode(mode) == FULL and query_len > 1 and support < UNIFORM_BATCH:
        mode = 'PIECEWISE' if split_attention else 'NONE'
        why.append('投机解码要 UNIFORM_BATCH')
    # compilation.py:1468 / vllm.py:1144：没有 piecewise 子图可捕获时 without_piecewise()
    if not vllm_compile and has_mode(mode, PIECEWISE):
        mode = {'PIECEWISE': 'NONE', 'FULL_AND_PIECEWISE': 'FULL_DECODE_ONLY'}[mode]
        why.append('没有 piecewise 子图')
    return mode, '；'.join(why)


# --------------------------------------------------------------------------- 2. 捕获尺寸
def vllm_capture_sizes(max_num_seqs, max_num_batched_tokens, blackwell=False, query_len=1,
                       performance_mode='balanced', user_sizes=None, mrv1_spec_round=True):
    """config/vllm.py:2470–2625 _set_cudagraph_sizes（用户没给 max_cudagraph_capture_size 的分支）。"""
    default_max = 1024 if blackwell else 512
    max_size = min(max_num_seqs * query_len * 2, default_max)
    uniform = []
    if query_len > 1:
        def request_counts(max_reqs):
            max_reqs = min(max_reqs, default_max)
            c = [n for n in (1, 2, 4) if n <= max_reqs]
            c += list(range(8, min(max_reqs + 1, 256), 8))
            c += list(range(256, max_reqs + 1, 16))
            return sorted(set(c + [max_reqs]))
        uniform = sorted({n * query_len for n in request_counts(max_num_seqs) if n * query_len <= max_size})
    elif max_num_seqs <= max_size:
        uniform = [max_num_seqs]
    max_tokens = max_num_batched_tokens
    max_size = min(max_tokens, max_size)
    if user_sizes is not None:
        sizes = sorted(i for i in set(user_sizes) if i <= max_tokens)
    else:
        if performance_mode == 'interactivity':
            sizes = list(range(1, min(max_size, 32) + 1))
        else:
            sizes = [i for i in (1, 2, 4) if i <= max_size]
        if max_size >= 8:
            sizes += list(range(8, min(max_size + 1, 256), 8))
        if max_size >= 256:
            sizes += list(range(256, max_size + 1, 16))
        if max_tokens <= max_size and max_tokens not in sizes:
            sizes.append(max_tokens)
        sizes += [s for s in uniform if s <= max_tokens]
        sizes = sorted(set(sizes))
    # compilation.py:1537：MRv1 + FULL decode + 投机解码时取 query_len 的倍数
    if mrv1_spec_round and query_len > 1:
        top = sizes[-1]
        r = sorted({-(-s // query_len) * query_len for s in sizes if -(-s // query_len) * query_len <= top})
        sizes = r or [query_len]
    return sizes


def sglang_decode_sizes(max_bs, spec=False, disable_padding=False):
    """cuda_graph_hook.py:500 generate_decode_cuda_graph_batch_sizes。单位是请求数。"""
    if disable_padding:
        bs = list(range(1, max_bs + 1))
    elif not spec:
        bs = [1, 2, 4, 8, 12] + list(range(16, 257, 8)) + list(range(272, 512, 16)) + list(range(512, max_bs + 1, 32))
    else:
        bs = (list(range(1, 9)) + list(range(10, 33, 2)) + list(range(40, 65, 4))
              + list(range(72, 257, 8)) + list(range(272, max_bs + 1, 16)))
    bs = [b for b in bs if b <= max_bs]
    if max_bs not in bs:
        bs.append(max_bs)
    return bs


def sglang_prefill_sizes(max_bs):
    """cuda_graph_hook.py:480 generate_prefill_cuda_graph_batch_sizes。单位是 token。"""
    s = (list(range(4, 33, 4)) + list(range(48, 257, 16)) + list(range(288, 513, 32))
         + list(range(576, 1025, 64)) + list(range(1280, 4097, 256)) + list(range(4608, max_bs + 1, 512)))
    return [x for x in s if x <= max_bs]


def sglang_defaults(gpu_mem_gb, tp=1):
    """memory_hook.py:89 handle_gpu_memory_settings：(chunked_prefill_size, decode max_bs)。"""
    m = gpu_mem_gb * 1024
    if m < 20 * 1024:
        return 2048, 8
    if m < 35 * 1024:
        return 4096, 48 if tp < 4 else 160
    if m < 60 * 1024:
        return 4096, 32 if tp < 4 else 160
    if m < 90 * 1024:
        return 8192, 256 if tp < 4 else 512
    if m < 160 * 1024:
        return 8192, 256 if tp < 4 else 512
    return 16384, 512


def compress(sizes):
    """把列表压成 1, 2, 4, 8..248/8, 256..512/16 这种写法，便于打印。"""
    out, i = [], 0
    while i < len(sizes):
        j = i
        if i + 2 < len(sizes):
            step = sizes[i + 1] - sizes[i]
            while j + 1 < len(sizes) and sizes[j + 1] - sizes[j] == step:
                j += 1
            if j - i >= 2:
                out.append(f'{sizes[i]}..{sizes[j]}/{step}')
                i = j + 1
                continue
        out.append(str(sizes[i]))
        i += 1
    return ', '.join(out)


# --------------------------------------------------------------------------- 3. dispatcher
@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int
    num_reqs: int = None
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0


def lora_cases(max_loras=None, specialize_lora=True, specialize_active=False):
    """cudagraph_dispatcher.py _get_lora_cases + lora/utils.py:55。"""
    if max_loras is None:
        return [0]
    if specialize_lora:
        counts = ([n for n in range(1, max_loras + 2) if (n & (n - 1)) == 0 or n == max_loras + 1]
                  if specialize_active else [max_loras + 1])
        return [0] + counts
    return [max_loras + 1]


class Dispatcher:
    def __init__(self, mode, sizes, max_num_seqs, query_len=1, max_loras=None, specialize_active=False):
        self.mode, self.sizes, self.max_num_seqs, self.q = mode, sizes, max_num_seqs, query_len
        self.max_size = sizes[-1] if sizes else 0
        self.keys = {PIECEWISE: set(), FULL: set()}
        self.max_loras, self.spec_active = max_loras, specialize_active
        if mode == 'NONE':
            return
        # :71 查表：bs → ≥ bs 的最小捕获尺寸
        self.pad = [0] * (self.max_size + 1)
        for end, start in zip(sizes + [self.max_size + 1], [0] + sizes):
            for bs in range(start, end):
                self.pad[bs] = start if bs == start else end
        self.lora = lora_cases(max_loras, True, specialize_active)
        self.captured_lora = [c for c in self.lora if c]
        if mixed_mode(mode) != NONE:
            for bs in sizes:
                for la in self.lora:
                    d = self._desc(bs, False, la > 0, la)
                    if mixed_mode(mode) == PIECEWISE:
                        d = replace(d, num_reqs=None, uniform=False)
                    self.keys[mixed_mode(mode)].add(d)
        if decode_mode(mode) == FULL and mode in SEPARATE:
            top = query_len * max_num_seqs
            for bs in [x for x in sizes if query_len <= x <= top]:
                for la in self.lora:
                    self.keys[FULL].add(self._desc(bs, True, la > 0, la))

    def _desc(self, n, uniform, has_lora, nla=0):
        padded = self.pad[n]
        if uniform and has_mode(self.mode, FULL):
            return BatchDescriptor(padded, min(padded // self.q, self.max_num_seqs), True, has_lora, nla)
        return BatchDescriptor(padded, min(padded, self.max_num_seqs), False, has_lora, nla)

    def dispatch(self, num_tokens, uniform_decode=False, num_active_loras=0):
        if self.mode == 'NONE' or num_tokens > self.max_size:
            return NONE, BatchDescriptor(num_tokens)
        nla = num_active_loras
        if nla > 0:
            if self.spec_active:
                i = bisect_left(self.captured_lora, nla)
                if i < len(self.captured_lora):
                    nla = self.captured_lora[i]
            else:
                nla = self.max_loras + 1
        d = self._desc(num_tokens, uniform_decode and self.mode in SEPARATE, nla > 0, nla)
        if d in self.keys[FULL]:
            return FULL, d
        r = replace(d, num_reqs=None, uniform=False)
        if r in self.keys[PIECEWISE]:
            return PIECEWISE, r
        return NONE, BatchDescriptor(num_tokens)


def pad_to(sizes, n):
    """SGLang 的 bisect 补齐（decode_cuda_graph_runner.py _pad_to_bucket 同一思路）。"""
    i = bisect_left(sizes, n)
    return sizes[i] if i < len(sizes) else None


# --------------------------------------------------------------------------- 4. padding 统计
def padding_stats(sizes, dist):
    """dist: [(n, 权重)]。返回 (补齐 token 占比, 走图的 batch 占比, 平均补齐)。"""
    tot_w = sum(w for _, w in dist)
    real = padded = covered = 0.0
    for n, w in dist:
        p = pad_to(sizes, n)
        if p is None:
            continue
        covered += w
        real += n * w
        padded += (p - n) * w
    return padded / real if real else 0.0, covered / tot_w, padded / covered if covered else 0.0


def dist_uniform(lo, hi):
    return [(n, 1.0) for n in range(lo, hi + 1)]


def dist_geometric(mean, hi):
    """离散几何分布截断在 hi：低并发在线服务里 running 请求数的粗略形状。"""
    p = 1.0 / mean
    return [(n, p * (1 - p) ** (n - 1)) for n in range(1, hi + 1)]


# --------------------------------------------------------------------------- 5. 场景
def scenario_sizes():
    print('== 1. vLLM 默认捕获尺寸（单位 token；decode 一个请求 1 个 token，所以也等于 decode 的请求数）')
    cases = [
        ('H100 API server（max_num_seqs 1024，预算 8192）', 1024, 8192, False, 1, 'balanced'),
        ('A100 API server（256，预算 2048）', 256, 2048, False, 1, 'balanced'),
        ('B200（1024，预算 16384，上限 1024）', 1024, 16384, True, 1, 'balanced'),
        ('H100 --performance-mode interactivity', 1024, 8192, False, 1, 'interactivity'),
        ('H100 max_num_seqs 64', 64, 8192, False, 1, 'balanced'),
        ('H100 EAGLE 3 个草稿（query_len 4，256 条）', 256, 8192, False, 4, 'balanced'),
    ]
    for name, seqs, mnbt, bw, q, pm in cases:
        s = vllm_capture_sizes(seqs, mnbt, bw, q, pm)
        print(f'  {name}')
        print(f'    {len(s)} 个，max {s[-1]}：{compress(s)}')
    print()
    print('== 2. SGLang 默认 decode 捕获尺寸（单位：请求数）与 prefill 捕获尺寸（单位：token）')
    for name, mem, tp in [('H100 80G tp1', 80, 1), ('H100 80G tp8', 80, 8), ('B200 180G', 180, 1), ('RTX 4090 24G', 24, 1)]:
        cps, mbs = sglang_defaults(mem, tp)
        d = sglang_decode_sizes(mbs)
        print(f'  {name}：chunked_prefill_size {cps}，decode max_bs {mbs}')
        print(f'    decode {len(d)} 个：{compress(d)}')
    p = sglang_prefill_sizes(8192)
    print(f'  prefill（max_bs = chunked_prefill_size 8192）{len(p)} 个：{compress(p)}')
    sp = sglang_decode_sizes(256, spec=True)
    print(f'  投机解码 decode（max_bs 256）{len(sp)} 个：{compress(sp)}')


def scenario_resolve():
    print('== 后端能力 × 请求的 cudagraph_mode → 实际模式（VLLM_COMPILE，splitting_ops 含注意力）')
    req = ['FULL_AND_PIECEWISE', 'FULL', 'FULL_DECODE_ONLY']
    hdr = f'  {"后端":<22}{"AttentionCGSupport":<30}' + ''.join(f'{r:<22}' for r in req) + '投机解码 q=4'
    print(hdr)
    for name, sup in BACKENDS:
        row = f'  {name:<22}{SUPPORT_NAME[sup]:<30}'
        for r in req:
            m, _ = resolve_mode(r, sup)
            row += f'{m:<22}'
        m, why = resolve_mode('FULL_AND_PIECEWISE', sup, query_len=4)
        row += m
        print(row)
    print()
    print('  compilation mode 不是 VLLM_COMPILE（比如 mode=0 但仍要 CUDA graph）时没有 piecewise 子图可退：')
    for name, sup in BACKENDS:
        m, why = resolve_mode('FULL_AND_PIECEWISE', sup, vllm_compile=False, split_attention=False)
        print(f'    {name:<22} FULL_AND_PIECEWISE → {m}')


def scenario_dispatch():
    sizes = vllm_capture_sizes(1024, 8192)
    d = Dispatcher('FULL_AND_PIECEWISE', sizes, 1024)
    print('== H100 默认：FULL_AND_PIECEWISE，max_num_seqs 1024，捕获尺寸 51 个，max 512')
    cases = [('decode 1 条', 1, True), ('decode 37 条', 37, True), ('decode 300 条', 300, True),
             ('decode 512 条', 512, True), ('decode 600 条', 600, True),
             ('混合 37 token（1 个短 prefill + decode）', 37, False), ('混合 300 token', 300, False),
             ('混合 2048 token（chunked prefill）', 2048, False)]
    for name, n, u in cases:
        m, bd = d.dispatch(n, u)
        if m == NONE:
            print(f'  {name:<34} → NONE   eager（超过 max {d.max_size}），不补齐')
        else:
            print(f'  {name:<34} → {m:<9} 补到 {bd.num_tokens}（+{bd.num_tokens - n}），num_reqs={bd.num_reqs}，uniform={bd.uniform}')
    print()
    print('== 同样的 batch，其他模式')
    for mode in ['PIECEWISE', 'FULL_DECODE_ONLY', 'FULL']:
        dd = Dispatcher(mode, sizes, 1024)
        out = []
        for name, n, u in [('decode 37', 37, True), ('混合 37', 37, False), ('混合 300', 300, False)]:
            m, bd = dd.dispatch(n, u)
            out.append(f'{name} → {m}' + (f' {bd.num_tokens}' if m != NONE else ''))
        print(f'  {mode:<20} ' + '；'.join(out))
    print()
    print('== SGLang H100 tp1：decode max_bs 256，36 个尺寸')
    sg = sglang_decode_sizes(256)
    for n in (1, 37, 300):
        p = pad_to(sg, n)
        print(f'  decode {n:>3} 条 → ' + (f'图 bs={p}（+{p - n}）' if p else '超过 256，不用图'))


def scenario_padding():
    v = vllm_capture_sizes(1024, 8192)
    vi = vllm_capture_sizes(1024, 8192, performance_mode='interactivity')
    s1 = sglang_decode_sizes(256)
    s8 = sglang_decode_sizes(512)
    v64 = vllm_capture_sizes(64, 8192)
    lists = [('vLLM 默认 (51)', v), ('vLLM interactivity', vi), ('vLLM max_num_seqs 64', v64),
             ('SGLang tp1 (36)', s1), ('SGLang tp≥4 (52)', s8)]
    dists = [('decode 1–32 均匀', dist_uniform(1, 32)), ('decode 几何 均值 12', dist_geometric(12, 1024)),
             ('decode 1–256 均匀', dist_uniform(1, 256)), ('decode 1–1024 均匀', dist_uniform(1, 1024))]
    print('== decode batch 的 padding 浪费（补的 token / 真实 token）与走图比例')
    print(f'  {"分布":<20}' + ''.join(f'{n:<24}' for n, _ in lists))
    for dn, dist in dists:
        row = f'  {dn:<20}'
        for _, s in lists:
            waste, cov, _ = padding_stats(s, dist)
            row += f'{waste * 100:5.1f}% / 走图 {cov * 100:5.1f}%    '
        print(row)
    print()
    print('== interactivity 只改 1–32 段：逐个 batch 看补齐')
    for n in (3, 5, 9, 13, 17, 33):
        print(f'  bs={n:<3} 默认 → {pad_to(v, n):<4} interactivity → {pad_to(vi, n):<4} SGLang → {pad_to(s1, n)}')


def scenario_keys(layers=36):
    sizes = vllm_capture_sizes(1024, 8192)
    print(f'== 每种模式的 dispatcher 键（H100 默认 51 个尺寸，模型 {layers} 层注意力）')
    print('   PIECEWISE 键每个对应一次分段前向：注意力把图切成约 层数 + 1 段，每段一个 torch.cuda.CUDAGraph')
    for mode in ['PIECEWISE', 'FULL', 'FULL_DECODE_ONLY', 'FULL_AND_PIECEWISE']:
        d = Dispatcher(mode, sizes, 1024)
        npw, nfull = len(d.keys[PIECEWISE]), len(d.keys[FULL])
        print(f'  {mode:<20} PIECEWISE 键 {npw:>3}（≈ {npw * (layers + 1):>4} 段图）  FULL 键 {nfull:>3}')
    print()
    print('== LoRA：键数按 LoRA 情况翻倍（cudagraph_specialize_lora 默认 True）')
    for ml, sa in [(None, False), (4, False), (4, True), (16, True)]:
        d = Dispatcher('FULL_AND_PIECEWISE', sizes, 1024, max_loras=ml, specialize_active=sa)
        tag = '无 LoRA' if ml is None else f'max_loras {ml}' + ('，specialize_active_lora' if sa else '')
        print(f'  {tag:<36} lora 情况 {d.lora}  → PIECEWISE {len(d.keys[PIECEWISE])} + FULL {len(d.keys[FULL])}')
    print()
    print('== 显存估算（profile_cudagraph_memory）：每种模式只真捕获最大的两个，其余线性外推')
    d = Dispatcher('FULL_AND_PIECEWISE', sizes, 1024)
    for mode in (PIECEWISE, FULL):
        n = len(d.keys[mode])
        print(f'  {mode:<9} n={n}：估算 = 第 1 个 + {n - 1} × max(第 2 个, 1 MiB) ≥ 第 1 个 + {n - 1} MiB')
    print('  捕获顺序：PIECEWISE 先、FULL 后，各自从大到小，小图复用大图留下的显存池')


def cache_dir(env_hash, config_hash, code_hash, compiler_hash, root='~/.cache/vllm', rank=0, dp=0, prefix='backbone'):
    factors = [env_hash, config_hash, code_hash, compiler_hash]
    key = hashlib.sha256(str(factors).encode()).hexdigest()[:10]
    return f'{root}/torch_compile_cache/{key}/rank_{rank}_{dp}/{prefix}'


def compile_ranges(endpoints):
    e = sorted(set(endpoints))
    return [(s + 1, t) for s, t in zip([0] + e[:-1], e)]


def scenario_cache():
    print('== 编译缓存目录 = sha256(str([env, config, code, compiler]))[:10]（四个因子用示例字符串）')
    base = ('env-a', 'cfg-qwen3-8b-tp1', 'code-v0', 'torch-2.10')
    print(f'  基线                 {cache_dir(*base)}')
    print(f'  换 max_num_seqs      {cache_dir(base[0], "cfg-qwen3-8b-tp1-seqs256", *base[2:])}')
    print(f'  改了模型文件一行     {cache_dir(*base[:2], "code-v1", base[3])}')
    print(f'  升级 torch           {cache_dir(*base[:3], "torch-2.11")}')
    print(f'  TP=2 的 rank 1       {cache_dir(*base, rank=1)}')
    print()
    print('== compile range：endpoints 切成左开右闭的区间，每个区间编译一份 Inductor 产物')
    for ep in ([8192], [2048], [512, 8192]):
        print(f'  endpoints {ep!s:<14} → ' + ', '.join(f'[{a}, {b}]' for a, b in compile_ranges(ep)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='sizes', choices=['sizes', 'resolve', 'dispatch', 'padding', 'keys', 'cache'])
    a = ap.parse_args()
    {'sizes': scenario_sizes, 'resolve': scenario_resolve, 'dispatch': scenario_dispatch,
     'padding': scenario_padding, 'keys': scenario_keys, 'cache': scenario_cache}[a.scenario]()


if __name__ == '__main__':
    main()
