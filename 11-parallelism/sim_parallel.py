"""并行切分模拟：vLLM 的 TP / PP / DP / EP / EPLB / DCP，对照 SGLang（只依赖标准库）。

对照的源码（vllm 主干 0d6f92a，sglang 主干 8fabf5e）：
  vllm   distributed/parallel_state.py:2150–2370  initialize_model_parallel：rank 排布 ExternalDP x DP x PP x PCP x TP，
                                                 每个维度 transpose 到最后再 reshape 成组；EP 只给 MoE 模型建
         config/parallel.py:137–470        ParallelConfig 字段（dp_sync_interval=16、all2all_backend、enable_dbo、DCP）
         config/parallel.py:542–630        _validate_parallel_config：EPLB 需要 EP、冗余专家需要 EPLB、tp % dcp
         config/parallel.py:889–914        sync_dp_state：一次 SUM all-reduce 同步 "还有没有活" 和 pause
         config/parallel.py:1050–1054      离线 DP 对稠密模型直接报错
         config/model.py:1585–1613         GQA 的 DCP 校验：tp > kv_heads、dcp <= tp // kv_heads、q_per_kv % dcp == 0
         config/model.py:1690              每卡 KV 头数 max(1, kv_heads // tp)：TP 超过 KV 头数时复制
         config/vllm.py:603–614            max_concurrent_batches：PP 需要 pp 个 batch 在飞，MRv2 异步再 +1
         distributed/utils.py:198–240      get_pp_indices：余数层分给倒数第 2、3… 段，可用 VLLM_PP_LAYER_PARTITION 覆盖
         distributed/eplb/policy/default.py  balanced_packing / replicate_experts / rebalance_experts_hierarchical
         distributed/eplb/eplb_state.py:66–71, 466–470, 561–716, 850  balancedness=avg/max、首次重排在 3/4 处、窗口记录
         model_executor/layers/fused_moe/router/base_router.py:45–60  token 到副本：Knuth 乘法哈希 % 副本数
         model_executor/layers/fused_moe/config.py:1120   FusedMoEParallelConfig.make：开 EP 时 ep = tp*dp*pcp，否则 MoE 按 tp*dp 做 TP
         v1/engine/core.py:1594–1603, 2567–2672  只有 MoE 走 DPEngineCoreProc；空闲 rank 跑 dummy batch；每 16 步同步一次
         v1/worker/dp_utils.py:122–187     一次 all-reduce 决定 ubatch、取最小 cudagraph 模式、按最大 token 数补齐
         v1/worker/ubatch_utils.py:129–148 DBO 阈值：decode 32 token、prefill 512 token
  sglang srt/arg_groups/fields/parallel.py:45–295   tp_size / dp_size / attn_dp_size / ep_size / moe_dense_tp_size / enable_dp_lm_head
         srt/arg_groups/parallel_hook.py:276–355     DP attention：schedule_conservativeness x0.3，chunked_prefill_size //= attn_dp_size
         srt/arg_groups/parallel_hook.py:750–790     EPLB 自动开 expert_distribution_recorder_mode=stat
         srt/arg_groups/overrides.py:1578–1621       deepep / mooncake / nixl 等 a2a 后端把 ep_size 改成 tp_size
         srt/arg_groups/fields/exec_.py:917–1068     ep_num_redundant_experts、eplb_rebalance_num_iterations=1000、TBO

假设：DeepSeek-V3 线性层 FP8（1 字节），embedding / lm_head / gate BF16；Llama、Qwen3 全 BF16；忽略 norm 和 FP8 scale；
每卡激活 + 非 torch + CUDA Graph 合计记 6 GiB（02-ledger 的实测量级）；KV cache 用 BF16。
通信只算字节，时间是 NVLink 单向 450 GB/s 的带宽下限，没有算启动延迟。

用法：
  python3 sim_parallel.py --scenario layout   # rank 排布：每种并行组里是哪些 GPU
  python3 sim_parallel.py --scenario memory   # 每卡权重 / KV 余量 / 整组能放多少 token
  python3 sim_parallel.py --scenario comm     # 每步通信次数和字节：TP all-reduce、EP dispatch/combine、PP send
  python3 sim_parallel.py --scenario dcp      # DCP 合法性和每卡 KV 字节
  python3 sim_parallel.py --scenario eplb     # 复现 EPLB 策略：倾斜负载下 balancedness 前后对比
  python3 sim_parallel.py --scenario dp       # DP wave：dummy batch、每 16 步同步、CUDA Graph 补齐
  python3 sim_parallel.py --scenario pp       # get_pp_indices 切层和每段显存
  python3 sim_parallel.py --scenario sglang   # 同样的 8 卡，vLLM 和 SGLang 的参数怎么写、默认值差在哪
"""
import argparse
import itertools
import math
import os
import random
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, MODELS  # noqa: E402

GiB = 1 << 30
OVERHEAD_GIB = 6.0
NVLINK_GBS = 450e9


# --------------------------------------------------------------------------- 1. rank 排布
def _arange_groups(shape, perm, size):
    """等价于 torch.arange(prod(shape)).reshape(shape).permute(perm).reshape(-1, size)。"""
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    pshape = [shape[p] for p in perm]
    flat = [sum(idx[j] * strides[perm[j]] for j in range(len(perm)))
            for idx in itertools.product(*[range(s) for s in pshape])]
    return [flat[i:i + size] for i in range(0, len(flat), size)]


def _swap(n, a, b):
    p = list(range(n))
    p[a], p[b] = p[b], p[a]
    return p


def build_groups(world, dp=1, pp=1, pcp=1, tp=1, dcp=1, is_moe=False):
    """复现 initialize_model_parallel 的分组。world 可以大于 dp*pp*pcp*tp（多出来的是 ExternalDP）。"""
    assert world % (dp * pp * pcp * tp) == 0
    shape = (world // (dp * pp * pcp * tp), dp, pp, pcp, tp)
    g = {'tp': _arange_groups(shape, list(range(5)), tp)}
    g['dcp'] = _arange_groups(shape, _swap(5, 3, 4) if dcp > 1 else list(range(5)), dcp)
    g['pcp'] = _arange_groups(shape, _swap(5, 3, 4), pcp)
    g['pp'] = _arange_groups(shape, _swap(5, 2, 4), pp)
    g['dp'] = _arange_groups(shape, _swap(5, 1, 4), dp)
    if is_moe:   # "Don't create EP group for dense models."
        g['ep'] = _arange_groups(shape, _swap(5, 1, 2), dp * pcp * tp)
    return g


# --------------------------------------------------------------------------- 2. 模型参数拆账
@dataclass(frozen=True)
class Arch:
    name: str
    layers: int
    hidden: int
    vocab: int
    attn_rep: int          # TP 下每卡都要完整复制的注意力参数（MLA 的 q_a_proj、kv_a_proj_with_mqa）
    attn_shard: int        # 按 TP 切的注意力参数（不含 GQA 的 K/V 投影）
    kv_proj: int           # GQA 的 K+V 投影参数（TP > kv_heads 时按 max(1, kv//tp) 头复制）
    kv_heads: int
    head_dim: int
    dense_layers: int      # 稠密 MLP 层数
    dense_mlp: int         # 每个稠密 MLP 的参数
    experts: int           # 每层路由专家数，稠密模型为 0
    expert: int            # 每个专家的参数
    shared: int            # 每层共享专家参数
    topk: int
    lin_bytes: float       # 线性层每参数字节
    mla_kv: int = 0        # MLA 每层每 token 的 latent + rope 维度，0 表示 GQA

    @property
    def moe_layers(self):
        return self.layers - self.dense_layers if self.experts else 0

    def kv_bytes_full(self):
        """一个 token 在所有层的 KV，BF16。"""
        if self.mla_kv:
            return self.layers * self.mla_kv * 2
        return 2 * self.layers * self.kv_heads * self.head_dim * 2

    def total_params(self):
        return (2 * self.vocab * self.hidden + self.layers * (self.attn_rep + self.attn_shard + self.kv_proj)
                + self.dense_layers * self.dense_mlp
                + self.moe_layers * (self.experts * self.expert + self.shared + self.hidden * self.experts))


def _deepseek_v3():
    h, heads = 7168, 128
    q_a, q_b = h * 1536, 1536 * heads * (128 + 64)
    kv_a, kv_b = h * (512 + 64), 512 * heads * (128 + 128)
    o = heads * 128 * h
    e = 3 * h * 2048
    return Arch('DeepSeek-V3', 61, h, 129280, q_a + kv_a, q_b + kv_b + o, 0, 128, 128,
                3, 3 * h * 18432, 256, e, e, 8, 1.0, mla_kv=512 + 64)


def _qwen3_235b():
    h = 4096
    return Arch('Qwen3-235B-A22B', 94, h, 151936, 0, h * 64 * 128 * 2, h * 4 * 128 * 2, 4, 128,
                0, 0, 128, 3 * h * 1536, 0, 8, 2.0)


def _dense(key, inter):
    m = MODELS[key]
    return Arch(m.name, m.layers, m.hidden, m.vocab, 0, m.hidden * m.heads * m.head_dim * 2,
                m.hidden * m.kv_heads * m.head_dim * 2, m.kv_heads, m.head_dim,
                m.layers, 3 * m.hidden * inter, 0, 0, 0, 0, 2.0)


Q_HEADS = {'dsv3': 128, 'qwen3-235b': 64, 'llama70b': 64}   # 注意力 Q 头数，DCP 校验 q_per_kv % dcp 用
ARCHS = {'dsv3': _deepseek_v3(), 'qwen3-235b': _qwen3_235b(), 'llama70b': _dense('llama3.1-70b', 28672)}


@dataclass(frozen=True)
class Plan:
    label: str
    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: bool = False
    dcp: int = 1
    redundant: int = 0

    @property
    def gpus(self):
        return self.tp * self.pp * self.dp


def pp_partition(layers, pp):
    """distributed/utils.py get_pp_indices，未设 VLLM_PP_LAYER_PARTITION 时。"""
    parts = [layers // pp] * pp
    for i in range(2, layers % pp + 2):
        parts[-i] += 1
    return parts


def kv_heads_per_rank(a: Arch, tp):
    return max(1, a.kv_heads // tp)


def kv_bytes_per_rank(a: Arch, p: Plan, stage_layers=None):
    """每张卡为一个 token 存的 KV 字节（BF16）。MLA 每个 TP rank 存完整 latent；DCP 再按 dcp 切 token。"""
    L = stage_layers if stage_layers is not None else a.layers
    if a.mla_kv:
        b = L * a.mla_kv * 2
    else:
        b = 2 * L * kv_heads_per_rank(a, p.tp) * a.head_dim * 2
    return b / p.dcp


def weight_bytes_per_gpu(a: Arch, p: Plan, stage=0):
    parts = pp_partition(a.layers, p.pp)
    L = parts[stage]
    first = sum(parts[:stage])
    dense_here = max(0, min(a.dense_layers, first + L) - first) if a.experts else L
    moe_here = L - dense_here if a.experts else 0
    lb = a.lin_bytes
    w = 0.0
    if stage == 0:
        w += a.vocab * a.hidden * 2 / p.tp                       # VocabParallelEmbedding
    if stage == p.pp - 1:
        w += a.vocab * a.hidden * 2 / p.tp                       # ParallelLMHead
    kv_w = a.kv_proj * kv_heads_per_rank(a, p.tp) / a.kv_heads if a.kv_heads and a.kv_proj else 0
    w += L * (a.attn_rep + a.attn_shard / p.tp + kv_w) * lb
    w += dense_here * a.dense_mlp / p.tp * lb
    if moe_here:
        moe_ranks = p.tp * p.dp                                  # 开不开 EP，路由专家都按 tp*dp 摊
        n_phys = a.experts + p.redundant
        w += moe_here * (n_phys * a.expert / moe_ranks) * lb
        w += moe_here * (a.shared / p.tp * lb + a.hidden * a.experts * 2)
    return w


def ledger(a: Arch, p: Plan, gpu_key, util=0.9):
    gpu = GPUS[gpu_key]
    requested = gpu.mem_gb * 0.99 * GiB * util
    parts = pp_partition(a.layers, p.pp)
    rows = []
    for s in range(p.pp):
        w = weight_bytes_per_gpu(a, p, s)
        free = requested - w - OVERHEAD_GIB * GiB
        kvb = kv_bytes_per_rank(a, p, parts[s])
        rows.append((w, free, kvb, free / kvb if free > 0 else 0))
    tokens_per_engine = min(r[3] for r in rows)
    replicas = p.dp
    return dict(weights=max(r[0] for r in rows), free=min(r[1] for r in rows),
                kv_per_tok=max(r[2] for r in rows), tokens_engine=tokens_per_engine,
                tokens_total=tokens_per_engine * replicas)


# --------------------------------------------------------------------------- 3. 通信
def expected_ranks_hit(experts, ranks, k):
    """top-k 个不同专家均匀落在 ranks 个 rank 上（每个 rank experts/ranks 个），期望命中多少个 rank。"""
    per = experts // ranks
    miss = math.comb(experts - per, k) / math.comb(experts, k)
    return ranks * (1 - miss)


def ring_allreduce_bytes(msg, n):
    return 2 * (n - 1) / n * msg if n > 1 else 0


# --------------------------------------------------------------------------- 4. EPLB（default policy 的纯 Python 版）
def balanced_packing(weights, num_packs):
    n = len(weights)
    per = n // num_packs
    if per == 1:
        return list(range(n)), [0] * n
    order = sorted(range(n), key=lambda i: -weights[i])
    pack_w, pack_n = [0.0] * num_packs, [0] * num_packs
    pack_index, rank_in = [-1] * n, [-1] * n
    for i in order:
        pk = min(range(num_packs), key=lambda j: pack_w[j])   # argmin，满的包已是 inf
        pack_index[i], rank_in[i] = pk, pack_n[pk]
        pack_w[pk] += weights[i]
        pack_n[pk] += 1
        if pack_n[pk] == per:
            pack_w[pk] = math.inf
    return pack_index, rank_in


def replicate_experts(weights, num_phy):
    num_log = len(weights)
    phy2log = list(range(num_log))
    cnt = [1] * num_log
    for _ in range(num_log, num_phy):
        j = max(range(num_log), key=lambda e: weights[e] / cnt[e])
        phy2log.append(j)
        cnt[j] += 1
    return phy2log, cnt


def rebalance_hierarchical(weights, num_phy, num_groups, num_nodes, num_gpus):
    """返回 phy2log（长度 num_phy），按 GPU 顺序连续排列，每卡 num_phy/num_gpus 个槽。"""
    E = len(weights)
    gsize = E // num_groups
    gpn = num_groups // num_nodes
    per_gpu = num_phy // num_gpus
    # Step 1：专家组装进节点
    gw = [sum(weights[g * gsize:(g + 1) * gsize]) for g in range(num_groups)]
    gpack, grank = balanced_packing(gw, num_nodes)
    log2mlog = [0] * E
    for e in range(E):
        g = e // gsize
        log2mlog[e] = (gpack[g] * gpn + grank[g]) * gsize + e % gsize
    mlog2log = [0] * E
    for e, m in enumerate(log2mlog):
        mlog2log[m] = e
    out = []
    per_node_log = E // num_nodes
    per_node_phy = num_phy // num_nodes
    gpus_per_node = num_gpus // num_nodes
    for node in range(num_nodes):
        w_node = [weights[mlog2log[node * per_node_log + i]] for i in range(per_node_log)]
        # Step 2：节点内复制热点专家
        phy2m, cnt = replicate_experts(w_node, per_node_phy)
        # Step 3：物理专家（负载 = 逻辑负载 / 副本数）装进节点内的 GPU
        w_phy = [w_node[m] / cnt[m] for m in phy2m]
        pk, rk = balanced_packing(w_phy, gpus_per_node)
        slots = [None] * per_node_phy
        for i in range(per_node_phy):
            slots[pk[i] * per_gpu + rk[i]] = phy2m[i]
        out += [mlog2log[node * per_node_log + m] for m in slots]
    return out


def rebalance(weights, num_phy, num_groups, num_nodes, num_gpus):
    if num_gpus % num_nodes:                      # eplb_state.py:850
        num_nodes = 1
    if num_groups % num_nodes == 0:
        return rebalance_hierarchical(weights, num_phy, num_groups, num_nodes, num_gpus)
    return rebalance_hierarchical(weights, num_phy, 1, 1, num_gpus)


def gpu_loads(weights, phy2log, num_gpus):
    """副本之间按哈希均分负载（base_router.py 的 Knuth 哈希 % 副本数，大 batch 下近似均分）。"""
    cnt = {}
    for e in phy2log:
        cnt[e] = cnt.get(e, 0) + 1
    per = len(phy2log) // num_gpus
    return [sum(weights[e] / cnt[e] for e in phy2log[g * per:(g + 1) * per]) for g in range(num_gpus)]


def linear_placement(E, num_gpus):
    """不开 EPLB：专家 e 放在第 e // (E/num_gpus) 张卡，即物理顺序 = 逻辑顺序。"""
    assert E % num_gpus == 0
    return list(range(E))


def round_robin_placement(E, num_gpus):
    per = E // num_gpus
    return [g + i * num_gpus for g in range(num_gpus) for i in range(per)]


def balancedness(loads_per_layer):
    """eplb_state.py：avg 和 max 先各自在层上求和再相除。"""
    avg = sum(sum(l) / len(l) for l in loads_per_layer)
    mx = sum(max(l) for l in loads_per_layer)
    return avg / mx


def skewed_load(E, tokens, seed, sigma=1.0):
    rng = random.Random(seed)
    raw = [rng.lognormvariate(0, sigma) for _ in range(E)]
    s = sum(raw)
    return [tokens * r / s for r in raw]


# --------------------------------------------------------------------------- 5. DP
CAPTURE = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))   # compilation.py:703，max 512


def cg_pad(n):
    for c in CAPTURE:
        if c >= n:
            return c, True
    return n, False


def dp_step(tokens):
    """dp_utils._synchronize_dp_ranks：各 rank 先按自己的 CG 尺寸补齐，再取最小 CG 模式；不是 NONE 就补到最大值。"""
    padded, modes = [], []
    for n in tokens:
        n_eff = max(n, 1)                      # 没活的 rank 跑 1 token 的 dummy batch
        p, cg = cg_pad(n_eff)
        padded.append(p)
        modes.append(2 if cg else 0)
    synced = min(modes)
    if synced != 0:
        final = [max(padded)] * len(tokens)
    else:
        final = [max(n, 1) for n in tokens]
    return final, synced


def dp_wave(finish_steps, sync_interval=16):
    """每个 rank 在 finish_steps[r] 步之后就没活了；返回 wave 结束的步数和每个 rank 的 dummy 步数。"""
    last = max(finish_steps)
    step = 0
    while True:
        step += 1
        if step == 1 or step % sync_interval == 0:
            if step >= last:
                break
    return step, [step - f for f in finish_steps]


# --------------------------------------------------------------------------- 场景
def scenario_layout():
    cases = [('TP2 x PP4（parallel_state.py docstring 的例子）', 8, dict(tp=2, pp=4)),
             ('DP2 x PP2 x TP2，MoE 模型', 8, dict(dp=2, pp=2, tp=2, is_moe=True)),
             ('DP2 x TP4 + DCP2，MoE 模型', 8, dict(dp=2, tp=4, dcp=2, is_moe=True)),
             ('TP8 x DP2（两机 16 卡，稠密模型）', 16, dict(dp=2, tp=8))]
    for title, world, kw in cases:
        print(f'== {title}  world={world}')
        g = build_groups(world, **kw)
        for k in ('tp', 'pp', 'dp', 'dcp', 'ep'):
            if k in g and len(g[k][0]) > 1:
                print(f'  {k:<4} {g[k]}')
        print()
    print('读法：rank = ((dp*PP + pp)*PCP + pcp)*TP + tp，TP 在最内层，所以一个 TP 组总是相邻的 GPU（同一台机器的 NVLink）。')


def scenario_memory():
    cases = [
        ('llama70b', 'h100-sxm', [Plan('TP8', tp=8), Plan('TP4 x DP2', tp=4, dp=2), Plan('TP2 x DP4', tp=2, dp=4),
                                  Plan('PP2 x TP4', tp=4, pp=2), Plan('PP8', pp=8)]),
        ('dsv3', 'h200', [Plan('TP8', tp=8), Plan('TP8 + EP', tp=8, ep=True), Plan('DP8 + EP8', dp=8, ep=True),
                          Plan('DP8 + EP8 + 32 冗余', dp=8, ep=True, redundant=32), Plan('TP8 + DCP8', tp=8, dcp=8),
                          Plan('DP16 + EP16（两机）', dp=16, ep=True)]),
        ('qwen3-235b', 'h200', [Plan('TP8', tp=8), Plan('TP8 + DCP2', tp=8, dcp=2), Plan('TP4 x DP2 + EP', tp=4, dp=2, ep=True),
                                Plan('DP8 + EP8', dp=8, ep=True)]),
    ]
    for key, g, plans in cases:
        a = ARCHS[key]
        dt = 'FP8 线性层' if a.lin_bytes == 1 else 'BF16'
        print(f'== {a.name}（{a.total_params() / 1e9:.1f}B，{dt}）@ {GPUS[g].name}，util 0.9，每卡杂项 {OVERHEAD_GIB:.0f} GiB')
        print(f'{"方案":<22} {"GPU":>4} {"每卡权重 GiB":>12} {"每卡 KV 余量 GiB":>16} {"每卡每 token KV B":>17} '
              f'{"一个引擎 token":>14} {"全部 GPU token":>15}')
        for p in plans:
            r = ledger(a, p, g)
            print(f'{p.label:<22} {p.gpus:>4} {r["weights"] / GiB:>12.1f} {r["free"] / GiB:>16.1f} '
                  f'{r["kv_per_tok"]:>17,.0f} {r["tokens_engine"]:>14,.0f} {r["tokens_total"]:>15,.0f}')
        print()
    print('DP 行的 "一个引擎" 指一个 DP rank；"全部 GPU" 是所有 DP rank 加起来。MLA 在 TP 下每卡存完整 latent，'
          '8 张卡存的是同一批 token。')


def scenario_comm():
    print('== 每个 decode step 的集合通信（一个 rank 的视角，batch = 每个 DP rank 上的 token 数）')
    print(f'{"配置":<34} {"次数":>5} {"每次消息":>10} {"本卡发送 MiB":>13} {"带宽下限 us":>12} {"全组每 token MiB":>16}')
    rows = []
    # Llama-70B TP8：每层 o_proj 后、down_proj 后各一次 all-reduce，外加 embedding 一次
    a = ARCHS['llama70b']
    for B in (64, 256):
        msg = B * a.hidden * 2
        n = 2 * a.layers + 1
        sent = n * ring_allreduce_bytes(msg, 8)
        rows.append((f'Llama-70B TP8 all-reduce B={B}', n, msg, sent, 8, B))
    # Llama-70B PP4：每个段边界送 hidden + residual
    for B in (64, 256):
        msg = B * a.hidden * 2 * 2
        rows.append((f'Llama-70B PP4 send B={B}', 3, msg, msg, 4, B))
    d = ARCHS['dsv3']
    for B in (64, 256, 512):
        msg = B * d.hidden * 2
        n = 2 * d.layers + 1
        rows.append((f'DSV3 TP8 all-reduce B={B}', n, msg, n * ring_allreduce_bytes(msg, 8), 8, B))
    for B in (64, 256):
        for R in (8, 16):
            hit = expected_ranks_hit(d.experts, R, d.topk)
            remote = hit * (R - 1) / R
            disp = d.hidden + d.hidden // 128 * 4          # FP8 + 每 128 维一个 fp32 scale
            comb = d.hidden * 2
            per_tok = remote * (disp + comb)
            n = 2 * d.moe_layers
            rows.append((f'DSV3 DP{R}+EP{R} a2a B={B}', n, B * remote * disp, d.moe_layers * B * per_tok, R, R * B))
    for label, n, msg, sent, ranks, tok in rows:
        print(f'{label:<34} {n:>5} {msg / 1024:>8.0f}KiB {sent / 2**20:>13.1f} {sent / NVLINK_GBS * 1e6:>12.0f} '
              f'{sent * ranks / tok / 2**20:>16.2f}')
    print()
    print('== DeepSeek-V3 top-8 / 256 专家：一个 token 平均要发往几个远端 rank')
    for R in (8, 16, 32, 64):
        hit = expected_ranks_hit(d.experts, R, d.topk)
        print(f'  EP{R:<3} 命中 {hit:5.2f} 个 rank，其中远端 {hit * (R - 1) / R:5.2f}；按专家发不去重则远端 {d.topk * (R - 1) / R:4.2f} 份')
    print()
    print('TP 的 all-reduce 次数和层数成正比、和 batch 无关；EP 的字节数和 batch、top-k 成正比；PP 只在段边界送一次。')
    print('"全组每 token" = 所有 rank 发送量之和 / 全组这一步处理的 token 数：TP8 的 B 是全组的 batch，DP8 的 B 是每个 rank 的 batch。')


def scenario_dcp():
    cases = [('qwen3-235b', 8, (1, 2, 4)), ('llama70b', 8, (2,)), ('llama70b', 16, (2,)), ('dsv3', 8, (1, 2, 4, 8))]
    print(f'{"模型":<18} {"TP":>3} {"DCP":>4} {"结果":<44} {"每卡每 token KV B":>17}')
    for key, tp, dcps in cases:
        a = ARCHS[key]
        heads = Q_HEADS[key]
        for dcp in dcps:
            err = None
            if tp % dcp:
                err = f'tp_size={tp} must be divisible by dcp_size={dcp}'
            elif dcp > 1 and not a.mla_kv:
                if tp <= a.kv_heads:
                    err = f'requires --tensor-parallel-size ({tp}) > KV heads ({a.kv_heads})'
                elif dcp > tp // a.kv_heads:
                    err = f'exceeds the maximum supported value ({tp // a.kv_heads})'
                elif (heads // a.kv_heads) % dcp:
                    err = f'q_per_kv {heads // a.kv_heads} not divisible by {dcp}'
            kvb = kv_bytes_per_rank(a, Plan('', tp=tp, dcp=dcp))
            rep = '' if a.mla_kv else f'，KV 头复制 {tp // a.kv_heads if tp > a.kv_heads else 1}x'
            res = ('ERROR ' + err) if err else ('OK' + (f'，{dcp} 张卡分摊同一条序列' if dcp > 1 else '') + rep)
            print(f'{a.name:<18} {tp:>3} {dcp:>4} {res:<44} {"-" if err else f"{kvb:,.0f}":>17}')
    print()
    print('GQA：TP 超过 KV 头数时多出来的卡只能复制 KV 头；DCP 把这份复制改成按 token 切，最大 dcp = tp // kv_heads。')
    print('MLA：每个 TP rank 本来就存完整 latent（等价于 1 个 KV 头），所以 dcp 最多到 tp。')


def scenario_eplb():
    E, G, layers = 256, 8, 4      # DeepSeek-V3：256 路由专家，n_group 8
    tokens = 4096 * 8             # 一个 step 4096 token x top-8
    loads = [skewed_load(E, tokens, seed=11 + l, sigma=1.0) for l in range(layers)]
    print(f'== 合成负载：{layers} 个 MoE 层，每层 {E} 专家，对数正态 sigma=1.0，一个 step {tokens:,} 次专家调用')
    print(f'   最热专家 / 平均 = {max(loads[0]) / (tokens / E):.1f}x（第 0 层）')
    print(f'{"EP":>4} {"节点":>4} {"放置":<26} {"物理专家":>8} {"balancedness":>13} {"最忙卡 / 平均":>13}')
    for gpus, nodes in ((16, 2), (32, 4), (64, 8)):
        variants = [('linear（默认，不开 EPLB）', None, 0), ('round_robin', 'rr', 0),
                    ('EPLB 0 冗余', 'eplb', 0), (f'EPLB {gpus} 冗余（每卡 +1）', 'eplb', gpus),
                    (f'EPLB {2 * gpus} 冗余（每卡 +2）', 'eplb', 2 * gpus)]
        for label, kind, red in variants:
            per_layer = []
            for w in loads:
                if kind is None:
                    phy = linear_placement(E, gpus)
                elif kind == 'rr':
                    phy = round_robin_placement(E, gpus)
                else:
                    phy = rebalance(w, E + red, G, nodes, gpus)
                per_layer.append(gpu_loads(w, phy, gpus))
            b = balancedness(per_layer)
            print(f'{gpus:>4} {nodes:>4} {label:<26} {E + red:>8} {b:>13.3f} {1 / b:>13.2f}')
        print()
    d = ARCHS['dsv3']
    print(f'每个冗余专家多占 {d.expert * d.lin_bytes / 2**20:.0f} MiB（FP8），58 层每卡 +1 冗余 = '
          f'{58 * d.expert * d.lin_bytes / GiB:.2f} GiB 权重。')
    print('vLLM 默认 window_size=1000、step_interval=3000，计数器从 3000 - 3000//4 = 2250 起步、到 3000 触发，首次重排在启动后第 750 步；'
          'SGLang eplb_rebalance_num_iterations=1000。')


def scenario_dp():
    print('== DP4 一个 wave：各 rank 的活在不同步数干完（dp_sync_interval=16）')
    for fin in ((40, 40, 40, 40), (40, 12, 3, 33), (97, 5, 5, 5), (1, 0, 0, 0)):
        end, dummy = dp_wave(fin)
        print(f'  干完的步数 {str(fin):<18} wave 在第 {end:>3} 步结束，各 rank dummy 步数 {dummy}')
    print()
    print('== 每步 token 数的 DP 补齐（CUDA Graph 尺寸 1,2,4,8..248 步长 8，256..512 步长 16）')
    print(f'{"各 rank 实际 token":<26} {"同步后 CG":<10} {"各 rank 实际跑":<26} {"补齐浪费":>8}')
    for t in ((37, 5, 0, 64), (120, 118, 121, 2), (300, 12, 12, 12), (2048, 16, 16, 16), (8, 8, 8, 8)):
        final, mode = dp_step(t)
        waste = sum(final) - sum(t)
        print(f'{str(t):<26} {["NONE", "PIECEWISE", "FULL"][mode]:<10} {str(final):<26} '
              f'{waste / sum(final):>7.0%}')
    print()
    print('没活的 rank 也要跑一个 1 token 的 dummy batch，因为 MoE 的 dispatch/combine 需要 EP 组里每个 rank 到场。')
    print('只要有一个 rank 走 eager（比如 2048 token 的 prefill 超过最大 capture），所有 rank 都退回 NONE，不再补齐。')
    print('稠密模型开 DP 时每个 rank 是独立引擎（EngineCoreProc），没有 wave、dummy 和补齐。')


def scenario_pp():
    print(f'{"层数":>4} {"PP":>3} {"切分（get_pp_indices）":<28} 说明')
    for L, pp, note in ((80, 4, 'Llama-3.1-70B，整除'), (61, 4, 'DeepSeek-V3：余 1 层给倒数第 2 段'),
                        (94, 4, 'Qwen3-235B：余 2 层给倒数第 2、3 段'), (80, 3, '余 2 层：倒数第 2、3 段，第 3 段就是第 1 段'),
                        (36, 5, 'Qwen3-8B：余 1')):
        print(f'{L:>4} {pp:>3} {str(pp_partition(L, pp)):<28} {note}')
    print()
    a = ARCHS['llama70b']
    p = Plan('', pp=4)
    print('== Llama-3.1-70B PP4 每段权重（BF16）：首段多 embedding，末段多 lm_head')
    for s in range(4):
        print(f'  stage {s}: {pp_partition(a.layers, 4)[s]} 层，{weight_bytes_per_gpu(a, p, s) / GiB:.2f} GiB')
    print()
    print('max_concurrent_batches：同步调度 = pp；V1 runner 异步 + PP 退回 pp；MRv2 异步 = pp + 1。')
    print('batch 不够 pp 个时流水线有空泡：in-flight 1 个 batch 的 PP4，每张卡 3/4 的时间在等。')


def scenario_sglang():
    print('== 8 x H200 跑 DeepSeek-V3：同一种部署两边怎么写')
    rows = [('纯 TP8', 'vllm serve DSV3 -tp 8', 'python -m sglang.launch_server --tp 8'),
            ('注意力 DP + 专家 EP', 'vllm serve DSV3 -dp 8 -ep', '--tp 8 --attn-dp-size 8 --ep-size 8'),
            ('同上 + DeepEP', '... --all2all-backend deepep_low_latency', '... --moe-a2a-backend deepep（ep_size 强制 = tp）'),
            ('EPLB + 冗余', '--enable-eplb --eplb-config \'{"num_redundant_experts":32}\'',
             '--enable-eplb --ep-num-redundant-experts 32'),
            ('计算/通信 overlap', '--enable-dbo（decode >= 32 token 才切）', '--enable-two-batch-overlap'),
            ('DCP', '-tp 8 -dcp 8', '--tp 8 --dcp-size 8')]
    for k, v, s in rows:
        print(f'  {k:<18} vLLM:   {v}')
        print(f'  {"":<18} SGLang: {s}')
    print()
    print('== world size 的口径')
    for v, s in (('-tp 8 -dp 8', 8 * 8), ('-tp 1 -dp 8', 8)):
        print(f'  vLLM {v:<12} → {s} 张卡（world_size = PP x TP x PCP，DP 另乘）')
    print('  SGLang --tp 8 --attn-dp-size 8 → 8 张卡（tp_size 就是总卡数，attn_dp 在里面切）')
    print()
    print('== 注意力 DP 下的默认值')
    chunk = 8192   # H200 141GB 落在 SGLang 的 <160GB 档
    for adp in (1, 2, 4, 8):
        print(f'  attn_dp_size={adp}: SGLang chunked_prefill_size {chunk} → {chunk // adp:>5}，'
              f'schedule_conservativeness {1.0 * (0.3 if adp > 1 else 1):.1f}；'
              f'vLLM 每个 DP rank 各自 max_num_batched_tokens 8192')
    print()
    print('  EPLB 节奏：vLLM window 1000 / interval 3000（计数器从 2250 起步，首次在第 750 步）；SGLang 每 1000 iteration 重排一次。')
    print('  enable_dp_lm_head：SGLang 在 attn TP 组内切词表，没有 DP 时自动关掉；vLLM DP rank 各自持有完整 lm_head（tp=1 时）。')


def main():
    ap = argparse.ArgumentParser()
    sc = {'layout': scenario_layout, 'memory': scenario_memory, 'comm': scenario_comm, 'dcp': scenario_dcp,
          'eplb': scenario_eplb, 'dp': scenario_dp, 'pp': scenario_pp, 'sglang': scenario_sglang}
    ap.add_argument('--scenario', default='memory', choices=list(sc))
    sc[ap.parse_args().scenario]()


if __name__ == '__main__':
    main()
