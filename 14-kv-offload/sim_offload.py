"""KV 卸载模拟：GPU/CPU/存储三级容量账本、回迁 vs 重算、物理块布局、多轮 agentic 负载下的分级命中、
LRU / ARC / store_threshold 的淘汰对照、连接器生命周期、SGLang HiCache 对照（只依赖标准库）。

对照的源码（vllm 主干 0d6f92a，sglang 主干 8fabf5e）：
  vllm   config/cache.py:383, 389              kv_offloading_size（GiB，TP 各 rank 合计）/ kv_offloading_backend
         config/vllm.py:1168–1196              native → OffloadingConnector，cpu_bytes_to_use = size·2^30；lmcache → LMCacheMPConnector
         config/kv_transfer.py:88              kv_load_failure_policy = "fail"（另一个取值 "recompute"）
         v1/core/sched/scheduler.py:1104–1170  本地前缀命中 → 按块对齐 → connector.get_num_new_matched_tokens；None 就跳过本请求
         v1/core/sched/scheduler.py:3465–3513  _update_waiting_for_remote_kv：cache_blocks；整段命中时回退 1 个 token 重算
         v1/core/sched/scheduler.py:3515       _handle_blocked_waiting_request：WAITING_FOR_REMOTE_KVS 等 finished_recving
         distributed/kv_transfer/kv_connector/v1/multi_connector.py:451–470   第一个命中 > 0 的连接器胜出；任一返回 None 就整体 None
         distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1258–1351
                                               get_num_new_matched_tokens / _reserve_sync_load（本文件逐行移植判定逻辑）
         v1/kv_offload/cpu/policies/arc.py      ARCCachePolicy：T1/T2/B1/B2、target_t1_size、evict 规则（本文件按语义移植）
         v1/kv_offload/cpu/policies/lru.py      LRUCachePolicy：请求粒度的 recency
         v1/kv_offload/cpu/manager.py:66, 261–300   store_threshold：计数达到阈值才写 CPU
  sglang srt/arg_groups/fields/memory.py       hicache_ratio（cache 模式默认 2.0）/ hicache_size（GB=1e9 B，每 rank）/
                                               write_policy 默认 write_through / io_backend 默认 kernel / mem_layout 默认 page_first /
                                               prefetch_policy 默认 timeout
         srt/mem_cache/unified_radix_cache.py:570, 588   write_through_threshold（write_through 为 1，否则 2）/ load_back_threshold = 10
         srt/managers/cache_controller.py:309, 623       prefetch_threshold = 256（不小于 page_size）
         srt/mem_cache/hybrid_cache/hybrid_cache_controller.py:404–407  prefetch_timeout_base=1 / per_ki_token=0.25
         srt/mem_cache/unified_radix_cache.py:2320–2330  timeout = base + pages·per_page，无上限
         docs/docs/advanced_features/hicache_design.mdx  文档写的是 min(30, 2 + 0.1·tokens/1024)，与代码默认不一致

假设：
  - GPU 侧 KV 池按 02-ledger/sim_ledger.py 的账本（H100 80GB、util 0.92）算出：Llama-3.1-8B 单卡 457,696 token。
  - 带宽：GPU↔CPU 取 vLLM 博客实测的 DMA 单向约 50 GB/s、双向 83.4 GB/s；本地 NVMe 取 12 GB/s（两块 Gen4 盘顺序读，
    假设值）；RDMA 取 45 GB/s（400 Gb/s 网卡线速的 90%，假设值）。
  - 重算按 prefill 的 FLOP 算：线性层 2·P·n，注意力 4·L·H·d·(n·p + n²/2)，MFU 取 0.5。
  - tiers / eviction 场景用 512 token 的 chunk 做哈希粒度（真实部署常见 64~256），只算前缀能否命中，不模拟传输排队，
    回迁时间按带宽串行累加，不和计算重叠，所以回迁那一项偏保守。
  - 多轮负载按 vLLM x Mooncake 博客的 Codex/SWE-bench Pro 统计：首轮 12K、每轮 +2,242、30 轮。

用法：
  python3 sim_offload.py --scenario capacity   # 每级能放多少 token、多少个 agentic 会话
  python3 sim_offload.py --scenario ledger     # 回迁 vs 重算：每 token 微秒数、80K 前缀一轮的 TTFT 账
  python3 sim_offload.py --scenario layout     # 旧布局 vs 新布局的物理块大小、一次回迁要发多少次拷贝
  python3 sim_offload.py --scenario tiers      # 并发会话数 × CPU 容量 → GPU 命中 / CPU 命中 / 重算
  python3 sim_offload.py --scenario eviction   # 混入一次性长文档后 LRU / ARC / store_threshold=2 的差别
  python3 sim_offload.py --scenario lifecycle  # 一个请求走完连接器的每一步
  python3 sim_offload.py --scenario sglang     # HiCache 与 vLLM 原生卸载的容量、写策略、预取超时对照
"""
import argparse
import os
import random
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, MODELS  # noqa: E402

GiB = 1 << 30
BLOCK = 16                  # vLLM 默认 GPU block_size
PCIE_GBS = 50.0             # vLLM 博客：DMA 与自定义 kernel 单向都能到 ~50 GB/s
PCIE_BIDIR_GBS = 83.4       # vLLM 博客：DMA 双向 83.4 GB/s（kernel 68.5）
NVME_GBS = 12.0             # 假设值
RDMA_GBS = 45.0             # 假设值
MFU = 0.5                   # prefill 的算力利用率，假设值
CHUNK = 512                 # tiers / eviction 场景的哈希粒度

# 02-ledger/sim_ledger.py 默认参数下的 GPU KV 池（每卡 token 数、每卡 KV 字节/token）
GPU_POOL = {
    ('llama3.1-8b', 1): (457_696, 'H100 ×1'),
    ('llama3.1-70b', 4): (500_000, 'H100 ×4'),
    ('qwen3-8b', 1): (None, 'H100 ×1'),
}


def kv_per_card(m, tp=1, dtype_bytes=2):
    b = m.kv_bytes_per_token(dtype_bytes)
    if not m.mla_kv_rank:
        b /= min(tp, m.kv_heads)
    return b


def prefill_flops(m, n_new, n_prefix=0, tp=1):
    """在已有 n_prefix 个 token 的 KV 上再算 n_new 个 token 的 FLOP（每卡）。"""
    lin = 2 * m.active_b * 1e9 * n_new
    attn = 4 * m.layers * m.heads * m.head_dim * (n_new * n_prefix + n_new * n_new / 2)
    return (lin + attn) / tp


def prefill_s(m, gpu, n_new, n_prefix=0, tp=1):
    return prefill_flops(m, n_new, n_prefix, tp) / (gpu.bf16_tflops * 1e12 * MFU)


def reload_s(m, n, gbs=PCIE_GBS, tp=1):
    return n * kv_per_card(m, tp) / (gbs * 1e9)


def agentic_ctx(turn, first=12_000, delta=2_242):
    return first + delta * turn


# --------------------------------------------------------------------------- 1. 容量
def scenario_capacity():
    m = MODELS['llama3.1-8b']
    bpt = kv_per_card(m)
    gpu_tok = 457_696
    print('== Llama-3.1-8B，H100 ×1，BF16 KV：每级能放多少')
    print(f'  每 token KV {bpt / 1024:.0f} KiB；1 GiB = {GiB / bpt:,.0f} token；80K 上下文 = {80_000 * bpt / GiB:.2f} GiB')
    sess = [agentic_ctx(t) for t in range(30)]
    avg = sum(sess) / len(sess)
    print(f'  agentic 会话（首轮 12K、每轮 +2,242、30 轮）：第 30 轮 {sess[-1]:,} token，30 轮平均 {avg:,.0f} token')
    print(f'  {"层级":22s} {"容量":>10s} {"token":>12s} {"放得下的会话（按平均）":>14s}  {"按第 30 轮":>8s}')
    rows = [('GPU KV 池（util 0.92）', gpu_tok * bpt / GiB, gpu_tok)]
    for g in (64, 128, 256, 512):
        rows.append((f'CPU --kv-offloading-size {g}', g, int(g * GiB / bpt)))
    rows.append(('NVMe 4 TB', 4e12 / GiB, int(4e12 / bpt)))
    for name, gib, tok in rows:
        print(f'  {name:22s} {gib:8.1f} GiB {tok:12,d} {tok / avg:14.1f}  {tok / sess[-1]:8.1f}')
    print()
    print('== 同一台 8×H100 节点（2 TB DRAM）怎么分')
    host = 2048 * 0.75
    print(f'  留 25% 给 OS、权重加载缓冲和 pinned 区，剩 {host:.0f} GiB；8 个 TP=1 实例各 {host / 8:.0f} GiB，'
          f'每卡 CPU 层 = GPU 池的 {host / 8 / (gpu_tok * bpt / GiB):.1f} 倍')
    print('  vLLM：--kv-offloading-size 是 TP 组内各 rank 合计（GiB）；TP=4 想每卡 192 GiB 要写 768')
    print('  SGLang：--hicache-size 是每 rank（GB = 1e9 B）；同样每卡 192 GiB 要写 206')
    print()
    print('== 多大才有用：CPU 层小于 GPU 池时它只是 GPU 内容的镜像（见 --scenario tiers）')
    print(f'  vLLM 文档：单层 CPU 卸载时 cpu_bytes_to_use 要大于全部 GPU KV，否则只是镜像，不加命中率')
    print(f'  SGLang：--hicache-ratio 必须 > 1；write_through 下 host 里真正 "GPU 没有" 的部分 = host - GPU')


# --------------------------------------------------------------------------- 2. 回迁 vs 重算
def scenario_ledger():
    gpu = GPUS['h100-sxm']
    print(f'== 每个 token：重算要几微秒，搬回来要几微秒（H100，MFU {MFU}，前缀 0）')
    print(f'  {"模型":14s} {"TP":>2s} {"KV/卡":>8s} {"重算":>8s} {"PCIe 50":>8s} {"NVMe 12":>8s} {"RDMA 45":>8s} {"重算/PCIe":>9s}')
    for key, tp in (('qwen3-0.6b', 1), ('llama3.1-8b', 1), ('qwen3-8b', 1), ('llama3.1-70b', 4)):
        m = MODELS[key]
        rc = prefill_s(m, gpu, 1, 0, tp) * 1e6
        p, n, r = (reload_s(m, 1, g, tp) * 1e6 for g in (PCIE_GBS, NVME_GBS, RDMA_GBS))
        print(f'  {m.name:14s} {tp:2d} {kv_per_card(m, tp) / 1024:6.0f}Ki {rc:7.2f}µ {p:7.2f}µ {n:7.2f}µ {r:7.2f}µ {rc / p:8.1f}x')
    print('  注意力项随前缀长度线性涨，下面按真实前缀长度再算一次。')
    print()
    m = MODELS['llama3.1-8b']
    print('== Llama-3.1-8B 一轮 agentic 请求：前缀 p 已算过，新增 2,242 token')
    print(f'  {"前缀 p":>8s} {"GPU 命中":>9s} {"CPU 回迁":>9s} {"NVMe 回迁":>10s} {"全部重算":>9s} {"重算/CPU":>8s}')
    for p in (4_000, 16_000, 32_000, 64_000, 80_000, 128_000):
        hit = prefill_s(m, gpu, 2242, p)
        cpu = hit + reload_s(m, p)
        nv = hit + reload_s(m, p, NVME_GBS)
        rc = prefill_s(m, gpu, p + 2242, 0)
        print(f'  {p:8,d} {hit * 1e3:7.0f}ms {cpu * 1e3:7.0f}ms {nv * 1e3:8.0f}ms {rc * 1e3:7.0f}ms {rc / cpu:7.1f}x')
    print('  CPU 回迁这一列按搬完再算，没有重叠；vLLM 按层流水（wait_for_layer_load）时会更短。')
    print()
    print('== 回迁什么时候不划算')
    for key in ('qwen3-0.6b', 'llama3.1-8b'):
        m = MODELS[key]
        for p in (1_000, 8_000, 32_000):
            rc = prefill_s(m, gpu, p, 0)
            print(f'  {m.name:12s} 前缀 {p:6,d}：重算 {rc * 1e3:7.1f} ms，PCIe 回迁 {reload_s(m, p) * 1e3:6.1f} ms，'
                  f'NVMe 回迁 {reload_s(m, p, NVME_GBS) * 1e3:6.1f} ms')
    print('  小模型 KV/FLOP 比大：0.6B 的短前缀从 NVMe 搬回来比重算还慢，这就是 vLLM 博客拿 1B 模型当最坏情况的原因。')


# --------------------------------------------------------------------------- 3. 物理块布局
BLOG_LAYOUT = {  # vLLM 博客 2026-01-08 表格（16 token 块）
    'Llama-3.1-8B': ('32 KB', '2 MB'), 'Llama-3.1-70B（TP=4）': ('8 KB', '1.25 MB'),
    'Qwen3-0.6B': ('32 KB', '1.75 MB'), 'Qwen3-8B': ('28 KB', '1.97 MB'),
}


def layout_sizes(m, tp=1, block=BLOCK, dtype_bytes=2):
    """旧布局：每层 K、V 各一段 → 一段 = block·kv_heads/tp·head_dim·dtype；新布局把 2·L 段拼成一块。"""
    kvh = max(m.kv_heads // tp, 1)
    frag = block * kvh * m.head_dim * dtype_bytes
    return frag, frag * 2 * m.layers, 2 * m.layers


def scenario_layout():
    print('== 一个 16 token 的块在 GPU 上是几段：旧布局每层 K、V 各一段，新布局（0.12.0 起）全层连续一段')
    print(f'  {"模型":22s} {"旧：一段":>9s} {"新：一块":>9s} {"段数":>5s}   博客表格（旧 / 新）')
    for name, key, tp in (('Llama-3.1-8B', 'llama3.1-8b', 1), ('Llama-3.1-70B（TP=4）', 'llama3.1-70b', 4),
                          ('Qwen3-0.6B', 'qwen3-0.6b', 1), ('Qwen3-8B', 'qwen3-8b', 1)):
        frag, blk, n = layout_sizes(MODELS[key], tp)
        o, nw = BLOG_LAYOUT[name]
        print(f'  {name:22s} {frag / 1024:7.0f}KB {blk / 2**20:7.2f}MB {n:5d}   {o} / {nw}')
    print('  Qwen3-8B 按 config.json 算出 32 KB / 2.25 MB，博客表格是 28 KB / 1.97 MB，两者倍数都是 2·36 = 72。')
    print()
    m = MODELS['llama3.1-8b']
    frag, blk, n = layout_sizes(m)
    tok = 80_000
    blocks = -(-tok // BLOCK)
    print(f'== Llama-3.1-8B 回迁 80K token = {blocks:,} 个块，{tok * kv_per_card(m) / GiB:.2f} GiB')
    print(f'  旧布局：{blocks * n:,} 次 {frag // 1024} KB 拷贝；新布局：{blocks:,} 次 {blk / 2**20:.0f} MB 拷贝'
          f'（offload block_size 256 时再降到 {-(-tok // 256):,} 次 32 MB）')
    print('  每次 cudaMemcpyAsync 若有 t0 的固定开销（假设值），纯开销部分：')
    for t0 in (2e-6, 5e-6):
        print(f'    t0 = {t0 * 1e6:.0f} µs：旧 {blocks * n * t0 * 1e3:6.0f} ms，新 {blocks * t0 * 1e3:5.1f} ms，'
              f'纯带宽 {tok * kv_per_card(m) / (PCIE_GBS * 1e9) * 1e3:.0f} ms')
    print('  博客结论：KB 级小块上自定义 kernel 吞吐明显更高，MB 级块上 DMA 追平；并发负载下 DMA 不占 SM，')
    print('  吞吐比 kernel 高 5.5%（0% 命中）到 15%（80% 命中），Llama-3.1-8B 上最多高 32%。')


# --------------------------------------------------------------------------- 4/5. 分级缓存模拟
class LRU:
    name = 'lru'

    def __init__(self, cap):
        self.cap, self.d = cap, OrderedDict()

    def __contains__(self, k):
        return k in self.d

    def touch(self, keys):
        for k in keys:
            if k in self.d:
                self.d.move_to_end(k)

    def insert(self, keys, protected):
        out = 0
        for k in keys:
            if k in self.d:
                continue
            while len(self.d) >= self.cap:
                v = next((x for x in self.d if x not in protected), None)
                if v is None:
                    return out
                del self.d[v]
            self.d[k] = None
            out += 1
        return out


class ARC:
    """按 v1/kv_offload/cpu/policies/arc.py 的语义：新块进 T1，被请求复用的就绪块升到 T2；
    B1 命中加大 target_t1_size，B2 命中减小；淘汰时 |T1| ≥ target 就从 T1 淘汰，否则从 T2。"""
    name = 'arc'

    def __init__(self, cap):
        self.cap = cap
        self.t1, self.t2, self.b1, self.b2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        self.p = 0.0

    def __contains__(self, k):
        return k in self.t1 or k in self.t2

    def _ghost(self, k):
        if k in self.b1:
            self.p = min(self.p + max(1, len(self.b2) / len(self.b1)), self.cap)
            return True
        if k in self.b2:
            self.p = max(self.p - max(1, len(self.b1) / len(self.b2)), 0)
            return True
        return False

    def touch(self, keys):  # on_request_finished 里 reused_keys 的处理
        for k in reversed(keys):
            if k in self.t1:
                self.t2[k] = self.t1.pop(k)
            elif k in self.t2:
                self.t2.move_to_end(k)

    def _evict_one(self, protected):
        order = (self.t1, self.t2) if len(self.t1) >= int(self.p) else (self.t2, self.t1)
        for src in order:
            for k in src:
                if k not in protected:
                    del src[k]
                    (self.b1 if src is self.t1 else self.b2)[k] = None
                    return True
        return False

    def insert(self, keys, protected):
        new = [k for k in keys if k not in self]
        for k in reversed(new):  # on_store_miss：先从 ghost 学习
            self._ghost(k)
        out = 0
        for k in new:
            while len(self.t1) + len(self.t2) >= self.cap:
                if not self._evict_one(protected):
                    return out
            self.t1[k] = None
            self.b1.pop(k, None)
            self.b2.pop(k, None)
            out += 1
        for g in (self.b1, self.b2):
            while len(g) > self.cap:
                g.popitem(last=False)
        return out


class Threshold:
    """manager.py 的 store_threshold：一个 key 被提交存储的次数达到阈值才真正写入（计数表有上限）。"""

    def __init__(self, inner, threshold=2, tracker=64_000):
        self.inner, self.th, self.tracker = inner, threshold, tracker
        self.counts = OrderedDict()
        self.name = f'{inner.name}+store_threshold={threshold}'

    def __contains__(self, k):
        return k in self.inner

    def touch(self, keys):
        self.inner.touch(keys)

    def insert(self, keys, protected):
        ok = []
        for k in keys:
            if k in self.inner:
                continue
            if k in self.counts:
                self.counts[k] += 1
                self.counts.move_to_end(k)
            else:
                if len(self.counts) >= self.tracker:
                    self.counts.popitem(last=False)
                self.counts[k] = 1
            if self.counts[k] >= self.th:
                ok.append(k)
        return self.inner.insert(ok, protected)


def run_tiers(n_sess, gpu_chunks, cpu_chunks, policy='lru', scan_every=0, scan_tokens=32_000,
              turns=30, events=None, seed=0, threshold=0):
    """返回 dict：各级命中 token、重算 token、写 CPU 的 chunk 数。"""
    rng = random.Random(seed)
    gpu = LRU(gpu_chunks)
    cpu = None
    if cpu_chunks:
        cpu = ARC(cpu_chunks) if policy == 'arc' else LRU(cpu_chunks)
        if threshold >= 2:
            cpu = Threshold(cpu, threshold)
    sessions = {}
    next_id = 0

    def new_session():
        nonlocal next_id
        next_id += 1
        return [next_id, rng.randrange(turns // 2) if events is None else 0]

    for _ in range(n_sess):
        s = new_session()
        s[1] = rng.randrange(turns)  # 稳态起点：各会话处在不同轮次
        sessions[s[0]] = s
    events = events or n_sess * turns * 2
    warm = events // 3
    st = dict(gpu=0, cpu=0, recompute=0, total=0, writes=0, turns=0)
    scan_id = 0
    for e in range(events):
        if scan_every and e % scan_every == 0:
            scan_id += 1
            keys = [('scan', scan_id, i) for i in range(scan_tokens // CHUNK)]
            ctx = scan_tokens
        else:
            sid = rng.choice(list(sessions))
            s = sessions[sid]
            ctx = agentic_ctx(s[1])
            keys = [(sid, i) for i in range(ctx // CHUNK)]
            s[1] += 1
            if s[1] >= turns:
                del sessions[sid]
                ns = new_session()
                sessions[ns[0]] = ns
        g = 0
        while g < len(keys) and keys[g] in gpu:
            g += 1
        c = g
        if cpu is not None:
            while c < len(keys) and keys[c] in cpu:
                c += 1
        prot = set(keys)
        gpu.touch(keys[:g])
        gpu.insert(keys, prot)
        gpu.touch(keys)
        w = 0
        if cpu is not None:
            cpu.touch([k for k in keys if k in cpu])
            w = cpu.insert(keys, prot)
        if e >= warm:
            st['gpu'] += g * CHUNK
            st['cpu'] += (c - g) * CHUNK
            st['recompute'] += ctx - c * CHUNK
            st['total'] += ctx
            st['writes'] += w
            st['turns'] += 1
    return st


def turn_cost(m, gpu, st):
    """把命中拆分换算成每轮平均耗时：重算部分按 prefill，CPU 命中部分按 PCIe 回迁（不重叠）。"""
    n = st['turns']
    rc = st['recompute'] / n
    pre = (st['gpu'] + st['cpu']) / n
    t = prefill_s(m, gpu, rc, pre) + reload_s(m, st['cpu'] / n)
    return t


def scenario_tiers():
    m, gpu = MODELS['llama3.1-8b'], GPUS['h100-sxm']
    bpt = kv_per_card(m)
    gpu_chunks = 457_696 // CHUNK
    print(f'== Llama-3.1-8B / H100：GPU 池 457,696 token（{gpu_chunks} 个 {CHUNK} token chunk），CPU 层 LRU，'
          f'多轮 agentic（12K 起、每轮 +2,242、30 轮），会话随机交错')
    print(f'  {"会话":>4s} {"CPU GiB":>8s} {"CPU/GPU":>7s} {"GPU 命中":>8s} {"CPU 命中":>8s} {"重算":>6s} {"每轮 prefill":>11s} {"相对无卸载":>9s}')
    for n in (8, 16, 32, 64, 128):
        base = None
        for cg in (0, 32, 64, 128, 256, 512):
            cpu_chunks = int(cg * GiB / bpt) // CHUNK
            st = run_tiers(n, gpu_chunks, cpu_chunks)
            t = turn_cost(m, gpu, st)
            base = base or t
            T = st['total']
            print(f'  {n:4d} {cg:8d} {cg / 55.87:7.2f} {st["gpu"] / T:8.1%} {st["cpu"] / T:8.1%} '
                  f'{st["recompute"] / T:6.1%} {t * 1e3:9.0f}ms {base / t:8.2f}x')
        print()
    print('  CPU/GPU < 1 时 CPU 命中几乎为 0：每次请求都把整段前缀同时留在 GPU 和 CPU，小的那层只是大的那层的子集。')


def scenario_eviction():
    m = MODELS['llama3.1-8b']
    bpt = kv_per_card(m)
    gpu_chunks = 457_696 // CHUNK
    cpu_chunks = int(128 * GiB / bpt) // CHUNK
    print(f'== 32 个 agentic 会话 + 一次性长文档（32K、不会再来），GPU 池 457,696 token，CPU 128 GiB')
    print(f'  {"长文档占比":>8s} {"策略":24s} {"GPU 命中":>8s} {"CPU 命中":>8s} {"重算":>6s} {"写 CPU GiB/轮":>12s}')
    for every in (0, 4, 2):
        frac = '0' if not every else f'1/{every}'
        for pol, th in (('lru', 0), ('arc', 0), ('lru', 2)):
            st = run_tiers(32, gpu_chunks, cpu_chunks, policy=pol, scan_every=every, threshold=th)
            T = st['total']
            name = pol + (f'+store_threshold={th}' if th else '')
            wg = st['writes'] * CHUNK * bpt / GiB / st['turns']
            print(f'  {frac:>8s} {name:24s} {st["gpu"] / T:8.1%} {st["cpu"] / T:8.1%} {st["recompute"] / T:6.1%} {wg:12.2f}')
        print()
    print('  长文档只进 T1 不升 T2，ARC 淘汰时先拿 T1 开刀；store_threshold=2 干脆不写只出现一次的 chunk，')
    print('  代价是会话的新 chunk 要第二次出现才落 CPU（这一轮之后被 GPU 淘汰就得重算），并且 TieringOffloadingSpec 不接受 ≥2。')


# --------------------------------------------------------------------------- 6. 连接器生命周期
def multi_connector_pick(results):
    """multi_connector.py 451–470：results 是各连接器的 (toks, load_async)。返回 (toks, load_async, 选中下标)。"""
    to_return, idx = (0, False), None
    for i, (toks, la) in enumerate(results):
        if toks is None:
            return None, False, None
        if to_return[0] == 0 and toks > 0:
            to_return, idx = (toks, la), i
    return to_return[0], to_return[1], idx


def offload_matched(in_flight, skip_or_failed, cpu_hit_tokens, num_computed, num_tokens,
                    kv_bpt, budget, used):
    """offloading/scheduler.py 1258–1351：返回 (hit, load_async, sync_used_after)。"""
    if in_flight:
        return None, False, used
    hit = 0 if skip_or_failed else max(cpu_hit_tokens - num_computed, 0)
    sync = False
    if budget and hit and num_computed + hit < num_tokens:
        nb = hit * kv_bpt
        if used + nb <= budget:
            used += nb
            sync = True
    return hit, bool(hit) and not sync, used


def schedule_external(local_hit, ext_tokens, block=BLOCK):
    """scheduler.py 1129–1175：给连接器的是按块对齐的本地命中；远端更长就丢掉本地的不满块。"""
    tail = local_hit % block
    aligned = local_hit - tail
    if tail and ext_tokens > tail:
        return aligned, ext_tokens
    return local_hit, max(ext_tokens - tail, 0) if ext_tokens else 0


def after_recv(num_computed, num_tokens):
    """scheduler.py 3511–3512：整段 prompt 都命中时回退 1 个 token，好采样下一个。"""
    return num_tokens - 1 if num_computed == num_tokens else num_computed


def scenario_lifecycle():
    m = MODELS['llama3.1-8b']
    bpt = kv_per_card(m)
    print('== 请求 A：prompt 82,242 token；GPU 前缀命中 20,008；CPU 层存着前 81,920 token（256 token 一块）')
    prompt, local = 82_242, 20_008
    print(f'  step 0  OffloadingConnector：上一轮的回迁还没结束 → get_num_new_matched_tokens 返回 '
          f'{offload_matched(True, False, 81_920, 20_000, prompt, bpt, 0, 0)[:2]}，调度器 skip_request，本步不排 A')
    tail = local % BLOCK
    hit, la, _ = offload_matched(False, False, 81_920, local - tail, prompt, bpt, 2 * GiB, 0)
    loc, ext = schedule_external(local, hit)
    print(f'  step 1  本地命中 {local:,} 的不满块尾巴 {tail} 个 → 给连接器的是 {local - tail:,}；连接器返回 '
          f'({hit:,}, load_async={la})')
    print(f'          远端 {hit:,} > 尾巴 {tail} → 丢掉本地不满块，num_computed = {loc:,} + {ext:,} = {loc + ext:,}')
    nb = hit * bpt
    print(f'          要搬 {nb / GiB:.2f} GiB，超过 sync_load_budget_bytes = 2 GiB → 走异步：update_state_after_alloc，'
          f'状态 WAITING_FOR_REMOTE_KVS')
    t = nb / (PCIE_GBS * 1e9)
    print(f'  step 2…  PCIe {PCIE_GBS:.0f} GB/s 搬 {t * 1e3:.0f} ms；期间其它请求照常 decode（一步约 10 ms 时是 {t / 0.01:.0f} 步）')
    print(f'  step k  worker 报 finished_recving → _update_waiting_for_remote_kv：cache_blocks({loc + ext:,})，'
          f'状态回 WAITING，下一步只算 {prompt - loc - ext:,} 个 token')
    print()
    print('== 请求 B：prompt 正好 81,920 token，CPU 全命中')
    hit, la, _ = offload_matched(False, False, 81_920, 0, 81_920, bpt, 2 * GiB, 0)
    print(f'  连接器返回 ({hit:,}, {la})；回迁完成后 num_computed = {after_recv(81_920, 81_920):,}，'
          '最后 1 个 token 重算，才能采样第一个输出')
    print()
    print('== 请求 C：短前缀走同步回迁（sync_load_budget_bytes = 2 GiB / 步 / 卡）')
    used = 0
    for name, h, n in (('C1', 4_096, 6_000), ('C2', 8_192, 9_000), ('C3', 4_096, 4_096)):
        hit, la, used2 = offload_matched(False, False, h, 0, n, bpt, 2 * GiB, used)
        how = '异步' if la else ('同步，本步就能算' if hit < n else '异步（不留 1 个 token 就没法调度）')
        print(f'  {name}: 命中 {hit:,} / prompt {n:,}，{hit * bpt / GiB:.2f} GiB → {how}；本步预算已用 {used2 / GiB:.2f} GiB')
        used = used2
    print()
    print('== MultiConnector（PD + 存储池）：第一个命中 > 0 的连接器胜出，任何一个说 "还不知道" 就整体等')
    for desc, res in (('[MooncakeConnector, MooncakeStoreConnector]', [(0, False), (30_000, True)]),
                      ('[OffloadingConnector, MooncakeStoreConnector]', [(12_000, True), (60_000, True)]),
                      ('[OffloadingConnector, LMCacheMPConnector]', [(12_000, True), (None, False)])):
        toks, la, idx = multi_connector_pick(res)
        print(f'  {desc:46s} 各自返回 {res} → ({toks}, {la})，选中 #{idx}')
    print('  第二行：后面的连接器命中更长也不会被选，顺序就是优先级。')
    print()
    print('== 回迁失败：kv_load_failure_policy')
    print('  "fail"（默认）    请求直接以错误结束，上游要自己重试')
    print('  "recompute"       num_computed_tokens 截到第一个坏块之前，坏块之后的 token 本地重算；'
          '_update_waiting_for_remote_kv 只 cache 有效前缀')


# --------------------------------------------------------------------------- 7. SGLang 对照
def hicache_timeout(tokens, base=1.0, per_ki=0.25, mx=None):
    """代码默认：base + per_ki·tokens/1024，无上限。传 (2.0, 0.1, 30.0) 得到设计文档里的公式。"""
    t = base + per_ki * tokens / 1024
    return t if mx is None else min(mx, t)


def scenario_sglang():
    m = MODELS['llama3.1-8b']
    bpt = kv_per_card(m)
    gpu_gib = 457_696 * bpt / GiB
    print('== 默认容量：Llama-3.1-8B / H100，GPU KV 池 %.1f GiB' % gpu_gib)
    host = 2.0 * gpu_gib
    print(f'  SGLang --enable-hierarchical-cache（cache 模式）：--hicache-ratio 默认 2.0 → host {host:.1f} GiB/卡')
    print(f'    write_through 下 host 先镜像一份 GPU 内容，"GPU 没有而 host 有" 的空间 = {host - gpu_gib:.1f} GiB'
          f' ≈ {(host - gpu_gib) * GiB / bpt:,.0f} token')
    print('  vLLM 不开就没有；开了以后容量完全由 --kv-offloading-size 决定，没有比例参数')
    print()
    print('== 同一份需求怎么写（TP=4，每卡 192 GiB 主机内存给 KV）')
    rows = [('打开 CPU 层', '--kv-offloading-size 768', '--enable-hierarchical-cache --hicache-size 206'),
            ('单位', 'GiB，TP 组合计', 'GB(1e9)，每 rank'),
            ('写入时机', '算完就写（store_threshold 默认不过滤）', 'write_through（默认）/ _selective / write_back'),
            ('只写 prompt', 'offload_prompt_only 默认 true', '—'),
            ('淘汰', 'eviction_policy lru（默认）/ arc / 自定义', 'host 侧 LRU'),
            ('GPU↔CPU 拷贝', 'DMA cudaMemcpyAsync，不占 SM', '--hicache-io-backend kernel（默认）/ direct'),
            ('主机布局', 'GPU 页原样；canonical_layout 可选', '--hicache-mem-layout page_first（默认）'),
            ('第三级', 'TieringOffloadingSpec: fs / obj / p2p；或 MooncakeStore、LMCache 连接器',
             '--hicache-storage-backend file / mooncake / hf3fs / nixl / aibrix …'),
            ('跨实例共享', 'secondary tier 或外部连接器', '只有 L3 能共享，L2 是实例私有')]
    for a, b, c in rows:
        print(f'  {a:12s} | vLLM: {b:44s} | SGLang: {c}')
    print()
    print('== SGLang 的三道门槛（vLLM 没有对应的硬阈值）')
    print('  load_back_threshold = 10    host → GPU 回迁少于 10 个 token 就不搬，直接重算')
    print('  prefetch_threshold  = 256   L3 命中不到 256 token 不发起预取（且不小于 page_size）')
    print('  write_through_threshold     write_through 为 1，write_through_selective 为 2（命中 2 次才写下一级）')
    print()
    print('== L3 预取超时（prefetch_policy=timeout，默认）')
    print('  代码默认（8fabf5e）：1 + 0.25·tokens/1024 秒，无上限')
    print('  设计文档写的是：min(30, 2 + 0.1·tokens/1024) 秒 —— 以代码为准，部署时显式传参')
    for t in (256, 4_096, 32_768, 81_920, 131_072, 300_000):
        bytes_ = t * bpt
        print(f'  {t:8,d} token  代码 {hicache_timeout(t):5.2f} s  文档 {hicache_timeout(t, 2.0, 0.1, 30.0):5.2f} s   '
              f'NVMe 12 GB/s 读完 {bytes_ / 12e9:5.2f} s   RDMA 45 GB/s {bytes_ / 45e9:5.2f} s')
    print('  vLLM 对应的是"异步就绪前不调度"：没有超时，请求停在 WAITING_FOR_REMOTE_KVS 直到 finished_recving。')


SCENARIOS = dict(capacity=scenario_capacity, ledger=scenario_ledger, layout=scenario_layout,
                 tiers=scenario_tiers, eviction=scenario_eviction, lifecycle=scenario_lifecycle,
                 sglang=scenario_sglang)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=list(SCENARIOS), default='capacity')
    SCENARIOS[ap.parse_args().scenario]()
