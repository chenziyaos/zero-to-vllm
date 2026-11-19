"""PagedAttention 的块分配器与块表模拟（只依赖标准库）。

对照的源码（vllm 主干 0d6f92a，sglang 主干 8fabf5e）：
  vllm   v1/core/kv_cache_utils.py:162      KVCacheBlock：block_id、ref_cnt、block_hash、prev/next_free_block
         v1/core/kv_cache_utils.py:258      FreeKVCacheBlockQueue：手写双向链表，带假头假尾，中间删除 O(1)
         v1/core/block_pool.py:137          BlockPool：建池时弹出 block 0 作为 null block
         v1/core/block_pool.py:706          get_new_blocks：从队头弹 n 块，ref_cnt 0 → 1
         v1/core/block_pool.py:792          touch：前缀命中时把 ref_cnt==0 的块从队列中间摘掉
         v1/core/block_pool.py:816          free_blocks：没有哈希的块 prepend（LIFO），有哈希的块 append（LRU）
         v1/core/single_type_kv_cache_manager.py:591  free：按 reversed(块列表) 交给 free_blocks
         v1/core/kv_cache_manager.py:379    allocate_slots：watermark、full_sequence_must_fit、reserved_blocks
         v1/core/sched/scheduler.py:911     running 循环分配失败 → 从 running 队尾抢占，重算
         v1/worker/block_table.py:29,49     块表宽度按 128 token 对齐；block_table int32、slot_mapping int64
         v1/worker/block_table.py:~300      slot = block_table[pos // bs] * bs + pos % bs，填充位写 -1
  sglang srt/mem_cache/memory_pool.py:376   ReqToTokenPool：[(max_num_reqs+1), context_len+4] int32，逐 token 存槽位
         srt/mem_cache/allocator/token.py:28 TokenToKVPoolAllocator：free_pages = arange(1, size+1)，page_size 1

四种显存管理（第 2 节的 waste 场景）：
  orca-max     按 max_len 整段预留（Orca 论文实现的近似，PagedAttention 论文的 Orca (Max)）
  orca-pow2    按真实总长向上取 2 的幂预留（Orca (Pow2)，最多多留 2×）
  orca-oracle  事先知道真实输出长度，按 prompt+output 精确预留（Orca (Oracle)，理论上界）
  paged        按块分配，用到才要；块不够时从 running 队尾抢占并重算（vLLM V1）

数据集是合成的：按 PagedAttention 论文 Fig.11 的均值（ShareGPT prompt 161 / output 338，
Alpaca 19 / 58，正文给的比例是 8.4× 与 5.8×）做对数正态采样，总长截断到 max_len。
一步 = 每条 running 请求前进 1 个 token（prefill 一步做完），不估时间，只看显存与批大小。

用法：
  python3 sim_block_pool.py                         # waste：论文条件（15.7K 槽位，max_len 2048），ShareGPT
  python3 sim_block_pool.py --dataset alpaca
  python3 sim_block_pool.py --scenario bsweep       # 块大小 1 … 256：内部碎片、批大小、块表宽度
  python3 sim_block_pool.py --scenario h100         # H100 + Qwen3-8B：容量、9.75x、块表 vs req_to_token
  python3 sim_block_pool.py --scenario trace        # 小池子逐步打印块号：LIFO 复用、LRU 驱逐顺序、slot mapping
"""
import argparse
import math
import random

PAD_SLOT_ID = -1


def cdiv(a, b):
    return -(-a // b)


# --------------------------------------------------------------------------- 块与空闲队列
class KVCacheBlock:
    __slots__ = ("block_id", "ref_cnt", "block_hash", "prev", "next", "is_null")

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash = None
        self.prev = None
        self.next = None
        self.is_null = False


class FreeKVCacheBlockQueue:
    """kv_cache_utils.py:258。队头先被复用；中间删除 O(1)，不分配新的 Python 对象。"""

    def __init__(self, blocks):
        self.head = KVCacheBlock(-1)
        self.tail = KVCacheBlock(-1)
        self.num_free_blocks = len(blocks)
        prev = self.head
        for b in blocks:
            prev.next, b.prev = b, prev
            prev = b
        prev.next, self.tail.prev = self.tail, prev

    def popleft_n(self, n):
        assert self.num_free_blocks >= n
        ret, cur = [], self.head.next
        for _ in range(n):
            ret.append(cur)
            nxt = cur.next
            cur.prev = cur.next = None
            cur = nxt
        self.head.next, cur.prev = cur, self.head
        self.num_free_blocks -= n
        return ret

    def popleft(self):
        return self.popleft_n(1)[0]

    def remove(self, b):
        b.prev.next, b.next.prev = b.next, b.prev
        b.prev = b.next = None
        self.num_free_blocks -= 1

    def prepend_n(self, blocks):
        if not blocks:
            return
        first, prev = self.head.next, self.head
        for b in blocks:
            b.prev, prev.next = prev, b
            prev = b
        prev.next, first.prev = first, prev
        self.num_free_blocks += len(blocks)

    def append_n(self, blocks):
        if not blocks:
            return
        last = self.tail.prev
        for b in blocks:
            b.prev, last.next = last, b
            last = b
        last.next, self.tail.prev = self.tail, last
        self.num_free_blocks += len(blocks)

    def ids(self):
        out, cur = [], self.head.next
        while cur is not self.tail:
            out.append(cur.block_id)
            cur = cur.next
        return out


class BlockPool:
    """block_pool.py:137。block 0 被弹出当 null block，不进空闲队列，也不维护 ref_cnt。"""

    def __init__(self, num_gpu_blocks, enable_caching=True):
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.blocks = [KVCacheBlock(i) for i in range(num_gpu_blocks)]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
        self.cached = {}          # block_hash -> block（第 08 篇展开）
        self.evicted = []         # 被重新分配时从缓存里踢掉的哈希

    def get_num_free_blocks(self):
        return self.free_block_queue.num_free_blocks

    def get_new_blocks(self, n):
        if n > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {n} free blocks from the pool")
        ret = self.free_block_queue.popleft_n(n)
        for b in ret:
            if self.enable_caching and b.block_hash is not None:   # _maybe_evict_cached_block
                self.cached.pop(b.block_hash, None)
                self.evicted.append(b.block_hash)
                b.block_hash = None
            assert b.ref_cnt == 0
            b.ref_cnt += 1
        return ret

    def touch(self, blocks):
        for b in blocks:
            if b.is_null:
                continue
            if b.ref_cnt == 0:
                self.free_block_queue.remove(b)
            b.ref_cnt += 1

    def free_blocks(self, ordered_blocks):
        first, last = [], []
        for b in ordered_blocks:
            if b.is_null:
                continue
            b.ref_cnt -= 1
            if b.ref_cnt == 0:
                (first if b.block_hash is None or not self.enable_caching else last).append(b)
        self.free_block_queue.prepend_n(first)   # LIFO：没哈希的块马上被复用，GPU 局部性更好
        self.free_block_queue.append_n(last)     # FIFO：有哈希的块排到队尾，形成 LRU

    def get_usage(self):
        return 1.0 - self.get_num_free_blocks() / (self.num_gpu_blocks - 1)


# --------------------------------------------------------------------------- 管理器（单一 full-attention 组）
class KVCacheManager:
    def __init__(self, num_gpu_blocks, block_size, max_model_len, watermark=0.0, enable_caching=False):
        self.pool = BlockPool(num_gpu_blocks, enable_caching)
        self.bs = block_size
        self.max_model_len = max_model_len
        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.req_blocks = {}

    def allocate_slots(self, req, num_new_tokens, full_sequence_must_fit=False, has_scheduled_reqs=True):
        """kv_cache_manager.py:379 的骨架：只算 full attention、无前缀命中、无 lookahead。"""
        held = self.req_blocks.setdefault(req.rid, [])
        wm = self.watermark_blocks if (has_scheduled_reqs and req.status in ("WAITING", "PREEMPTED")) else 0
        if full_sequence_must_fit:
            full = min(req.num_tokens, self.max_model_len)
            if cdiv(full, self.bs) - len(held) + wm > self.pool.get_num_free_blocks():
                return None
        need_slot = min(req.computed + num_new_tokens, self.max_model_len)
        n = max(0, cdiv(need_slot, self.bs) - len(held))
        if n + wm > self.pool.get_num_free_blocks():
            return None
        new = self.pool.get_new_blocks(n)
        held.extend(new)
        return new

    def free(self, req):
        blocks = self.req_blocks.pop(req.rid, [])
        self.pool.free_blocks(reversed(blocks))   # 尾块先驱逐

    def held_blocks(self, req):
        return len(self.req_blocks.get(req.rid, ()))


# --------------------------------------------------------------------------- 块表与 slot mapping
def block_table_width(max_num_blocks, block_size, token_alignment=128):
    """block_table.py:29：宽度对齐到 128 token。"""
    align = token_alignment // math.gcd(token_alignment, block_size)
    return cdiv(max_num_blocks, align) * align


def slot_mapping(block_table_rows, query_spans, block_size, pad_to=None):
    """block_table.py 的 Triton kernel 的纯 Python 版。

    block_table_rows: 每条请求一行块号
    query_spans: 每条请求本步算的位置区间 (start, end)
    返回每个 token 的物理槽位；pad_to 之后的填充位写 PAD_SLOT_ID（-1），reshape_and_cache 会跳过。
    """
    out = []
    for row, (s, e) in zip(block_table_rows, query_spans):
        for pos in range(s, e):
            out.append(row[pos // block_size] * block_size + pos % block_size)
    if pad_to is not None:
        out += [PAD_SLOT_ID] * (pad_to - len(out))
    return out


# --------------------------------------------------------------------------- 负载
DATASETS = {
    # PagedAttention 论文 Fig.11 的均值；正文：ShareGPT 输入长 8.4×、输出长 5.8×
    "sharegpt": (161.31, 337.99),
    "alpaca": (19.31, 58.45),
}


def lognormal_int(rng, mean, sigma):
    mu = math.log(mean) - sigma * sigma / 2
    return max(1, int(round(rng.lognormvariate(mu, sigma))))


def make_requests(dataset, n, max_len, seed=0, sigma=1.0):
    rng = random.Random(seed)
    pm, om = DATASETS[dataset]
    reqs = []
    for i in range(n):
        p = min(max(4, lognormal_int(rng, pm, sigma)), max_len - 1)
        o = min(lognormal_int(rng, om, sigma), max_len - p)
        reqs.append(Req(i, p, o))
    return reqs


class Req:
    __slots__ = ("rid", "prompt", "out", "gen", "computed", "status", "reserve", "preempted")

    def __init__(self, rid, prompt, out):
        self.rid, self.prompt, self.out = rid, prompt, out
        self.gen = 0            # 已生成的 token 数
        self.computed = 0       # 已写入 KV 的 token 数（num_computed_tokens）
        self.status = "WAITING"
        self.reserve = 0        # orca-* 的预留长度
        self.preempted = 0

    @property
    def num_tokens(self):
        return self.prompt + self.gen

    def fresh(self):
        return Req(self.rid, self.prompt, self.out)


def pow2_ceil(x):
    return 1 << (x - 1).bit_length()


# --------------------------------------------------------------------------- 动态模拟
def run(reqs, mode, slots, max_len, block_size=16, max_num_seqs=1024, watermark=0.0):
    """返回 dict：steps、平均/峰值批大小、显存口径（stored / reserved / frag / free）、抢占次数。"""
    reqs = [r.fresh() for r in reqs]
    waiting = list(reqs)
    running = []
    done = 0
    steps = 0
    acc = {"batch": 0, "stored": 0, "reserved": 0, "frag": 0, "free": 0}
    peak = 0
    preempts = 0
    if mode == "paged":
        num_blocks = slots // block_size + 1           # +1：null block
        mgr = KVCacheManager(num_blocks, block_size, max_len, watermark)
        cap = (num_blocks - 1) * block_size
    else:
        cap = slots
        used = 0

    while done < len(reqs):
        steps += 1
        if mode == "paged":
            # running 循环：每条要 1 个 token 的槽位；不够就从队尾抢占
            i = 0
            while i < len(running):
                r = running[i]
                while mgr.allocate_slots(r, 1) is None:
                    victim = running.pop()
                    mgr.free(victim)
                    victim.status, victim.computed = "PREEMPTED", 0
                    victim.preempted += 1
                    preempts += 1
                    waiting.insert(0, victim)
                    if victim is r:
                        break
                else:
                    i += 1
            # waiting 循环：整条（prompt + 已生成）放得下才收；被抢占的请求重算
            while waiting and len(running) < max_num_seqs:
                r = waiting[0]
                if mgr.allocate_slots(r, r.num_tokens - r.computed, full_sequence_must_fit=True,
                                      has_scheduled_reqs=bool(running)) is None:
                    break
                waiting.pop(0)
                r.status = "RUNNING"
                r.computed = r.num_tokens - 1        # 这一步把 prompt（及已生成部分）算进 KV，并采样下一个 token
                running.append(r)
        else:
            while waiting and len(running) < max_num_seqs:
                r = waiting[0]
                total = r.prompt + r.out
                need = {"orca-max": max_len, "orca-pow2": min(pow2_ceil(total), max_len), "orca-oracle": total}[mode]
                if used + need > cap:
                    break
                waiting.pop(0)
                r.reserve, r.status = need, "RUNNING"
                r.computed = r.prompt - 1
                used += need
                running.append(r)

        # 前向：每条 running 前进一个 token
        nxt = []
        for r in running:
            r.computed += 1
            r.gen += 1
            if r.gen >= r.out:
                done += 1
                if mode == "paged":
                    mgr.free(r)
                else:
                    used -= r.reserve
            else:
                nxt.append(r)
        # 统计（前向那一刻的显存）
        stored = sum(r.computed for r in running)
        if mode == "paged":
            alloc = sum(mgr.held_blocks(r) for r in running) * block_size
            # 已完成的请求在本步刚释放，按释放前口径统计
            alloc += sum(cdiv(r.computed, block_size) * block_size for r in running if r.gen >= r.out)
            reserved, frag = 0, alloc - stored
        else:
            alloc = sum(r.reserve for r in running)
            future = sum(r.prompt + r.out - r.computed for r in running)
            reserved, frag = future, alloc - stored - future
        acc["batch"] += len(running)
        acc["stored"] += stored
        acc["reserved"] += reserved
        acc["frag"] += frag
        acc["free"] += cap - alloc
        peak = max(peak, len(running))
        running = nxt
    out = {k: v / steps for k, v in acc.items()}
    out.update(steps=steps, peak=peak, preempts=preempts, cap=cap,
               tokens=sum(r.out for r in reqs))
    return out


def fmt_row(name, s):
    cap = s["cap"]
    pct = lambda k: 100 * s[k] / cap  # noqa: E731
    return (f"{name:<14}{s['batch']:>8.1f}{s['peak']:>6}{s['steps']:>7}{pct('stored'):>8.1f}%"
            f"{pct('reserved'):>8.1f}%{pct('frag'):>8.1f}%{pct('free'):>8.1f}%{s['preempts']:>6}")


HEADER = f"{'模式':<12}{'平均批':>8}{'峰值':>6}{'步数':>7}{'有效':>9}{'预留':>9}{'碎片':>9}{'空闲':>9}{'抢占':>6}"


def scenario_waste(args):
    reqs = make_requests(args.dataset, args.n, args.max_len, args.seed)
    pm = sum(r.prompt for r in reqs) / len(reqs)
    om = sum(r.out for r in reqs) / len(reqs)
    print(f"负载：{args.dataset} 合成 {args.n} 条，同时到达；prompt 均值 {pm:.0f}，output 均值 {om:.0f}；"
          f"KV {args.slots:,} 个 token 槽位，max_len {args.max_len}")
    print("显存口径（按步平均，占总槽位的比例）：有效 = 已写入的 KV；预留 = 给未来 token 留的位置；"
          "碎片 = 永远用不上的位置（Orca 多留的 / 最后一块的空位）；空闲 = 没分出去")
    print(HEADER)
    res = {}
    for mode in ("orca-max", "orca-pow2", "orca-oracle", "paged"):
        s = run(reqs, mode, args.slots, args.max_len, args.block_size, args.max_num_seqs, args.watermark)
        res[mode] = s
        print(fmt_row(mode if mode != "paged" else f"paged bs={args.block_size}", s))
    base = res["paged"]["batch"]
    print(f"paged 的平均批是 orca-max 的 {base / res['orca-max']['batch']:.1f}×、"
          f"orca-pow2 的 {base / res['orca-pow2']['batch']:.1f}×、orca-oracle 的 {base / res['orca-oracle']['batch']:.1f}×；"
          f"步数 {res['orca-max']['steps']} → {res['paged']['steps']}")
    return res


def scenario_bsweep(args):
    print(f"块大小扫描：KV {args.slots:,} 个 token 槽位，max_len {args.max_len}，各 {args.n} 条同时到达")
    print(f"{'数据集':<10}{'bs':>5}{'平均批':>8}{'步数':>7}{'有效':>8}{'碎片':>8}{'抢占':>6}{'块表宽':>8}{'池里块数':>9}")
    res = {}
    for ds in ("sharegpt", "alpaca"):
        reqs = make_requests(ds, args.n, args.max_len, args.seed)
        for bs in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            s = run(reqs, "paged", args.slots, args.max_len, bs, args.max_num_seqs, args.watermark)
            res[(ds, bs)] = s
            w = block_table_width(cdiv(args.max_len, bs), bs)
            print(f"{ds:<10}{bs:>5}{s['batch']:>8.1f}{s['steps']:>7}{100 * s['stored'] / s['cap']:>7.1f}%"
                  f"{100 * s['frag'] / s['cap']:>7.1f}%{s['preempts']:>6}{w:>8}{args.slots // bs:>9,}")
    return res


def scenario_h100(args):
    kv_per_tok = 2 * 36 * 8 * 128 * 2                  # Qwen3-8B BF16：147,456 B
    blocks, bs, mml = 24957, 16, 40960
    usable = (blocks - 1) * bs
    per_req = cdiv(mml, bs)
    print("H100 80GB + Qwen3-8B BF16，gpu_memory_utilization 0.92")
    print(f"  KV 池：{blocks:,} 块 × {bs} = {blocks * bs:,} token（日志口径）；扣 null block 可分配 {usable:,}")
    print(f"  每 token {kv_per_tok:,} B = {kv_per_tok // 1024} KiB；每块 {kv_per_tok * bs // 1024:,} KiB（36 层合计）")
    print(f"  打满 max_model_len {mml:,} 要 {per_req:,} 块 → 日志并发 {blocks / per_req:.2f}x")
    print(f"  如果像 Orca (Max) 那样整段预留：只能同时跑 {usable // mml} 条")
    for ctx in (1024, 4096, 16384):
        print(f"  按块分配、平均上下文 {ctx:>6,}：约 {usable // ctx:,} 条（最后一块平均空 {bs // 2} 个槽，占 {bs / 2 / ctx:.2%}）")
    print()
    max_seqs = 1024
    w = block_table_width(per_req, bs)
    bt = max_seqs * w * 4
    print("块表（vLLM，worker 侧，CPU pinned + GPU 各一份）")
    print(f"  block_table int32 [{max_seqs}, {w}] = {bt / 2**20:.1f} MiB；slot_mapping int64 [8192] = {8192 * 8 // 1024} KiB")
    sgl_tokens = 364487                               # VLLM-02：SGLang 默认 0.830 时的 KV token 数
    est = max(min(int(sgl_tokens / mml * 512), 4096), 2048)
    sgl_reqs = min(est, sgl_tokens // 2)
    row = mml + 4
    r2t = (sgl_reqs + 1) * row * 4
    print("req_to_token（SGLang，GPU）")
    print(f"  max_num_reqs = min(clamp({sgl_tokens:,} / {mml:,} × 512, 2048, 4096), {sgl_tokens:,} // 2) = {sgl_reqs}")
    print(f"  req_to_token int32 [{sgl_reqs + 1}, {row:,}] = {r2t / 2**20:.1f} MiB；free_pages int64 [{sgl_tokens:,}] = {sgl_tokens * 8 / 2**20:.1f} MiB")
    print(f"  同样 1024 行时：[{max_seqs + 1}, {row:,}] = {(max_seqs + 1) * row * 4 / 2**20:.1f} MiB，是块表的 {(max_seqs + 1) * row / (max_seqs * w):.0f}×")
    return dict(usable=usable, per_req=per_req, bt=bt, r2t=r2t, sgl_reqs=sgl_reqs)


def scenario_trace(args):
    print("小池子：10 块（block 0 是 null），block_size 4，关前缀缓存")
    mgr = KVCacheManager(10, 4, 64)
    q = mgr.pool.free_block_queue
    print(f"  建池后空闲队列 {q.ids()}，null block = {mgr.pool.null_block.block_id}")
    a, b = Req(0, 10, 4), Req(1, 6, 4)
    mgr.allocate_slots(a, 10)
    mgr.allocate_slots(b, 6)
    print(f"  A prompt 10 → 块 {[x.block_id for x in mgr.req_blocks[0]]}；B prompt 6 → 块 {[x.block_id for x in mgr.req_blocks[1]]}；队列 {q.ids()}")
    a.computed, b.computed = 10, 6
    for step in range(1, 4):
        got_a = mgr.allocate_slots(a, 1)
        got_b = mgr.allocate_slots(b, 1)
        a.computed += 1
        b.computed += 1
        print(f"  decode {step}：A 写位置 {a.computed - 1}{' 新块 ' + str([x.block_id for x in got_a]) if got_a else ''}；"
              f"B 写位置 {b.computed - 1}{' 新块 ' + str([x.block_id for x in got_b]) if got_b else ''}")
    rows = [[x.block_id for x in mgr.req_blocks[0]], [x.block_id for x in mgr.req_blocks[1]]]
    print(f"  块表：A {rows[0]}，B {rows[1]}")
    sm = slot_mapping(rows, [(12, 13), (8, 9)], 4, pad_to=4)
    print(f"  第 4 步 decode：A pos 12 → {rows[0][3]}×4+0 = {sm[0]}；B pos 8 → {rows[1][2]}×4+0 = {sm[1]}；"
          f"填充到 4 个 token：slot_mapping = {sm}")
    mgr.free(a)
    print(f"  A 结束，释放 {rows[0]}（按 reversed 交回）→ 队列 {q.ids()}：尾块 {rows[0][-1]} 排第一，马上被复用")
    c = Req(2, 5, 1)
    got = mgr.allocate_slots(c, 5)
    print(f"  C prompt 5 → 块 {[x.block_id for x in got]}（LIFO：刚还回来的块）")
    print()
    print("同样的流程打开前缀缓存，A 的 3 个满块各带一个哈希")
    mgr = KVCacheManager(10, 4, 64, enable_caching=True)
    q = mgr.pool.free_block_queue
    a, b = Req(0, 12, 1), Req(1, 6, 1)
    mgr.allocate_slots(a, 13)
    mgr.allocate_slots(b, 6)
    for i, blk in enumerate(mgr.req_blocks[0][:3]):
        blk.block_hash = f"h{i}"
        mgr.pool.cached[f"h{i}"] = blk
    ra = [x.block_id for x in mgr.req_blocks[0]]
    print(f"  A 块 {ra}（前 3 块满、有哈希 h0 h1 h2，第 4 块只写了 1 个 token），B 块 {[x.block_id for x in mgr.req_blocks[1]]}")
    mgr.free(a)
    print(f"  A 结束 → 队列 {q.ids()}：无哈希的块 {ra[3]} 进队头；h2 h1 h0 进队尾（链尾先被驱逐）")
    hit = mgr.pool.cached["h0"]
    mgr.pool.touch([hit])
    print(f"  新请求命中 h0：touch 把块 {hit.block_id} 从队列中间摘掉，ref_cnt=1 → 队列 {q.ids()}")
    mgr.pool.get_new_blocks(q.num_free_blocks - 1)
    last = mgr.pool.get_new_blocks(1)[0]
    print(f"  把空闲块全部要走：倒数第二个被分出去的是 h2 所在块，最后一个是 h1 所在块 {last.block_id}；被踢出缓存的哈希 {mgr.pool.evicted}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="waste", choices=["waste", "bsweep", "h100", "trace"])
    ap.add_argument("--dataset", default="sharegpt", choices=list(DATASETS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--slots", type=int, default=15696, help="KV token 槽位；默认 ≈ 论文 Table 1 OPT-13B/A100-40GB 的 15.7K")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-num-seqs", type=int, default=1024)
    ap.add_argument("--watermark", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"waste": scenario_waste, "bsweep": scenario_bsweep, "h100": scenario_h100, "trace": scenario_trace}[args.scenario](args)


if __name__ == "__main__":
    main()
