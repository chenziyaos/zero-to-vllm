"""前缀缓存模拟：vLLM 的块哈希 + LRU 空闲队列 vs SGLang 的 radix tree（只依赖标准库）。

对照的源码（vllm 主干 0d6f92a，sglang 主干 8fabf5e）：
  vllm   v1/core/kv_cache_utils.py:707   hash_block_tokens：hash((parent or NONE_HASH, tuple(tokens), extra_keys))
         v1/core/kv_cache_utils.py:108   DEFAULT_NONE_HASH_SEED = "vllm-none-hash"（sha256 这类密码学哈希的默认种子）
         utils/hashing.py:26             sha256 = sha256(pickle.dumps(obj, HIGHEST_PROTOCOL))
         v1/core/kv_cache_utils.py:637   extra_keys：lora ("lora", name, path)、多模态、("cache_salt", s) 只进第一个块
         v1/core/kv_cache_utils.py:60    make_block_hash_with_group_id：block_hash + group_id 4 字节大端
         v1/core/block_pool.py:36        BlockHashToBlockMap：同一个哈希可以挂多个块，不去重（块表只追加）
         v1/core/block_pool.py:816       free_blocks：无哈希块 prepend，有哈希块 append → 队头就是 LRU
         v1/core/kv_cache_manager.py:266 get_computed_blocks：命中上限 num_tokens - 1，最后一个 token 必须重算出 logits
         v1/core/single_type_kv_cache_manager.py:743   full attention：从左往右，未命中就停
         v1/core/single_type_kv_cache_manager.py:969   sliding window：从右往左，凑够 cdiv(w-1, bs) 个连续块就停
         v1/core/single_type_kv_cache_manager.py:1066  reachable_block_mask：retention 0 只给重放边界留窗口尾巴
         v1/core/kv_cache_planning.py:522 分组：group_size = 最少的那类层数；max < 1.5×min 时用 max；不足补 padding 层
  sglang srt/mem_cache/radix_cache.py:651  _match_prefix_helper：命中停在节点中间就 split
         srt/mem_cache/radix_cache.py:450  _insert_cache：插入时发现已有的前缀，把自己那份重复 KV 释放掉
         srt/mem_cache/radix_cache.py:552  evict：可驱逐叶子进堆，按 last_access_time 弹
         srt/mem_cache/common.py:271       checkpoint_kv_cache：每个 extend chunk 之后把已算部分插进树
         srt/mem_cache/common.py:418       release_kv_cache(checkpoint=False)：retract 时不插树
         srt/managers/schedule_policy.py:106 in-batch 前缀：prefix_len ≤ 32 的请求在等待队列树里匹配，≥ 32 就降级
         srt/mem_cache/pool_configurator.py:798  SWA：cell = full_cell + ratio × swa_cell，ratio 默认 0.8

工作负载是合成的 token 序列：
  chat     多轮对话：64 个用户共用 1000 token 系统提示，每轮 prompt = 上一轮 prompt + 上一轮输出 + 新消息
  fewshot  4 套 1234 token 的 few-shot 模板，问题与输出各自独立
  random   没有任何共享
一步 = prefill 一批（受 token 预算限制）+ 所有 running 请求各 decode 1 个 token；不估时间。

用法：
  python3 sim_prefix_cache.py --scenario trace     # 哈希链、重复块、文档 10 块例子的两种重放
  python3 sim_prefix_cache.py --scenario hit       # 池子足够大时的命中率：块粒度 vs token 粒度
  python3 sim_prefix_cache.py --scenario pool      # 池子变小：驱逐、抢占与命中率
  python3 sim_prefix_cache.py --scenario hybrid    # 滑窗命中、retention 掩码、分组 padding、SWA 显存账
"""
import argparse
import hashlib
import heapq
import math
import os
import pickle
import random
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "07-paged-kv"))
from sim_block_pool import FreeKVCacheBlockQueue, KVCacheBlock, cdiv, lognormal_int  # noqa: E402


# --------------------------------------------------------------------------- 1. 块哈希
def sha256_pickle(obj):
    return hashlib.sha256(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)).digest()


NONE_HASH = sha256_pickle("vllm-none-hash")


def hash_block_tokens(parent, tokens, extra_keys=None):
    return sha256_pickle((parent if parent is not None else NONE_HASH, tuple(tokens), extra_keys))


def extra_keys_for(block_idx, lora=None, cache_salt=None):
    keys = []
    if lora is not None:
        keys.append(("lora",) + tuple(lora))          # (name, path)
    if cache_salt is not None and block_idx == 0:      # salt 只进第一个块，靠链式哈希传下去
        keys.append(("cache_salt", cache_salt))
    return tuple(keys) if keys else None


def block_hashes(tokens, bs, lora=None, cache_salt=None):
    out, parent = [], None
    for i in range(len(tokens) // bs):
        parent = hash_block_tokens(parent, tokens[i * bs:(i + 1) * bs], extra_keys_for(i, lora, cache_salt))
        out.append(parent)
    return out


def with_group(h, group_id):
    return h + group_id.to_bytes(4, "big")


# --------------------------------------------------------------------------- 2. vLLM：哈希表 + 空闲队列
class PrefixBlockPool:
    """BlockPool + BlockHashToBlockMap。哈希表值是 {block_id: block}，允许重复。"""

    def __init__(self, num_gpu_blocks, lifo_uncached=True):
        self.blocks = [KVCacheBlock(i) for i in range(num_gpu_blocks)]
        self.q = FreeKVCacheBlockQueue(self.blocks)
        self.null = self.q.popleft()
        self.null.is_null = True
        self.map = {}
        self.lifo_uncached = lifo_uncached   # False = 设计文档写成时的行为：释放的块一律 append
        self.evictions = 0
        self.dup_inserts = 0

    def num_free(self):
        return self.q.num_free_blocks

    def get_cached(self, key):
        d = self.map.get(key)
        return next(iter(d.values())) if d else None

    def insert(self, key, blk):
        d = self.map.setdefault(key, {})
        if d:
            self.dup_inserts += 1
        d[blk.block_id] = blk
        blk.block_hash = key

    def get_new_blocks(self, n):
        ret = self.q.popleft_n(n)
        for b in ret:
            if b.block_hash is not None:               # _maybe_evict_cached_block
                d = self.map[b.block_hash]
                del d[b.block_id]
                if not d:
                    del self.map[b.block_hash]
                b.block_hash = None
                self.evictions += 1
            b.ref_cnt = 1
        return ret

    def touch(self, blks):
        for b in blks:
            if b.ref_cnt == 0:
                self.q.remove(b)
            b.ref_cnt += 1

    def free_blocks(self, ordered):
        first, last = [], []
        for b in ordered:
            if b.is_null:
                continue
            b.ref_cnt -= 1
            if b.ref_cnt == 0:
                (first if (b.block_hash is None and self.lifo_uncached) else last).append(b)
        self.q.prepend_n(first)
        self.q.append_n(last)

    def live_duplicates(self):
        return sum(len(d) - 1 for d in self.map.values())


class VllmCache:
    """单个 full-attention 组的 KVCacheManager（prefix caching 打开）。"""
    name = "vllm"

    def __init__(self, num_tokens_pool, block_size, lifo_uncached=True, group_id=0):
        self.bs = block_size
        self.pool = PrefixBlockPool(num_tokens_pool // block_size + 1, lifo_uncached)
        self.gid = group_id
        self.st = {}            # rid -> dict(blocks, hashes, cached)
        self.max_dup = 0

    def label(self):
        return f"vLLM bs={self.bs}"

    def _hashes(self, req, upto):
        st = self.st.setdefault(req.rid, {"blocks": [], "hashes": [], "cached": 0})
        toks = req.tokens
        while len(st["hashes"]) < upto // self.bs:     # 请求对象里增量算块哈希
            i = len(st["hashes"])
            parent = st["hashes"][-1] if st["hashes"] else None
            st["hashes"].append(hash_block_tokens(parent, toks[i * self.bs:(i + 1) * self.bs],
                                                  extra_keys_for(i, req.lora, req.salt)))
        return st

    def lookup(self, req):
        st = self._hashes(req, len(req.tokens))
        hit = []
        for h in st["hashes"][:(len(req.tokens) - 1) // self.bs]:   # 上限 num_tokens - 1
            b = self.pool.get_cached(with_group(h, self.gid))
            if b is None:
                break
            hit.append(b)
        return hit

    def peek(self, req):
        return len(self.lookup(req)) * self.bs

    def admit(self, req):
        hit = self.lookup(req)
        n = len(req.tokens)
        need = cdiv(n, self.bs) - len(hit)
        evictable_hits = sum(1 for b in hit if b.ref_cnt == 0)
        if need + evictable_hits > self.pool.num_free():
            return None
        self.pool.touch(hit)
        st = self.st[req.rid]
        st["blocks"] = hit + self.pool.get_new_blocks(need)
        st["cached"] = len(hit)
        self._cache_full(req, n)                         # 分配时立刻缓存满块，同一批后面的请求就能命中
        return len(hit) * self.bs

    def _cache_full(self, req, num_tokens):
        st = self._hashes(req, num_tokens)
        full = num_tokens // self.bs
        for i in range(st["cached"], full):
            self.pool.insert(with_group(st["hashes"][i], self.gid), st["blocks"][i])
        st["cached"] = max(st["cached"], full)
        self.max_dup = max(self.max_dup, self.pool.live_duplicates())

    def extend(self, req, computed):
        st = self.st[req.rid]
        if cdiv(computed, self.bs) > len(st["blocks"]):
            if self.pool.num_free() < 1:
                return False
            st["blocks"] += self.pool.get_new_blocks(1)
        self._cache_full(req, computed)
        return True

    def after_step(self, reqs):
        pass

    def release(self, req, finished):
        st = self.st.pop(req.rid)
        self.pool.free_blocks(reversed(st["blocks"]))    # 尾块先被驱逐

    def stats(self):
        return {"evictions": self.pool.evictions, "dup": self.max_dup}


# --------------------------------------------------------------------------- 3. SGLang：radix tree
class Node:
    __slots__ = ("key", "value", "children", "parent", "lock_ref", "last_access")
    _clock = 0

    def __init__(self, key=(), value=(), parent=None):
        self.key, self.value = list(key), list(value)
        self.children, self.parent, self.lock_ref = {}, parent, 0
        Node._clock += 1
        self.last_access = Node._clock

    def touch(self):
        Node._clock += 1
        self.last_access = Node._clock

    def __lt__(self, other):
        return self.last_access < other.last_access


class RadixCache:
    """树的边存“页”：page_size=1 时就是逐 token。value 是页号。"""

    def __init__(self, num_pages):
        self.root = Node()
        self.root.lock_ref = 1
        self.free_pages = list(range(num_pages - 1, -1, -1))
        self.evictable = 0          # 页数
        self.evicted = 0

    def match(self, keys):
        node, value = self.root, []
        node.touch()
        i = 0
        while i < len(keys) and keys[i] in node.children:
            child = node.children[keys[i]]
            child.touch()
            n = 0
            while n < len(child.key) and i + n < len(keys) and child.key[n] == keys[i + n]:
                n += 1
            if n < len(child.key):
                child = self._split(child, n)
            value += child.value
            node, i = child, i + n
        return value, node

    def _split(self, child, n):
        new = Node(child.key[:n], child.value[:n], child.parent)
        new.lock_ref, new.last_access = child.lock_ref, child.last_access
        new.children = {child.key[n]: child}
        child.parent.children[child.key[0]] = new
        child.key, child.value, child.parent = child.key[n:], child.value[n:], new
        return new

    def insert(self, keys, pages):
        """返回已经在树里的页数（调用方那部分是重复的，要释放）。"""
        node, i, existed = self.root, 0, 0
        node.touch()
        while i < len(keys) and keys[i] in node.children:
            child = node.children[keys[i]]
            child.touch()
            n = 0
            while n < len(child.key) and i + n < len(keys) and child.key[n] == keys[i + n]:
                n += 1
            if n < len(child.key):
                child = self._split(child, n)
            existed += n
            node, i = child, i + n
        if i < len(keys):
            new = Node(keys[i:], pages[i:], node)
            node.children[keys[i]] = new
            self.evictable += len(keys) - i
        return existed

    def inc_lock(self, node):
        while node is not self.root:
            if node.lock_ref == 0:
                self.evictable -= len(node.value)
            node.lock_ref += 1
            node = node.parent

    def dec_lock(self, node):
        while node is not self.root:
            if node.lock_ref == 1:
                self.evictable += len(node.value)
            node.lock_ref -= 1
            node = node.parent

    def evict(self, num_pages):
        leaves, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if not n.children and n is not self.root and n.lock_ref == 0:
                leaves.append(n)
            stack.extend(n.children.values())
        heapq.heapify(leaves)
        freed = 0
        while freed < num_pages and leaves:
            x = heapq.heappop(leaves)
            self.free_pages.extend(x.value)
            freed += len(x.value)
            self.evictable -= len(x.value)
            self.evicted += len(x.value)
            p = x.parent
            del p.children[x.key[0]]
            if not p.children and p is not self.root and p.lock_ref == 0:
                heapq.heappush(leaves, p)
        return freed

    def alloc(self, n):
        if len(self.free_pages) < n:
            self.evict(n - len(self.free_pages))
        if len(self.free_pages) < n:
            return None
        return [self.free_pages.pop() for _ in range(n)]


class SglangCache:
    name = "sglang"

    def __init__(self, num_tokens_pool, page_size):
        self.p = page_size
        self.tree = RadixCache(num_tokens_pool // page_size)
        self.st = {}
        self.dup_freed = 0

    def label(self):
        return f"SGLang page={self.p}"

    def _keys(self, req, n):
        t, p = req.tokens, self.p
        extra = (req.lora, req.salt)
        return [(extra,) + tuple(t[i * p:(i + 1) * p]) for i in range(n // p)]

    def peek(self, req):
        return len(self.tree.match(self._keys(req, len(req.tokens) - 1))[0]) * self.p

    def admit(self, req):
        n = len(req.tokens)
        pages, node = self.tree.match(self._keys(req, n - 1))      # _compute_max_prefix_len = n - 1
        self.tree.inc_lock(node)
        new = self.tree.alloc(cdiv(n, self.p) - len(pages))
        if new is None:
            self.tree.dec_lock(node)
            return None
        self.st[req.rid] = {"pages": pages + new, "protected": len(pages), "node": node}
        return len(pages) * self.p

    def _checkpoint(self, req, computed):
        st = self.st[req.rid]
        full = computed // self.p
        if full <= st["protected"]:
            return
        keys = self._keys(req, full * self.p)
        existed = self.tree.insert(keys, st["pages"][:full])
        if existed > st["protected"]:                               # 别人先插进去了：自己那份是重复的
            dup = st["pages"][st["protected"]:existed]
            self.tree.free_pages.extend(dup)
            self.dup_freed += len(dup)
        tree_pages, node = self.tree.match(keys)
        st["pages"][:full] = tree_pages                             # req_to_token 那一行改指向树里的页
        self.tree.dec_lock(st["node"])
        self.tree.inc_lock(node)
        st["node"], st["protected"] = node, full

    def extend(self, req, computed):
        st = self.st[req.rid]
        if cdiv(computed, self.p) > len(st["pages"]):
            new = self.tree.alloc(1)
            if new is None:
                return False
            st["pages"] += new
        return True

    def after_step(self, prefilled):
        for req in prefilled:                                       # checkpoint_kv_cache：extend 之后插树
            self._checkpoint(req, req.computed)

    def release(self, req, finished):
        if finished:
            self._checkpoint(req, req.computed)                    # cache_finished_req
        st = self.st.pop(req.rid)
        self.tree.free_pages.extend(st["pages"][st["protected"]:])   # retract：不插树，直接还
        self.tree.dec_lock(st["node"])

    def stats(self):
        return {"evictions": self.tree.evicted, "dup": self.dup_freed}


# --------------------------------------------------------------------------- 4. 请求、负载、引擎
class Req:
    def __init__(self, rid, prompt_tokens, output_tokens, user=None, lora=None, salt=None):
        self.rid, self.prompt, self.output = rid, list(prompt_tokens), list(output_tokens)
        self.user, self.lora, self.salt = user, lora, salt
        self.gen, self.computed, self.first, self.preempted = 0, 0, True, 0

    @property
    def tokens(self):
        return self.prompt + self.output[:self.gen]


def _rand_tokens(rng, n):
    return [rng.randrange(1, 1 << 40) for _ in range(n)]


def workload(kind, seed=0):
    """返回 (初始请求列表, 后续轮次生成器 next_turn(req) -> Req | None)。"""
    rng = random.Random(seed)
    if kind == "chat":
        system = _rand_tokens(rng, 1000)
        users, turns = 64, 6
        plan = {u: [(lognormal_int(rng, 120, 0.6), lognormal_int(rng, 200, 0.6)) for _ in range(turns)]
                for u in range(users)}
        counter = [0]

        def mk(u, k, history):
            m, o = plan[u][k]
            counter[0] += 1
            r = Req(counter[0], history + _rand_tokens(rng, m), _rand_tokens(rng, o), user=(u, k))
            return r

        first = [mk(u, 0, list(system)) for u in range(users)]

        def next_turn(r):
            u, k = r.user
            return mk(u, k + 1, r.prompt + r.output) if k + 1 < turns else None
        return first, next_turn
    if kind == "fewshot":
        templates = [_rand_tokens(rng, 1234) for _ in range(4)]
        reqs = [Req(i, templates[rng.randrange(4)] + _rand_tokens(rng, lognormal_int(rng, 80, 0.6)),
                    _rand_tokens(rng, lognormal_int(rng, 150, 0.6))) for i in range(400)]
        return reqs, lambda r: None
    reqs = [Req(i, _rand_tokens(rng, lognormal_int(rng, 1300, 0.6)),
                _rand_tokens(rng, lognormal_int(rng, 150, 0.6))) for i in range(400)]
    return reqs, lambda r: None


def _common(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def run(cache, kind, max_running=32, token_budget=8192, seed=0, in_batch_defer=False):
    first, next_turn = workload(kind, seed)
    waiting, running = deque(first), []
    q = h = prefill = recompute = steps = 0
    while waiting or running:
        steps += 1
        # decode：每条 running 请求算 1 个 token 的 KV
        i = 0
        while i < len(running):
            r = running[i]
            if r.gen >= len(r.output):
                i += 1
                continue
            if cache.extend(r, r.computed + 1):
                r.computed += 1
                r.gen += 1
                i += 1
            else:                                   # 块不够：从队尾抢占（vLLM 与 SGLang 都是重算）
                victim = running.pop()
                cache.release(victim, finished=False)
                victim.computed, victim.preempted = 0, victim.preempted + 1
                waiting.appendleft(victim)
        done = [r for r in running if r.gen >= len(r.output)]
        for r in done:
            running.remove(r)
            cache.release(r, finished=True)
            nxt = next_turn(r)
            if nxt is not None:
                waiting.append(nxt)
        # prefill：FCFS，token 预算内尽量多收
        budget, prefilled, deferred = token_budget, [], []
        while waiting and len(running) < max_running:
            r = waiting[0]
            n, hit = len(r.tokens), cache.peek(r)
            if in_batch_defer and hit <= 32 and any(_common(r.tokens, p.tokens) >= 32 for p in prefilled):
                deferred.append(waiting.popleft())  # schedule_policy.py:356：同批已有人在算这段前缀，先让一让
                continue
            if n - hit > budget and prefilled:      # 预算不够：下一步再来
                break
            hit = cache.admit(r)
            if hit is None:
                break
            waiting.popleft()
            budget -= n - hit
            if r.first:
                q, h = q + len(r.prompt), h + hit
                r.first = False
                prefill += n - hit
            else:
                recompute += n - hit
            r.computed = n
            if r.gen < len(r.output):
                r.gen += 1                          # prefill（含抢占后重算）顺带采样出下一个 token
            prefilled.append(r)
            running.append(r)
        waiting.extendleft(reversed(deferred))
        cache.after_step(prefilled)
        if not running and waiting and not prefilled:
            raise RuntimeError("池子装不下单条请求")
    s = cache.stats()
    return {"hit": h / q, "q": q, "prefill": prefill, "recompute": recompute, "steps": steps, **s}


# --------------------------------------------------------------------------- 5. 场景
def hx(b, n=8):
    return b.hex()[:n]


def scenario_trace(args):
    print("== 1. 哈希链：同样 32 个 token，bs=16 → 2 个块")
    toks = list(range(100, 132))
    base = block_hashes(toks, 16)
    salt = block_hashes(toks, 16, cache_salt="tenant-a")
    lora = block_hashes(toks, 16, lora=("sql", "/lora/sql"))
    print(f"  NONE_HASH = sha256(pickle('vllm-none-hash')) = {hx(NONE_HASH)}")
    print(f"  无 extra      : {hx(base[0])} → {hx(base[1])}")
    print(f"  cache_salt    : {hx(salt[0])} → {hx(salt[1])}   salt 只进块 0，块 1 因父哈希不同也变了")
    print(f"  LoRA(sql)     : {hx(lora[0])} → {hx(lora[1])}   LoRA 进每个块")
    print(f"  group 0 / 1   : …{with_group(base[0], 0).hex()[-8:]} / …{with_group(base[0], 1).hex()[-8:]}"
          f"   键长 {len(with_group(base[0], 0))} 字节 = 32 + 4")
    toks2 = list(range(100, 116)) + [999] + list(range(117, 132))
    b2 = block_hashes(toks2, 16)
    print(f"  改第 17 个 token：块 0 {'相同' if b2[0] == base[0] else '不同'}，块 1 {'相同' if b2[1] == base[1] else '不同'}")

    print("\n== 2. 重复块：bs=4，prompt ABCDEF，输出 GHI（设计文档 Duplicated blocks）")
    A = {c: i + 1 for i, c in enumerate("ABCDEFGHI")}
    c = VllmCache(4 * 10, 4)
    ids = lambda r: [b.block_id - 1 for b in c.st[r.rid]["blocks"]]   # 减 1：文档不算 null block
    r1 = Req(1, [A[x] for x in "ABCDEF"], [A[x] for x in "GHI"])
    c.admit(r1)
    r1.computed, r1.gen = 6, 1
    for _ in range(3):                       # 文档的 Tokens 含“正在算的那个 token”：Time 2 是 9 个
        c.extend(r1, r1.computed + 1)
        r1.computed += 1
        r1.gen = min(r1.gen + 1, 3)
    print(f"  请求 1 Time 2：块表 {ids(r1)}，已写 KV {r1.computed} token")
    r2 = Req(2, [A[x] for x in "ABCDEF"], [A[x] for x in "GHI"])
    hit = c.admit(r2)
    r2.computed, r2.gen = 6, 1
    c.extend(r2, 7)
    r2.computed, r2.gen = 7, 2
    print(f"  请求 2 Time 0：命中 {hit} token（ABCDEF 的上限是 5，只够 1 块），块表 {ids(r2)}")
    c.extend(r2, 8)
    r2.computed, r2.gen = 8, 3
    key = with_group(c.st[r1.rid]["hashes"][1], 0)
    print(f"  请求 2 Time 1：块表 {ids(r2)}；哈希 EFGH 挂着块 {sorted(i - 1 for i in c.pool.map[key])}，重复 {c.pool.live_duplicates()} 个")
    c.release(r1, True)
    c.release(r2, True)
    print(f"  两个请求都释放后：哈希 EFGH 仍挂着块 {sorted(i - 1 for i in c.pool.map[key])}"
          f"（free_blocks 只动 ref_cnt 和队列，不删哈希；多余那份等 LRU 轮到它）")

    for lifo in (False, True):
        tag = "设计文档的写法（释放一律 append）" if not lifo else "0d6f92a 的 free_blocks（无哈希块 prepend）"
        print(f"\n== 3{'b' if lifo else 'a'}. 设计文档 10 块例子，{tag}")
        c = VllmCache(4 * 10, 4, lifo_uncached=lifo)
        ids = lambda r: [b.block_id - 1 for b in c.st[r.rid]["blocks"]]
        qids = lambda: [i - 1 for i in c.pool.q.ids()]
        shared = list(range(1, 13))
        r0 = Req(0, shared + [13, 14, 15], [16, 17, 18])
        c.admit(r0)
        r0.computed, r0.gen = 15, 1
        for _ in range(2):
            c.extend(r0, r0.computed + 1)
            r0.computed += 1
            r0.gen += 1
        print(f"  T2 请求 0：块表 {ids(r0)}")
        r1 = Req(1, shared[:10] + [50, 51, 52, 53], [60])
        hit = c.admit(r1)
        r1.computed = 14
        print(f"  T3 请求 1（14 token，前 10 个相同）：命中 {hit} token，块表 {ids(r1)}")
        c.release(r0, True)
        print(f"  T4 释放请求 0：空闲队列 {'-'.join(map(str, qids()))}")
        c.release(r1, True)
        print(f"  T5 释放请求 1：空闲队列 {'-'.join(map(str, qids()))}")
        r2 = Req(2, shared + list(range(200, 217)), [1])
        before = c.pool.evictions
        hit = c.admit(r2)
        print(f"  T6 请求 2（29 token，前 12 个相同）：命中 {hit} token，块表 {ids(r2)}，驱逐 {c.pool.evictions - before} 个缓存块")


def _table(rows, cols):
    print("  " + " | ".join(f"{c:>12}" for c in cols))
    for r in rows:
        print("  " + " | ".join(f"{x:>12}" for x in r))


def scenario_hit(args):
    print(f"== 池子足够大（{args.big} token 槽位），不驱逐，只看粒度与同批共享")
    for kind in ("chat", "fewshot", "random"):
        rows = []
        for cache, defer in ((VllmCache(args.big, 16), False), (VllmCache(args.big, 64), False),
                             (SglangCache(args.big, 1), False), (SglangCache(args.big, 16), False),
                             (SglangCache(args.big, 1), True)):
            s = run(cache, kind, seed=args.seed, in_batch_defer=defer)
            rows.append([cache.label() + (" +defer" if defer else ""), f"{s['hit'] * 100:.2f}%", s["prefill"], s["dup"], s["steps"]])
        print(f"\n[{kind}] prompt token 总数 {s['q']}")
        _table(rows, ["配置", "命中率", "实际 prefill", "重复块/页", "步数"])


def scenario_pool(args):
    print("== chat 负载，池子从小到大（token 槽位）")
    rows = []
    for slots in args.slots:
        for cache in (VllmCache(slots, 16), SglangCache(slots, 1), SglangCache(slots, 16)):
            s = run(cache, "chat", seed=args.seed)
            rows.append([slots, cache.label(), f"{s['hit'] * 100:.2f}%", s["recompute"], s["evictions"], s["steps"]])
    _table(rows, ["槽位", "配置", "命中率", "抢占重算", "驱逐块/页", "步数"])


def sw_find_longest(cached, max_length, w, bs):
    """single_type_kv_cache_manager.py:969 的骨架：从右往左凑 cdiv(w-1, bs) 个连续命中块。"""
    need = cdiv(w - 1, bs)
    n = max_length // bs
    run_len = 0
    for i in range(n - 1, -1, -1):
        if i in cached:
            run_len += 1
            if run_len >= need:
                return (i + run_len) * bs
        else:
            run_len = 0
    return 0


def reachable_mask(start, end, alignment, w, bs, retention, boundaries, use_eagle=False):
    need = cdiv(w - 1, bs) + (1 if use_eagle else 0)
    shift = 1 if use_eagle else 0
    mask = [False] * (end - start)
    seg = alignment if retention is None else (None if retention == 0 else retention)
    if seg is not None:
        per = seg // bs
        if need >= per:
            return None
        for i in range(start, end):
            if i >= shift and (i - shift) % per >= per - need:
                mask[i - start] = True
    if retention is not None:
        for b in boundaries:
            e = (b // alignment * alignment) // bs + shift
            for j in range(max(start, e - need), min(end, e)):
                mask[j - start] = True
    return mask


def plan_groups(buckets):
    """kv_cache_planning.py:522：返回 (group_size, [(类型, 组数, padding 层, padding%)])。"""
    mn, mx = min(buckets.values()), max(buckets.values())
    gs = mx if mx < mn * 1.5 else mn
    out = []
    for k, n in buckets.items():
        pad = gs - n % gs
        pad = 0 if pad == gs else pad
        out.append((k, cdiv(n, gs), pad, pad / n * 100))
    return gs, out


MODELS = {
    # 名字: (sw 层, full 层, 窗口, kv_heads, head_dim)
    "gpt-oss-20b": (12, 12, 128, 8, 64),
    "gpt-oss-20b+eagle": (12, 13, 128, 8, 64),
    "gemma-3-27b": (52, 10, 1024, 16, 128),
}


def sw_blocks_held(L, w, bs):
    """窗口覆盖的块数：最后一个 token 看 [L-w, L-1]。"""
    return (L - 1) // bs - max(0, L - w) // bs + 1


def scenario_hybrid(args):
    print("== 1. 滑窗命中（设计文档：w=4，bs=1，15 token prompt）")
    cached = {2, 3, 4, 5, 11, 12, 13}
    print(f"  缓存位置 {sorted(cached)}，max_length=14（15-1）")
    print(f"  从右往左：命中长度 {sw_find_longest(cached, 14, 4, 1)}")
    print(f"  块 11-13 被驱逐后：{sw_find_longest(cached - {11, 12, 13}, 14, 4, 1)}")
    print(f"  再驱逐块 5：{sw_find_longest(cached - {11, 12, 13, 5}, 14, 4, 1)}")

    print("\n== 2. 一个 32K prompt 在 SW 组里缓存几块（w=128，bs=16，对齐 16）")
    L = 32768
    nblk = L // 16
    for ret in (None, 1024, 0):
        m = reachable_mask(0, nblk, 16, 128, 16, ret, [L - 1])
        k = nblk if m is None else sum(m)
        tag = {None: "None（稠密）", 1024: "1024", 0: "0（默认）"}[ret]
        print(f"  retention={tag:>9}：{k:5d} / {nblk} 块进哈希表")
    print(f"  full 组：{nblk} / {nblk} 块（全部可命中）")

    print("\n== 3. 分组与 padding（kv_cache_planning.py:522）")
    for name, (sw, full, w, kvh, hd) in MODELS.items():
        gs, out = plan_groups({"sw": sw, "full": full})
        desc = "，".join(f"{k} {n} 组 + {p} padding 层 ({pp:.2f}%)" for k, n, p, pp in out)
        print(f"  {name:18s} group_size={gs:2d}：{desc}")

    print("\n== 4. 设计文档 112 token 例子（bs=16，w=32，1 full 组 + 2 sw 组）")
    full_b = cdiv(112, 16)
    swb = sw_blocks_held(112, 32, 16)
    print(f"  full 组 {full_b} 块，每个 sw 组 {swb} 块，共 {full_b + 2 * swb} 块")

    print("\n== 5. 每请求 KV 占用（bf16，bs=16），相对全部按 full attention 存")
    for name in ("gpt-oss-20b", "gemma-3-27b"):
        sw, full, w, kvh, hd = MODELS[name]
        per_layer_tok = 2 * kvh * hd * 2
        gs, out = plan_groups({"sw": sw, "full": full})
        n_sw_groups = next(n for k, n, _, _ in out if k == "sw")
        n_full_groups = next(n for k, n, _, _ in out if k == "full")
        page = 16 * gs * per_layer_tok
        for L in (4096, 32768, 131072):
            hyb = (n_full_groups * cdiv(L, 16) + n_sw_groups * sw_blocks_held(L, w, 16)) * page
            allf = cdiv(L, 16) * 16 * (sw + full) * per_layer_tok
            print(f"  {name:12s} L={L:6d}：混合 {hyb / 2**20:8.1f} MiB，全 full {allf / 2**20:8.1f} MiB，省 {(1 - hyb / allf) * 100:5.1f}%"
                  f"（页 {page // 1024} KiB）")

    print("\n== 6. 同样 40 GiB，能同时放下几条 L 长请求（gpt-oss-20b）")
    sw, full, w, kvh, hd = MODELS["gpt-oss-20b"]
    M = 40 * 2**30
    cell = full * 2 * kvh * hd * 2            # 每 token 一类层的字节数（两类层数相同）
    page = 16 * 12 * 2 * kvh * hd * 2
    hold = w + 128                             # SGLang：窗口 + SGLANG_SWA_EVICTION_INTERVAL 的懒驱逐余量（假设）
    for L in (8192, 32768):
        v = (M // page) // (cdiv(L, 16) + sw_blocks_held(L, w, 16))
        row = [f"vLLM {v}"]
        for r in (0.8, 0.2, 0.05):
            F = M / (cell + r * cell)
            row.append(f"ratio {r}: {int(min(F / L, r * F / hold))}")
        print(f"  L={L:5d}：" + "；".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="trace", choices=["trace", "hit", "pool", "hybrid"])
    ap.add_argument("--big", type=int, default=400000)
    ap.add_argument("--slots", type=int, nargs="+", default=[40000, 60000, 100000, 200000])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"trace": scenario_trace, "hit": scenario_hit, "pool": scenario_pool, "hybrid": scenario_hybrid}[args.scenario](args)


if __name__ == "__main__":
    main()
