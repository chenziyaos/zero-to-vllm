"""多副本路由、K8s 探针、sleep mode 与 RL 权重同步的离线推演（只依赖标准库，不需要 GPU）。

对照的源码与文档（vllm 主干 0d6f92a，sglang 主干 8fabf5e）：
  vllm   v1/engine/core_client.py:1767        eng_start_index = n * client_index // client_count
         v1/engine/core_client.py:1771-1830   内置 DP 负载均衡：score = max(client_count × inflight, waiting + running)，
                                              有排队时再加 waiting × 6 × max(0, kv_usage − 0.5)；选完给本地 waiting 计数 += client_count，扫描起点 +1
         v1/engine/coordinator.py:152         统计快照最短 100 ms 发一次
         docs/serving/data_parallel_deployment.md:77  内置 LB 只看 running / waiting，"could be made ... KV cache aware"
         docs/deployment/k8s.md:244-255       示例探针：/health，initialDelaySeconds 60，periodSeconds 10 / 5
         docs/deployment/k8s.md:436-443       failureThreshold 太小 → "KeyboardInterrupt: terminated"，容器被反复重启
         docs/features/sleep_mode.md:21       level 1 权重卸到 CPU、丢 KV；level 2 权重和 KV 都丢
         docs/training/weight_transfer/nccl.md:80-108  rank 0 广播；packed 缓冲 1 GiB × 2，两端各常驻 2 GiB
         docs/training/async_rl.md:26-28      pause 三种 mode：abort / wait / keep
  sglang sgl-router/README.md:120-150         power_of_two / cache_aware（最少命中 1024 token）/ affinity prefer · balanced（倍数 2 + 差值）
         sgl-router/src/policies/cache_aware.rs  候选按命中前缀 token 数降序，再按负载
         docs/advanced_features/sglang_for_rl.mdx:618-660  pause_generation：abort / retract / in_place
  k8s    concepts/workloads/pods/probes.md    periodSeconds 默认 10，failureThreshold 默认 3，initialDelaySeconds 默认 0
         concepts/workloads/pods/pod-lifecycle.md  重启退避 10 s、20 s、40 s …… 封顶 300 s
         concepts/storage/volumes.md:170      emptyDir 的数据在容器崩溃重启后仍在

场景：
  route    N 个副本、共享前缀的负载：轮询 / vLLM 内置打分 / P2C / 前缀感知（prefer、balanced）的命中率、TTFT、负载倾斜
  hot      一个前缀占一半流量：prefer 把热点堆到一台，balanced 怎么分流
  stale    100 ms 快照下的突发：只看快照、加本地计数、再加 inflight 下限，三种写法各把请求分到哪
  probe    启动时间线 × 探针配置：哪种配置会陷入 CrashLoopBackOff，编译缓存放在卷上为什么能救回来
  sleep    换模型：冷启动 / level 1 / level 2 各花多少秒，level 1 要多少主机内存
  nccl     RL 权重同步：按带宽算下限，packed 缓冲占多少显存
  pause    更新权重前的 pause：abort / wait / keep 与 SGLang abort / retract / in_place 的等待、重算和 stale KV

用法：
  python3 sim_router.py --scenario route
  python3 sim_router.py --scenario route --replicas 8 --rate 64
  python3 sim_router.py --scenario hot
  python3 sim_router.py --scenario stale --clients 2
  python3 sim_router.py --scenario probe
  python3 sim_router.py --scenario sleep
  python3 sim_router.py --scenario nccl
  python3 sim_router.py --scenario pause
  python3 sim_router.py --scenario all

步长沿用 06 章的 roofline 模型（H100 SXM + Qwen3-8B BF16），只用来比较策略的相对差异，不要拿去报数。
"""
import argparse
import math
import os
import random
import sys
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '06-chunked-prefill'))
import sim_chunked as sc  # noqa: E402
from common.specs import MODELS  # noqa: E402

GiB = 1024 ** 3
BLOCK = 16


# --------------------------------------------------------------------------- 负载
class Req:
    __slots__ = ('id', 'pfx', 'plen', 'sfx', 'out', 'arrival', 'computed', 'gen',
                 'first', 'done', 'hit', 'shared', 'replica')

    def __init__(self, rid, pfx, plen, sfx, out, arrival):
        self.id, self.pfx, self.plen, self.sfx, self.out, self.arrival = rid, pfx, plen, sfx, out, arrival
        self.computed = self.gen = self.hit = self.shared = 0
        self.first = self.done = None
        self.replica = -1

    @property
    def prompt(self):
        return self.plen + self.sfx

    @property
    def prefilling(self):
        return self.computed < self.prompt


def prefix_workload(a, hot=0.0):
    """n_prefix 个共享前缀（system prompt / few-shot / 文档），Zipf 热度；每条请求 = 前缀 + 独有后缀。"""
    rng = random.Random(a.seed)
    sizes = [rng.randint(3000, 8000) for _ in range(a.n_prefix)]
    w = [1 / (i + 1) ** a.zipf for i in range(a.n_prefix)]
    if hot:
        rest = sum(w[1:])
        w = [rest * hot / (1 - hot)] + w[1:]
    tot = sum(w)
    cum, acc = [], 0.0
    for x in w:
        acc += x / tot
        cum.append(acc)
    reqs, t, i = [], 0.0, 0
    while t < a.duration * 1e3:
        t += rng.expovariate(a.rate) * 1e3
        u = rng.random()
        p = next(k for k, c in enumerate(cum) if c >= u)
        reqs.append(Req(i, p, sizes[p], rng.randint(100, 400), rng.randint(64, 256), t))
        i += 1
    return reqs, sizes


# --------------------------------------------------------------------------- 单个副本（vLLM V1 调度 + 前缀缓存）
class Replica:
    def __init__(self, idx, a):
        self.idx, self.a = idx, a
        self.waiting, self.running = [], []
        self.cache = OrderedDict()      # prefix id → token 数；LRU，最近用过的在尾部
        self.inuse = Counter()          # prefix id → 正在引用缓存前缀的 running 请求数
        self.busy_until = None
        self.batch = None
        self.busy_ms = 0.0
        self.prefill_computed = 0

    def private(self, r):
        """请求独占的 KV：共享前缀已在缓存里时只算后缀和输出。"""
        return r.prompt + r.gen - (r.plen if r.shared else 0)

    def usage(self):
        # vLLM 的 kv_cache_usage 不把 ref_cnt 为 0 的缓存块算作占用（18 章）
        used = sum(self.private(r) for r in self.running) + sum(self.cache[p] for p in self.inuse)
        return min(1.0, used / sc.KV_CAP)

    def pending_prefill(self):
        """还没算的 prompt token：waiting 里命中缓存的前缀不算。"""
        run = sum(r.prompt - r.computed for r in self.running if r.prefilling)
        wait = sum(r.prompt - (r.plen if r.pfx in self.cache else 0) for r in self.waiting)
        return run + wait

    def matched(self, pfx):
        return self.cache.get(pfx, 0)

    def _make_room(self, need, keep=None):
        free = sc.KV_CAP - sum(self.private(r) for r in self.running) - sum(self.cache.values())
        for p in list(self.cache):
            if free >= need:
                break
            if self.inuse[p] or p == keep:
                continue
            free += self.cache.pop(p)
        return free >= need

    def schedule(self):
        a = self.a
        budget, out = a.budget, []
        for r in self.running:
            if budget <= 0:
                break
            if r.prefilling:
                n = min(r.prompt - r.computed, budget)
                out.append((r, n, r.computed))
            else:
                n = 1
                out.append((r, 1, r.prompt + r.gen))
            budget -= n
        while self.waiting and budget > 0 and len(self.running) < a.max_num_seqs:
            r = self.waiting[0]
            hit = r.pfx in self.cache
            if not self._make_room(r.sfx + r.out + (0 if hit else r.plen), keep=r.pfx if hit else None):
                break
            self.waiting.pop(0)
            if hit:
                self.cache.move_to_end(r.pfx)
                r.computed = r.plen
                r.hit = r.shared = 1
                self.inuse[r.pfx] += 1
            self.running.append(r)
            n = min(r.prompt - r.computed, budget)
            out.append((r, n, r.computed))
            budget -= n
        return out

    def start(self, now):
        batch = self.schedule()
        if not batch:
            self.busy_until, self.batch = None, None
            return
        wall = sc.gpu_ms([(n, s) for _, n, s in batch]) + sc.cpu_ms(len(batch))
        self.busy_ms += wall
        self.batch, self.busy_until = batch, now + wall

    def complete(self):
        now, finished = self.busy_until, []
        for r, n, _ in self.batch:
            if r.prefilling:
                self.prefill_computed += n
                r.computed += n
                if r.prefilling:
                    continue
                if not r.shared and r.pfx not in self.cache:   # prefill 完成，前缀块进缓存，请求改为引用缓存
                    self.cache[r.pfx] = r.plen
                    r.shared = 1
                    self.inuse[r.pfx] += 1
                r.first = now
            r.gen += 1
            if r.gen >= r.out:
                r.done = now
                finished.append(r)
        for r in finished:
            self.running.remove(r)
            if r.shared:
                self.inuse[r.pfx] -= 1
                if not self.inuse[r.pfx]:
                    del self.inuse[r.pfx]
            if r.pfx in self.cache:
                self.cache.move_to_end(r.pfx)
        return finished


# --------------------------------------------------------------------------- 路由策略
class Router:
    """所有策略看同一份周期快照（--snapshot-ms，默认 100，对应 coordinator.py:152）。"""

    def __init__(self, policy, reps, a, client_index=0):
        self.policy, self.reps, self.a = policy, reps, a
        self.rng = random.Random(a.seed + 7)
        self.rr = 0
        n = len(reps)
        self.eng_start = n * client_index // a.clients
        self.inflight = Counter()
        self.snap_t = -1e18
        self.snap = None
        self.lb = None

    def refresh(self, now):
        if now - self.snap_t >= self.a.snapshot_ms:
            self.snap_t = now
            self.snap = [dict(waiting=len(r.waiting), running=len(r.running), kv=r.usage(),
                              prefill=r.pending_prefill()) for r in self.reps]
            self.lb = [[s['waiting'], s['running'], s['kv']] for s in self.snap]

    def vllm_pick(self):
        n, best, idx = len(self.reps), float('inf'), 0
        for i in range(n):
            j = (self.eng_start + i) % n
            waiting, running, kv = self.lb[j]
            score = max(self.a.clients * self.inflight[j], waiting + running)
            if waiting:
                score += waiting * 6.0 * max(0.0, kv - 0.5)
            if score < best:
                best, idx = score, j
        self.lb[idx][0] += self.a.clients
        self.eng_start = (self.eng_start + 1) % n
        return idx

    def p2c_key(self, j):
        s = self.snap[j]
        return (s['prefill'], s['waiting'], s['running'], self.inflight[j])

    def p2c_pick(self, exclude=None):
        cand = [j for j in range(len(self.reps)) if j != exclude]
        x, y = self.rng.sample(cand, 2) if len(cand) >= 2 else (cand[0], cand[0])
        kx, ky = self.p2c_key(x), self.p2c_key(y)
        if kx == ky:
            return self.rng.choice((x, y))
        return x if kx < ky else y

    def route(self, r, now):
        self.refresh(now)
        p = self.policy
        if p == 'rr':
            j = self.rr % len(self.reps)
            self.rr += 1
        elif p == 'vllm':
            j = self.vllm_pick()
        elif p == 'p2c':
            j = self.p2c_pick()
        else:                                            # cache_aware：KV 事件实时推送，索引不滞后
            cands = [(rep.matched(r.pfx), i) for i, rep in enumerate(self.reps)
                     if rep.matched(r.pfx) >= self.a.min_match]
            if not cands:
                j = self.p2c_pick()
            else:
                cands.sort(key=lambda c: (-c[0], self.p2c_key(c[1])))
                j = cands[0][1]
                if p == 'balanced':
                    alt = self.p2c_pick(exclude=j)
                    lp, la = self.snap[j]['prefill'], self.snap[alt]['prefill']
                    if lp > la * self.a.factor and lp > la + self.a.gap:
                        j = alt
        self.inflight[j] += 1
        return j


def simulate(a, policy, reqs, warm=None):
    reps = [Replica(i, a) for i in range(a.replicas)]
    if warm:                                 # 开场时某个前缀只在 0 号副本的缓存里（低峰期第一次见到它）
        pfx, plen = warm
        reps[0].cache[pfx] = plen
    routers = [Router(policy, reps, a, c) for c in range(a.clients)]
    pending = sorted(reqs, key=lambda r: r.arrival)
    k = 0
    while k < len(pending) or any(rep.busy_until is not None for rep in reps):
        nxt_rep = min((rep for rep in reps if rep.busy_until is not None), key=lambda x: x.busy_until, default=None)
        if k < len(pending) and (nxt_rep is None or pending[k].arrival <= nxt_rep.busy_until):
            r = pending[k]
            k += 1
            router = routers[r.id % a.clients]
            j = router.route(r, r.arrival)
            r.replica = j
            reps[j].waiting.append(r)
            if reps[j].busy_until is None:
                reps[j].start(r.arrival)
        else:
            for r in nxt_rep.complete():
                routers[r.id % a.clients].inflight[nxt_rep.idx] -= 1
            nxt_rep.start(nxt_rep.busy_until)
    return reps


def summarize(reqs, reps):
    ttft = [r.first - r.arrival for r in reqs]
    hit_tok = sum(r.plen for r in reqs if r.hit)
    prompt_tok = sum(r.prompt for r in reqs)
    per = Counter(r.replica for r in reqs)
    pf = [rep.prefill_computed for rep in reps]
    span = max(r.done for r in reqs) - min(r.arrival for r in reqs)
    tpot = [(r.done - r.first) / (r.out - 1) for r in reqs if r.out > 1]
    return dict(tpot99=sc.pct(tpot, 0.99), hit=hit_tok / prompt_tok, ttft50=sc.pct(ttft, 0.5), ttft99=sc.pct(ttft, 0.99),
                imb=max(per.values()) / (len(reqs) / len(reps)),
                work_max=max(pf) / (sum(pf) / len(pf)), prefill=sum(rep.prefill_computed for rep in reps), n=len(reqs))


POLICIES = [('rr', '轮询'), ('vllm', 'vLLM 内置打分'), ('p2c', 'P2C'),
            ('prefer', '前缀感知 prefer'), ('balanced', '前缀感知 balanced')]


def table(a, hot=0.0):
    warm = None
    print(f'{"策略":<20}{"前缀命中":>10}{"TTFT p50":>11}{"TTFT p99":>11}{"TPOT p99":>11}{"请求倾斜":>10}{"prefill 倾斜":>13}{"实算 prefill":>14}')
    res = {}
    for pol, name in POLICIES:
        reqs, sizes = prefix_workload(a, hot)
        if hot:
            warm = (0, sizes[0])
        reps = simulate(a, pol, reqs, warm)
        s = summarize(reqs, reps)
        res[pol] = s
        print(f'{name:<20}{s["hit"]:>10.1%}{s["ttft50"]:>9.0f}ms{s["ttft99"]:>9.0f}ms{s["tpot99"]:>9.1f}ms{s["imb"]:>10.2f}'
              f'{s["work_max"]:>13.2f}{s["prefill"] / 1e6:>12.2f}M')
    return res


def show_route(a):
    _, sizes = prefix_workload(a)
    print(f'== route：{a.replicas} 副本 × Qwen3-8B，{a.rate:g} req/s × {a.duration:g} s，'
          f'{a.n_prefix} 个前缀（{min(sizes)}–{max(sizes)} token，合计 {sum(sizes) / 1e3:.0f}k，Zipf {a.zipf:g}），'
          f'单副本 KV 容量 {sc.KV_CAP / 1e3:.0f}k token，快照 {a.snapshot_ms:g} ms')
    res = table(a)
    rr, ca = res['rr'], res['prefer']
    print(f'   前缀感知把命中率从 {rr["hit"]:.1%} 提到 {ca["hit"]:.1%}，实际要算的 prefill 少 '
          f'{1 - ca["prefill"] / rr["prefill"]:.0%}；轮询和内置打分只看队列，前缀在每台副本上各存一份，'
          f'合计前缀 {sum(sizes) / 1e3:.0f}k 超过单副本容量，LRU 在每台上都在换')
    print()
    return res


def show_hot(a):
    print(f'== hot：一个前缀占 {a.hot:.0%} 流量，开场时只在 0 号副本上有缓存，其余同 route（{a.rate:g} req/s）')
    res = table(a, a.hot)
    p, b = res['prefer'], res['balanced']
    print(f'   prefer：首选副本只要还能收就一直往它送，热点前缀的请求全压在 0 号上，请求倾斜 {p["imb"]:.2f}，'
          f'TTFT p50 {p["ttft50"]:.0f} ms、TPOT p99 {p["tpot99"]:.1f} ms；命中率最高，但那一台的 decode batch 被撑大')
    print(f'   balanced：首选的待算 prefill 超过备选 {a.factor:g} 倍且多出 {a.gap} token 才改投，热点前缀被复制到别的副本，'
          f'倾斜 {b["imb"]:.2f}、TTFT p50 {b["ttft50"]:.0f} ms，命中率只少 {p["hit"] - b["hit"]:.1%}')
    print()
    return res


# --------------------------------------------------------------------------- stale：快照间隔里的突发
def burst_split(n_eng, n_req, clients, local, rotate, floor, rebind=0):
    """快照全是 0（engine 刚空下来），n_req 条请求在下一份真实快照前全部到达。
    rebind=k：每 k 条请求被一份过期的全 0 快照覆盖一次（快照与路由竞争，core_client.py:1797-1804 注释里的 rebind）。"""
    counts = Counter()
    for c in range(clients):
        lb = [[0, 0, 0.0] for _ in range(n_eng)]
        inflight = Counter()
        start = n_eng * c // clients
        for k in range(n_req // clients):
            if rebind and k and k % rebind == 0:
                lb = [[0, 0, 0.0] for _ in range(n_eng)]
            best, idx = float('inf'), 0
            for i in range(n_eng):
                j = (start + i) % n_eng
                w, run_, kv = lb[j]
                score = w + run_
                if floor:
                    score = max(clients * inflight[j], score)
                if score < best:
                    best, idx = score, j
            if local:
                lb[idx][0] += clients
            if rotate:
                start = (start + 1) % n_eng
            inflight[idx] += 1
            counts[idx] += 1
    return [counts[i] for i in range(n_eng)]


STALE_ROWS = [('只看快照', dict(local=False, rotate=False, floor=False)),
              ('+ 本地 waiting += client_count', dict(local=True, rotate=False, floor=False)),
              ('+ 扫描起点轮转', dict(local=True, rotate=True, floor=False)),
              ('+ inflight 下限（当前主干）', dict(local=True, rotate=True, floor=True))]


def show_stale(a):
    n_eng, n_req, rebind = a.replicas, 64, 6
    print(f'== stale：{n_eng} 个 engine，{a.clients} 个 API server，{n_req} 条请求挤在同一个 {a.snapshot_ms:g} ms 快照窗口里')
    print(f'{"写法":<34}{"快照不变":>22}{"每 " + str(rebind) + " 条被过期快照覆盖":>30}')
    out = {}
    for name, kw in STALE_ROWS:
        x = burst_split(n_eng, n_req, a.clients, **kw)
        y = burst_split(n_eng, n_req, a.clients, rebind=rebind, **kw)
        out[name] = (x, y)
        print(f'{name:<32}{str(x):>24}{str(y):>30}')
    print('   只看快照：窗口内每条请求看到的都是同一份 0，全落在扫描起点；本地计数让路由立刻“看见”自己刚派出去的请求；')
    print('   过期快照一覆盖，本地计数清零，又从头堆；inflight 是 API server 自己精确记的数，快照抹不掉，所以最后一行两种情况都是均匀的')
    for snap in (0, 100, 1000):
        a2 = argparse.Namespace(**vars(a))
        a2.snapshot_ms = snap
        reqs, _ = prefix_workload(a2)
        s = summarize(reqs, simulate(a2, 'vllm', reqs))
        out[f'snap{snap}'] = s
        print(f'   稳态 route 负载，vLLM 打分，快照间隔 {snap:>4} ms：TTFT p50 {s["ttft50"]:.0f} ms，p99 {s["ttft99"]:.0f} ms')
    print()
    return out


# --------------------------------------------------------------------------- probe：启动时间线 × 探针
# 01 章实测（H100，Qwen3-8B，首启）：解析 6 s、拉起 EngineCore 8 s、建 runner 1 s、读权重 11.28 s、
# torch.compile 41.09 s、显存 profile 约 8 s、CUDA graph 7 s、API server 2 s
CH01 = [('parse', 6), ('spawn', 8), ('runner', 1), ('weights', 11.28), ('compile', 41.09),
        ('profile', 8), ('cudagraph', 7), ('api', 2)]


def startup(compile_cached, weight_scale=1.0, download_s=0.0):
    t, marks = download_s, {}
    for name, s in CH01:
        if name == 'compile' and compile_cached:
            s = 0.0
        if name == 'weights':
            s *= weight_scale
        t += s
        marks[name] = t
    return t, marks


def kill_deadline(cfg):
    """最晚在第几秒被杀：探针在 delay + k·period 打，连续 failureThreshold 次失败触发重启。"""
    p = cfg.get('startup') or cfg['liveness']
    return p['delay'] + (p['ft'] - 1) * p['period']


def probe_timeline(cfg, case, max_restarts=8):
    """返回 (Ready 时刻, 重启次数, 事件)。编译缓存写在卷上时，第一次跑完编译的那次启动之后都命中缓存。"""
    clock, restarts, backoff, compiled, events = 0.0, 0, 10.0, False, []
    deadline = kill_deadline(cfg)
    while restarts <= max_restarts:
        dl = case['download'] if (restarts == 0 or not case['model_volume']) else 0.0
        ready_after, marks = startup(compiled and case['cache_volume'], case['wscale'], dl)
        if ready_after <= deadline:
            events.append(f'第 {restarts + 1} 次启动 {ready_after:.0f} s 就绪')
            return clock + ready_after, restarts, events
        if marks['compile'] <= deadline:
            compiled = True
        events.append(f'第 {restarts + 1} 次启动在 {deadline:.0f} s 被杀（需要 {ready_after:.0f} s）')
        clock += deadline + backoff
        backoff = min(backoff * 2, 300.0)
        restarts += 1
    return None, restarts, events


PROBES = {
    'doc-liveness': dict(liveness=dict(delay=60, period=10, ft=3)),
    'default-3x10': dict(liveness=dict(delay=0, period=10, ft=3)),
    'startup-30x10': dict(startup=dict(delay=0, period=10, ft=30), liveness=dict(delay=0, period=10, ft=3)),
}


def show_probe(a):
    cold, _ = startup(False)
    warm, _ = startup(True)
    big, _ = startup(False, weight_scale=70.6 / 8.19 / 4)
    dl = 16.4e9 * 8 / (a.download_gbps * 1e9)
    print(f'== probe：01 章实测时间线 Qwen3-8B H100 冷启动 {cold:.0f} s、编译缓存命中 {warm:.0f} s；'
          f'Llama-3.1-70B TP4 估算 {big:.0f} s（每卡读 1/4 权重，按 8B 的读盘速度线性放大，其余阶段照搬 8B）')
    print(f'   kubelet 判定：探针在 initialDelay + k × period 打，连续 failureThreshold 次失败才杀，'
          f'最晚在 initialDelay + (failureThreshold − 1) × period 被杀；重启退避 10 → 20 → 40 … → 300 s')
    cases = [
        ('8B，编译缓存在容器层', dict(cache_volume=False, model_volume=True, wscale=1.0, download=0)),
        ('8B，VLLM_CACHE_ROOT 挂 emptyDir', dict(cache_volume=True, model_volume=True, wscale=1.0, download=0)),
        (f'8B，无 PVC 每次拉权重 {a.download_gbps:g} Gb/s', dict(cache_volume=True, model_volume=False, wscale=1.0, download=dl)),
        ('70B TP4，编译缓存在卷上', dict(cache_volume=True, model_volume=True, wscale=70.6 / 8.19 / 4, download=0)),
    ]
    out = {}
    print(f'{"":<34}' + ''.join(f'{k:>22}' for k in PROBES))
    for cname, case in cases:
        row = f'{cname:<32}'
        for pname, cfg in PROBES.items():
            t, rs, ev = probe_timeline(cfg, case)
            out[(cname, pname)] = (t, rs)
            cell = f'CrashLoop（重启 {rs} 次）' if t is None else (f'{t:.0f} s' + (f'（重启 {rs}）' if rs else ''))
            row += f'{cell:>22}'
        print(row)
    dd = kill_deadline(PROBES['doc-liveness'])
    _, m = startup(False)
    print(f'   doc-liveness 是 vLLM k8s 文档的示例（60 s 后每 10 s 打一次，failureThreshold 取默认 3）：最晚 {dd} s 被杀，8B 冷启动要 {cold:.0f} s。')
    print(f'   torch.compile 在第 {m["compile"]:.0f} s 结束，产物写进 VLLM_CACHE_ROOT/torch_compile_cache；目录挂在 emptyDir 上能活过容器重启，'
          f'第二次 {warm:.0f} s 起来；放在容器可写层里，每次重启都是冷启动，永远 CrashLoopBackOff。')
    print(f'   70B 的编译要到第 {startup(False, 70.6 / 8.19 / 4)[1]["compile"]:.1f} s 才结束，比 {dd} s 晚，缓存永远写不完。'
          f'startupProbe 30 × 10 s 给足 {kill_deadline(PROBES["startup-30x10"]) + 10} s，成功后 liveness 才接手。')
    print()
    return out


# --------------------------------------------------------------------------- sleep：换模型
# 博客（vLLM 0.11.0，A100）：Qwen3-0.6B 与 Phi-3-vision 来回切 5 次，No Sleep 357.1 s，L1 112.6 s，L2 124.6 s；
# L1 唤醒 0.26 s / 0.82 s，L2 唤醒 0.85 s / 2.58 s
BLOG = dict(nosleep=357.1, l1=112.6, l2=124.6, l1_wake=(0.26, 0.82), l2_wake=(0.85, 2.58))


def show_sleep(a):
    print('== sleep：换模型的三种做法（每次切换的固定开销）')
    phi_gb = 4.2e9 * 2 / 1e9
    l1_bw = phi_gb / BLOG['l1_wake'][1]
    l2_bw = phi_gb / BLOG['l2_wake'][1]
    print(f'   用博客 Phi-3-vision（4.2B，约 {phi_gb:.1f} GB）的唤醒时间反推：L1 换回权重 ≈ {l1_bw:.1f} GB/s，'
          f'L2 重读权重 ≈ {l2_bw:.1f} GB/s（含其余初始化，只当量级）')
    w8 = 16.4
    cold, _ = startup(False)
    warm, _ = startup(True)
    rows = [('重启进程，无编译缓存', cold), ('重启进程，编译缓存命中', warm),
            ('sleep level 1 → wake_up', w8 / l1_bw), ('sleep level 2 → wake + reload_weights', w8 / l2_bw)]
    print(f'   Qwen3-8B（{w8} GB 权重）每次切换：')
    for name, s in rows:
        print(f'     {name:<38}{s:>7.1f} s')
    print(f'   博客实测 5 次切换总时长：No Sleep {BLOG["nosleep"]} s，L1 {BLOG["l1"]} s，L2 {BLOG["l2"]} s'
          f'（L1 比重启快 {BLOG["nosleep"] / BLOG["l1"]:.1f}×，含推理时间）')
    print('   L1 把权重卸到 CPU 的 pinned 内存，主机内存要装得下所有睡着的模型：')
    for name, gb, per_node in (('8 卡各睡 1 个 8B', 16.4, 8), ('8 卡各睡 2 个 8B（3 个模型轮转）', 16.4, 16),
                               ('2 组 TP4 各睡 1 个 70B', 141.2, 2)):
        print(f'     {name:<28} {per_node:>2} 份 × {gb:g} GB → {gb * per_node:>4.0f} GB pinned 主机内存')
    print('   L2 不占主机内存，但唤醒后要 collective_rpc("reload_weights") 并 reset_prefix_cache；KV 两档都丢，前缀缓存要重新预热')
    print()
    return dict(l1_bw=l1_bw, l2_bw=l2_bw, rows=rows)


# --------------------------------------------------------------------------- nccl：权重同步
def show_nccl(a):
    print('== nccl：训练端 rank 0 广播一次全量权重（packed，1 GiB × 2 轮转缓冲）的时间下限 = 字节 / 链路带宽')
    links = [('NVLink 节点内 ~350 GB/s', 350), ('400G IB 单网卡 50 GB/s', 50), ('100G RoCE 12.5 GB/s', 12.5)]
    models = [('Qwen3-8B BF16', MODELS['qwen3-8b'].params_b * 2), ('Llama-3.1-70B BF16', MODELS['llama3.1-70b'].params_b * 2),
              ('DeepSeek-V3 FP8', MODELS['deepseek-v3'].params_b * 1)]
    print(f'{"模型":<22}{"字节":>9}{"packed 次数":>12}' + ''.join(f'{n:>26}' for n, _ in links))
    out = {}
    for name, gb in models:
        chunks = math.ceil(gb * 1e9 / GiB)
        row = f'{name:<22}{gb:>7.0f}GB{chunks:>12}'
        for ln, bw in links:
            t = gb / bw
            out[(name, ln)] = t
            row += f'{t:>24.2f} s'
        print(row)
    print('   packed 两端各常驻 1 GiB × 2 = 2 GiB 显存，整个传输期间都在；推理端 gpu_memory_utilization 要留出这 2 GiB')
    print('   全部字节都要穿过 rank 0 的出口链路，所以单网卡带宽就是下限；同卡共置用 IPC 只传句柄，不搬字节')
    print()
    return out


# --------------------------------------------------------------------------- pause：更新权重前怎么停
def show_pause(a):
    rng = random.Random(a.seed)
    n = 64
    reqs = []
    for i in range(n):
        p, o = rng.randint(500, 2000), rng.randint(256, 4096)        # RL rollout：输出长度长尾
        g = rng.randint(0, o - 1)
        reqs.append((p, o, g))
    itl = sc.gpu_ms([(1, p + g) for p, o, g in reqs]) + sc.cpu_ms(n)
    remain = [o - g for p, o, g in reqs]
    ctx = sum(p + g for p, o, g in reqs)
    gen_done = sum(g for p, o, g in reqs)
    prefill_ms = sc.gpu_ms([(min(ctx, 8192), 0)]) * math.ceil(ctx / 8192)
    print(f'== pause：{n} 条 rollout 在飞（prompt 500–2000，输出 256–4096，已生成 {gen_done} token），decode 步长 {itl:.1f} ms')
    rows = [
        ('vLLM abort（默认）', 0, '在飞请求带部分结果返回', f'重投要把 prompt + 已生成共 {ctx} token 重新 prefill', '503', 1),
        ('vLLM wait', max(remain) * itl / 1e3, f'等最长的一条再跑 {max(remain)} 步', '无', '503', 1),
        ('vLLM keep + clear_cache', 0, '冻结在队列里', f'resume 后重算 {ctx} token 的 KV，约 {prefill_ms / 1e3:.1f} s', '照收照排', 2),
        ('vLLM keep, clear_cache=False', 0, '冻结在队列里', f'{ctx} token 的 KV 是旧权重算的（stale）', '照收照排', 2),
        ('SGLang abort（默认）', 0, 'waiting + running 全部返回', '同 vLLM abort', '文档未写', 1),
        ('SGLang retract', 0, 'running 退回 waiting', 'KV 可 flush，continue 后重算', '文档未写', 2),
        ('SGLang in_place', 0, '原地冻结', 'flush_cache 会失败，只能沿用旧 KV', '文档未写', 2),
    ]
    print(f'{"模式":<30}{"换权重前等":>10}  {"在飞请求":<24}{"新请求":<10}{"版本段":>6}  代价')
    for name, wait, inflight, cost, new, spans in rows:
        print(f'{name:<30}{wait:>8.1f} s  {inflight:<24}{new:<10}{spans:>6}  {cost}')
    print(f'   wait 的等待由最长尾决定：平均剩 {sum(remain) / n:.0f} 步，最长剩 {max(remain)} 步；keep 不等，但一条输出会拼着新旧两版权重')
    print()
    return dict(itl=itl, wait=max(remain) * itl / 1e3, ctx=ctx, gen=gen_done, prefill_s=prefill_ms / 1e3)


SCEN = dict(route=show_route, hot=show_hot, stale=show_stale, probe=show_probe, sleep=show_sleep,
            nccl=show_nccl, pause=show_pause)


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', choices=list(SCEN) + ['all'], default='route')
    ap.add_argument('--replicas', type=int, default=4)
    ap.add_argument('--clients', type=int, default=1, help='API server 个数（--api-server-count）')
    ap.add_argument('--rate', type=float, default=32.0)
    ap.add_argument('--duration', type=float, default=60.0)
    ap.add_argument('--n-prefix', dest='n_prefix', type=int, default=128)
    ap.add_argument('--zipf', type=float, default=0.8)
    ap.add_argument('--hot', type=float, default=0.5)
    ap.add_argument('--snapshot-ms', dest='snapshot_ms', type=float, default=100.0)
    ap.add_argument('--min-match', dest='min_match', type=int, default=1024)
    ap.add_argument('--factor', type=float, default=2.0)
    ap.add_argument('--gap', type=int, default=1024)
    ap.add_argument('--budget', type=int, default=8192)
    ap.add_argument('--max-num-seqs', dest='max_num_seqs', type=int, default=256)
    ap.add_argument('--download-gbps', dest='download_gbps', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=0)
    return ap


def main():
    a = parser().parse_args()
    for name in (SCEN if a.scenario == 'all' else [a.scenario]):
        SCEN[name](a)


if __name__ == '__main__':
    main()
