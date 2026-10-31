"""vLLM / SGLang 指标口径的离线复现（只依赖标准库，不需要 GPU）。

对照的源码（vllm 主干 0d6f92a，sglang 主干 8fabf5e）：
  vllm   v1/metrics/stats.py:39-118     CachingMetrics：最近 1000 条请求的命中率，reset_prefix_cache 时清零
         v1/metrics/stats.py:426-470    PromptTokenStats：computed + local_cache_hit + external_kv_transfer = total
         v1/metrics/stats.py:515-560    TTFT = 帧到达时刻 − arrival_time；ITL = 本次输出事件 − 上次输出事件（每个事件一个样本）
         v1/metrics/stats.py:594-613    queued / prefill / decode / inference 四段；TPOT = decode / (n_gen − 1)
         v1/metrics/buckets.py          各直方图族的默认桶
         v1/metrics/loggers.py:160-165  日志里的 "Avg prompt throughput" 只算 computed（不含命中缓存的 token）
         v1/metrics/loggers.py:1262     vllm:prompt_tokens 计入全部 prompt token
         v1/core/block_pool.py:921      get_usage = 1 − free / (num_gpu_blocks − 1)；ref_cnt 为 0 的缓存块在 free 队列里
         profiler/wrapper.py:36-200     WorkerProfiler.start/step/stop：delay_iterations、max_iterations
         profiler/wrapper.py:279-321    torch.profiler.schedule(wait, warmup, active, repeat=1)
         benchmarks/serve.py:628-661    vllm bench serve：TPOT = (latency − ttft) / (out − 1)，out ≤ 1 的请求不进 TPOT 统计
  sglang srt/observability/metrics_collector.py:1965-2040  TTFT / e2e / ITL 默认桶
         srt/observability/metrics_collector.py:2161       ITL 按 token 加权：interval / num_new_tokens，记 num_new_tokens 次
         srt/managers/scheduler_components/pool_stats_observer.py:151-160  token_usage = 各池最大值，num_used = total − (available + evictable)

场景：
  intervals    一条请求的事件时间线怎么拆成 queue / prefill / decode / inference / TTFT / TPOT，抢占落在哪一段
  itl          同一批 token 时间戳，vLLM 事件加权 ITL、请求加权 TPOT、SGLang token 加权 ITL 三种口径；投机解码时差多少
  buckets      histogram_quantile 在两家默认桶上的插值误差，以及自定义桶的修正
  hitrate      日志里的 1000 条窗口命中率 vs PromQL rate() 窗口，在流量高低两档下的滞后
  kvusage      kv_cache_usage_perc 不把空闲的前缀缓存块算作占用
  throughput   日志 prompt 吞吐（computed） vs rate(vllm:prompt_tokens)（total）
  profiler     delay / max_iterations / wait / warmup / active 落到第几个 engine step
  cardinality  直方图时间序列数：桶数 × label 组合 × engine 数

用法：
  python3 sim_metrics.py --scenario intervals
  python3 sim_metrics.py --scenario itl --rate 10
  python3 sim_metrics.py --scenario buckets --rate 10
  python3 sim_metrics.py --scenario hitrate
  python3 sim_metrics.py --scenario kvusage
  python3 sim_metrics.py --scenario throughput
  python3 sim_metrics.py --scenario profiler --delay 3 --max-iters 5 --wait 1 --warmup 2 --active 5
  python3 sim_metrics.py --scenario cardinality --dp 8
  python3 sim_metrics.py --scenario all

延迟数据来自 06 章的步进模拟器（H100 SXM + Qwen3-8B BF16 的 roofline 步长），只用来看口径差异，不要拿去报数。
"""
import argparse
import math
import os
import random
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '06-chunked-prefill'))
import sim_chunked as sc  # noqa: E402

# --------------------------------------------------------------------------- 默认桶（秒）
VLLM_BUCKETS = dict(
    request_latency=[0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                     120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0],
    time_to_first_token=[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0,
                         7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0],
    inter_token_latency=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5,
                         10.0, 20.0, 40.0, 80.0],
    iteration_tokens=[1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
    request_params_n=[1, 2, 5, 10, 20],
    request_num_preemptions=[1, 2, 3, 4, 5, 10, 20],
)
SGL_BUCKETS = dict(
    time_to_first_token=[0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400],
    inter_token_latency=[0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.060,
                         0.080, 0.100, 0.200, 0.400, 0.600, 0.800, 1.000, 2.000, 4.000, 6.000, 8.000],
    e2e_request_latency=[0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400, 600,
                         1200, 1800, 2400],
)
# loggers.py:862-1030 每个直方图用哪一族桶
VLLM_HISTOGRAMS = [
    ('request_prompt_tokens', 'request_tokens'), ('request_generation_tokens', 'request_tokens'),
    ('iteration_tokens_total', 'iteration_tokens'), ('request_max_num_generation_tokens', 'request_tokens'),
    ('request_params_n', 'request_params_n'), ('request_params_max_tokens', 'request_tokens'),
    ('time_to_first_token_seconds', 'time_to_first_token'),
    ('inter_token_latency_seconds', 'inter_token_latency'),
    ('request_time_per_output_token_seconds', 'inter_token_latency'),
    ('e2e_request_latency_seconds', 'request_latency'), ('request_queue_time_seconds', 'request_latency'),
    ('request_inference_time_seconds', 'request_latency'), ('request_prefill_time_seconds', 'request_latency'),
    ('request_decode_time_seconds', 'request_latency'), ('request_num_preemptions', 'request_num_preemptions'),
    ('request_prefill_kv_computed_tokens', 'request_tokens'),
]


def build_1_2_5(max_value):
    """buckets.py build_1_2_5_buckets：1, 2, 5, 10, 20, 50 … 不超过 max_value。"""
    out, e = [], 0
    while True:
        for m in (1, 2, 5):
            v = m * 10 ** e
            if v > max_value:
                return out
            out.append(v)
        e += 1


def bucket_count(family, max_model_len):
    if family == 'request_tokens':
        return len(build_1_2_5(max_model_len))
    return len(VLLM_BUCKETS[family])


# --------------------------------------------------------------------------- Prometheus 直方图
def observe(bounds, xs):
    """返回累计计数列表，最后一个是 +Inf。"""
    cnt = [0] * (len(bounds) + 1)
    for x in xs:
        for i, b in enumerate(bounds):
            if x <= b:
                cnt[i] += 1
                break
        else:
            cnt[-1] += 1
    cum, s = [], 0
    for c in cnt:
        s += c
        cum.append(s)
    return cum


def histogram_quantile(q, bounds, cum):
    """PromQL histogram_quantile 的线性插值（promql/quantile.go bucketQuantile）。"""
    total = cum[-1]
    if total == 0:
        return float('nan')
    rank = q * total
    for i, c in enumerate(cum):
        if c >= rank:
            break
    if i == len(bounds):              # 落进 +Inf 桶：返回最后一个有限上界
        return bounds[-1]
    end = bounds[i]
    start = bounds[i - 1] if i > 0 else (0.0 if end > 0 else end)
    prev = cum[i - 1] if i > 0 else 0
    inb = c - prev
    if inb == 0:
        return end
    return start + (end - start) * (rank - prev) / inb


def exact_q(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1)] if xs else float('nan')


# --------------------------------------------------------------------------- 负载
def sim_args(**kw):
    a = dict(scenario='mixed', mode='vllm', budget=8192, threshold=0, max_num_seqs=1024, max_model_len=40960,
             max_prefill_tokens=16384, rate=10.0, duration=120.0, long_at=2000.0, n=2000, bs=256, sync=False,
             sweep=False, seed=0)
    a.update(kw)
    return SimpleNamespace(**a)


_CACHE = {}


def sim_reqs(rate, mode='vllm', budget=8192, duration=120.0):
    key = (rate, mode, budget, duration)
    if key not in _CACHE:
        _CACHE[key] = sc.run(sim_args(rate=rate, budget=budget, duration=duration), mode)['reqs']
    return _CACHE[key]


def ms(x):
    return sc.fmt(x * 1e3)


# --------------------------------------------------------------------------- intervals
def request_intervals(ev):
    """ev: dict(arrival, queued, scheduled, tokens=[t1, t2, ...])（秒）。stats.py:515-613 的算法。

    arrival 是前端墙钟（开始 tokenize 的时刻），其余是 engine core 单调时钟；这里假设两者对齐。
    scheduled 只记第一次 SCHEDULED（被抢占后再调度不更新，stats.py:577）。
    """
    toks = ev['tokens']
    n = len(toks)
    ttft = toks[0] - ev['arrival']
    itl = [b - a for a, b in zip(toks, toks[1:])]
    queued = ev['scheduled'] - ev['queued']
    prefill = toks[0] - ev['scheduled']
    decode = toks[-1] - toks[0]
    inference = toks[-1] - ev['scheduled']
    tpot = decode / (n - 1) if n - 1 > 0 else 0.0
    e2e = toks[-1] - ev['arrival']
    return dict(ttft=ttft, queued=queued, prefill=prefill, decode=decode, inference=inference, tpot=tpot,
                e2e=e2e, itl_max=max(itl) if itl else 0.0, n=n)


def timeline(preempt=None, pause=1.5, step=0.02, n_out=200, prefill_s=0.30, tokenize=0.01, queue_s=0.40):
    """构造一条请求：tokenize 10 ms，排队 400 ms，prefill 300 ms，decode 每步 20 ms。

    preempt='prefill'：prefill 中途被抢占，暂停 pause 秒后重算；preempt='decode'：第 100 个 token 后暂停 pause 秒。
    """
    arrival = 0.0
    queued = arrival + tokenize
    scheduled = queued + queue_s
    first = scheduled + prefill_s + (pause if preempt == 'prefill' else 0.0)
    toks = [first]
    for i in range(1, n_out):
        t = toks[-1] + step
        if preempt == 'decode' and i == n_out // 2:
            t += pause
        toks.append(t)
    return dict(arrival=arrival, queued=queued, scheduled=scheduled, tokens=toks)


def show_intervals(a):
    print('== intervals：一条请求 200 个输出 token，事件时间线 → 各段指标（stats.py:594-613） ==')
    print('   tokenize 10 ms，排队 400 ms，prefill 300 ms，decode 20 ms/token；抢占暂停 1.5 s')
    rows = [('无抢占', timeline()), ('prefill 中抢占', timeline('prefill')), ('decode 中抢占', timeline('decode'))]
    print(f"{'场景':<14}{'TTFT':>10}{'queue':>10}{'prefill':>10}{'decode':>10}{'inference':>11}"
          f"{'e2e':>10}{'TPOT':>10}{'ITL max':>10}")
    for name, ev in rows:
        r = request_intervals(ev)
        print(f"{name:<12}{ms(r['ttft']):>10}{ms(r['queued']):>10}{ms(r['prefill']):>10}{ms(r['decode']):>10}"
              f"{ms(r['inference']):>11}{ms(r['e2e']):>10}{ms(r['tpot']):>10}{ms(r['itl_max']):>10}")
    print('   读法：TTFT 比 queue+prefill 多出的 10 ms 是 tokenize（arrival 从前端开始计）。')
    print('   prefill 中抢占 → TTFT、prefill、inference 变大，TPOT 不变；decode 中抢占 → decode、TPOT、ITL max 变大。')
    print('   queue 只记第一次排队；被抢占后重新排队的时间不进 queue，而是进 prefill 或 decode。')
    print()


# --------------------------------------------------------------------------- itl
def spec_events(emits, rng, accept=0.7, k=3):
    """把逐 token 时间戳并成投机解码的输出事件：每步 1 + 接受数个 token，接受数服从截断几何分布（≤ k）。

    事件时刻取这组 token 里最后一个的时刻，token 的平均间隔不变，只改变"一个事件带几个 token"。
    """
    ev = [(emits[0], 1)]                          # prefill 吐出的首 token
    i = 1
    while i < len(emits):
        m = 1
        while m <= k and rng.random() < accept:
            m += 1
        m = min(m, len(emits) - i)
        ev.append((emits[i + m - 1], m))
        i += m
    return ev


def itl_views(reqs, spec=False, seed=0):
    rng = random.Random(seed)
    v_itl, s_itl, tpot = [], [], []
    for r in reqs:
        e = [t / 1e3 for t in r.emits]
        ev = spec_events(e, rng) if spec else [(t, 1) for t in e]
        for (t0, _), (t1, m) in zip(ev, ev[1:]):
            v_itl.append(t1 - t0)                         # vLLM：一个输出事件一个样本
            s_itl.extend([(t1 - t0) / m] * m)              # SGLang：interval / num_new_tokens，记 m 次
        if len(e) > 1:
            tpot.append((e[-1] - e[0]) / (len(e) - 1))     # 两家 TPOT 都按请求
    return v_itl, s_itl, tpot


def show_itl(a):
    reqs = sim_reqs(a.rate)
    print(f'== itl：同一批 token 时间戳的三种口径（06 章混合负载，rate {a.rate:g}，{len(reqs)} 条请求） ==')
    print(f"{'解码方式':<10}{'口径':<30}{'样本数':>9}{'均值':>10}{'p50':>10}{'p99':>10}")
    ratio = 0.0
    for spec in (False, True):
        v, s, t = itl_views(reqs, spec)
        if spec:
            ratio = len(s) / len(v)
        tag = '投机 k=3' if spec else '逐 token'
        for name, xs in (('vLLM ITL（事件加权）', v), ('SGLang ITL（token 加权）', s), ('TPOT（请求加权）', t)):
            mean = sum(xs) / len(xs)
            print(f"{tag:<10}{name:<26}{len(xs):>9,}{ms(mean):>10}{ms(exact_q(xs, .5)):>10}{ms(exact_q(xs, .99)):>10}")
    print('   读法：逐 token 时 vLLM 与 SGLang 的 ITL 完全相同；投机解码时 vLLM ITL 反映"每步多久"，')
    print(f'   SGLang ITL 反映"每个 token 摊多少"，两者均值之比 = 每个输出事件平均带的 token 数 {ratio:.2f}。TPOT 不受影响。')
    print('   单 token 请求：Prometheus 的 request_time_per_output_token 记 0（stats.py:613），')
    print('   vllm bench serve 的 TPOT 直接剔除（serve.py:628-661），SGLang 的 TPOT 直方图也剔除 completion < 2 的请求。')
    print()


# --------------------------------------------------------------------------- buckets
def show_buckets(a):
    reqs = sim_reqs(a.rate)
    ttft = [(r.emits[0] - r.arrival) / 1e3 for r in reqs]
    itl = [(b - x) / 1e3 for r in reqs for x, b in zip(r.emits, r.emits[1:])]
    custom_itl = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.5]
    custom_ttft = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
    print(f'== buckets：histogram_quantile 在默认桶上的插值误差（rate {a.rate:g}） ==')
    print(f"{'指标':<6}{'分位':>5}{'真实值':>11}{'vLLM 默认桶':>18}{'SGLang 默认桶':>18}{'自定义桶':>18}")
    for name, xs, vb, sb, cb in (('TTFT', ttft, VLLM_BUCKETS['time_to_first_token'], SGL_BUCKETS['time_to_first_token'],
                                  custom_ttft),
                                 ('ITL', itl, VLLM_BUCKETS['inter_token_latency'], SGL_BUCKETS['inter_token_latency'],
                                  custom_itl)):
        cv, cs, cc = observe(vb, xs), observe(sb, xs), observe(cb, xs)
        for q in (0.5, 0.9, 0.99):
            tv = exact_q(xs, q)
            ev, es, ec = histogram_quantile(q, vb, cv), histogram_quantile(q, sb, cs), histogram_quantile(q, cb, cc)
            cell = lambda e: f"{ms(e)} ({(e / tv - 1) * 100:+.0f}%)"
            print(f"{name:<6}{'p%d' % round(q * 100):>5}{ms(tv):>11}{cell(ev):>20}{cell(es):>20}{cell(ec):>20}")
    lo = sum(1 for x in itl if x <= 0.01) / len(itl)
    print(f'   ITL ≤ 10 ms 的样本占 {lo:.0%}，在 vLLM 默认桶里全挤在第一个桶 (0, 0.01]，p50 只能在 0–10 ms 线性插值。')
    print('   自定义桶（JSON 里的数值就是上面两列的边界）：')
    print(f'   --custom-histogram-buckets \'{{"time_to_first_token": {custom_ttft},')
    print(f'                                 "inter_token_latency": {custom_itl}}}\'')
    print()
    return dict(itl=itl, ttft=ttft)


# --------------------------------------------------------------------------- hitrate
class CachingMetrics:
    """stats.py:39-118 的逐行移植。"""

    def __init__(self, max_recent_requests=1000):
        self.max = max_recent_requests
        self.req = self.q = self.h = 0
        self.dq = []

    def observe(self, requests, queries, hits, reset=False):
        if reset:
            self.req = self.q = self.h = 0
            self.dq.clear()
        if requests == 0:
            return
        self.dq.append((requests, queries, hits))
        self.req += requests
        self.q += queries
        self.h += hits
        while len(self.dq) > 1 and self.req > self.max:
            r, q, h = self.dq.pop(0)
            self.req -= r
            self.q -= q
            self.h -= h

    @property
    def hit_rate(self):
        return self.h / self.q if self.q else 0.0


def hitrate_series(rate, t_change=600.0, before=0.8, after=0.2, horizon=1200.0, prompt=2000, seed=0):
    """泊松到达，t_change 时刻命中率从 before 掉到 after。返回 (时刻, 窗口命中率, rate1m, rate5m) 的采样函数。"""
    rng = random.Random(seed)
    t, events = 0.0, []
    while t < horizon:
        t += rng.expovariate(rate)
        p = before if t < t_change else after
        hits = int(prompt * min(1.0, max(0.0, rng.gauss(p, 0.1))) // 16 * 16)
        events.append((t, prompt, hits))
    return events


def window_rate(events, now, win):
    q = h = 0
    for t, qq, hh in events:
        if now - win < t <= now:
            q += qq
            h += hh
    return h / q if q else float('nan')


def log_rate(events, now):
    cm = CachingMetrics()
    for t, q, h in events:
        if t > now:
            break
        cm.observe(1, q, h)
    return cm.hit_rate


def show_hitrate(a):
    print('== hitrate：t = 600 s 时命中率从 80% 掉到 20%，各口径多久看到 ==')
    print('   日志 "Prefix cache hit rate" = 最近 1000 条请求（stats.py:39）；PromQL = rate(hits[w]) / rate(queries[w])')
    print(f"{'流量':>9}{'t − 600':>9}{'日志窗口':>10}{'rate[1m]':>10}{'rate[5m]':>10}")
    for rate in (2.0, 20.0):
        ev = hitrate_series(rate)
        for dt in (-1, 30, 60, 120, 300, 500):
            now = 600 + dt
            print(f"{rate:>6g}/s{dt:>9}{log_rate(ev, now):>10.1%}{window_rate(ev, now, 60):>10.1%}"
                  f"{window_rate(ev, now, 300):>10.1%}")
    print('   读法：日志窗口按请求条数算，低流量（2 req/s）下 1000 条 ≈ 8 分钟，掉线后几分钟还在报高命中率；')
    print('   高流量（20 req/s）下 1000 条 ≈ 50 s，反而比 rate[5m] 更快。告警用 rate[1m]，看板用 rate[5m]。')
    print()


# --------------------------------------------------------------------------- kvusage
def show_kvusage(a):
    n = 24957                                   # 02 章 H100 + Qwen3-8B 的块数
    print('== kvusage：kv_cache_usage_perc 与前缀缓存（block_pool.py:921） ==')
    print(f'   num_gpu_blocks = {n}；usage = 1 − free / (num_gpu_blocks − 1)，ref_cnt = 0 的缓存块排在 free 队列里')
    print(f"{'时刻':<26}{'运行中占用块':>12}{'空闲但有缓存':>13}{'真正空块':>10}{'vLLM usage':>12}{'SGLang usage':>14}")
    states = [('峰值：200 条在跑', 21000, 1500), ('流量退潮：20 条在跑', 2100, 19800),
              ('全部结束', 0, 22000), ('reset_prefix_cache 之后', 0, 0)]
    for name, used, cached in states:
        free = n - 1 - used                     # null block 不计
        empty = free - cached
        v = 1 - free / (n - 1)
        s = round(used / (n - 1), 2)            # SGLang：num_used = total − (available + evictable)，round 到 2 位
        print(f"{name:<22}{used:>12,}{cached:>13,}{empty:>10,}{v:>12.1%}{s:>14.2f}")
    print('   读法：两家的"使用率"都只算被运行中请求引用的块，前缀缓存块不算。退潮时 usage 8.4%，但 79% 的块里装着可复用的前缀，')
    print('   要看缓存状况，得看 prefix_cache_hits / queries 和 --kv-cache-metrics 打开后的 kv_block_* 直方图。')
    print()


# --------------------------------------------------------------------------- throughput
def show_throughput(a):
    print('== throughput：日志 prompt 吞吐 vs rate(vllm:prompt_tokens) ==')
    print('   每秒 10 条请求、每条 prompt 4000 token；日志只累加 computed（loggers.py:163-164），counter 累加 total（1262）')
    print(f"{'本地命中':>8}{'外部 KV':>9}{'日志 prompt tok/s':>19}{'rate(prompt_tokens)':>21}"
          f"{'by_source local_compute':>25}{'cache_hit':>11}{'external':>10}")
    for loc, ext in ((0.0, 0.0), (0.5, 0.0), (0.9, 0.0), (0.3, 0.6)):
        total = 10 * 4000
        hit = int(total * loc)
        exk = int(total * ext)
        comp = total - hit - exk
        print(f"{loc:>8.0%}{ext:>9.0%}{comp:>19,}{total:>21,}{comp:>25,}{hit:>11,}{exk:>10,}")
    print('   读法：命中率 90% 时日志里的 prompt 吞吐只有 counter 的十分之一，两者都没错，问的问题不同。')
    print('   看 GPU 真干了多少 prefill，用日志口径或 prompt_tokens_by_source{source="local_compute"}。')
    print()


# --------------------------------------------------------------------------- profiler
def torch_schedule_action(idx, wait, warmup, active):
    """torch.profiler.schedule(wait, warmup, active, repeat=1) 在第 idx 步的动作。"""
    if idx < wait:
        return 'NONE'
    if idx < wait + warmup:
        return 'WARMUP'
    if idx < wait + warmup + active:
        return 'RECORD_SAVE' if idx == wait + warmup + active - 1 else 'RECORD'
    return 'NONE'


def profile_steps(delay=0, max_iters=0, wait=0, warmup=0, active=5, n_steps=20):
    """逐行模拟 WorkerProfiler.step()（wrapper.py:130-160）+ TorchProfilerWrapper._profiler_step（437-454）。

    /start_profile 在第 1 个 engine step 之前到达。每个 execute_model 前调一次 step()（gpu_worker.py:1222）。
    返回每个 step 的 (step 号, 是否在 profiler 运行中执行, schedule 动作, 是否计入 max_iterations)。
    """
    uses_schedule = warmup > 0 or wait > 0
    warm_left = max(wait + warmup - 1, 0)
    active_on, running = True, False
    sched_idx = 0
    count = prof_iters = 0
    if delay == 0:
        running = True                           # start() 时 _call_start；schedule 处于第 0 步
    rows = []
    for s in range(1, n_steps + 1):
        if not active_on:
            rows.append((s, False, '-', False))
            continue
        count += 1
        if not running and delay > 0 and count == delay:
            running = True
            sched_idx = 0
        counted = False
        action = 'ALL' if not uses_schedule else '-'
        if running:
            if uses_schedule:
                sched_idx += 1
                action = torch_schedule_action(sched_idx, wait, warmup, active)
                if warm_left > 0:
                    warm_left -= 1
                else:
                    counted = True
            else:
                counted = True
            if counted:
                prof_iters += 1
        if max_iters > 0 and running and prof_iters > max_iters:
            running = False
            active_on = False
            rows.append((s, False, 'STOP', False))
            continue
        rows.append((s, running, action, counted))
    return rows


def show_profiler(a):
    cfgs = [dict(delay=0, max_iters=0), dict(delay=3, max_iters=5),
            dict(delay=0, max_iters=0, wait=1, warmup=2, active=5),
            dict(delay=a.delay, max_iters=a.max_iters, wait=a.wait, warmup=a.warmup, active=a.active)]
    print('== profiler：--profiler-config 的迭代控制落到第几个 engine step（wrapper.py:36-200, 279-321） ==')
    for c in cfgs:
        rows = profile_steps(n_steps=14, **c)
        rec = [s for s, run, act, _ in rows if run and act in ('ALL', 'RECORD', 'RECORD_SAVE')]
        tag = ' '.join(f'{k}={v}' for k, v in c.items())
        line = ''.join('R' if (run and act in ('ALL', 'RECORD', 'RECORD_SAVE')) else
                       ('w' if run and act in ('WARMUP', 'NONE') else ('S' if act == 'STOP' else '.'))
                       for s, run, act, _ in rows)
        print(f'  {tag:<40} step 1..14: {line}   落盘的 step：{rec if rec else "无"}')
    print('   图例：R 记录并落盘；w profiler 已启动但不记录（wait / warmup / schedule 已走完）；S 自动 stop；. 未运行')
    print('   读法：delay=3 时第 3 个 step 就开始记（计数先 +1 再比较，跳过的是前 2 个）；max_iterations=5 记满 5 个 step 后，')
    print('   在第 6 个计数 step 的 execute 之前 stop。warmup 只有 wait + warmup − 1 个 step 被丢弃，因为 start() 已占掉 schedule 第 0 步。')
    print('   不设 max_iterations 时 schedule 跑完 active 也不会自动 stop，必须调 /stop_profile 才会写 profiler_out 表。')
    print()


# --------------------------------------------------------------------------- cardinality
def series_count(n_buckets, label_sets):
    return (n_buckets + 1 + 2) * label_sets      # le 桶 + le="+Inf" + _sum + _count


def show_cardinality(a):
    print(f'== cardinality：直方图时间序列数（max_model_len {a.max_model_len}，labels = model_name × engine） ==')
    fams = {}
    for name, fam in VLLM_HISTOGRAMS:
        fams.setdefault(fam, []).append(name)
    print(f"{'桶族':<26}{'桶数':>6}{'直方图个数':>10}{'每 engine 序列':>15}")
    per_engine = 0
    for fam, names in fams.items():
        nb = bucket_count(fam, a.max_model_len)
        k = len(names) * series_count(nb, 1)
        per_engine += k
        print(f"{fam:<26}{nb:>6}{len(names):>10}{k:>15}")
    print(f"{'合计':<26}{'':>6}{len(VLLM_HISTOGRAMS):>10}{per_engine:>15}")
    for dp in (1, a.dp):
        print(f'   DP={dp}：{per_engine * dp:,} 条直方图序列（engine label 每个 DP rank 一份）')
    extra = 40 - len(VLLM_BUCKETS['inter_token_latency'])
    print(f'   把 inter_token_latency 族改成 40 个桶：每 engine 多 {2 * extra} 条（ITL 和 TPOT 共用这一族），DP={a.dp} 多 {2 * extra * a.dp} 条')
    sg = series_count(len(SGL_BUCKETS['time_to_first_token']), 2) + series_count(len(SGL_BUCKETS['inter_token_latency']), 1) \
        + series_count(len(SGL_BUCKETS['e2e_request_latency']), 1)
    print(f'   对照 SGLang 三个延迟直方图（TTFT 多一个 is_streaming label）：每个 model_name 约 {sg} 条')
    print()


SCEN = dict(intervals=show_intervals, itl=show_itl, buckets=show_buckets, hitrate=show_hitrate,
            kvusage=show_kvusage, throughput=show_throughput, profiler=show_profiler, cardinality=show_cardinality)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', choices=list(SCEN) + ['all'], default='intervals')
    ap.add_argument('--rate', type=float, default=10.0)
    ap.add_argument('--delay', type=int, default=2)
    ap.add_argument('--max-iters', dest='max_iters', type=int, default=4)
    ap.add_argument('--wait', type=int, default=0)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--active', type=int, default=3)
    ap.add_argument('--dp', type=int, default=8)
    ap.add_argument('--max-model-len', dest='max_model_len', type=int, default=40960)
    a = ap.parse_args()
    for name in (SCEN if a.scenario == 'all' else [a.scenario]):
        SCEN[name](a)


if __name__ == '__main__':
    main()
