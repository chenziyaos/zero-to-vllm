"""用一个最小的连续批处理模拟器生成逐请求时间线，再按 vllm bench serve 的口径算 TTFT、TPOT、ITL、goodput。

不需要 GPU，只依赖标准库。两件事分开：
  1. metrics()：逐行对照 vllm/benchmarks/serve.py calculate_metrics 的口径（主干 0d6f92a）。
     也可以直接喂 vllm bench serve --save-result --save-detailed 产出的 JSON：
         python3 sim_goodput.py --from-json result.json --goodput ttft:500 tpot:30
  2. simulate()：H100 + Qwen3-8B 的粗粒度引擎模型（step 时间 = max(读字节/带宽, 算 FLOP/算力) + 固定开销），
     调度规则取 vLLM V1 的默认：先排 running 的 decode，再用剩余 token 预算做 chunked prefill，KV 不够时抢占最后加入的请求。
     模拟器只为说明指标的形状，不替代真机压测。

用法：
    python3 sim_goodput.py                         # 开环 request-rate 扫描 + 闭环 max-concurrency 扫描
    python3 sim_goodput.py --burstiness 0.5        # 更突发的到达
    python3 sim_goodput.py --isl 4096 --osl 256    # 换负载
    python3 sim_goodput.py --demo-itl              # 一条请求被长 prefill 卡住时，TPOT 和 ITL 的差别
    python3 sim_goodput.py --steady 64             # 全程吞吐 vs SGLang 式稳态窗口吞吐
"""
import argparse
import heapq
import json
import math
import random
import statistics

# ---------------------------------------------------------------- 口径：calculate_metrics


def pct(xs, p):
    """numpy.percentile 的默认 linear 插值。"""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = math.floor(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def metrics(outs, dur, slo=None, percentiles=(50, 99)):
    """outs: [{'ok','ttft','itl':[...],'latency','output_len','prompt_len'}]，时间单位秒。"""
    ttfts, tpots, all_tpots, itls, e2els = [], [], [], [], []
    done = good = total_out = total_in = 0
    for o in outs:
        if not o['ok']:
            continue
        done += 1
        n = o['output_len']
        total_out += n
        total_in += o['prompt_len']
        tpot = 0.0
        if n > 1:
            tpot = (o['latency'] - o['ttft']) / (n - 1)   # 每条请求先平均，再跨请求统计
            tpots.append(tpot)
        all_tpots.append(tpot)                             # goodput 用的列表：单 token 请求记 0
        itls += o['itl']                                   # 所有请求的所有间隔混在一起
        ttfts.append(o['ttft'])
        e2els.append(o['latency'])
    if slo:
        cols, lim = [], []
        for k, arr in (('ttft', ttfts), ('tpot', all_tpots), ('e2el', e2els)):
            if k in slo:
                cols.append(arr)
                lim.append(slo[k] / 1000)
        for row in zip(*cols):
            if all(s >= r for s, r in zip(lim, row)):
                good += 1
    r = dict(completed=done, dur=dur, req_s=done / dur, goodput=good / dur, out_tok_s=total_out / dur,
             slo_attain=good / len(outs) if outs else 0,   # SGLang 的 slo_attainment：good / 发出的请求
             total_tok_s=(total_in + total_out) / dur,
             mean_ttft=statistics.fmean(ttfts) * 1e3 if ttfts else 0,
             mean_tpot=statistics.fmean(tpots) * 1e3 if tpots else 0,
             mean_itl=statistics.fmean(itls) * 1e3 if itls else 0,
             mean_e2el=statistics.fmean(e2els) * 1e3 if e2els else 0)
    for p in percentiles:
        r[f'p{p}_ttft'] = pct(ttfts, p) * 1e3
        r[f'p{p}_tpot'] = pct(tpots, p) * 1e3
        r[f'p{p}_itl'] = pct(itls, p) * 1e3
        r[f'p{p}_e2el'] = pct(e2els, p) * 1e3
    return r


def load_json(path):
    d = json.load(open(path))
    outs = []
    for i, pl in enumerate(d['input_lens']):
        ok = not d['errors'][i]
        outs.append(dict(ok=ok, ttft=d['ttfts'][i], itl=d['itls'][i], latency=d['latencies'][i],
                         output_len=d['output_lens'][i], prompt_len=pl))
    return outs, d['duration']


# ---------------------------------------------------------------- 引擎模型

H100 = dict(bw=3.35e12, flops=989e12)
QWEN3_8B = dict(params=8.19e9, weight_bytes=16.38e9, kv_tok=147_456)


class Engine:
    def __init__(self, budget=8192, max_seqs=1024, kv_tokens=399_296, block=16, bw_eff=0.80, fl_eff=0.55,
                 overhead=1.2e-3, model=QWEN3_8B, gpu=H100):
        self.budget, self.max_seqs, self.block = budget, max_seqs, block
        self.free_blocks = kv_tokens // block
        self.bw, self.fl = gpu['bw'] * bw_eff, gpu['flops'] * fl_eff
        self.ovh, self.m = overhead, model

    def step_time(self, prefill_tokens, decode_seqs, ctx_tokens):
        byts = self.m['weight_bytes'] + ctx_tokens * self.m['kv_tok']
        flop = 2 * self.m['params'] * (prefill_tokens + decode_seqs)
        return max(byts / self.bw, flop / self.fl) + self.ovh


def arrivals(n, rate, burstiness, seed):
    """vllm/benchmarks/serve.py get_request 的到达间隔：gamma(shape=burstiness, scale=1/(rate*burstiness))，再整体缩放到 n/rate。"""
    rnd = random.Random(seed)
    if rate == math.inf:
        return [0.0] * n
    d, t = [], 0.0
    for _ in range(n):
        t += 1 / rate if burstiness == math.inf else rnd.gammavariate(burstiness, 1 / (rate * burstiness))
        d.append(t)
    scale = (n / rate) / d[-1]
    return [0.0] + [x * scale for x in d[:-1]]


def simulate(n=1000, isl=1024, osl=128, rate=math.inf, burstiness=1.0, max_concurrency=None, seed=0, eng=None,
             force_long=None):
    """返回 (outs, dur, stats)。force_long=(index, length) 在第 index 个请求处插一条超长 prompt。"""
    eng = eng or Engine()
    arr = arrivals(n, rate, burstiness, seed)
    reqs = []
    for i in range(n):
        pl = force_long[1] if force_long and i == force_long[0] else isl
        reqs.append(dict(i=i, arrive=arr[i], send=None, pl=pl, ol=osl, done_tok=0, gen=0, times=[], blocks=0))
    pending = list(range(n))             # 还没发出去的请求（客户端侧）
    waiting, running = [], []            # 服务端
    inflight = 0
    t = 0.0
    finished = 0
    preempt = 0
    steps = []
    while finished < n:
        # 客户端：到点的请求在并发闸门允许时发出（闸门就是 bench 里的 asyncio.Semaphore）
        while pending and reqs[pending[0]]['arrive'] <= t and (max_concurrency is None or inflight < max_concurrency):
            r = reqs[pending.pop(0)]
            r['send'] = max(t, r['arrive'])
            waiting.append(r)
            inflight += 1
        if not waiting and not running:
            nxt = reqs[pending[0]]['arrive']
            t = max(t, nxt)
            continue
        budget = eng.budget
        sched_dec, sched_pre = [], []
        ctx = 0
        # 1) running 优先：decode 每条 1 token；prefill 未完的继续切块
        for r in list(running):
            need = 1 if r['done_tok'] >= r['pl'] + r['gen'] else min(budget, r['pl'] + r['gen'] - r['done_tok'])
            if need <= 0 or budget <= 0:
                continue
            new_total = r['done_tok'] + need
            want = math.ceil(new_total / eng.block) - r['blocks']
            while want > eng.free_blocks:
                victim = running.pop()   # 抢占最后加入的
                if victim is r:
                    break
                eng.free_blocks += victim['blocks']
                victim['blocks'] = 0
                victim['done_tok'] = 0
                waiting.insert(0, victim)
                preempt += 1
            if r not in running:
                eng.free_blocks += r['blocks']
                r['blocks'] = 0
                r['done_tok'] = 0
                waiting.insert(0, r)
                preempt += 1
                continue
            eng.free_blocks -= want
            r['blocks'] += want
            budget -= need
            ctx += new_total
            (sched_dec if need == 1 and r['done_tok'] >= r['pl'] else sched_pre).append((r, need))
        # 2) 再从 waiting 里拉新请求做 prefill
        while waiting and budget > 0 and len(running) < eng.max_seqs:
            r = waiting[0]
            remain = r['pl'] + r['gen'] - r['done_tok']
            need = min(budget, remain)
            want = math.ceil((r['done_tok'] + need) / eng.block) - r['blocks']
            if want > eng.free_blocks:
                break
            waiting.pop(0)
            running.append(r)
            eng.free_blocks -= want
            r['blocks'] += want
            budget -= need
            ctx += r['done_tok'] + need
            sched_pre.append((r, need))
        p_tok = sum(nd for _, nd in sched_pre)
        dt = eng.step_time(p_tok, len(sched_dec), ctx)
        t += dt
        steps.append((t, p_tok, len(sched_dec), dt))
        for r, need in sched_pre + sched_dec:
            r['done_tok'] += need
            if r['done_tok'] >= r['pl'] + r['gen']:   # 本步把已知 token 全算完 → 采样出一个新 token
                r['gen'] += 1
                r['times'].append(t)
                if r['gen'] >= r['ol']:
                    running.remove(r)
                    eng.free_blocks += r['blocks']
                    r['blocks'] = 0
                    finished += 1
                    inflight -= 1
    outs = []
    for r in reqs:
        ts = r['times']
        outs.append(dict(ok=True, ttft=ts[0] - r['send'], itl=[b - a for a, b in zip(ts, ts[1:])],
                         latency=ts[-1] - r['send'], output_len=len(ts), prompt_len=r['pl'],
                         client_queue=r['send'] - r['arrive'], start=r['send']))
    start = min(r['send'] for r in reqs)
    return outs, t - start, dict(preempt=preempt, steps=len(steps))


def token_times(o):
    t = [o['start'] + o['ttft']]
    for g in o['itl']:
        t.append(t[-1] + g)
    return t


def steady_window(outs, ratio):
    """SGLang sglang/benchmark/steady_state.py find_steady_state_window：并发 ≥ ceil(峰值 × ratio) 的最长连续区间。"""
    ev = {}
    for o in outs:
        ev[o['start']] = ev.get(o['start'], 0) + 1
        e = o['start'] + o['latency']
        ev[e] = ev.get(e, 0) - 1
    ts = sorted(ev)
    cur, iv = 0, []
    for i, t in enumerate(ts[:-1]):
        cur += ev[t]
        if ts[i + 1] > t:
            iv.append((t, ts[i + 1], cur))
    peak = max(c for _, _, c in iv)
    thr = max(1, math.ceil(peak * ratio))
    spans, s0, s1 = [], None, None
    for a, b, c in iv:
        if c >= thr:
            s0 = a if s0 is None else s0
            s1 = b
        elif s0 is not None:
            spans.append((s0, s1))
            s0 = s1 = None
    if s0 is not None:
        spans.append((s0, s1))
    lo, hi = max(spans, key=lambda s: s[1] - s[0])
    return lo, hi, thr, peak


def row(label, m, extra=''):
    return (f'{label:>10} {m["req_s"]:7.2f} {m["out_tok_s"]:9,.0f} {m["p50_ttft"]:9.0f} {m["p99_ttft"]:9.0f} '
            f'{m["mean_tpot"]:8.1f} {m["p99_tpot"]:8.1f} {m["p99_itl"]:8.1f} {m["goodput"]:8.2f}{extra}')


HEAD = f'{"":>10} {"req/s":>7} {"out tok/s":>9} {"p50 TTFT":>9} {"p99 TTFT":>9} {"mTPOT":>8} {"p99TPOT":>8} {"p99 ITL":>8} {"goodput":>8}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from-json')
    ap.add_argument('--goodput', nargs='+', default=['ttft:1000', 'tpot:40'])
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--isl', type=int, default=1024)
    ap.add_argument('--osl', type=int, default=128)
    ap.add_argument('--burstiness', type=float, default=1.0)
    ap.add_argument('--budget', type=int, default=8192)
    ap.add_argument('--demo-itl', action='store_true')
    ap.add_argument('--steady', type=int, metavar='C', help='C=0 表示不设并发闸门；闭环 max-concurrency=C，对比全程吞吐与 SGLang 稳态窗口吞吐')
    a = ap.parse_args()
    slo = {k: float(v) for k, v in (s.split(':') for s in a.goodput)}

    if a.steady is not None:
        outs, dur, st = simulate(a.n, a.isl, a.osl, math.inf, max_concurrency=a.steady or None, eng=Engine(budget=a.budget))
        w = steady_window(outs, 0.9)
        full = metrics(outs, dur, slo)
        inside = [o for o in outs if o['start'] >= w[0] and o['start'] + o['latency'] <= w[1]]
        tok = sum(1 for o in outs for t in token_times(o) if w[0] <= t < w[1])
        print(f'c={a.steady}：全程 {dur:.1f} s，峰值并发 {w[3]}，阈值 {w[2]}（0.9 × 峰值），稳态窗口 [{w[0]:.1f}, {w[1]:.1f}] s')
        print(f'  全程 out tok/s = {full["out_tok_s"]:,.0f}；窗口内 out tok/s = {tok / (w[1] - w[0]):,.0f}；窗口内完整请求 {len(inside)} 条')
        return

    if a.from_json:
        outs, dur = load_json(a.from_json)
        m = metrics(outs, dur, slo)
        for k in sorted(m):
            print(f'{k:>12} = {m[k]:,.3f}')
        return

    if a.demo_itl:
        outs, dur, st = simulate(n=64, isl=a.isl, osl=a.osl, rate=8, force_long=(20, 32768), eng=Engine(budget=a.budget))
        victim = max(range(len(outs)), key=lambda i: max(outs[i]['itl'] or [0]))
        o = outs[victim]
        tpot = (o['latency'] - o['ttft']) / (o['output_len'] - 1)
        print(f'第 20 个请求是 32K 的长 prompt；被它拖得最狠的是请求 {victim}：')
        print(f'  TPOT = (latency − TTFT)/(n − 1) = {tpot * 1e3:.1f} ms')
        print(f'  最大 ITL = {max(o["itl"]) * 1e3:.1f} ms，中位 ITL = {statistics.median(o["itl"]) * 1e3:.1f} ms')
        m = metrics(outs, dur, slo)
        print(f'  全体 p99 TPOT = {m["p99_tpot"]:.1f} ms，p99 ITL = {m["p99_itl"]:.1f} ms')
        return

    print(f'负载：random 数据集 ISL={a.isl} OSL={a.osl}，{a.n} 条，burstiness={a.burstiness}；'
          f'引擎：H100 + Qwen3-8B 粗模型，budget={a.budget}；goodput SLO {slo}')
    print('\n[开环] --request-rate 扫描（--max-concurrency 不设）')
    print(HEAD)
    for rate in (1, 2, 4, 8, 12, 16, 20, 24, 32, math.inf):
        outs, dur, st = simulate(a.n, a.isl, a.osl, rate, a.burstiness, eng=Engine(budget=a.budget))
        m = metrics(outs, dur, slo)
        print(row('inf' if rate == math.inf else f'{rate}', m, f'  抢占 {st["preempt"]}' if st['preempt'] else ''))
    print('\n[闭环] --request-rate inf + --max-concurrency 扫描')
    print(HEAD)
    for c in (1, 4, 16, 32, 64, 128, 256, 512):
        outs, dur, st = simulate(a.n, a.isl, a.osl, math.inf, max_concurrency=c, eng=Engine(budget=a.budget))
        m = metrics(outs, dur, slo)
        cq = statistics.fmean(o['client_queue'] for o in outs) * 1e3
        print(row(f'c={c}', m, f'  客户端排队均值 {cq:,.0f} ms'))
    print('\n[突发度] 同一 --request-rate 16，只改 --burstiness')
    print(HEAD + '  SLO 达成率')
    for b in (0.25, 0.5, 1.0, 4.0, math.inf):
        outs, dur, st = simulate(a.n, a.isl, a.osl, 16, b, eng=Engine(budget=a.budget))
        m = metrics(outs, dur, slo)
        print(row(f'b={b:g}', m, f'  {m["slo_attain"] * 100:5.1f}%'))


if __name__ == '__main__':
    main()
