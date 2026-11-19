"""抓两次 /metrics，按 PromQL 的口径算窗口内的 p50/p99、命中率和 token 吞吐。

只用标准库。对应 vLLM 0d6f92a 的指标名（vllm/v1/metrics/loggers.py）：
  vllm:time_to_first_token_seconds        histogram, buckets.py TTFT 22 档
  vllm:inter_token_latency_seconds        histogram, 每个 output 事件一个样本（stats.py 548-560）
  vllm:request_time_per_output_token_seconds  histogram, decode/(n-1)
  vllm:e2e_request_latency_seconds        histogram
  vllm:prefix_cache_queries/hits          counter（_total）, 662/673
  vllm:prompt_tokens                      counter, 按 total 累加（1262）
  vllm:prompt_tokens_by_source{source=local_compute|local_cache_hit|external_kv_transfer}
  vllm:generation_tokens                  counter
  vllm:kv_cache_usage_perc                gauge, 1 - free/(num_gpu_blocks-1)

用法：
  python3 scrape_metrics.py --url http://127.0.0.1:8000/metrics --interval 30
  python3 scrape_metrics.py --files before.txt after.txt --interval 30
  python3 scrape_metrics.py --prefix sglang:      # SGLang 的同名指标前缀
"""
import argparse
import math
import re
import sys
import time
import urllib.request

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)(\s+\S+)?$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse(text):
    """返回 {(name, frozenset(labels.items())): value}。忽略 # 行。"""
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        m = LINE.match(line)
        if not m:
            continue
        name, _, body, val = m.group(1), m.group(2), m.group(3), m.group(4)
        labels = dict(LABEL.findall(body or ''))
        try:
            v = float(val)
        except ValueError:
            continue
        out[(name, frozenset(labels.items()))] = v
    return out


def _le(x):
    return math.inf if x == '+Inf' else float(x)


def merged_hist(series, name, where=None):
    """把所有 engine / label 组合的 _bucket 按 le 相加（等价 sum by (le)）。
    返回 (有限上界, 累计计数含 +Inf)。"""
    acc = {}
    for (n, lab), v in series.items():
        if n != name + '_bucket':
            continue
        d = dict(lab)
        if where and any(d.get(k) != w for k, w in where.items()):
            continue
        le = _le(d['le'])
        acc[le] = acc.get(le, 0.0) + v
    les = sorted(acc)
    return [x for x in les if x != math.inf], [acc[x] for x in les]


def counter_sum(series, name, where=None):
    s = 0.0
    for (n, lab), v in series.items():
        if n != name:
            continue
        d = dict(lab)
        if where and any(d.get(k) != w for k, w in where.items()):
            continue
        s += v
    return s


def gauge_max(series, name):
    vals = [v for (n, _), v in series.items() if n == name]
    return max(vals) if vals else float('nan')


def ratio_delta(s0, s1, num, den):
    dn = counter_sum(s1, num) - counter_sum(s0, num)
    dd = counter_sum(s1, den) - counter_sum(s0, den)
    return dn / dd if dd > 0 else float('nan')


def histogram_quantile(q, bounds, cum):
    """与 PromQL 一致：桶内线性插值；落进 +Inf 桶时返回最后一个有限上界。"""
    total = cum[-1] if cum else 0
    if total <= 0:
        return float('nan')
    rank = q * total
    lo, prev = 0.0, 0.0
    for i, b in enumerate(bounds):
        if cum[i] >= rank:
            width = cum[i] - prev
            return lo + (b - lo) * ((rank - prev) / width if width else 0)
        lo, prev = b, cum[i]
    return bounds[-1] if bounds else float('nan')


def delta_hist(s0, s1, name):
    b0, c0 = merged_hist(s0, name)
    b1, c1 = merged_hist(s1, name)
    if not c0:
        return b1, c1
    if b0 != b1:
        raise ValueError(f'{name}: 两次抓取的桶边界不同（重启过或改了 --custom-histogram-buckets）')
    return b1, [y - x for x, y in zip(c0, c1)]


def fetch(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.read().decode()


def report(s0, s1, dt, p):
    rows = []
    for short in ('time_to_first_token_seconds', 'inter_token_latency_seconds',
                  'request_time_per_output_token_seconds', 'e2e_request_latency_seconds',
                  'request_queue_time_seconds'):
        name = p + short
        b, c = delta_hist(s0, s1, name)
        if not c or c[-1] <= 0:
            continue
        p50, p99 = histogram_quantile(0.5, b, c), histogram_quantile(0.99, b, c)
        inf_share = (c[-1] - c[-2]) / c[-1] if len(c) > 1 else 0
        flag = '  <- p99 落进 +Inf 桶，只是下界' if inf_share > 0.01 else ''
        rows.append(f'  {short:<42} n={int(c[-1]):>7}  p50={p50*1000:9.1f} ms  p99={p99*1000:9.1f} ms{flag}')
    print(f'窗口 {dt:.0f} s 内的分位数（sum by (le) 后插值）：')
    print('\n'.join(rows) if rows else '  （窗口内没有新样本）')

    hit = ratio_delta(s0, s1, p + 'prefix_cache_hits_total', p + 'prefix_cache_queries_total')
    print(f'前缀缓存命中率（窗口 delta，token 口径）：{hit:.3f}' if hit == hit else '前缀缓存命中率：窗口内无查询')

    tot = counter_sum(s1, p + 'prompt_tokens_total') - counter_sum(s0, p + 'prompt_tokens_total')
    comp = (counter_sum(s1, p + 'prompt_tokens_by_source_total', {'source': 'local_compute'})
            - counter_sum(s0, p + 'prompt_tokens_by_source_total', {'source': 'local_compute'}))
    gen = counter_sum(s1, p + 'generation_tokens_total') - counter_sum(s0, p + 'generation_tokens_total')
    print(f'prompt tok/s：total {tot/dt:9.1f}   真正算过的 {comp/dt:9.1f}   generation tok/s {gen/dt:9.1f}')
    print('  （日志里的 "Avg prompt throughput" 对应"真正算过的"那列，Prometheus 的 prompt_tokens 对应 total）')
    print(f'KV usage（各 engine 取 max）：{gauge_max(s1, p + "kv_cache_usage_perc"):.3f}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default='http://127.0.0.1:8000/metrics')
    ap.add_argument('--files', nargs=2, metavar=('BEFORE', 'AFTER'))
    ap.add_argument('--interval', type=float, default=30.0)
    ap.add_argument('--prefix', default='vllm:')
    a = ap.parse_args(argv)
    if a.files:
        s0, s1 = (parse(open(f).read()) for f in a.files)
    else:
        s0 = parse(fetch(a.url))
        time.sleep(a.interval)
        s1 = parse(fetch(a.url))
    report(s0, s1, a.interval, a.prefix)


if __name__ == '__main__':
    sys.exit(main())
