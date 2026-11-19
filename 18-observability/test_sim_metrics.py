import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_metrics as sm  # noqa: E402
import scrape_metrics as scr  # noqa: E402


class Intervals(unittest.TestCase):
    def test_no_preempt(self):
        r = sm.request_intervals(sm.timeline())
        self.assertAlmostEqual(r['queued'], 0.40)
        self.assertAlmostEqual(r['prefill'], 0.30)
        self.assertAlmostEqual(r['ttft'], 0.71)                  # 多出 10 ms tokenize
        self.assertAlmostEqual(r['tpot'], 0.02)
        self.assertAlmostEqual(r['inference'], r['prefill'] + r['decode'])

    def test_preempt_in_prefill_goes_to_ttft(self):
        base, p = sm.request_intervals(sm.timeline()), sm.request_intervals(sm.timeline('prefill'))
        self.assertAlmostEqual(p['ttft'] - base['ttft'], 1.5)
        self.assertAlmostEqual(p['queued'], base['queued'])      # 只记第一次排队
        self.assertAlmostEqual(p['tpot'], base['tpot'])

    def test_preempt_in_decode_goes_to_tpot(self):
        base, d = sm.request_intervals(sm.timeline()), sm.request_intervals(sm.timeline('decode'))
        self.assertAlmostEqual(d['ttft'], base['ttft'])
        self.assertAlmostEqual(d['decode'] - base['decode'], 1.5)
        self.assertAlmostEqual(d['tpot'], (3.98 + 1.5) / 199)

    def test_single_token_tpot_zero(self):
        r = sm.request_intervals(dict(arrival=0, queued=0, scheduled=0.1, tokens=[0.3]))
        self.assertEqual(r['tpot'], 0.0)


class Itl(unittest.TestCase):
    def setUp(self):
        class R:
            pass
        rng = random.Random(1)
        self.reqs = []
        for _ in range(30):
            r = R()
            t = 0.0
            r.emits = []
            for _ in range(rng.randint(2, 60)):
                t += rng.uniform(5, 30)
                r.emits.append(t)
            self.reqs.append(r)

    def test_identical_without_spec(self):
        v, s, _ = sm.itl_views(self.reqs, spec=False)
        self.assertEqual(sorted(v), sorted(s))

    def test_spec_sum_preserved(self):
        v, s, _ = sm.itl_views(self.reqs, spec=True)
        self.assertAlmostEqual(sum(v), sum(s), places=6)         # 总时间相同，样本数不同
        self.assertGreater(len(s), len(v))

    def test_tpot_independent_of_spec(self):
        _, _, t0 = sm.itl_views(self.reqs, spec=False)
        _, _, t1 = sm.itl_views(self.reqs, spec=True)
        self.assertEqual(t0, t1)


class Buckets(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(sm.VLLM_BUCKETS['request_latency']), 21)
        self.assertEqual(len(sm.VLLM_BUCKETS['time_to_first_token']), 22)
        self.assertEqual(len(sm.VLLM_BUCKETS['inter_token_latency']), 19)
        self.assertEqual(len(sm.SGL_BUCKETS['time_to_first_token']), 18)
        self.assertEqual(len(sm.SGL_BUCKETS['inter_token_latency']), 23)
        self.assertEqual(len(sm.SGL_BUCKETS['e2e_request_latency']), 22)

    def test_1_2_5(self):
        self.assertEqual(sm.build_1_2_5(100), [1, 2, 5, 10, 20, 50, 100])
        self.assertEqual(len(sm.build_1_2_5(40960)), 14)

    def test_quantile_interpolation(self):
        bounds = [1.0, 2.0, 4.0]
        cum = sm.observe(bounds, [0.5] * 10 + [1.5] * 10 + [3.0] * 20)
        self.assertEqual(cum, [10, 20, 40, 40])
        self.assertAlmostEqual(sm.histogram_quantile(0.5, bounds, cum), 2.0)
        self.assertAlmostEqual(sm.histogram_quantile(0.75, bounds, cum), 3.0)
        self.assertAlmostEqual(sm.histogram_quantile(0.125, bounds, cum), 0.5)

    def test_inf_bucket_returns_last_bound(self):
        bounds = [1.0, 2.0]
        cum = sm.observe(bounds, [0.5, 5.0, 6.0, 7.0])
        self.assertEqual(sm.histogram_quantile(0.99, bounds, cum), 2.0)


class Hitrate(unittest.TestCase):
    def test_window_1000(self):
        cm = sm.CachingMetrics()
        for _ in range(1000):
            cm.observe(1, 100, 80)
        for _ in range(500):
            cm.observe(1, 100, 20)
        self.assertAlmostEqual(cm.hit_rate, (500 * 80 + 500 * 20) / 100000)

    def test_reset(self):
        cm = sm.CachingMetrics()
        cm.observe(1, 100, 90)
        cm.observe(1, 100, 10, reset=True)
        self.assertAlmostEqual(cm.hit_rate, 0.1)

    def test_empty_not_appended(self):
        cm = sm.CachingMetrics(max_recent_requests=2)
        cm.observe(2, 100, 50)
        cm.observe(0, 0, 0)
        self.assertEqual(len(cm.dq), 1)

    def test_low_traffic_lags(self):
        ev = sm.hitrate_series(2.0)
        self.assertGreater(sm.log_rate(ev, 720), 0.5)
        self.assertLess(sm.window_rate(ev, 720, 60), 0.3)


class Profiler(unittest.TestCase):
    def rec(self, **kw):
        return [s for s, run, act, _ in sm.profile_steps(n_steps=14, **kw)
                if run and act in ('ALL', 'RECORD', 'RECORD_SAVE')]

    def test_delay_and_limit(self):
        self.assertEqual(self.rec(delay=3, max_iters=5), [3, 4, 5, 6, 7])

    def test_warmup_discard(self):
        self.assertEqual(self.rec(wait=1, warmup=2, active=5), [3, 4, 5, 6, 7])

    def test_unbounded(self):
        self.assertEqual(len(self.rec()), 14)


class Cardinality(unittest.TestCase):
    def test_per_engine(self):
        total = sum(sm.series_count(sm.bucket_count(f, 40960), 1) for _, f in sm.VLLM_HISTOGRAMS)
        self.assertEqual(total, 308)
        self.assertEqual(len(sm.VLLM_HISTOGRAMS), 16)


class Scrape(unittest.TestCase):
    TEXT = '''# HELP vllm:time_to_first_token_seconds x
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",model_name="m"} 10.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.5",model_name="m"} 30.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="m"} 40.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="m"} 40.0
vllm:time_to_first_token_seconds_sum{engine="0",model_name="m"} 12.0
vllm:time_to_first_token_seconds_bucket{engine="1",le="0.1",model_name="m"} 0.0
vllm:time_to_first_token_seconds_bucket{engine="1",le="0.5",model_name="m"} 10.0
vllm:time_to_first_token_seconds_bucket{engine="1",le="+Inf",model_name="m"} 10.0
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 1000.0
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 250.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.42
'''

    def test_parse_and_merge(self):
        s = scr.parse(self.TEXT)
        b, cum = scr.merged_hist(s, 'vllm:time_to_first_token_seconds')
        self.assertEqual(b, [0.1, 0.5])
        self.assertEqual(cum, [10.0, 40.0, 50.0])
        self.assertAlmostEqual(scr.counter_sum(s, 'vllm:prefix_cache_hits_total'), 250.0)
        self.assertAlmostEqual(scr.gauge_max(s, 'vllm:kv_cache_usage_perc'), 0.42)

    def test_delta(self):
        s0, s1 = scr.parse(self.TEXT), scr.parse(self.TEXT.replace('} 250.0', '} 450.0').replace('} 1000.0', '} 1400.0'))
        self.assertAlmostEqual(scr.ratio_delta(s0, s1, 'vllm:prefix_cache_hits_total', 'vllm:prefix_cache_queries_total'), 0.5)


if __name__ == '__main__':
    unittest.main()
