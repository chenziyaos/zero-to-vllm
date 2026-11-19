"""sim_router.py 的回归测试：python3 -m unittest test_sim_router -v"""
import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_router as sr  # noqa: E402


def args(**kw):
    a = sr.parser().parse_args([])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


class VllmScore(unittest.TestCase):
    """core_client.py:1806-1826 的打分与副作用。"""

    def router(self, lb, clients=1):
        a = args(replicas=len(lb), clients=clients)
        reps = [sr.Replica(i, a) for i in range(len(lb))]
        r = sr.Router('vllm', reps, a)
        r.lb = [list(x) for x in lb]
        return r

    def test_kv_penalty_off_below_half(self):
        # 同样 4 条排队，KV 50% 时不加罚分，按扫描顺序挑第一台
        r = self.router([[4, 0, 0.5], [4, 0, 0.3]])
        self.assertEqual(r.vllm_pick(), 0)

    def test_kv_penalty_ramps(self):
        # 0 号：4 waiting，KV 100% → 4 + 4×6×0.5 = 16；1 号：10 running → 10
        r = self.router([[4, 0, 1.0], [0, 10, 1.0]])
        self.assertEqual(r.vllm_pick(), 1)

    def test_local_increment_and_rotate(self):
        r = self.router([[0, 0, 0.0]] * 4, clients=2)
        r.eng_start = 0
        j = r.vllm_pick()
        self.assertEqual(r.lb[j][0], 2)          # += client_count
        self.assertEqual(r.eng_start, 1)

    def test_start_index_spreads_clients(self):
        a = args(replicas=8, clients=4)
        reps = [sr.Replica(i, a) for i in range(8)]
        self.assertEqual([sr.Router('vllm', reps, a, c).eng_start for c in range(4)], [0, 2, 4, 6])


class Stale(unittest.TestCase):
    def test_snapshot_only_piles_up(self):
        self.assertEqual(sr.burst_split(4, 64, 1, False, False, False), [64, 0, 0, 0])

    def test_local_count_spreads(self):
        self.assertEqual(sr.burst_split(4, 64, 1, True, False, False), [16] * 4)

    def test_rebind_breaks_local_count_but_not_floor(self):
        self.assertNotEqual(sr.burst_split(4, 64, 1, True, False, False, rebind=6), [16] * 4)
        self.assertEqual(sr.burst_split(4, 64, 1, True, True, True, rebind=6), [16] * 4)


class Routing(unittest.TestCase):
    def run_all(self, **kw):
        a = args(duration=20.0, **kw)
        out = {}
        for pol, _ in sr.POLICIES:
            reqs, _ = sr.prefix_workload(a)
            out[pol] = sr.summarize(reqs, sr.simulate(a, pol, reqs))
        return out

    def test_prefix_aware_hits_more(self):
        res = self.run_all()
        self.assertGreater(res['prefer']['hit'], res['rr']['hit'] + 0.1)
        self.assertLess(res['prefer']['prefill'], res['rr']['prefill'])

    def test_round_robin_is_even(self):
        res = self.run_all()
        self.assertAlmostEqual(res['rr']['imb'], 1.0, delta=0.01)

    def test_every_request_finishes(self):
        a = args(duration=10.0)
        reqs, _ = sr.prefix_workload(a)
        sr.simulate(a, 'balanced', reqs)
        self.assertTrue(all(r.done is not None and r.gen == r.out for r in reqs))

    def test_hot_prefix_prefer_skews(self):
        a = args(duration=30.0, hot=0.5)
        reqs, sizes = sr.prefix_workload(a, 0.5)
        p = sr.summarize(reqs, sr.simulate(a, 'prefer', reqs, (0, sizes[0])))
        reqs, sizes = sr.prefix_workload(a, 0.5)
        b = sr.summarize(reqs, sr.simulate(a, 'balanced', reqs, (0, sizes[0])))
        self.assertGreater(p['imb'], 1.5)
        self.assertLess(b['imb'], 1.2)


class Probe(unittest.TestCase):
    def test_deadline(self):
        self.assertEqual(sr.kill_deadline(sr.PROBES['doc-liveness']), 80)
        self.assertEqual(sr.kill_deadline(sr.PROBES['default-3x10']), 20)
        self.assertEqual(sr.kill_deadline(sr.PROBES['startup-30x10']), 290)

    def test_cache_volume_rescues(self):
        base = dict(model_volume=True, wscale=1.0, download=0)
        t0, _, _ = sr.probe_timeline(sr.PROBES['doc-liveness'], dict(base, cache_volume=False))
        t1, rs, _ = sr.probe_timeline(sr.PROBES['doc-liveness'], dict(base, cache_volume=True))
        self.assertIsNone(t0)
        self.assertEqual(rs, 1)

    def test_startup_probe_ok(self):
        t, rs, _ = sr.probe_timeline(sr.PROBES['startup-30x10'],
                                     dict(cache_volume=False, model_volume=True, wscale=1.0, download=0))
        self.assertEqual(rs, 0)
        self.assertAlmostEqual(t, 84.37, places=1)


class Pause(unittest.TestCase):
    def test_wait_longer_than_keep(self):
        out = sr.show_pause(args())
        self.assertGreater(out['wait'], 10)
        self.assertGreater(out['ctx'], out['gen'])


if __name__ == '__main__':
    unittest.main()
