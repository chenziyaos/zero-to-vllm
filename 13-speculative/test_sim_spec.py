"""sim_spec.py 的回归测试：python3 -m unittest test_sim_spec -v"""
import random
import unittest

import sim_spec as S


class TestTheory(unittest.TestCase):
    def test_expected_tokens_closed_form(self):
        # E = (1 - α^(k+1)) / (1 - α)，k=0 时只有 bonus token
        self.assertAlmostEqual(S.expected_tokens(0.8, 0), 1.0)
        self.assertAlmostEqual(S.expected_tokens(0.8, 3), (1 - 0.8 ** 4) / 0.2)
        self.assertAlmostEqual(S.expected_tokens(1.0, 5), 6.0)

    def test_round_up(self):
        self.assertEqual(S.round_up(560, 17), 561)
        self.assertEqual(S.round_up(561, 17), 561)


class TestNgram(unittest.TestCase):
    def test_kmp_matches_brute_force(self):
        rng = random.Random(0)
        for _ in range(400):
            toks = [rng.randrange(4) for _ in range(rng.randrange(1, 40))]
            mn = rng.randrange(1, 4)
            mx = rng.randrange(mn, 6)
            k = rng.randrange(1, 6)
            self.assertEqual(S.find_longest_ngram_propose(toks, mn, mx, 1000, k),
                             S.brute_force_propose(toks, mn, mx, 1000, k), (toks, mn, mx, k))


class TestDynamicSD(unittest.TestCase):
    def test_dense_lookup_gap_tail_clamp(self):
        d = S.build_dynamic_sd_schedule_lookup([[1, 16, 3], [32, 128, 2]], 160, 3)
        self.assertEqual(d[1], 3)
        self.assertEqual(d[31], 3)      # 空档沿用前一档
        self.assertEqual(d[32], 2)
        self.assertEqual(d[160], 2)     # 尾部沿用最后一档
        d = S.build_dynamic_sd_schedule_lookup([[1, 8, 7], [9, 32, 4]], 64, 5)
        self.assertEqual(d[1], 5)       # 被 num_speculative_tokens 截顶
        self.assertEqual(d[64], 4)

    def test_invalid_schedules(self):
        for bad in ([[2, 8, 3]], [[1, 8, 3], [8, 16, 1]], [[1, 8, -1]]):
            with self.assertRaises(ValueError):
                S.validate_and_normalize_dynamic_sd_schedule(bad)


class TestCudaGraphSizes(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(S.capture_sizes(1024, 8192, 0)), 51)
        self.assertEqual(len(S.capture_sizes(1024, 8192, 3)), 51)
        self.assertEqual(len(S.capture_sizes(512, 16384, 16)), 55)

    def test_source_comment_numbers(self):
        k = 16
        grid_ext = set([1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 512 * (k + 1) + 1, 16)))
        rc17 = {n * (k + 1) for n in S.request_counts(512, 512)}
        grid512 = {s for s in grid_ext if s <= 512}
        self.assertEqual(len(grid_ext), 563)
        self.assertEqual(len(grid_ext | rc17), 581)
        self.assertEqual(len(grid512 | rc17), 100)

    def test_mrv1_round_up_multiples(self):
        sizes = S.mrv1_round_up(S.capture_sizes(1024, 8192, 3), 4)
        self.assertTrue(all(s % 4 == 0 for s in sizes))
        self.assertEqual(len(sizes), 49)


class TestSGLangAdaptive(unittest.TestCase):
    def test_routing(self):
        p = S.AdaptiveSpeculativeParams(initial_steps=3, cuda_graph_bs=[1, 2, 4, 8, 16, 24, 32, 40, 48])
        self.assertEqual(p.closest(p.pad(3)), 1)
        self.assertEqual(p.closest(p.pad(5)), 8)
        self.assertEqual(p.closest(p.pad(100)), 64)

    def test_step_up_when_all_accepted(self):
        slot = S.AdaptiveStepSlot(3, S.SGL_DEFAULT_ADAPTIVE_CONFIG['1'])
        for _ in range(60):
            slot.update([7])
        self.assertEqual(slot.current_steps, 7)

    def test_zero_step_probe(self):
        slot = S.AdaptiveStepSlot(3, S.SGL_DEFAULT_ADAPTIVE_CONFIG['8'])
        for _ in range(15):
            slot.update([0])
        self.assertEqual(slot.current_steps, 0)
        for _ in range(5):
            slot.update([0])
        self.assertEqual(slot.current_steps, 1)   # 0 步之后强制探一次最小正步数


class TestMetrics(unittest.TestCase):
    def test_synthetic_rates(self):
        self.assertEqual(S.acceptance_length_to_rates(2.5, 5), [1, 0.5, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
