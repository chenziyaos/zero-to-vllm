"""sim_rejection.py 的回归测试：python3 -m unittest test_sim_rejection -v"""
import random
import unittest

import sim_rejection as R

N = 20000


class TestLossless(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target, cls.draft = R.make_models()
        cls.joint = R.exact_joint(cls.target)
        cls.floor = R.tv(R.direct(cls.target, N, 1), cls.joint, N)

    def tv_of(self, fn, seed=9):
        counts, per_round = R.run(fn, self.target, N, seed)
        return R.tv(counts, self.joint, N), per_round

    def test_lossless_rules(self):
        t, d = self.target, self.draft
        for fn in (lambda r, pv: R.round_standard(r, t, d, pv),
                   lambda r, pv: R.round_standard(r, t, d, pv, True),
                   lambda r, pv: R.round_block(r, t, d, pv),
                   lambda r, pv: R.round_block(r, t, d, pv, True),
                   lambda r, pv: R.round_sgl_target_only(r, t, d, pv)):
            self.assertLess(self.tv_of(fn)[0], 2.5 * self.floor)

    def test_lossy_rules(self):
        t, d = self.target, self.draft
        for fn in (lambda r, pv: R.round_synthetic(r, t, d, pv, [0.9, 0.8, 0.7]),
                   lambda r, pv: R.round_fly(r, t, d, pv, 1, 0.3),
                   lambda r, pv: R.round_sgl_target_only(r, t, d, pv, 1.0, 0.5)):
            self.assertGreater(self.tv_of(fn)[0], 5 * self.floor)

    def test_block_beats_standard_probabilistic(self):
        t, d = self.target, self.draft
        _, std = self.tv_of(lambda r, pv: R.round_standard(r, t, d, pv), 3)
        _, blk = self.tv_of(lambda r, pv: R.round_block(r, t, d, pv), 3)
        self.assertGreater(blk, std * 1.03)

    def test_fly_high_threshold_is_standard(self):
        t, d = self.target, self.draft
        a = R.run(lambda r, pv: R.round_fly(r, t, d, pv, 1, 99.0), t, 3000, 5)
        b = R.run(lambda r, pv: R.round_standard(r, t, d, pv), t, 3000, 5)
        self.assertEqual(a, b)   # 同一随机流、不挂起 → 逐位相同


class TestRecovered(unittest.TestCase):
    def test_exponential_argmax_matches_residual(self):
        rng = random.Random(3)
        p, q = [0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.1, 0.6]
        r = R.norm_relu(p, q)
        counts = [0] * R.V
        n = 40000
        rel = [max(a - b, 0.0) for a, b in zip(p, q)]
        for _ in range(n):
            e = [rng.expovariate(1.0) for _ in range(R.V)]
            counts[R.argmax([rel[j] / e[j] for j in range(R.V)])] += 1
        for c, x in zip(counts, r):
            self.assertAlmostEqual(c / n, x, delta=0.01)


if __name__ == '__main__':
    unittest.main()
