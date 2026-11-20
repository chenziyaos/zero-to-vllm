"""测 kv_cache_timing：python3 test_kv_cache_timing.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_cache_timing as kt  # noqa: E402


class KVCacheTiming(unittest.TestCase):
    def setUp(self):
        self.w = kt.init(layers=2, d=32, heads=4, d_ff=64, vocab=50)
        self.prompt = [3, 14, 15, 9, 26, 5]

    def test_same_tokens(self):
        on = kt.generate(self.w, self.prompt, 12, use_cache=True)
        off = kt.generate(self.w, self.prompt, 12, use_cache=False)
        self.assertEqual(len(on), 12)
        self.assertEqual(on, off)

    def test_cached_logits_match_full_forward(self):
        import numpy as np
        ids = self.prompt + [7, 8, 9]
        cache = {'len': 0, 'k': [np.zeros((0, 32))] * 2, 'v': [np.zeros((0, 32))] * 2}
        kt.forward(self.w, self.prompt, cache)
        for t in ids[len(self.prompt):]:
            last = kt.forward(self.w, [t], cache)
        np.testing.assert_allclose(last, kt.forward(self.w, ids), atol=1e-10)
        self.assertEqual(cache['len'], len(ids))
        self.assertEqual(cache['k'][0].shape, (len(ids), 32))

    def test_flops_single_token(self):
        # prompt 1、生成 1：两种做法都只有一次 1 个 token 的前向
        on = kt.flops(1, 1, 2, 32, 64, 50, use_cache=True)
        off = kt.flops(1, 1, 2, 32, 64, 50, use_cache=False)
        self.assertEqual(on, off)
        self.assertEqual(on, 2 * 2 * (4 * 32 * 32 + 2 * 32 * 64) + 2 * 2 * 2 * 32 * 1 + 2 * 32 * 50)

    def test_flops_ratio_grows_with_length(self):
        r = [kt.flops(p, g, 4, 128, 512, 512, use_cache=False) / kt.flops(p, g, 4, 128, 512, 512, use_cache=True)
             for p, g in ((32, 32), (32, 96), (128, 256))]
        self.assertTrue(1 < r[0] < r[1] < r[2])

    def test_params_mode(self):
        on = kt.flops(512, 256, 28, 1024, params=0.6e9, use_cache=True)
        off = kt.flops(512, 256, 28, 1024, params=0.6e9, use_cache=False)
        # 带 cache 的投影部分正好是 (prompt + gen - 1) 个 token 各算一次
        self.assertGreater(on, 2 * 0.6e9 * (512 + 255))
        self.assertGreater(off / on, 100)

    def test_main_runs(self):
        kt.main(['--layers', '1', '--d-model', '16', '--heads', '2', '--vocab', '20',
                 '--prompt', '4', '--gen', '5', '--repeat', '1'])


if __name__ == '__main__':
    unittest.main()
