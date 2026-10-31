"""sim_tuning.py 的回归测试：python3 -m unittest test_sim_tuning -v"""
import unittest

import sim_tuning as S


class TestVllmDefaults(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(S.vllm_batch_defaults('h100-sxm', 'api'), (8192, 1024))
        self.assertEqual(S.vllm_batch_defaults('h100-sxm', 'llm'), (16384, 1024))
        self.assertEqual(S.vllm_batch_defaults('b200', 'api'), (16384, 1024))
        self.assertEqual(S.vllm_batch_defaults('h200', 'api'), (8192, 1024))

    def test_a100_name_check(self):
        # 80G 显存够 70 GiB，但卡名含 a100：落到小档（PR #17885）
        self.assertEqual(S.vllm_batch_defaults('a100-80g', 'api'), (2048, 256))
        self.assertEqual(S.vllm_batch_defaults('l40s', 'llm'), (8192, 256))

    def test_throughput_doubles_only_unset(self):
        self.assertEqual(S.vllm_resolve('h100-sxm', 'api', 'throughput'), (16384, 2048))
        self.assertEqual(S.vllm_resolve('h100-sxm', 'api', 'throughput', user_seqs=256), (16384, 256))

    def test_clamps(self):
        self.assertEqual(S.vllm_resolve('h100-sxm', 'api', chunked=False), (40960, 1024))
        self.assertEqual(S.vllm_resolve('h100-sxm', 'api', max_model_len=4, user_seqs=8), (32, 8))
        # seqs = min(seqs, tokens)
        self.assertEqual(S.vllm_resolve('h100-sxm', 'api', user_tokens=512), (512, 512))


class TestCaptureSizes(unittest.TestCase):
    def test_default_grid_matches_ch02(self):
        sz = S.vllm_capture_sizes(1024, 8192)
        self.assertEqual(len(sz), 51)
        self.assertEqual(sz[:5], [1, 2, 4, 8, 16])
        self.assertEqual(sz[-1], 512)

    def test_cap_is_twice_max_num_seqs(self):
        self.assertEqual(S.vllm_capture_sizes(64, 8192)[-1], 128)
        self.assertEqual(S.vllm_capture_sizes(1024, 8192, cc=100)[-1], 1024)

    def test_budget_below_cap_is_added(self):
        sz = S.vllm_capture_sizes(256, 100)
        self.assertEqual(sz[-1], 100)
        self.assertIn(100, sz)

    def test_interactivity(self):
        sz = S.vllm_capture_sizes(1024, 8192, mode='interactivity')
        self.assertEqual(sz[:32], list(range(1, 33)))
        self.assertEqual(len(sz), 76)

    def test_pad_and_eager(self):
        w, e = S.pad_stats(S.vllm_capture_sizes(1024, 8192), 1024)
        self.assertAlmostEqual(e, 0.5)
        w, e = S.pad_stats([1, 2, 4], 4)
        self.assertEqual(e, 0)
        self.assertAlmostEqual(w, (0 + 0 + 0.25 + 0) / 4)


class TestSglang(unittest.TestCase):
    def test_tier(self):
        self.assertEqual(S.sglang_tier('h100-sxm'), (8192, 256))
        self.assertEqual(S.sglang_tier('h100-sxm', tp=8), (8192, 512))
        self.assertEqual(S.sglang_tier('rtx4090'), (4096, 48))
        self.assertEqual(S.sglang_tier('b200'), (16384, 512))

    def test_mem_fraction_matches_ch02(self):
        reserved, mf = S.sglang_mem_fraction('h100-sxm')
        self.assertEqual(reserved, 13904)
        self.assertEqual(mf, 0.830)

    def test_floor_10g(self):
        reserved, _ = S.sglang_mem_fraction('h100-sxm', chunked=2048, max_bs=8)
        self.assertEqual(reserved, 10240)

    def test_grids(self):
        self.assertEqual(len(S.sglang_decode_sizes(256)), 36)
        self.assertEqual(len(S.sglang_prefill_sizes(8192)), 58)
        self.assertEqual(S.sglang_decode_sizes(50)[-1], 50)


class TestMem(unittest.TestCase):
    def test_ch02_numbers(self):
        g, blocks, tok = S.vllm_kv(0.92)
        self.assertEqual(blocks, 24957)
        self.assertEqual(tok, 399312)

    def test_kv_bytes_bypasses_util(self):
        self.assertEqual(S.vllm_kv(0.5, kv_bytes=50 * S.GiB), S.vllm_kv(0.99, kv_bytes=50 * S.GiB))

    def test_monotonic(self):
        toks = [S.vllm_kv(u)[2] for u in (0.85, 0.9, 0.95)]
        self.assertEqual(toks, sorted(toks))


class TestSim(unittest.TestCase):
    def test_goodput_bounds(self):
        r = S.sc.run(S.sim_args(rate=4, duration=10), 'vllm')
        gp, frac = S.goodput(r, 1e9, 1e9)
        self.assertEqual(frac, 1.0)
        gp, frac = S.goodput(r, 0, 0)
        self.assertEqual(frac, 0.0)

    def test_admission_rejects_under_overload(self):
        a = S.sim_args(rate=20, duration=20)
        r0 = S.run_admission(a, 'vllm', None)
        r1 = S.run_admission(S.sim_args(rate=20, duration=20), 'vllm', 30000)
        self.assertEqual(r0['rejected'], 0)
        self.assertGreater(r1['rejected'], 0)
        self.assertLess(r1['ttft99'], r0['ttft99'])

    def test_cpu_formula(self):
        a = S.SimpleNamespace(tp=2, dp=4, api_servers=0)
        n = a.tp * a.dp
        self.assertEqual(a.dp + a.dp + n + 1, 17)


if __name__ == '__main__':
    unittest.main()
