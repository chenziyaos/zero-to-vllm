"""ds_impact.py 的回归测试：python3 -m unittest test_ds_impact -v

字节数钉在 vLLM 0d6f92a 源码里的常量上，EPLB 移植钉在 vLLM 自己的单测期望上，
标题计数钉在 ../21-frontier-month/data 的快照上。正文里的数都从这里来。
"""
import contextlib
import io
import math
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds_impact as di  # noqa: E402


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        di.main(argv)
    return buf.getvalue()


class KVBytes(unittest.TestCase):
    def test_mla_layouts(self):
        self.assertEqual(di.MLA_BF16, 1152)
        self.assertEqual(di.MLA_FP8_VLLM, 656)        # V3.2 博客与 vLLM FP8 MLA 布局
        self.assertEqual(di.V3_MHA_EQUIV, 81920)
        self.assertAlmostEqual(di.V3_MHA_EQUIV / di.MLA_BF16, 71.1, places=1)

    def test_indexer(self):
        # deepseek_v41/attention.py:_indexer_k_cache_head_dim
        self.assertEqual(di.INDEXER_FP8, 132)
        self.assertEqual(di.INDEXER_FP4, 68)

    def test_v32_bf16_matches_blog(self):
        out = run(["--scenario", "kvbytes"])
        self.assertIn("= 83.9 GiB（博客 83.9）", out)

    def test_swa_snapshot_dominates_block(self):
        swa = di.V41_SWA["fp8"] * di.V41_WINDOW * di.V41_SWA_LAYERS
        self.assertGreater(swa / (di.V41_GLOBAL * 256), 10)   # 博客：>10x
        out = run(["--scenario", "kvbytes", "--swa-dtype", "mxfp8"])
        self.assertIn("528 B", out)


class EPLB(unittest.TestCase):
    def test_matches_vllm_unit_test(self):
        for w, exp in zip(di.VLLM_TEST_W, di.VLLM_TEST_PHY2LOG):
            self.assertEqual(di.rebalance(w, 16, 4, 2, 8), exp)

    def test_logcnt(self):
        exp = [[1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1], [1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1]]
        for w, e in zip(di.VLLM_TEST_W, exp):
            p = di.rebalance(w, 16, 4, 2, 8)
            self.assertEqual([p.count(i) for i in range(12)], e)

    def test_balanced_packing_fills_evenly(self):
        idx, rank = di.balanced_packing([5, 1, 4, 2, 3, 6], 3)
        self.assertEqual(sorted(idx), [0, 0, 1, 1, 2, 2])
        for p in range(3):
            self.assertEqual(sorted(r for i, r in zip(idx, rank) if i == p), [0, 1])

    def test_balancedness_formula(self):
        # vLLM 单测：avg 50 / max 200
        self.assertAlmostEqual(di.balancedness([[200, 0, 0, 0]]), 0.25)

    def test_redundancy_helps(self):
        W = di.skewed_weights(64, 10, 7, 0.9)
        b0 = di.balancedness([di.gpu_loads(w, list(range(64)), 16) for w in W])
        b16 = di.balancedness([di.gpu_loads(w, di.rebalance(w, 80, 8, 2, 16), 16) for w in W])
        self.assertLess(b0, 0.5)
        self.assertGreater(b16, 0.9)

    def test_fallback_when_groups_not_divisible(self):
        w = di.VLLM_TEST_W[0]
        p = di.rebalance(w, 16, 3, 2, 8)          # 3 组不能被 2 节点整除 → 全局策略
        self.assertEqual(len(p), 16)
        self.assertEqual(set(p), set(range(12)))


class Replay(unittest.TestCase):
    def test_formula(self):
        out = run(["--scenario", "replay", "--hit", "0", "--new", "8192"])
        self.assertIn("174464", out)
        self.assertNotIn("encoder 侧重放补算命中末尾 128", out)

    def test_short_prompt_saves_nothing(self):
        out = run(["--scenario", "replay"])
        self.assertRegex(out, r"新 token\s+128：省\s+0\.0%")

    def test_upper_bound(self):
        L, cut = 40, 20
        self.assertAlmostEqual(1 - (cut + 1) / L, 0.475)
        out = run(["--scenario", "replay", "--new", "32768"])
        pct = [float(x) for x in re.findall(r"省\s+([\d.]+)%", out)]
        self.assertTrue(all(p < 47.5 for p in pct))


class Adaptive(unittest.TestCase):
    def test_argmax_beats_endpoints(self):
        conf = [[0.9, 0.7, 0.4, 0.2, 0.1]] * 8
        cost = lambda t: 6.0 + max(0, t - 16) * 0.2
        B, e, c = di.adaptive_budget(conf, [5] * 8, [1] * 8, 8, 1.5, cost)
        r = [x / y for x, y in zip(e, c)]
        self.assertEqual(r[B], max(r))
        self.assertLess(0, B)
        self.assertLess(B, 40)

    def test_flat_cost_verifies_all(self):
        conf = [[0.9, 0.8, 0.7]] * 4
        B, _, _ = di.adaptive_budget(conf, [3] * 4, [1] * 4, 4, 1.0, lambda t: 5.0)
        self.assertEqual(B, 12)

    def test_survival_is_cumprod(self):
        conf = [[0.5, 0.5]]
        _, e, _ = di.adaptive_budget(conf, [2], [1], 1, 1.0, lambda t: 1.0)
        self.assertEqual(e, [1.0, 1.5, 1.75])

    def test_max_budget_truncates(self):
        conf = [[0.9] * 5] * 4
        B, e, _ = di.adaptive_budget(conf, [5] * 4, [1] * 4, 4, 1.0, lambda t: 1.0, max_budget=7)
        self.assertEqual(len(e), 8)
        self.assertLessEqual(B, 7)

    def test_default_numbers(self):
        out = run(["--scenario", "adaptive"])
        self.assertIn("B=105", out)
        self.assertIn("k1 0.73", out)


class Grep(unittest.TestCase):
    def test_counts(self):
        v, s = di.load_titles("vllm_commits.tsv.gz"), di.load_titles("sglang_commits.tsv.gz")
        self.assertEqual((len(v), len(s)), (1942, 2135))
        c = {k: (a, b) for k, a, b in di.topic_counts(v, s)}
        self.assertEqual(c["deepgemm"], (21, 9))
        self.assertEqual(c["megamoe"], (18, 17))
        self.assertEqual(c["dspark"], (25, 50))
        self.assertEqual(c["replay"], (28, 53))
        self.assertEqual(c["dsv4.1"], (68, 100))
        self.assertEqual(c["dsv4"], (30, 127))
        self.assertEqual(c["deepinfer"], (0, 0))

    def test_dsv4_excludes_41(self):
        rx4 = re.compile(dict(di.TOPICS)["dsv4"], re.I)
        self.assertFalse(rx4.search("[DSv4.1] x"))
        self.assertTrue(rx4.search("DeepSeek-V4-Pro"))


class Misc(unittest.TestCase):
    def test_lines_total(self):
        self.assertEqual(sum(n for _, n, _ in di.LINES[:3]), 78412)

    def test_main_all_runs(self):
        out = run(["--scenario", "all"])
        for sc in di.SCEN:
            self.assertIn(f"== {sc}", out)


if __name__ == "__main__":
    unittest.main()
