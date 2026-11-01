"""deploy_planner.py 的回归测试：python3 -m unittest test_deploy_planner -v

字节数钉在 vLLM 0d6f92a 的源码常量和 recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml 的注释上，
两条校准（H200 81.2 / 38.5 GiB、4×PRO 6000 74.29 GiB / 8,050,969 token）钉在实测上。正文里的数都从这里来。
"""
import contextlib
import io
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deploy_planner as dp  # noqa: E402

GiB = dp.GiB


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dp.main(argv)
    return buf.getvalue()


class Checkpoint(unittest.TestCase):
    def test_total_matches_recipe(self):
        self.assertAlmostEqual(dp.TOTAL_GIB, 475.3, places=1)
        self.assertEqual(dp.VRAM_MIN_GB, 614)
        self.assertEqual(round(dp.CKPT_GB * 1.2), 613)      # recipe 写 614，是向上取整

    def test_engram_bytes(self):
        self.assertAlmostEqual(dp.ENGRAM_VALUES_GIB, 183.11, places=2)
        self.assertAlmostEqual(dp.ENGRAM_GIB, 188.83, places=2)
        self.assertAlmostEqual(dp.ENGRAM_GIB / 8, 23.60, places=2)   # recipe：23.6 GiB per rank

    def test_tp8_resident_static(self):
        # recipe H100 注释：~36 GiB/GPU of resident weights at TP8
        self.assertAlmostEqual(dp.NON_ENGRAM_GIB / 8, 35.8, places=1)

    def test_expert_size(self):
        self.assertEqual(dp.EXPERT_PARAMS, 35_389_440)
        self.assertAlmostEqual(dp.EXPERT_BYTES / 1e6, 18.8, places=1)


class KVBytes(unittest.TestCase):
    def test_entries(self):
        self.assertEqual(dp.ENTRIES_PER_TOKEN, 2.5)

    def test_sm100_matches_blog(self):
        self.assertEqual(dp.kv_bytes_per_token("sm100"), dp.BLOG_GLOBAL)

    def test_sm90_sm120_double(self):
        for arch in ("sm90", "sm120", "sm89"):
            self.assertEqual(dp.kv_bytes_per_token(arch), 1790)
        self.assertAlmostEqual(1790 / 890, 2.01, places=2)

    def test_swa(self):
        self.assertAlmostEqual(dp.swa_bytes_per_req("sm90") / 2 ** 20, 2.85, places=2)
        self.assertAlmostEqual(dp.swa_bytes_per_req("sm100") / 2 ** 20, 2.58, places=2)


class Indexer(unittest.TestCase):
    def test_v031_fixed(self):
        self.assertEqual(dp.indexer_reserve("sm90", 1 << 20, 16, 4096, 5, "v031"), 512 * dp.MiB)

    def test_recipe_shape(self):
        self.assertEqual(dp.indexer_reserve("sm90", 1 << 20, 16, 8192, 5, "recipe"), 16 * GiB)
        self.assertEqual(dp.indexer_reserve("sm90", 1 << 20, 16, 4096, 5, "recipe"), 8 * GiB)

    def test_row_align(self):
        self.assertEqual(math.lcm(384, 1024 // 2), 1536)
        self.assertEqual(math.lcm(256, 1024 // 4), 256)

    def test_main_floor_and_growth(self):
        self.assertEqual(dp.indexer_reserve("sm120", 1 << 20, 16, 4096, 5, "main"), 512 * dp.MiB)
        self.assertEqual(dp.indexer_reserve("sm90", 1 << 20, 256, 4096, 5, "main"), 6 * GiB)

    def test_gather_workspace(self):
        self.assertEqual(dp.prefill_buffer_entries(1 << 20), (1 << 20) * 8 * 2)
        self.assertEqual(dp.prefill_buffer_entries(256 * 1024), 256 * 1024 * 8 * 5)
        self.assertAlmostEqual(dp.gather_workspace("sm90", 1 << 20) / GiB, 2.06, places=2)


class Calibration(unittest.TestCase):
    def test_h200_recipe(self):
        p = dp.plan("h200", 8, 4, "host", 0, 0.92, 1 << 20, 16, 8192, 5, "v031")
        self.assertAlmostEqual(p["resident"], 81.2, places=1)
        self.assertAlmostEqual(p["kv"], 38.5, places=1)

    def test_pro6000_commdata(self):
        p = dp.plan("rtxpro6000", 4, 4, "host", 0, 0.93, 1 << 20, 16, 4096, 5, "v031")
        self.assertAlmostEqual(p["resident"], 74.29, places=2)
        self.assertAlmostEqual(p["tokens"] / 1e6, 8.05, delta=0.03)


class Host(unittest.TestCase):
    def test_pinned_pow2(self):
        exact, pinned = dp.engram_host_bytes(4, 4)
        self.assertAlmostEqual(exact / GiB, 188.83, places=2)
        self.assertAlmostEqual(pinned / GiB, 264.0, places=1)

    def test_thp_exact(self):
        exact, pinned = dp.engram_host_bytes(4, 4, thp=True)
        self.assertEqual(exact, pinned)

    def test_dp_shared(self):
        self.assertEqual(dp.engram_shards(8, 4, "host"), 4)
        self.assertEqual(dp.engram_shards(8, 4, "gpu"), 8)
        exact, pinned = dp.engram_host_bytes(8, 4)
        self.assertEqual(exact, pinned)


class Ranking(unittest.TestCase):
    def setUp(self):
        self.rows = {r["name"]: r for r in dp.rank_rows()}

    def test_4090_24g_infeasible(self):
        self.assertFalse(self.rows["8×4090 24G"]["p"]["fits"])

    def test_feasible(self):
        for name in ("4×PRO 6000", "8×PRO 6000", "8×H20 96G Engram 主机", "8×H20-3e TP8 Engram 显存",
                     "8×4090 48G（改装）", "8×5090", "1×PRO 6000 + 768G 内存"):
            self.assertTrue(self.rows[name]["p"]["fits"], name)

    def test_h20_3e_beats_h20(self):
        a = self.rows["8×H20-3e TP8 Engram 显存"]["p"]["tokens"]
        b = self.rows["8×H20 96G Engram 显存"]["p"]["tokens"]
        self.assertGreater(a, 3 * b)


class CLI(unittest.TestCase):
    def test_all_runs(self):
        out = run(["--scenario", "all"])
        for head in ("== ckpt", "== mem", "== kv", "== indexer", "== host", "== decode", "== rank"):
            self.assertIn(head, out)
        self.assertIn("(288 + 68) × 2.5 = 890 B", out)
        self.assertIn("(584 + 132) × 2.5 = 1790 B", out)


if __name__ == "__main__":
    unittest.main()
