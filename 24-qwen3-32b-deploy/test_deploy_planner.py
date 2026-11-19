"""deploy_planner.py 的回归测试：python3 -m unittest test_deploy_planner -v

字节数钉在 Qwen3-32B / Qwen3.5-27B 的 config.json、HF 仓库文件大小和 vLLM 0d6f92a 的源码默认值上，
校准钉在 #22344 的启动日志上（5.68 GiB / 23,264 token / 1.42x）。正文里的数都从这里来。
"""
import contextlib
import io
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


class Weights(unittest.TestCase):
    def test_params_split(self):
        self.assertEqual(dp.PARAMS_LINEAR + dp.PARAMS_BF16, dp.PARAMS_TOTAL)
        self.assertEqual(2 * dp.VOCAB * dp.HIDDEN, 1_555_824_640)    # embed + lm_head，余下 677K 是 norm

    def test_ckpt_gib(self):
        self.assertAlmostEqual(dp.CKPT_BYTES["bf16"] / GiB, 61.02, places=2)
        self.assertAlmostEqual(dp.CKPT_BYTES["fp8"] / GiB, 31.97, places=2)
        self.assertAlmostEqual(dp.CKPT_BYTES["awq"] / GiB, 18.00, places=2)
        self.assertAlmostEqual(dp.CKPT_BYTES["bf16"] * 8 / dp.PARAMS_TOTAL, 16.0, places=1)

    def test_recipe_min(self):
        self.assertEqual(dp.RECIPE_MIN_GB, {"bf16": 77, "fp8": 39, "awq": 20, "nvfp4": 22})


class KVBytes(unittest.TestCase):
    def test_per_token(self):
        self.assertEqual(dp.kv_per_token("bf16"), 256 * 1024)
        self.assertEqual(dp.kv_per_token("fp8"), 128 * 1024)
        self.assertEqual(dp.kv_per_token("bf16", tp=2), 128 * 1024)
        self.assertEqual(dp.kv_per_token("bf16", tp=8), 32 * 1024)

    def test_full_context(self):
        self.assertEqual(dp.MAX_POS * dp.kv_per_token() / GiB, 10.0)

    def test_qwen35_quarter(self):
        self.assertEqual(dp.q35_kv_per_token(), 64 * 1024)
        self.assertEqual(dp.kv_per_token() // dp.q35_kv_per_token(), 4)


class Defaults(unittest.TestCase):
    def test_batch_tiers(self):
        self.assertEqual(dp.default_batch("h100"), (8192, 1024))
        self.assertEqual(dp.default_batch("a100"), (2048, 256))       # 名字里有 a100
        self.assertEqual(dp.default_batch("rtx4090"), (2048, 256))
        self.assertEqual(dp.default_batch("b200"), (16384, 1024))

    def test_capture_sizes(self):
        s = dp.capture_sizes(1024, 8192)
        self.assertEqual((len(s), s[:4], s[-1]), (51, [1, 2, 4, 8], 512))
        self.assertEqual(dp.capture_sizes(32, 8192, max_capture=8), [1, 2, 4, 8])
        self.assertEqual(dp.capture_sizes(32, 2048, 4)[-1], 256)          # eagle3 k=3：32 × 4 × 2
        self.assertEqual(len(dp.capture_sizes(1024, 8192, mode="interactivity")), 76)
        self.assertEqual(dp.capture_sizes(1024, 16384, family100=True)[-1], 1024)

    def test_kernel_path(self):
        self.assertIn("Marlin", dp.kernel_path("sm89", "fp8"))
        self.assertIn("DeepGEMM", dp.kernel_path("sm90", "fp8"))
        self.assertIn("Marlin", dp.kernel_path("sm89", "awq"))
        self.assertIn("回退", dp.kernel_path("sm90", "nvfp4"))


class Calibration(unittest.TestCase):
    def test_issue_22344(self):
        p = dp.plan("rtx5090", 1, 1, "awq", "bf16", 0.95, 16384, 100, 32768, eager=True)
        self.assertAlmostEqual(p["kv"], 5.68, places=2)
        self.assertAlmostEqual(p["tokens"], 23264, delta=5)
        self.assertAlmostEqual(p["conc"], 1.42, places=2)

    def test_lmcache_tokens_consistent(self):
        # FAQ：33.66 GiB × 2 卡 / 每 token 0.2441 MiB ≈ 275,760
        self.assertAlmostEqual(33.66 * 2 * 1024 / (dp.kv_per_token() / 2 ** 20), 275_743, delta=50)

    def test_eagle3_extra(self):
        self.assertAlmostEqual(dp.EAGLE3_EXTRA_GIB, 1.37, places=2)


class Fit(unittest.TestCase):
    def test_h100(self):
        self.assertFalse(dp.plan("h100", fmt="bf16")["fits"])
        p = dp.plan("h100", fmt="fp8")
        self.assertTrue(p["fits"])
        self.assertAlmostEqual(p["kv"], 37.8, places=1)

    def test_4090_awq(self):
        self.assertFalse(dp.plan("rtx4090", fmt="awq")["fits"])
        self.assertTrue(dp.plan("rtx4090", fmt="awq", kv_dtype="fp8", gmu=0.95, max_len=16384, seqs=8, batched=2048)["fits"])

    def test_5090_awq(self):
        self.assertTrue(dp.plan("rtx5090", fmt="awq", kv_dtype="fp8", seqs=32, batched=4096)["fits"])

    def test_tp2_doubles_tokens(self):
        one = dp.plan("h100", 1, 1, "fp8")["tokens"]
        two = dp.plan("h100", 2, 2, "bf16")["tokens"]
        self.assertGreater(two, 2 * one)


class Decode(unittest.TestCase):
    def test_h100_single_stream(self):
        _, pu, _ = dp.decode_bound("h100", 1, "fp8", 1)
        self.assertAlmostEqual(72.5 / pu, 1.03, places=2)


class Qwen35(unittest.TestCase):
    def test_gptq_barely_smaller(self):
        self.assertAlmostEqual(dp.Q35_CKPT["fp8"] - dp.Q35_CKPT["gptq"], 0.59, places=2)
        self.assertGreater(dp.Q35_CKPT["gptq"] * GiB / 1e9, 24)

    def test_parts_sum(self):
        for i, fmt in ((1, "bf16"), (2, "fp8"), (3, "gptq")):
            self.assertAlmostEqual(sum(r[i] for r in dp.Q35_PARTS), dp.Q35_CKPT[fmt], delta=0.05)

    def test_block_size(self):
        self.assertEqual(dp.q35_mamba_page(1), 3_207_168)
        self.assertEqual(dp.q35_block_size(1, "bf16", 0, 16), 784)
        self.assertEqual(dp.q35_block_size(1, "bf16", 0, 64), 832)
        self.assertEqual(dp.q35_block_size(2, "fp8", 1, 64), 1600)

    def test_state(self):
        self.assertAlmostEqual(dp.q35_state_per_seq() / dp.MiB, 146.8, places=1)
        self.assertEqual(dp.q35_state_per_seq(mode="align"), 2 * dp.q35_state_per_seq())


class CLI(unittest.TestCase):
    def test_all(self):
        out = run(["--scenario", "all"])
        for name in dp.SCEN:
            self.assertIn(f"== {name}", out)
        self.assertIn("23,265 token（日志 23,264）", out)
        self.assertIn("块 784 token", out)

    def test_mem_4090(self):
        out = run(["--scenario", "mem", "--gpu", "rtx4090", "--fmt", "awq"])
        self.assertIn("装不下一条满长请求", out)


if __name__ == "__main__":
    unittest.main()
