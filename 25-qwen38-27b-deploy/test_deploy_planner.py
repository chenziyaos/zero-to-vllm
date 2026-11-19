"""deploy_planner.py 的回归测试：python3 -m unittest test_deploy_planner -v

字节数钉在 Qwen/Qwen3.8-27B 的 config.json（text_config）、六家 HF 仓库
safetensors 合计和 vLLM 0d6f92a 的源码默认值上：
  - conv 状态通道数 10240 来自 mamba_utils.py:282 gated_delta_net_state_shape
  - 时间状态 48 个 v 头 × 128 × 128 × fp32，mamba_ssm_dtype=float32
  - 混合层块大小对齐逻辑来自 platforms/interface.py:1050-1125
  - recipe 实测数字只出现在 fit 场景的打印里，规划器不做反推拟合
正文和画板里的数都从这里来。
"""
import contextlib
import io
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deploy_planner as dp  # noqa: E402

GiB, MiB = dp.GiB, dp.MiB


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dp.main(argv)
    return buf.getvalue()


class Config(unittest.TestCase):
    def test_layer_split(self):
        # layer_types：3 层 linear_attention 后 1 层 full_attention，64 层一轮不缺
        self.assertEqual(dp.GDN_LAYERS + dp.FULL_ATTN, dp.LAYERS)
        self.assertEqual(dp.GDN_LAYERS, 48)
        self.assertEqual(dp.FULL_ATTN, dp.LAYERS // 4)

    def test_conv_dim_matches_source(self):
        # mamba_utils.py:282：head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
        self.assertEqual(dp.GDN_CONV_DIM, 128 * 16 * 2 + 128 * 48)
        self.assertEqual(dp.GDN_CONV_DIM, 10240)
        self.assertEqual(dp.GDN_CONV_DIM, 2 * dp.HIDDEN)

    def test_embed_and_lm_head(self):
        # embed 和 lm_head 各 2.37 GiB（共享权重时只加载一份，这里按仓库文件算）
        self.assertAlmostEqual(2 * dp.VOCAB * dp.HIDDEN * 2 / GiB, 4.74, places=2)


class Checkpoint(unittest.TestCase):
    def test_six_ckpts(self):
        # HF 树上 safetensors 相加，2026-11-03 核对
        self.assertAlmostEqual(dp.CKPT_GIB["bf16"], 51.75, places=2)
        self.assertAlmostEqual(dp.CKPT_GIB["fp8"], 28.75, places=2)
        self.assertAlmostEqual(dp.CKPT_GIB["nvfp4"], 24.57, places=2)
        self.assertAlmostEqual(dp.CKPT_GIB["nvnv"], 20.42, places=2)
        self.assertAlmostEqual(dp.CKPT_GIB["int4"], 18.12, places=2)
        self.assertAlmostEqual(dp.CKPT_GIB["mxfp4"], 18.47, places=2)

    def test_no_uniform_4bit(self):
        # 18.1 GiB INT4 对应 5.76 bit/参数：embed / lm_head / 视觉塔保持 BF16
        for fmt in ("int4", "mxfp4"):
            self.assertGreater(dp.CKPT_BYTES[fmt] * 8 / dp.PARAMS, 5.5)

    def test_legacy_gptq_smaller_than_fp8_bigger_than_4bit(self):
        # Qwen3.5-27B-GPTQ-Int4 只量化 MLP，28.16 GiB：比 3.8 FP8（28.75）小，
        # 但比 3.8 四种 4 bit 档都大；不能写成「比 FP8 大」。
        self.assertAlmostEqual(dp.LEGACY_GPTQ_GIB, 28.16, places=2)
        self.assertLess(dp.LEGACY_GPTQ_GIB, dp.CKPT_GIB["fp8"])
        for fmt in ("nvfp4", "nvnv", "int4", "mxfp4"):
            self.assertGreater(dp.LEGACY_GPTQ_GIB, dp.CKPT_GIB[fmt])

    def test_recipe_min_is_declared_not_derived(self):
        # recipe vram_minimum_gb 是上游声明值；注释公式 ceil(磁盘 GB×1.2) 对部分档位
        # 恰好对上（nvfp4 32），对 nvidia 变体对不上（磁盘约 21.9 GB，公式 27≠声明 32），
        # 所以这里只钉声明值，不把它当通用公式。
        self.assertEqual(dp.RECIPE_MIN_GB["nvfp4"], 32)
        self.assertEqual(dp.RECIPE_MIN_GB["int4"], 24)
        nvnv_formula = math.ceil(dp.CKPT_BYTES["nvnv"] / 1e9 * 1.2)
        self.assertNotEqual(nvnv_formula, dp.RECIPE_MIN_GB["nvnv"])

    def test_w8a8_ascend(self):
        self.assertAlmostEqual(dp.W8A8_BYTES / 1e9, 32.1, places=1)


class GDNState(unittest.TestCase):
    def test_one_layer_page_bytes(self):
        # TP1 一页 GDN 状态 3,207,168 B = conv 60 KiB + 时间状态 3072 KiB
        self.assertEqual(dp.gdn_state_one_layer(1, 0), 3_207_168)
        conv = (dp.GDN_CONV_DIM // 1) * (dp.CONV_K - 1) * 2
        temporal = dp.GDN_V_HEADS * dp.GDN_HEAD_DIM ** 2 * 4
        self.assertEqual((conv, temporal), (61_440, 3_145_728))

    def test_tp2_mtp1_page(self):
        # TP2 + MTP k=1：conv 减半且多一个历史位，时间状态减半 = 1,613,824 B
        self.assertEqual(dp.gdn_state_one_layer(2, 1), 1_613_824)

    def test_per_seq_none(self):
        self.assertAlmostEqual(dp.gdn_state_per_seq(1, 0, "none") / MiB, 146.8, places=0)
        # TP2 MTP1 每条仍约 148 MiB（一半 v 头，但 MTP 多一份 conv）
        self.assertAlmostEqual(dp.gdn_state_per_seq(2, 1, "none") / MiB, 147.8, places=0)

    def test_per_seq_align_doubles_tp1(self):
        # 开 prefix caching 走 align，TP1 每条 2 页，正好翻倍到 294 MiB
        self.assertAlmostEqual(dp.gdn_state_per_seq(1, 0, "align") / MiB, 293.6, places=0)
        self.assertEqual(dp.gdn_state_per_seq(1, 0, "align"),
                         2 * dp.gdn_state_per_seq(1, 0, "none"))

    def test_per_seq_align_tp2_mtp1(self):
        self.assertAlmostEqual(dp.gdn_state_per_seq(2, 1, "align") / MiB, 221.6, places=0)

    def test_state_independent_of_context(self):
        # 固定大小状态，和上下文长度没关系，这是混合模型最关键的一条。
        # 用两个不同 max_len 的 plan 验证每条序列状态相同，而不是自己和自己比。
        short = dp.plan("h100", fmt="bf16", max_len=32768)
        long = dp.plan("h100", fmt="bf16", max_len=dp.MAX_POS)
        self.assertEqual(short["state_each"], long["state_each"])
        self.assertAlmostEqual(short["state_each"] * 1024, 146.8, places=0)


class BlockSize(unittest.TestCase):
    def test_reference_values(self):
        self.assertEqual(dp.block_size(1, "bf16", 0, 16), 784)
        self.assertEqual(dp.block_size(1, "bf16", 0, 64), 832)
        self.assertEqual(dp.block_size(2, "fp8", 1, 16), 1584)
        self.assertEqual(dp.block_size(2, "fp8", 1, 64), 1600)

    def test_block_holds_one_gdn_page(self):
        # 一块注意力 KV 字节必须 ≥ 一层 GDN 状态页
        for tp, kvd, spec in ((1, "bf16", 0), (2, "fp8", 1)):
            bs = dp.block_size(tp, kvd, spec, 16)
            attn = bs * 2 * max(dp.KV_HEADS // tp, 1) * dp.HEAD_DIM * dp.KV_ELEM[kvd]
            self.assertGreaterEqual(attn, dp.gdn_state_one_layer(tp, spec))
            prev = (bs - 16) * 2 * max(dp.KV_HEADS // tp, 1) * dp.HEAD_DIM * dp.KV_ELEM[kvd]
            self.assertLess(prev, dp.gdn_state_one_layer(tp, spec))

    def test_block_is_multiple_of_alignment(self):
        for al in (16, 64):
            self.assertEqual(dp.block_size(1, "bf16", 0, al) % al, 0)


class KV(unittest.TestCase):
    def test_per_token(self):
        self.assertEqual(dp.kv_per_token("bf16"), 64 * 1024)
        self.assertEqual(dp.kv_per_token("fp8"), 32 * 1024)
        self.assertEqual(dp.kv_per_token("bf16", tp=2), 32 * 1024)

    def test_full_context(self):
        self.assertAlmostEqual(dp.MAX_POS * dp.kv_per_token("bf16") / GiB, 16.0, places=1)
        self.assertAlmostEqual(dp.MAX_POS * dp.kv_per_token("fp8") / GiB, 8.0, places=1)

    def test_quarter_of_dense_32b(self):
        # Qwen3-32B：64 层全注意力 × 8 KV 头 × 128 头维 = 256 KiB/token，是这里 4 倍
        dense = 2 * 64 * 8 * 128 * 2
        self.assertEqual(dense // dp.kv_per_token("bf16"), 4)


class Plan(unittest.TestCase):
    def test_h100_fp8_pool(self):
        # seqs=32 采样器预热可以忽略；可用 72.8 − 权重 28.75 − 其他 ≈ 42.0
        p = dp.plan("h100", fmt="fp8", seqs=32)
        self.assertAlmostEqual(p["pool"], 42.0, places=1)

    def test_h100_fp8_concurrency(self):
        p = dp.plan("h100", fmt="fp8", seqs=32)
        per32 = 32768 * p["per_token"] / GiB + p["state_each"]
        per262 = dp.MAX_POS * p["per_token"] / GiB + p["state_each"]
        self.assertAlmostEqual(per32, 2.14, places=2)
        self.assertAlmostEqual(per262, 16.14, places=2)
        self.assertAlmostEqual(p["pool"] / per32, 19.6, places=1)
        self.assertAlmostEqual(p["pool"] / per262, 2.6, places=1)

    def test_4090_cannot_boot(self):
        # NVFP4 缺 sm89 kernel 且权重超出可用；FP8 在 rank 表里同样起不来
        p = dp.plan("rtx4090", fmt="nvfp4", kv_dtype="fp8", seqs=32)
        self.assertLessEqual(p["pool"], 0)
        self.assertFalse(p["fits"])

    def test_5090_fp8_cannot_boot(self):
        # 32 GB 装 28.75 GiB FP8 权重 + 其他占用后没有 KV 池
        p = dp.plan("rtx5090", fmt="fp8", kv_dtype="fp8", seqs=32)
        self.assertLessEqual(p["pool"], 0)

    def test_5090_nvfp4_eager_32k(self):
        # recipe verified：1×5090 NVFP4，eager，砍到 32K，fp8 KV
        p = dp.plan("rtx5090", fmt="nvfp4", kv_dtype="fp8", seqs=8,
                    eager=True, max_len=32768)
        self.assertGreater(p["pool"], 0)
        per = 32768 * p["per_token"] / GiB + p["state_each"]
        self.assertGreaterEqual(p["pool"] / per, 1)

    def test_5090x2_tp2_full_context(self):
        p = dp.plan("rtx5090", n=2, tp=2, fmt="nvfp4", kv_dtype="fp8",
                    seqs=32, spec=3)
        self.assertGreater(p["pool"], 0)
        per = dp.MAX_POS * p["per_token"] / GiB + p["state_each"]
        self.assertGreaterEqual(p["pool"] / per, 3)

    def test_l40s_int4_bf16kv_barely_one(self):
        # rank 表档位：L40S INT4/W4A16 + bf16 KV，262K 原始字节预算约 1 条（sm89 kernel 未验证）
        p = dp.plan("l40s", fmt="int4", kv_dtype="bf16", seqs=32)
        per32 = 32768 * p["per_token"] / GiB + p["state_each"]
        self.assertGreaterEqual(p["pool"] / per32, 5)
        self.assertLess(p["conc"], 2)

    def test_text_only_saves_vision(self):
        full = dp.plan("h100", fmt="fp8")
        text = dp.plan("h100", fmt="fp8", text_only=True)
        self.assertAlmostEqual(full["resident"] - text["resident"],
                               dp.VISUAL_GIB, places=2)


class Decode(unittest.TestCase):
    def test_h100_fp8_single_stream(self):
        ms, tps, _ = dp.decode_bound("h100", fmt="fp8")
        self.assertAlmostEqual(ms, 13.7, places=1)
        self.assertAlmostEqual(tps, 73.0, places=0)

    def test_tp2_adds_ar_overhead(self):
        ms1, _, _ = dp.decode_bound("h100", tp=1, fmt="bf16")
        ms2, _, _ = dp.decode_bound("h100", tp=2, fmt="bf16")
        self.assertGreater(ms2, 0.55 * ms1)     # 权重减半但 64 层 all-reduce 要补回来一部分


class CLI(unittest.TestCase):
    def test_all(self):
        out = run(["--scenario", "all"])
        for marker in ("六种 checkpoint", "整条序列 48 层", "混合层块大小",
                       "全注意力 KV", "各档配置", "自带 MTP", "单流 decode"):
            self.assertIn(marker, out)
        self.assertIn("块   784 token", out)
        self.assertIn("NVFP4 Inferact", out)

    def test_recipe_source_is_recipes_repo(self):
        out = run(["--scenario", "fit"])
        self.assertIn("vllm-project/recipes/blob/0511d96", out)
        self.assertNotIn("vllm-project/vllm/blob/0d6f92a/recipes", out)

    def test_rank_shows_per_row_context_and_no_absolute_single5090(self):
        out = run(["--scenario", "rank"])
        self.assertIn("每条@ctx", out)
        self.assertIn("上游记录：eager、32K 可起；非唯一可行配置", out)
        self.assertNotIn("必须 eager，只能开 32K", out)

    def test_mem_h100(self):
        out = run(["--scenario", "mem", "--gpu", "h100", "--fmt", "fp8"])
        self.assertIn("KV 池", out)

    def test_mem_4090_dead(self):
        out = run(["--scenario", "mem", "--gpu", "rtx4090", "--fmt", "nvfp4",
                   "--kv-dtype", "fp8"])
        self.assertIn("起不来", out)


class Commands(unittest.TestCase):
    def setUp(self):
        import commands as cm  # noqa: E402
        self.cm = cm

    def test_tp2_profiles_enable_mtp3_by_default(self):
        for profile in ("5090x2-fp8", "5090x2-nvfp4"):
            cmd = self.cm.command(profile)
            self.assertIn('"num_speculative_tokens":3', cmd[cmd.index("--speculative-config") + 1])

    def test_single_5090_no_mtp_by_default(self):
        cmd = self.cm.command("5090-nvfp4")
        self.assertNotIn("--speculative-config", cmd)
        self.assertIn("--enforce-eager", cmd)
        self.assertIn("32768", cmd)

    def test_h100_length_32768_matches_serve(self):
        cmd = self.cm.command("h100-fp8")
        self.assertEqual(cmd[cmd.index("--max-model-len") + 1], "32768")

    def test_explicit_mtp_zero_disables(self):
        cmd = self.cm.command("5090x2-fp8", mtp=0)
        self.assertNotIn("--speculative-config", cmd)

    def test_sglang_only_h200_bf16(self):
        self.cm.command("h200-bf16", engine="sglang")
        with self.assertRaises(ValueError):
            self.cm.command("h100-fp8", engine="sglang")

    def test_bench_command_has_seed_and_rates(self):
        cmd = self.cm.bench("http://127.0.0.1:8000", "qwen38")
        self.assertIn("--seed", cmd)
        self.assertIn("42", cmd)
        self.assertIn("--request-rate", cmd)


class BenchChecks(unittest.TestCase):
    def setUp(self):
        import bench_checks as bc  # noqa: E402
        self.bc = bc

    def _payload(self, content=None, calls=None):
        msg = {"role": "assistant", "content": content}
        if calls is not None:
            msg["tool_calls"] = calls
        return {"choices": [{"message": msg}]}

    def test_arithmetic_pass_and_fail(self):
        self.assertTrue(self.bc.check_smoke(self._payload(content="323"), "arithmetic"))
        with self.assertRaises(ValueError):
            self.bc.check_smoke(self._payload(content="324"), "arithmetic")

    def test_tool_call_parsed(self):
        calls = [{"function": {"name": "get_weather",
                               "arguments": '{"city": "北京"}'}}]
        self.assertTrue(self.bc.check_smoke(self._payload(calls=calls), "tool"))

    def test_tool_call_rejects_wrong_function(self):
        calls = [{"function": {"name": "other", "arguments": '{"city": "北京"}'}}]
        with self.assertRaises(ValueError):
            self.bc.check_smoke(self._payload(calls=calls), "tool")

    def test_acceptance_from_counters(self):
        before = ("# help\n"
                  "vllm:spec_decode_num_draft_tokens_total 100\n"
                  "vllm:spec_decode_num_accepted_tokens_total 80\n")
        after = ("# help\n"
                 "vllm:spec_decode_num_draft_tokens_total 200\n"
                 "vllm:spec_decode_num_accepted_tokens_total 155\n")
        result = self.bc.acceptance(before, after)
        self.assertEqual(result["draft"], 100)
        self.assertEqual(result["accepted"], 75)
        self.assertAlmostEqual(result["acceptance"], 0.75)

    def test_acceptance_no_draft_delta_raises(self):
        same = ("vllm:spec_decode_num_draft_tokens_total 5\n"
                "vllm:spec_decode_num_accepted_tokens_total 3\n")
        with self.assertRaises(ValueError):
            self.bc.acceptance(same, same)


if __name__ == "__main__":
    unittest.main()
