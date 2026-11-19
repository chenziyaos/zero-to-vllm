"""sim_quant_mem.py 的用例：python3 -m unittest test_sim_quant_mem -v"""
import unittest

from sim_quant_mem import (
    ARCHS, FORMATS, MLA_KV, FP8KV, Env, Layer, boundary_skip_layers, break_even, fit, kv_head_bytes,
    prioritize_humming, resolve_backend, select, tp_alignment, tq_slot_bytes, LISTS,
)


class TestWeights(unittest.TestCase):
    def test_bytes_per_param(self):
        self.assertEqual(FORMATS['bf16'][0], 2)
        self.assertAlmostEqual(FORMATS['gptq-int4'][0], 0.515625)
        self.assertAlmostEqual(FORMATS['nvfp4'][0], 0.5625)
        self.assertAlmostEqual(FORMATS['mxfp4'][0], 0.53125)

    def test_dsv3_fp8_fits_h200_tp8_bf16_does_not(self):
        self.assertGreater(fit(ARCHS['dsv3'], 'fp8-block', 'h200', 8)['tokens'], 0)
        self.assertEqual(fit(ARCHS['dsv3'], 'bf16', 'h200', 8)['tokens'], 0)

    def test_int4_70b_single_h100(self):
        self.assertGreater(fit(ARCHS['llama70b'], 'awq-int4', 'h100-sxm', 1)['tokens'], 0)
        self.assertEqual(fit(ARCHS['llama70b'], 'fp8-tensor', 'h100-sxm', 1)['tokens'], 0)


class TestKV(unittest.TestCase):
    def test_turboquant_slots(self):
        self.assertEqual(tq_slot_bytes(128, 8, 4), 196)
        self.assertEqual(tq_slot_bytes(128, 4, 4), 134)
        self.assertEqual(tq_slot_bytes(128, 3, 4), 118)
        self.assertEqual(tq_slot_bytes(128, 3, 3), 102)

    def test_boundary_skip(self):
        self.assertEqual(boundary_skip_layers(32), [0, 1, 30, 31])
        self.assertEqual(boundary_skip_layers(32, hybrid=True), [])
        self.assertEqual(boundary_skip_layers(3), [0, 2])

    def test_head_bytes(self):
        self.assertEqual(kv_head_bytes('auto'), 512)
        self.assertEqual(kv_head_bytes('fp8'), 256)
        self.assertEqual(kv_head_bytes('nvfp4'), 144)
        self.assertEqual(MLA_KV['fp8_ds_mla'], 656)
        self.assertEqual(MLA_KV['nvfp4_ds_mla'], 352)


class TestKernel(unittest.TestCase):
    def test_humming_only_sm90(self):
        names = ['Cutlass', 'Marlin', 'Humming']
        self.assertEqual(prioritize_humming(names, 90), ['Cutlass', 'Humming', 'Marlin'])
        self.assertEqual(prioritize_humming(names, 100), names)

    def test_fp8_block_sm90(self):
        l = Layer('fp8_block', act='dyn_group')
        self.assertEqual(select(Env(90), l)[0], 'FlashInferFp8DeepGEMMDynamicBlockScaled')
        self.assertEqual(select(Env(90, nvcc=False), l)[0], 'CutlassFp8BlockScaledMM')
        self.assertEqual(select(Env(100), l)[0], 'DeepGemmFp8BlockScaledMM')
        self.assertEqual(select(Env(100, model_type='qwen3_5_moe_text'), l)[0], 'CutlassFp8BlockScaledMM')

    def test_nvfp4_a16_forcing(self):
        l = Layer('nvfp4', act='a16')
        self.assertEqual(select(Env(100), l)[0], 'FlashInferCuteDslNvFp4W4A16')
        self.assertEqual(select(Env(120), l)[0], 'MarlinNvFp4')
        self.assertEqual(select(Env(90), l)[0], 'HummingNvFp4')
        self.assertEqual(select(Env(100), l, per_quant={'nvfp4_w4a16': 'humming'})[0], 'HummingNvFp4')

    def test_batch_invariant(self):
        l = Layer('nvfp4', act='a4')
        self.assertEqual(select(Env(100, batch_invariant=True), l)[0], 'CutlassNvFp4')
        self.assertEqual(select(Env(90, batch_invariant=True), l)[0], 'EmulationNvFp4')

    def test_backend_fallback(self):
        log = []
        out = resolve_backend(LISTS['mxfp4'], 90, 'deep_gemm', log)
        self.assertEqual(len(log), 1)
        self.assertEqual(out, prioritize_humming(LISTS['mxfp4'], 90))

    def test_mp(self):
        awq = Layer('mp', weight_type='uint4', zero_points=True)
        self.assertEqual(select(Env(90), awq)[0], 'Machete')
        self.assertEqual(select(Env(80), awq)[0], 'Marlin')
        self.assertEqual(select(Env(70), Layer('mp', act_dtype='fp16'))[0], 'Exllama')


class TestTP(unittest.TestCase):
    def test_qwen72b_padding(self):
        self.assertFalse(tp_alignment(29568, 8)[2])
        self.assertEqual(tp_alignment(29568, 8)[1], 112)
        for tp in (1, 2, 4, 8):
            self.assertTrue(tp_alignment(29696, tp)[2])


class TestFP8KV(unittest.TestCase):
    def test_break_even_llama_h100(self):
        _, _, sb, ib, sf, i_f = FP8KV[0]
        self.assertAlmostEqual(break_even(sb, ib, sf, i_f), 7000, delta=50)


if __name__ == '__main__':
    unittest.main()
