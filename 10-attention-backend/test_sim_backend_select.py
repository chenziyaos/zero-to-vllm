"""sim_backend_select.py 的用例：python3 -m unittest test_sim_backend_select -v"""
import unittest

from sim_backend_select import (
    ALWAYS, BACKENDS, UNIFORM_BATCH, UNIFORM_SINGLE_TOKEN_DECODE, Cfg, MLADims, SglEnv, ARCHS,
    fa_version, mla_decode_cost, mla_prefill_attn_flops, priorities, resolve_mode, select, sglang_default,
    validate,
)


class TestPriorities(unittest.TestCase):
    def test_hopper_dense_fa_first(self):
        self.assertEqual(priorities(Cfg('qwen3-8b', 'h100-sxm'))[:2], ['FLASH_ATTN', 'FLASHINFER'])

    def test_blackwell_dense_flashinfer_first(self):
        self.assertEqual(priorities(Cfg('qwen3-8b', 'b200'))[:2], ['FLASHINFER', 'FLASH_ATTN'])

    def test_mla_order_by_sm(self):
        self.assertEqual(priorities(Cfg('deepseek-v3', 'b200'))[0], 'FLASHINFER_MLA')
        self.assertEqual(priorities(Cfg('deepseek-v3', 'h100-sxm'))[0], 'FLASH_ATTN_MLA')
        self.assertEqual(priorities(Cfg('deepseek-v3', 'rtx-pro-6000'))[0], 'TRITON_MLA')


class TestSelect(unittest.TestCase):
    def pick(self, *a, **k):
        return select(Cfg(*a, **k))

    def test_fa_version(self):
        self.assertEqual([fa_version(Cfg('qwen3-8b', g)) for g in ('a100-80g', 'h100-sxm', 'b200')], [2, 3, 4])

    def test_a100_fp8_falls_to_flashinfer(self):
        r = self.pick('qwen3-8b', 'a100-80g', kv='fp8')
        self.assertEqual(r.backend, 'FLASHINFER')
        self.assertIn('kv_cache_dtype not supported', r.invalid['FLASH_ATTN'][1])

    def test_e5m2_never_fa(self):
        self.assertEqual(self.pick('qwen3-8b', 'h100-sxm', kv='fp8_e5m2').backend, 'FLASHINFER')

    def test_sinks_on_ampere_only_triton(self):
        r = self.pick('gpt-oss-120b', 'a100-80g')
        self.assertEqual(r.backend, 'TRITON_ATTN')
        self.assertEqual([n for _, n in r.valid], ['TRITON_ATTN'])

    def test_sinks_hopper_and_blackwell(self):
        self.assertEqual(self.pick('gpt-oss-120b', 'h100-sxm').backend, 'FLASH_ATTN')
        self.assertEqual(self.pick('gpt-oss-120b', 'b200').backend, 'FLASHINFER')

    def test_batch_invariant_drops_flashinfer(self):
        r = self.pick('qwen3-8b', 'h100-sxm', batch_invariant=True)
        self.assertNotIn('FLASHINFER', [n for _, n in r.valid])

    def test_mla_defaults(self):
        self.assertEqual(self.pick('deepseek-v3', 'h100-sxm').backend, 'FLASH_ATTN_MLA')
        self.assertEqual(self.pick('deepseek-v3', 'a100-80g').backend, 'TRITON_MLA')
        r = self.pick('deepseek-v3', 'b200')
        self.assertEqual((r.backend, r.block, r.cg), ('FLASHINFER_MLA', 32, ALWAYS))

    def test_mla_fp8_on_hopper_goes_flashmla(self):
        r = self.pick('deepseek-v3', 'h100-sxm', kv='fp8')
        self.assertEqual((r.backend, r.block), ('FLASHMLA', 64))
        self.assertTrue(any('Setting kv cache block size to 64' in s for s in r.logs))

    def test_user_backend_invalid_raises(self):
        r = self.pick('deepseek-v3', 'a100-80g', backend='FLASHMLA')
        self.assertIsNone(r.backend)
        self.assertIn('compute capability not supported', r.error)

    def test_user_backend_skips_priority(self):
        r = self.pick('qwen3-8b', 'h100-sxm', backend='TRITON_ATTN')
        self.assertEqual(r.backend, 'TRITON_ATTN')
        self.assertEqual(r.valid, [])


class TestBlockSize(unittest.TestCase):
    def test_block16_precludes_on_b200(self):
        r = select(Cfg('deepseek-v3', 'b200', block_size=16))
        self.assertEqual(r.backend, 'TRITON_MLA')
        warn = [s for s in r.logs if s.startswith('WARNING')]
        self.assertEqual(len(warn), 1)
        self.assertIn('FLASHINFER_MLA, CUTLASS_MLA', warn[0])
        self.assertNotIn('TOKENSPEED_MLA', warn[0])     # 还有别的原因，不算"被 block 挤掉"

    def test_multiple_of_kernel_block(self):
        self.assertTrue(BACKENDS['FLASHINFER_MLA'].supports_block(Cfg('deepseek-v3', 'b200'), 128))
        self.assertFalse(BACKENDS['CUTLASS_MLA'].supports_block(Cfg('deepseek-v3', 'b200'), 64))

    def test_block8_no_backend(self):
        self.assertEqual(select(Cfg('qwen3-8b', 'h100-sxm', block_size=8)).error, 'No valid attention backend found')

    def test_flashinfer_large_blocks_only_trtllm(self):
        b = BACKENDS['FLASHINFER']
        self.assertTrue(b.supports_block(Cfg('qwen3-8b', 'b200'), 1024))
        self.assertTrue(b.supports_block(Cfg('qwen3-8b', 'b200'), 96))       # 96 = 6 × 16，混合 block 规则
        self.assertFalse(b.supports_block(Cfg('qwen3-8b', 'b200'), 24))
        self.assertTrue(b.supports_block(Cfg('qwen3-8b', 'h100-sxm'), 128))     # 128 是 64 的倍数


class TestCudagraph(unittest.TestCase):
    def test_fa2_vs_fa3(self):
        self.assertEqual(BACKENDS['FLASH_ATTN'].cg(Cfg('qwen3-8b', 'a100-80g')), UNIFORM_BATCH)
        self.assertEqual(BACKENDS['FLASH_ATTN'].cg(Cfg('qwen3-8b', 'h100-sxm')), ALWAYS)

    def test_flashinfer_without_cubin(self):
        s = BACKENDS['FLASHINFER'].cg(Cfg('qwen3-8b', 'h100-sxm', artifactory=False))
        self.assertEqual(s, UNIFORM_SINGLE_TOKEN_DECODE)
        self.assertEqual(resolve_mode('FULL_AND_PIECEWISE', s, query_len=4), 'PIECEWISE')

    def test_full_downgrade(self):
        self.assertEqual(resolve_mode('FULL', UNIFORM_BATCH), 'FULL_AND_PIECEWISE')
        self.assertEqual(resolve_mode('FULL', ALWAYS), 'FULL')


class TestMLA(unittest.TestCase):
    def test_decode_absorb_wins(self):
        c = mla_decode_cost(MLADims(), 8192)
        self.assertLess(c['mqa'][0] * 100, c['mha'][0])
        self.assertEqual(c['mqa'][1], c['mha'][1])          # 两条路径读的都是 latent
        self.assertAlmostEqual(c['mha_cached'][0] / c['mha_cached'][1], 1.0)

    def test_prefill_mha_wins(self):
        mha, mqa = mla_prefill_attn_flops(MLADims(), 8192, 8192)
        self.assertGreater(mqa / mha, 3)


class TestSGLang(unittest.TestCase):
    def test_defaults(self):
        q, d, g = ARCHS['qwen3-8b'], ARCHS['deepseek-v3'], ARCHS['gpt-oss-120b']
        self.assertEqual(sglang_default(q, SglEnv('h100-sxm')), 'fa3')
        self.assertEqual(sglang_default(q, SglEnv('h100-sxm', spec_topk=4)), 'flashinfer')
        self.assertEqual(sglang_default(q, SglEnv('b200')), 'trtllm_mha')
        self.assertEqual(sglang_default(g, SglEnv('a100-80g')), 'triton')
        self.assertEqual(sglang_default(d, SglEnv('b200', arch_name='DeepseekV3ForCausalLM')), 'trtllm_mla')

    def test_validate_reason_order(self):
        r = validate(BACKENDS['FLASH_ATTN'], Cfg('gpt-oss-120b', 'a100-80g'))
        self.assertEqual(r, ['attention sinks not supported', 'sink not supported on compute capability < 9.0'])


if __name__ == '__main__':
    unittest.main()
