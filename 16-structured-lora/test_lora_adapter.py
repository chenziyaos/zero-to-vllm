"""测 check_adapter（只要标准库）和 train_lora（装了 torch、transformers、peft 才跑）：python3 test_lora_adapter.py"""
import importlib.util
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_adapter as ca  # noqa: E402

HAS_TRAIN_DEPS = all(importlib.util.find_spec(m) for m in ('torch', 'transformers', 'peft'))


def fake_adapter(root, name, cfg, modules=('q_proj', 'v_proj'), added=None):
    d = os.path.join(root, name)
    os.makedirs(d)
    base = {'r': 8, 'lora_alpha': 16, 'target_modules': list(modules), 'bias': 'none',
            'base_model_name_or_path': 'Qwen/Qwen3-0.6B', 'peft_type': 'LORA'}
    base.update(cfg)
    json.dump(base, open(os.path.join(d, 'adapter_config.json'), 'w'))
    header = {'__metadata__': {'format': 'pt'}}
    for m in modules:
        for ab, shape in (('lora_A', [base['r'], 1024]), ('lora_B', [1024, base['r']])):
            header[f'base_model.model.model.layers.0.self_attn.{m}.{ab}.weight'] = {
                'dtype': 'F32', 'shape': shape, 'data_offsets': [0, 0]}
    raw = json.dumps(header).encode()
    with open(os.path.join(d, 'adapter_model.safetensors'), 'wb') as f:
        f.write(struct.pack('<Q', len(raw)) + raw)
    if added:
        json.dump(added, open(os.path.join(d, 'added_tokens.json'), 'w'))
    return d


def run(d, **kw):
    args = SimpleNamespace(max_lora_rank=kw.get('max_lora_rank'), lora_target_modules=kw.get('targets'))
    buf = io.StringIO()
    ok = ca.report(d, args, out=buf)
    return ok, buf.getvalue()


class CheckAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_plain_lora_ok(self):
        ok, out = run(fake_adapter(self.root, 'a', {}))
        self.assertTrue(ok)
        self.assertIn('--max-lora-rank 8 --lora-modules a=', out)
        self.assertIn("实际模块  ['q_proj', 'v_proj']（4 个张量）", out)

    def test_vllm_rank_rounds_up_to_allowed_value(self):
        self.assertEqual(ca.vllm_rank_flag(8), 8)
        self.assertEqual(ca.vllm_rank_flag(12), 16)
        self.assertEqual(ca.vllm_rank_flag(300), 320)
        self.assertIsNone(ca.vllm_rank_flag(513))

    def test_rank_over_limit(self):
        d = fake_adapter(self.root, 'big', {'r': 64})
        ok, out = run(d, max_lora_rank=16)
        self.assertFalse(ok)
        self.assertIn('rank 64 大于 --max-lora-rank 16', out)

    def test_rank_pattern_counts_for_vllm(self):
        cfg = {'r': 8, 'rank_pattern': {'q_proj': 32}}
        self.assertEqual(ca.max_rank(cfg), 32)
        self.assertTrue(ca.check_vllm(cfg, max_lora_rank=16))
        self.assertFalse(ca.check_vllm(cfg, max_lora_rank=32))

    def test_vllm_only_allows_listed_rank_values(self):
        errs = ca.check_vllm({'r': 8}, max_lora_rank=24)
        self.assertTrue(any('不在可选值' in e for e in errs))

    def test_dora_rejected_by_both(self):
        ok, out = run(fake_adapter(self.root, 'dora', {'use_dora': True}))
        self.assertFalse(ok)
        self.assertIn('vLLM    不能加载：不支持 DoRA', out)
        self.assertIn('SGLang  不能加载：不支持 DoRA', out)

    def test_modules_to_save(self):
        self.assertTrue(ca.check_vllm({'r': 8, 'modules_to_save': ['lm_head']}))
        self.assertFalse(ca.check_vllm({'r': 8, 'modules_to_save': ['score']}))

    def test_bias_and_init(self):
        self.assertTrue(ca.check_vllm({'r': 8, 'bias': 'lora_only'}))
        self.assertTrue(ca.check_vllm({'r': 8, 'init_lora_weights': 'pissa'}))
        self.assertFalse(ca.check_vllm({'r': 8, 'init_lora_weights': 'gaussian'}))
        self.assertFalse(ca.check_vllm({'r': 8, 'init_lora_weights': True}))

    def test_added_tokens_only_block_sglang(self):
        ok, out = run(fake_adapter(self.root, 'tok', {}, added={'<city>': 151669}))
        self.assertFalse(ok)
        self.assertIn('vLLM    可以加载', out)
        self.assertIn('SGLang  不能加载：adapter 往词表里加了 1 个 token', out)

    def test_sglang_target_modules_normalized(self):
        d = fake_adapter(self.root, 'qkv', {}, modules=('q_proj', 'k_proj', 'v_proj', 'gate_proj'))
        # q/k/v 都归一成 qkv_proj，gate_proj 归一成 gate_up_proj
        ok, _ = run(d, targets=['qkv_proj', 'gate_up_proj'])
        self.assertTrue(ok)
        ok, out = run(d, targets=['q_proj'])
        self.assertFalse(ok)
        self.assertIn("模块 ['gate_up_proj'] 不在 --lora-target-modules 里", out)
        self.assertTrue(run(d, targets=['all-linear'])[0])

    def test_declared_but_missing_module_warned(self):
        d = fake_adapter(self.root, 'decl', {'target_modules': ['q_proj', 'v_proj', 'c_attn']})
        _, out = run(d)
        self.assertIn("声明了 ['c_attn'] 但 safetensors 里没有", out)

    def test_main_exit_code(self):
        good = fake_adapter(self.root, 'g', {})
        bad = fake_adapter(self.root, 'b', {'use_dora': True})
        sys.stdout, saved = io.StringIO(), sys.stdout
        try:
            self.assertEqual(ca.main([good]), 0)
            self.assertEqual(ca.main([good, bad]), 1)
        finally:
            sys.stdout = saved


@unittest.skipUnless(HAS_TRAIN_DEPS, '没装 torch / transformers / peft')
class TrainTiny(unittest.TestCase):
    def test_train_and_check(self):
        import train_lora as tl
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'city-r4')
            try:
                res = tl.train(out=out, rank=4, alpha=8, steps=20, log=lambda *a: None)
            except OSError as e:  # 离线、拉不到 tiny 模型
                self.skipTest(f'拿不到 {tl.TINY}：{e}')
            self.assertLess(sum(res['losses'][-5:]) / 5, res['losses'][0])
            self.assertLess(res['after']['loss'], res['before']['loss'])
            self.assertTrue(os.path.exists(os.path.join(out, 'adapter_model.safetensors')))
            ok, text = run(out)
            self.assertTrue(ok, text)
            cfg = json.load(open(os.path.join(out, 'adapter_config.json')))
            self.assertEqual(cfg['r'], 4)
            self.assertEqual(cfg['bias'], 'none')


if __name__ == '__main__':
    unittest.main()
