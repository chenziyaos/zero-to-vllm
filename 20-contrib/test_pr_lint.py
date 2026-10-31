"""pr_lint.py 的回归测试：python3 -m unittest test_pr_lint -v

SGLang 的合法 / 非法标题原样搬自 sglang 8fabf5e .github/workflows/pr-title.test.cjs:60-132。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pr_lint as pl  # noqa: E402

TYPES = ['Feat', 'Fix', 'Refactor', 'Perf', 'Doc', 'Test', 'CI', 'Chore']

SG_VALID = [
    '[Fix] Attention: Fix output',
    '[Feat] Qwen3.5: add model support',
    '[Fix] Flash-Attn: FIX OUTPUT',
    '[Fix] [Apple] MLX: fix output',
    '[Fix] [HYGON] cpu: Fix output',
    '[Fix] [QUALCOMM] cpu: Fix output',
    '[Fix] [Huawei] model/Qwen3: Fix output',
    '[Fix] [Ascend] attention: Fix output',
    '[Perf] [TPU] Attention: optimize decode',
    'Revert "[Fix] attention: Fix output"',
    'Revert "A legacy title without tags or scope"',
    'Revert "Revert "[Fix] attention: Fix output""',
    'Revert "x"',
    *[f'[{t}] attention: Update implementation' for t in TYPES],
    '[Feat] qwen3: Add model support',
    '[Perf] [AMD] flash-attn: Optimize decode performance',
    '[Fix] [Intel] sgl-kernel: Fix output',
    '[Fix] [NVIDIA] model/qwen3: Fix output',
    '[Test] qwen3.5: Add coverage',
    '[CI] pr_title: Check format',
    '[Perf] [MI300X] moe: Optimize dispatch',
    '[Doc] contribution: 文档更新',
    '[Chore] ci: x',
    '[Fix] ci: Handle `quotes`, "text", and $(commands) as data',
]

SG_INVALID = [
    '[Fix] [Hygon] cpu: Fix output',
    '[Fix] [Qualcomm] cpu: Fix output',
    *[f'[Fix] [{h}] attention: Fix output' for h in (
        'amd', 'Amd', 'nvidia', 'Nvidia', 'intel', 'INTEL', 'apple', 'APPLE', 'huawei', 'HUAWEI',
        'ascend', 'ASCEND', 'tpu', 'Tpu', 'mi300x', 'Mi300X')],
    '[Revert] attention: Revert output fix',
    'Revert ""',
    'Revert "   "',
    'Revert " title"',
    'Revert "title "',
    'Revert "title',
    'Revert title',
    'revert "title"',
    'Revert  "title"',
    'Revert "title" extra text',
    'Revert "title"\n',
    'Revert "multi\nline"',
    'Fix KV cache indexing',
    '[Fix] Fix KV cache indexing',
    '[Feature] model: Add support',
    '[fix] attention: Fix output',
    '[Fix][AMD] attention: Fix output',
    '[Fix] [AMD][NVIDIA] attention: Fix output',
    '[Fix] [AMD] [NVIDIA] attention: Fix output',
    '[Fix] [] attention: Fix output',
    '[Fix] [ AMD ] attention: Fix output',
    '[Fix] [AMD GPU] attention: Fix output',
    '[Fix] : Fix output',
    '[Fix] flash attn: Fix output',
    '[Fix] attention : Fix output',
    '[Fix] attention:Fix output',
    '[Fix] attention:  Fix output',
    '[Fix] attention: ',
    '[Fix] attention:    ',
    '[Fix]  attention: Fix output',
    '[Fix]\tattention: Fix output',
    '[Fix] attention:\tFix output',
    ' [Fix] attention: Fix output',
    '[Fix] attention: Fix output ',
    '[Fix] attention: Fix output\n',
    '[Fix] attention: Fix\noutput',
]


class SglangTitle(unittest.TestCase):
    """pr-title.yml:20-60 + pr-title.test.cjs:60-132。"""

    def test_valid(self):
        for t in SG_VALID:
            with self.subTest(t=t):
                self.assertTrue(pl.sglang_title(t)[0])

    def test_invalid(self):
        for t in SG_INVALID:
            with self.subTest(t=t):
                self.assertFalse(pl.sglang_title(t)[0])

    def test_reasons(self):
        self.assertIn('[AMD]', pl.sglang_title('[Fix] [Amd] attention: Fix output')[1])
        self.assertIn('[Intel]', pl.sglang_title('[Fix] [INTEL] attention: Fix output')[1])
        self.assertIn('Type', pl.sglang_title('[Bugfix] scheduler: Fix priority handling')[1])
        self.assertIn('首尾', pl.sglang_title('[Fix] attention: Fix output ')[1])


class VllmTitle(unittest.TestCase):
    """vllm pr-title.yml:34：^(\\[[^][]+\\])+ +[^[:space:]].*$（bash ERE，. 也匹配换行）。"""

    def test_accept(self):
        for t in ('[Bugfix][Scheduler] Fix priority handling', '[Perf][ROCm][Kernel] Optimize fused MoE',
                  '[Misc] x', '[Fix] attention: Fix output', '[Doc]   多个空格也行',
                  '[Bugfix] trailing space ', '[Bugfix] multi\nline'):
            with self.subTest(t=t):
                self.assertTrue(pl.vllm_title(t)[0])

    def test_reject(self):
        for t in ('Fix priority handling', '[Bugfix]Fix priority handling',
                  '[] x', '[a[b]] x', ' [Bugfix] x', '[Bugfix] ', '[Bugfix]\tx',
                  'Revert "[Bugfix] x"'):
            with self.subTest(t=t):
                self.assertFalse(pl.vllm_title(t)[0])

    def test_space_between_tags(self):
        # "[Bugfix] [Scheduler] Fix" 里第二个 [ 前有空格，按规则 [Scheduler] 是描述的一部分，所以其实合法
        self.assertTrue(pl.vllm_title('[Bugfix] [Scheduler] Fix priority handling')[0])

    def test_same_title_differs(self):
        t = '[Bugfix][Scheduler] Fix priority handling'
        self.assertTrue(pl.vllm_title(t)[0])
        self.assertFalse(pl.sglang_title(t)[0])
        r = 'Revert "[Fix] scheduler: Fix priority handling"'
        self.assertFalse(pl.vllm_title(r)[0])
        self.assertTrue(pl.sglang_title(r)[0])


class Commit(unittest.TestCase):
    """docs/contributing/README.md:190-196 DCO，206-228 AI 署名。"""

    def test_signoff(self):
        self.assertEqual(len(pl.check_commit('[Bugfix] x\n')), 1)
        self.assertEqual(pl.check_commit('[Bugfix] x\n\nSigned-off-by: A Dev <a@example.com>\n'), [])

    def test_signoff_needs_email(self):
        self.assertEqual(len(pl.check_commit('x\n\nSigned-off-by: A Dev\n')), 1)

    def test_ai_trailer(self):
        msg = 'x\n\nSigned-off-by: A Dev <a@example.com>\n'
        self.assertEqual(len(pl.check_commit(msg, ai_assisted=True)), 1)
        self.assertEqual(pl.check_commit('x\n\nCo-authored-by: Claude\n' + msg[3:], ai_assisted=True), [])


class Diff(unittest.TestCase):
    """README.md:271-287 文档同 PR，309-312 >500 行先 RFC。"""

    def test_small_with_tests(self):
        self.assertEqual(pl.check_diff([(12, 3, 'vllm/plugins/__init__.py'),
                                        (40, 0, 'tests/plugins_tests/test_x.py')]), [])

    def test_kernel_not_counted(self):
        tips = pl.check_diff([(820, 0, 'csrc/attention/new_kernel.cu'), (60, 4, 'vllm/_custom_ops.py'),
                              (180, 0, 'tests/kernels/test_new.py')])
        self.assertEqual(tips, [])

    def test_rfc(self):
        tips = pl.check_diff([(430, 260, 'vllm/v1/core/sched/scheduler.py'), (120, 0, 'tests/v1/core/t.py')])
        self.assertTrue(any('690' in t and 'RFC' in t for t in tips))

    def test_user_visible_without_docs(self):
        tips = pl.check_diff([(8, 0, 'vllm/envs.py'), (15, 2, 'vllm/v1/worker/gpu_model_runner.py')])
        self.assertTrue(any('envs.py' in t for t in tips))
        self.assertTrue(any('tests/' in t for t in tips))
        tips = pl.check_diff([(8, 0, 'vllm/envs.py'), (5, 0, 'docs/configuration/env_vars.md'),
                              (9, 0, 'tests/test_envs.py')])
        self.assertEqual(tips, [])


if __name__ == '__main__':
    unittest.main()
