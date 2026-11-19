"""ci_select.py 的回归测试：python3 -m unittest test_ci_select -v"""
import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_select as cs  # noqa: E402

STEPS = cs.load_steps()


def call(src):
    return ast.parse(src).body[0].value


class Matching(unittest.TestCase):
    """buildkite_step.py:1001-1005 与 1544-1561。"""

    def test_prefix_needs_slash(self):
        self.assertTrue(cs.matches('vllm/plugins', 'vllm/plugins/__init__.py'))
        self.assertTrue(cs.matches('vllm/plugins/', 'vllm/plugins/__init__.py'))
        self.assertFalse(cs.matches('vllm/plug', 'vllm/plugins/__init__.py'))
        self.assertFalse(cs.matches('tests/test_umbp_', 'tests/test_umbp_async_lookup.py'))
        self.assertFalse(cs.matches('/', 'anything'))

    def test_exclude(self):
        deps = ['vllm/', '!vllm/distributed/kv_transfer/']
        self.assertTrue(cs.deps_match(deps, ['vllm/envs.py']))
        self.assertFalse(cs.deps_match(deps, ['vllm/distributed/kv_transfer/kv_connector/v1/base.py']))
        self.assertTrue(cs.deps_match(deps, ['vllm/distributed/kv_transfer/x.py', 'vllm/envs.py']))
        self.assertFalse(cs.deps_match([], ['vllm/envs.py']))

    def test_yaml_appended(self):
        s = next(s for s in STEPS if s['file'].endswith('plugins.yaml'))
        self.assertIn(s['file'], s['deps'])
        # rust_frontend_cargo.yaml 顶层没有 depends_on，不追加；它在 step 自己的依赖里显式写了一次
        r = next(s for s in STEPS if s['file'] in cs.NO_GROUP_DEPENDS)
        self.assertEqual(r['deps'].count(r['file']), 1)


class DocsOnly(unittest.TestCase):
    """pipeline_generator.py:486-501。"""

    def test_cases(self):
        self.assertFalse(cs.is_docs_only([]))
        self.assertTrue(cs.is_docs_only(['README.md']))
        self.assertTrue(cs.is_docs_only(['docs/design/plugin_system.md', 'mkdocs.yaml']))
        self.assertTrue(cs.is_docs_only(['docs/mkdocs/hooks/url_schemes.py']))
        self.assertTrue(cs.is_docs_only(['recipes/Qwen3.md']))
        self.assertFalse(cs.is_docs_only(['docs/design/plugin_system.md', 'vllm/envs.py']))
        self.assertFalse(cs.is_docs_only(['examples/offline_inference/basic.py']))

    def test_disable(self):
        state, sel, _, _ = cs.select(STEPS, ['README.md'], env={'DOCS_ONLY_DISABLE': '1'})
        self.assertNotIn('docs-only', state)


class RunAll(unittest.TestCase):
    """global_config.py:143-170 + vllm .buildkite/ci_config.yaml。"""

    def test_patterns(self):
        for f in ('setup.py', 'setup.pyc', 'CMakeLists.txt', 'csrc/attention/paged_attention_v1.cu',
                  'cmake/utils.cmake', 'docker/Dockerfile', 'requirements/cuda.txt'):
            with self.subTest(f=f):
                self.assertTrue(cs.should_run_all([f]))
        for f in ('csrc/cpu/attention.cpp', 'csrc/rocm/attention.cu', 'cmake/hipify.py',
                  'docker/Dockerfile.cpu', 'requirements/rocm.txt', 'vllm/envs.py'):
            with self.subTest(f=f):
                self.assertFalse(cs.should_run_all([f]))

    def test_label_and_env(self):
        self.assertTrue(cs.should_run_all(['vllm/envs.py'], labels=['ready-run-all-tests']))
        self.assertTrue(cs.should_run_all(['vllm/envs.py'], env={'RUN_ALL': '1'}))

    def test_label_adds_optional(self):
        _, a, _, _ = cs.select(STEPS, ['vllm/envs.py'], labels=['ready-run-all-tests'])
        _, b, _, _ = cs.select(STEPS, ['csrc/attention/paged_attention_v1.cu'])
        self.assertGreater(len(a), len(b))
        self.assertTrue(any(s['optional'] for s in a))
        self.assertFalse(any(s['optional'] for s in b))


class Select(unittest.TestCase):
    """快照上的几组典型 diff。"""

    def n(self, diff):
        return len(cs.select(STEPS, diff)[1])

    def test_snapshot_shape(self):
        self.assertEqual(len(STEPS), 254)
        self.assertEqual(sum(s['optional'] for s in STEPS), 82)
        self.assertEqual(sum(cs.is_amd(s) for s in STEPS), 21)

    def test_counts(self):
        self.assertEqual(self.n(['vllm/plugins/__init__.py']), 39)
        self.assertEqual(self.n(['tests/plugins_tests/test_platform_plugins.py']), 1)
        self.assertEqual(self.n(['.buildkite/test_areas/plugins.yaml']), 1)
        self.assertEqual(self.n(['vllm/distributed/kv_transfer/kv_connector/v1/base.py']), 0)

    def test_kernel_run_all(self):
        state, sel, _, _ = cs.select(STEPS, ['csrc/attention/paged_attention_v1.cu'])
        self.assertEqual(state, 'run_all')
        self.assertEqual(len(sel), sum(1 for s in STEPS if not s['optional'] and not cs.is_amd(s)))

    def test_noauto(self):
        self.assertEqual(len(cs.select(STEPS, ['vllm/envs.py'], env={'NOAUTO': '1'})[1]), 0)

    def test_dead(self):
        dead = cs.dead_deps(STEPS, cs.load_paths())
        self.assertEqual(len(dead), 10)


class SglangRegister(unittest.TestCase):
    """sglang python/sglang/test/ci/ci_register.py（8fabf5e）。"""

    def test_new_style(self):
        r = cs.parse_call(call('register_cuda_ci(est_time=300, stage="stage-b", runner_config="1-gpu-large")'))
        self.assertEqual(r['effective_suite'], 'stage-b-test-1-gpu-large')
        self.assertEqual(r['backend'], 'cuda')

    def test_positional(self):
        r = cs.parse_call(call('register_amd_ci(120, "stage-a-test-1-gpu-amd", True)'))
        self.assertEqual((r['suite'], r['nightly']), ('stage-a-test-1-gpu-amd', True))

    def test_errors(self):
        for src, msg in (
                ('register_cuda_ci(est_time=T, suite="x")', 'est_time is a required constant'),
                ('register_cuda_ci(est_time=1, stage="a", runner_config="b", suite="x")', 'exactly one'),
                ('register_cuda_ci(est_time=1, stage="a")', 'exactly one'),
                ('register_cuda_ci(300, "x", "true")', 'nightly must be a boolean'),
                ('register_cuda_ci(**CFG)', '**kwargs'),
                ('register_cuda_ci(*A)', 'starred'),
                ('register_cuda_ci(1, "x", False, None, 5)', 'too many positional'),
                ('register_cuda_ci(1, suite="x")', None),
                ('register_cuda_ci(1, "x", est_time=2)', 'duplicated'),
                ('register_cuda_ci(est_time=1, suite="x", foo=1)', 'unknown argument')):
            with self.subTest(src=src):
                if msg is None:
                    cs.parse_call(call(src))
                else:
                    with self.assertRaisesRegex(ValueError, msg.replace('*', r'\*')):
                        cs.parse_call(call(src))

    def test_module_level_only(self):
        src = 'if True:\n    register_cuda_ci(est_time=60, suite="x")\n'
        with self.assertRaisesRegex(ValueError, 'No CI registry'):
            cs.collect(src)

    def test_main_required_unless_disabled(self):
        reg = 'register_cuda_ci(est_time=60, suite="x"{})\n'
        with self.assertRaisesRegex(ValueError, 'missing'):
            cs.collect(reg.format(''))
        with self.assertRaisesRegex(ValueError, 'missing'):
            cs.collect(reg.format('') + 'if __name__ == "__main__":\n    pass\n')
        self.assertEqual(len(cs.collect(reg.format('') + 'if __name__ == "__main__":\n    main()\n')), 1)
        self.assertEqual(len(cs.collect(reg.format(', disabled="flaky #1, until 2026-12-01"'))), 1)

    def test_weighted_cost(self):
        self.assertEqual(cs.weighted_cost(dict(est_time=150, runner_config='8-gpu-h200')), 1200)
        self.assertEqual(cs.weighted_cost(dict(est_time=900, runner_config=None)), 900)

    def test_lpt(self):
        files = [('a', 600), ('b', 500), ('c', 400), ('d', 300), ('e', 300), ('f', 200), ('g', 100)]
        parts, sums = cs.auto_partition(files, 3)
        self.assertEqual(sums, [800.0, 800.0, 800.0])
        self.assertEqual(sorted(sum(parts, [])), sorted(f for f, _ in files))
        parts, sums = cs.auto_partition([('x', 10), ('y', 10)], 2)
        self.assertEqual(parts, [['x'], ['y']])


if __name__ == '__main__':
    unittest.main()
