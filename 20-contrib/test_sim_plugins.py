"""sim_plugins.py 的回归测试：python3 -m unittest test_sim_plugins -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_plugins as sp  # noqa: E402


class VllmAllowList(unittest.TestCase):
    """envs.py:1175-1179 与 plugins/__init__.py:36-74。"""

    def test_unset_loads_all(self):
        gen, ep, log = sp.v_env_effect(None)
        self.assertEqual(gen, list(sp.V_GENERAL))
        self.assertEqual(ep, [])                     # endpoint 组必须显式放行
        self.assertIn('VLLM_PLUGINS is not set', log[-1][1])

    def test_empty_string_loads_nothing(self):
        self.assertEqual(sp.v_allowed({'VLLM_PLUGINS': ''}), [''])
        gen, ep, _ = sp.v_env_effect('')
        self.assertEqual((gen, ep), ([], []))

    def test_space_is_part_of_name(self):
        gen, ep, _ = sp.v_env_effect('zv_demo, zv_demo_api')
        self.assertEqual((gen, ep), (['zv_demo'], []))

    def test_setting_list_drops_builtin_resolver(self):
        gen, _, _ = sp.v_env_effect('zv_demo')
        self.assertNotIn('lora_filesystem_resolver', gen)


class SglangAllowList(unittest.TestCase):
    """srt/plugins/__init__.py:52-55。"""

    def test_empty_equals_unset(self):
        self.assertIsNone(sp.s_allowed({'SGLANG_PLUGINS': ''}))
        self.assertEqual(sp.s_env_effect(''), sp.s_env_effect(None))

    def test_strip(self):
        self.assertEqual(sp.s_env_effect('zv_demo, other_hooks'), ['zv_demo', 'other_hooks'])


class GeneralLoad(unittest.TestCase):
    """plugins/__init__.py:77-90：每进程一次，func() 不包 try。"""

    def test_once_per_process(self):
        calls = []
        p = sp.VProc('api')
        eps = [sp.EP('zv_demo', lambda: calls.append(1))]
        sp.v_load_general(p, eps, {}, sp.Log())
        sp.v_load_general(p, eps, {}, sp.Log())
        self.assertEqual(len(calls), 1)

    def test_vllm_raise_propagates(self):
        def boom():
            raise ValueError('x')
        with self.assertRaises(ValueError):
            sp.v_load_general(sp.VProc('api'), [sp.EP('a', boom)], {}, sp.Log())

    def test_sglang_raise_logged(self):
        def boom():
            raise ValueError('x')
        log = sp.Log()
        ran = sp.s_load_plugins(sp.VProc('main'), [sp.EP('a', boom), sp.EP('b', lambda: None)], {}, log)
        self.assertEqual(ran, ['b'])
        self.assertIn('Failed to execute general plugin', log[0][1])

    def test_load_error_is_soft_on_both(self):
        ep = sp.EP('a', load_error=ImportError('no'))
        self.assertEqual(sp.v_load_general(sp.VProc('x'), [ep], {}, sp.Log()), [])
        self.assertEqual(sp.s_load_plugins(sp.VProc('y'), [ep], {}, sp.Log()), [])

    def test_failure_matrix_shape(self):
        rows = sp.failure_matrix()
        res = {(w, who): r for w, who, r in rows}
        self.assertTrue(res[('general 插件 func() 抛错', 'vLLM')].startswith('启动失败'))
        self.assertTrue(res[('general 插件 func() 抛错', 'SGLang')].startswith('继续启动'))
        self.assertIn('Duplicate op name', res[('两个插件 register_oot 同名', 'vLLM')])


class Processes(unittest.TestCase):
    """哪些进程会再跑一次 register()。"""

    def test_spawn_tp4(self):
        rows = sp.v_launch(tp=4, method='spawn')
        self.assertEqual(sum(r for _, r, _ in rows), 7)   # API + modelinfo + core + 4 worker

    def test_fork_inherits(self):
        rows = sp.v_launch(tp=4, method='fork', cache_hit=True)
        self.assertEqual(sum(r for _, r, _ in rows), 1)

    def test_uni_executor(self):
        rows = sp.v_launch(tp=1, method='spawn', cache_hit=True)
        self.assertEqual(sum(r for _, r, _ in rows), 2)   # API + core，worker 同进程

    def test_sglang_tp_pp(self):
        rows = sp.s_launch(tp=4, pp=2)
        self.assertEqual(len(rows), 1 + 8 + 1)


class Endpoint(unittest.TestCase):
    def test_required_tasks(self):
        class Gen:
            required_tasks = ('generate',)
        env = {'VLLM_PLUGINS': 'p'}
        self.assertEqual(sp.v_load_endpoint([sp.EP('p', Gen)], env, ('embed',), sp.Log()), [])
        self.assertEqual(sp.v_load_endpoint([sp.EP('p', Gen)], env, None, sp.Log()), [])
        self.assertEqual(len(sp.v_load_endpoint([sp.EP('p', Gen)], env, ('generate',), sp.Log())), 1)

    def test_rpc_codes(self):
        self.assertEqual(sp.endpoint_call(None)[0], 503)
        self.assertEqual(sp.endpoint_call(sp.FakeEngine({}))[0], 500)
        ok = sp.FakeEngine({'zv_get_scheduler_config': lambda r: r}, n_workers=3)
        self.assertEqual(sp.endpoint_call(ok), (200, [0, 1, 2]))


class Platform(unittest.TestCase):
    def test_vllm(self):
        self.assertEqual(sp.v_platform([], []), 'UnspecifiedPlatform')
        self.assertTrue(sp.v_platform(['cuda'], ['a']).startswith('a'))
        self.assertTrue(sp.v_platform(['cuda'], ['a', 'b']).startswith('RuntimeError'))
        self.assertTrue(sp.v_platform(['cuda', 'cpu'], []).startswith('RuntimeError'))

    def test_sglang(self):
        self.assertTrue(sp.s_platform(None, ['a', 'b'], ['a', 'b']).startswith('RuntimeError'))
        self.assertEqual(sp.s_platform('a', ['a', 'b'], ['a', 'b']), 'a（指定）')
        self.assertIn('not found', sp.s_platform('c', ['a'], ['a']))
        self.assertIn('returned None', sp.s_platform('b', ['a', 'b'], ['a']))


class HookOrder(unittest.TestCase):
    """hook_registry.py：REPLACE 排最里层，后注册的在外层。"""

    def test_last_registered_outermost(self):
        t = sp.hook_trace([('AFTER', 'h1'), ('AROUND', 'h2'), ('BEFORE', 'h3')])
        self.assertEqual(t, ['h3(BEFORE)', 'h2(AROUND 进)', '原函数', 'h1(AFTER)', 'h2(AROUND 出)'])

    def test_replace_skips_original(self):
        t = sp.hook_trace([('AROUND', 'timer'), ('REPLACE', 'mine')])
        self.assertNotIn('原函数', t)
        self.assertEqual(t[1], 'mine(REPLACE)')

    def test_after_can_override_result(self):
        h = sp.Hooks()
        h.targets['f'] = lambda x: x + 1
        h.register('f', lambda r, x: r * 10, 'AFTER')
        h.apply()
        h.apply()                                   # 幂等
        self.assertEqual(h.targets['f'](1), 20)


class Bench(unittest.TestCase):
    """random 数据集复跑：命中取决于容量。"""

    def test_same_seed_same_prompts(self):
        self.assertEqual(sp.random_prompts(3, 32, 0), sp.random_prompts(3, 32, 0))
        self.assertNotEqual(sp.random_prompts(3, 32, 0), sp.random_prompts(3, 32, 1))

    def test_fits_hits(self):
        r = sp.bench_rerun(20, 64, 16, 0, False, concurrency=8, cap=1000)
        self.assertAlmostEqual(r, 48 / 64)           # 命中上限 (64-1)//16*16 = 48

    def test_thrash_misses(self):
        # 一轮要 20×5=100 块，容量 60：顺序扫描下 LRU 全部错过
        self.assertEqual(sp.bench_rerun(20, 64, 16, 0, False, concurrency=4, cap=60), 0.0)

    def test_reset_and_new_seed(self):
        self.assertEqual(sp.bench_rerun(20, 64, 16, 0, True, concurrency=8, cap=1000), 0.0)
        self.assertEqual(sp.bench_rerun(20, 64, 16, 1, False, concurrency=8, cap=1000), 0.0)


if __name__ == '__main__':
    unittest.main()
