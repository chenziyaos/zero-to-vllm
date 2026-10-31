"""插件加载、平台选择、hook 链和 benchmark 复跑的离线推演（只依赖标准库，不需要 GPU）。

对照的源码与文档（vllm 主干 0d6f92a，sglang 主干 8fabf5e）：
  vllm   plugins/__init__.py:18-30            五个 entry point 组：general / io_processor / platform / stat_logger / endpoint
         plugins/__init__.py:36-74            load_plugins_by_group：VLLM_PLUGINS 为 None 全加载，否则 name in allowed；load() 抛错只记日志
         plugins/__init__.py:77-90            load_general_plugins：每进程一次（plugins_loaded），func() 没有 try，抛错直接打断启动
         plugins/__init__.py:93-159           load_endpoint_plugins：VLLM_PLUGINS 未设置 → 警告后返回 []；工厂抛错跳过；required_tasks 过滤
         envs.py:1175-1179                    VLLM_PLUGINS = os.environ[...].split(",")，不 strip；"" → [""]
         pyproject.toml:45-47                 自带两个 general 插件：lora_filesystem_resolver / lora_hf_hub_resolver
         engine/arg_utils.py:930-932          API server 进程建 config 时加载
         v1/engine/core.py:139-142            engine core 进程加载
         v1/worker/worker_base.py:287-289     每个 worker 进程加载
         model_executor/models/registry.py:1135, 1636-1640  modelinfo 缓存未命中时起子进程检查模型类，子进程里也加载
         entrypoints/common/startup.py:147-149  CLI 没设 VLLM_WORKER_MULTIPROC_METHOD 时改成 spawn；envs 默认 fork（envs.py:944-945）
         config/parallel.py:1074-1120         world_size_across_dp > 1 用 mp executor，world_size == 1 用 uni（worker 与 engine core 同进程）
         model_executor/custom_op.py:86, 338  register_oot 重名直接 assert
         platforms/__init__.py:240-292        平台：OOT ≥2 报错 / 1 个胜出 / 内置 ≥2 报错 / 1 个 / 0 个 → UnspecifiedPlatform
         utils/serial_utils.py:485-508        collective_rpc 的方法名在 worker 上找不到 → NotImplementedError
         benchmarks/datasets/datasets.py:748-750, 1585  random 数据集 token = (offset + index + arange) % len；--seed 默认 0
         v1/request.py:314-318                前缀命中最多 num_tokens - 1
         v1/core/single_type_kv_cache_manager.py:599  请求结束时块倒序放回空闲队列
  sglang srt/plugins/__init__.py:35-86         SGLANG_PLUGINS：strip、丢空串；"" 等于没设
         srt/plugins/__init__.py:103-141      load_plugins：func() 包在 try 里只记日志，最后 HookRegistry.apply_hooks()
         srt/entrypoints/engine.py:286, 1995  主进程加载；强制 spawn
         srt/managers/scheduler.py:6408       每个 scheduler 进程加载；detokenizer_manager.py:552 也加载
         srt/platforms/__init__.py:108-198    SGLANG_PLATFORM 指定 / 自动发现：0 个按硬件探测回退，1 个用它，多个报错
         srt/plugins/hook_registry.py:84-137, 146-300, 350-400  hook 注册、REPLACE 优先的稳定排序、后注册的包在外层
         benchmark/bench_serving.py:1014-1023, 1520-1523, 2554, 2697  flush 仅 CI 或 --flush-cache；--seed 默认 42

场景：
  procs     一次启动里 general 插件的 register() 被执行几次、在哪些进程
  env       VLLM_PLUGINS / SGLANG_PLUGINS 的取值各加载了什么，自带 LoRA resolver 什么时候被顺手关掉
  endpoint  endpoint 插件的放行、required_tasks、render server 的 None、缺 worker 方法时的报错
  platform  平台插件 0 / 1 / 2 个被激活时两边的选择
  failure   插件抛异常：哪些打断启动，哪些只留一行日志
  hooks     SGLang hook 链的执行顺序
  bench     同一个 seed 复跑：前缀缓存命中多少，取决于 KV 容量能不能装下整轮 prompt

用法：
  python3 sim_plugins.py --scenario procs --tp 4
  python3 sim_plugins.py --scenario procs --tp 4 --method fork
  python3 sim_plugins.py --scenario env
  python3 sim_plugins.py --scenario bench --num-prompts 200
  python3 sim_plugins.py --scenario all

加载器照抄两边源码的分支，entry point 用字典伪造；bench 的 KV 容量沿用 06 章 Qwen3-8B on H100 的 24957 块。
"""
import argparse
import os
import random
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '06-chunked-prefill'))
import sim_chunked as sc  # noqa: E402

BLOCK = 16


# --------------------------------------------------------------------------- 伪造的 entry point
class EP:
    """importlib.metadata.EntryPoint 的替身：load() 返回 func，或抛出 load_error。"""

    def __init__(self, name, func=None, dist='pkg', load_error=None):
        self.name, self.func, self.dist, self.load_error = name, func, dist, load_error

    def load(self):
        if self.load_error:
            raise self.load_error
        return self.func


class Log(list):
    def add(self, level, msg):
        self.append((level, msg))


# --------------------------------------------------------------------------- vLLM 加载器
def v_allowed(env):
    """envs.py:1175-1179。"""
    return None if 'VLLM_PLUGINS' not in env else env['VLLM_PLUGINS'].split(',')


def v_load_by_group(eps, env, log):
    """plugins/__init__.py:36-74。"""
    allowed = v_allowed(env)
    out = OrderedDict()
    for ep in eps:
        if allowed is None or ep.name in allowed:
            try:
                out[ep.name] = ep.load()
            except Exception as e:  # noqa: BLE001
                log.add('ERROR', f'Failed to load plugin {ep.name}: {e!r}')
    return out


class VProc:
    """一个进程的 plugins_loaded 状态。fork 出来的子进程复制父进程的这个全局变量。"""

    def __init__(self, name, loaded=False):
        self.name, self.loaded = name, loaded


def v_load_general(proc, eps, env, log):
    """plugins/__init__.py:77-90：没有 try，func() 抛什么启动就死在哪。"""
    if proc.loaded:
        return []
    proc.loaded = True
    plugins = v_load_by_group(eps, env, log)
    for f in plugins.values():
        f()
    return list(plugins)


def v_load_endpoint(eps, env, supported_tasks, log):
    """plugins/__init__.py:93-159。返回 [(entry point 名, 插件对象)]。"""
    if v_allowed(env) is None:
        if eps:
            log.add('WARNING', f'Found endpoint plugin(s) {[e.name for e in eps]} but VLLM_PLUGINS is not set.')
        return []
    out = []
    for name, factory in v_load_by_group(eps, env, log).items():
        try:
            plugin = factory()
        except Exception as e:  # noqa: BLE001
            log.add('ERROR', f'Failed to instantiate endpoint plugin {name}: {e!r}')
            continue
        req = plugin.required_tasks
        if req is not None and (supported_tasks is None or not set(req) & set(supported_tasks)):
            log.add('INFO', f'Skipping endpoint plugin {name}: requires one of tasks {req}')
            continue
        out.append((name, plugin))
    return out


V_BUILTIN_PLATFORMS = ('tpu', 'cuda', 'rocm', 'xpu', 'cpu')


def v_platform(builtin_active, oot_active, target_device=None):
    """platforms/__init__.py:240-292。参数是“探测函数返回非 None”的名字列表。"""
    if target_device == 'cpu':
        return 'cpu（VLLM_TARGET_DEVICE=cpu 直接短路）'
    if len(oot_active) >= 2:
        return f'RuntimeError: Only one platform plugin can be activated, but got: {sorted(oot_active)}'
    if len(oot_active) == 1:
        return f'{oot_active[0]}（OOT 胜出）'
    if len(builtin_active) >= 2:
        return f'RuntimeError: Only one platform plugin can be activated, but got: {sorted(builtin_active)}'
    if len(builtin_active) == 1:
        return builtin_active[0]
    return 'UnspecifiedPlatform'


# --------------------------------------------------------------------------- SGLang 加载器
def s_allowed(env):
    """srt/plugins/__init__.py:52-55：空串被 if 挡掉，等于没设。"""
    s = env.get('SGLANG_PLUGINS', '')
    if not s:
        return None
    return {x.strip() for x in s.split(',') if x.strip()}


def s_load_by_group(eps, env, log, excluded=()):
    allowed = s_allowed(env)
    out = OrderedDict()
    for ep in eps:
        if allowed is not None and ep.name not in allowed:
            continue
        if ep.dist in excluded:
            continue
        try:
            out[ep.name] = ep.load()
        except Exception as e:  # noqa: BLE001
            log.add('ERROR', f'Failed to load plugin {ep.name}: {e!r}')
    return out


def s_load_plugins(proc, eps, env, log, hooks=None):
    """srt/plugins/__init__.py:103-141：func() 有 try，失败只记日志；之后统一 apply_hooks。"""
    if proc.loaded:
        return []
    proc.loaded = True
    ran = []
    for name, f in s_load_by_group(eps, env, log).items():
        try:
            f()
            ran.append(name)
        except Exception as e:  # noqa: BLE001
            log.add('ERROR', f'Failed to execute general plugin: {name}: {e!r}')
    if hooks is not None:
        hooks.apply()
    return ran


def s_platform(selected, installed, active, hw_fallback='CUDA'):
    """srt/platforms/__init__.py:108-198。installed：装了哪些平台插件；active：其中探测成功的。"""
    if selected:
        if selected not in installed:
            return f'RuntimeError: SGLANG_PLATFORM={selected!r} not found'
        if selected not in active:
            return f'RuntimeError: Platform plugin {selected!r} is installed but activate() returned None'
        return f'{selected}（指定）'
    if not active:
        return f'{hw_fallback}（无插件，按 CPU→SAIL→CUDA→NPU→ROCm→XPU→MUSA→MPS 探测）'
    if len(active) == 1:
        return active[0]
    return f'RuntimeError: Multiple platform plugins activated: {", ".join(active)}. Set SGLANG_PLATFORM to select one.'


# --------------------------------------------------------------------------- SGLang hook 链（hook_registry.py）
class Hooks:
    def __init__(self):
        self.hooks = OrderedDict()   # target → [(type, fn)]
        self.targets = {}
        self.patched = set()

    def register(self, target, fn, typ='AFTER'):
        self.hooks.setdefault(target, []).append((typ, fn))

    def apply(self):
        for target, hooks in self.hooks.items():
            if target in self.patched:
                continue
            wrapped = self.targets[target]
            for typ, fn in sorted(hooks, key=lambda h: 0 if h[0] == 'REPLACE' else 1):
                wrapped = wrap(wrapped, fn, typ)
            self.targets[target] = wrapped
            self.patched.add(target)


def wrap(orig, hook, typ):
    """_wrap_fn（hook_registry.py:350-400）。"""
    if typ == 'REPLACE':
        return lambda *a, **k: hook(*a, **k)
    if typ == 'BEFORE':
        def w(*a, **k):
            r = hook(*a, **k)
            if r is not None:
                a, k = r
            return orig(*a, **k)
        return w
    if typ == 'AFTER':
        def w(*a, **k):
            r = orig(*a, **k)
            m = hook(r, *a, **k)
            return m if m is not None else r
        return w
    if typ == 'AROUND':
        return lambda *a, **k: hook(orig, *a, **k)
    raise ValueError(typ)


# --------------------------------------------------------------------------- 场景：procs
def v_launch(tp=1, pp=1, dp=1, method='spawn', cache_hit=False):
    """返回 [(进程, 是否执行 register, 出处)]。"""
    rows = [('API server / LLM() 主进程', True, 'arg_utils.py:930')]
    if not cache_hit:
        rows.append(('modelinfo 检查子进程', True, 'registry.py:1636（subprocess，总是全新解释器）'))
    fresh = method == 'spawn'
    world = tp * pp
    for d in range(dp):
        tag = f'[dp{d}]' if dp > 1 else ''
        rows.append((f'engine core{tag}', fresh, 'core.py:139' if fresh else 'fork 继承 plugins_loaded=True'))
        if world * dp > 1:
            for r in range(world):
                rows.append((f'  worker{tag} rank {r}', fresh,
                             'worker_base.py:289' if fresh else 'fork 继承 plugins_loaded=True'))
        else:
            rows.append(('  worker（uni executor，与 engine core 同进程）', False, 'plugins_loaded 已是 True'))
    return rows


def s_launch(tp=1, pp=1):
    rows = [('Engine 主进程（tokenizer manager）', True, 'engine.py:286')]
    for p in range(pp):
        for t in range(tp):
            rows.append((f'  scheduler pp{p} tp{t}', True, 'scheduler.py:6408'))
    rows.append(('detokenizer', True, 'detokenizer_manager.py:552'))
    return rows


def show_procs(a):
    print(f'== procs：一次启动里 general 插件 register() 的执行次数（TP={a.tp} PP={a.pp} DP={a.dp}）')
    for title, rows in ((f'vLLM（{a.method}，modelinfo 缓存{"命中" if a.cache_hit else "未命中"}）',
                         v_launch(a.tp, a.pp, a.dp, a.method, a.cache_hit)),
                        ('SGLang（engine.py:1995 强制 spawn）', s_launch(a.tp, a.pp))):
        print(f'-- {title}')
        for name, ran, src in rows:
            print(f'   {name:<40}{"执行" if ran else "跳过":<6}{src}')
        print(f'   合计执行 {sum(r for _, r, _ in rows)} 次，分布在 {len({n for n, r, _ in rows if r})} 个进程')
    print('   每个进程只跑一次是靠模块级 plugins_loaded；同一个进程里第二次注册同名 OOT op 会撞上 custom_op.py:338 的 assert')
    print()


# --------------------------------------------------------------------------- 场景：env
V_GENERAL = ('lora_filesystem_resolver', 'lora_hf_hub_resolver', 'zv_demo')
V_ENDPOINT = ('zv_demo_api',)


def v_env_effect(value):
    env = {} if value is None else {'VLLM_PLUGINS': value}
    log = Log()
    gen = list(v_load_by_group([EP(n, lambda: None) for n in V_GENERAL], env, log))

    class P:
        required_tasks = None
    ep = [n for n, _ in v_load_endpoint([EP(n, P) for n in V_ENDPOINT], env, ('generate',), log)]
    return gen, ep, log


def s_env_effect(value):
    env = {} if value is None else {'SGLANG_PLUGINS': value}
    return list(s_load_by_group([EP(n, lambda: None) for n in ('zv_demo', 'other_hooks')], env, Log()))


def show_env(a):
    print('== env：VLLM_PLUGINS 的取值决定加载什么（自带 general 插件 + zv_demo + endpoint 插件 zv_demo_api）')
    print(f'   {"VLLM_PLUGINS":<48}{"general 组":<50}{"endpoint 组":<16}')
    for v in (None, '', 'zv_demo', 'zv_demo,zv_demo_api', 'zv_demo, zv_demo_api',
              'zv_demo,zv_demo_api,lora_filesystem_resolver'):
        gen, ep, _ = v_env_effect(v)
        shown = '未设置' if v is None else repr(v)
        print(f'   {shown:<48}{", ".join(gen) or "（无）":<50}{", ".join(ep) or "（无）":<16}')
    print('   VLLM_PLUGINS 对 general / platform / io_processor / stat_logger 四组同时生效，一旦设置，'
          '自带的 lora_filesystem_resolver 也要写进去')
    print('   逗号后的空格不会被去掉：" zv_demo_api" 和 "zv_demo_api" 是两个名字')
    print()
    print(f'   {"SGLANG_PLUGINS":<34}SGLang general 组（装了 zv_demo、other_hooks）')
    for v in (None, '', 'zv_demo', 'zv_demo, other_hooks'):
        shown = '未设置' if v is None else repr(v)
        print(f'   {shown:<34}{", ".join(s_env_effect(v))}')
    print('   SGLang 先 strip 再丢空串，"" 等于没设；vLLM 的 "" 解析成 [""]，什么都不加载')
    print()


# --------------------------------------------------------------------------- 场景：endpoint
class FakeEngine:
    def __init__(self, worker_methods, n_workers=2):
        self.methods, self.n = worker_methods, n_workers

    def collective_rpc(self, method):
        """serial_utils.py:485-508：getattr 失败 → NotImplementedError。"""
        if method not in self.methods:
            raise NotImplementedError(f'Method {method!r} is not implemented.')
        return [self.methods[method](rank) for rank in range(self.n)]


def endpoint_call(engine, method='zv_get_scheduler_config'):
    if engine is None:
        return 503, 'requires an engine, which this server does not have'
    try:
        return 200, engine.collective_rpc(method)
    except NotImplementedError as e:
        return 500, str(e)


def show_endpoint(a):
    print('== endpoint：路由插件的四道关')

    class Open:
        required_tasks = None

    class GenOnly:
        required_tasks = ('generate',)

    rows = [
        ('VLLM_PLUGINS 未设置', {}, Open, ('generate',)),
        ('VLLM_PLUGINS=zv_demo（漏写 api）', {'VLLM_PLUGINS': 'zv_demo'}, Open, ('generate',)),
        ('VLLM_PLUGINS=zv_demo,zv_demo_api', {'VLLM_PLUGINS': 'zv_demo,zv_demo_api'}, Open, ('generate',)),
        ('required_tasks=generate，pooling 服务', {'VLLM_PLUGINS': 'zv_demo_api'}, GenOnly, ('embed',)),
        ('required_tasks=None，render server', {'VLLM_PLUGINS': 'zv_demo_api'}, Open, None),
        ('required_tasks=generate，render server', {'VLLM_PLUGINS': 'zv_demo_api'}, GenOnly, None),
    ]
    for name, env, cls, tasks in rows:
        log = Log()
        got = v_load_endpoint([EP('zv_demo_api', cls)], env, tasks, log)
        note = log[-1][1][:110] if log else ''
        print(f'   {name:<40}{"挂上" if got else "不挂":<6}{note}')
    print('-- 路由挂上之后，handler 调 collective_rpc("zv_get_scheduler_config")：')
    ext = {'zv_get_scheduler_config': lambda r: {'rank': r, 'max_num_seqs': 256}}
    for name, eng in (('render server（engine_client=None）', None),
                      ('没配 --worker-extension-cls', FakeEngine({})),
                      ('配了 ZvWorkerExtension', FakeEngine(ext))):
        code, body = endpoint_call(eng)
        print(f'   {name:<40}{code}  {body}')
    print('   上游 dummy 示例调的 get_scheduler_config 在 vllm 里不存在，照抄只会拿到 NotImplementedError')
    print()


# --------------------------------------------------------------------------- 场景：platform
def show_platform(a):
    print('== platform：几个平台插件同时“认领”这台机器时')
    cases = [([], [], 'CPU 机器没装插件'), (['cuda'], [], 'NVIDIA 机器'), (['cuda'], ['my_npu'], 'NVIDIA 机器 + 装了一个 OOT'),
             (['cuda'], ['my_npu', 'my_npu_v2'], '两个 OOT 都探测成功'), (['cuda', 'cpu'], [], '内置探测出两个')]
    for b, o, name in cases:
        print(f'   vLLM   {name:<28}→ {v_platform(b, o)}')
    print(f'   vLLM   {"VLLM_TARGET_DEVICE=cpu":<28}→ {v_platform(["cuda"], ["my_npu"], target_device="cpu")}')
    scases = [(None, [], [], '没装插件'), (None, ['my_npu'], ['my_npu'], '一个插件探测成功'),
              (None, ['a', 'b'], ['a', 'b'], '两个插件探测成功'), ('a', ['a', 'b'], ['a', 'b'], 'SGLANG_PLATFORM=a'),
              ('c', ['a', 'b'], ['a'], 'SGLANG_PLATFORM=c（没装）'), ('b', ['a', 'b'], ['a'], 'SGLANG_PLATFORM=b（探测失败）')]
    for sel, inst, act, name in scases:
        print(f'   SGLang {name:<28}→ {s_platform(sel, inst, act)}')
    print('   vLLM 里 OOT 插件压过内置；SGLang 里平台插件只和平台插件比，指定 SGLANG_PLATFORM 时其余平台包的 general 插件也不加载')
    print()


# --------------------------------------------------------------------------- 场景：failure
def failure_matrix():
    def boom():
        raise ValueError('VLLM_LORA_RESOLVER_CACHE_DIR must be set to a valid directory')
    rows = []
    # vLLM general：func() 抛错
    try:
        v_load_general(VProc('api'), [EP('lora_filesystem_resolver', boom)], {}, Log())
        rows.append(('general 插件 func() 抛错', 'vLLM', '继续启动'))
    except ValueError as e:
        rows.append(('general 插件 func() 抛错', 'vLLM', f'启动失败：ValueError: {str(e)[:40]}'))
    log = Log()
    s_load_plugins(VProc('main'), [EP('zv_demo', boom)], {}, log)
    rows.append(('general 插件 func() 抛错', 'SGLang', f'继续启动，日志：{log[-1][1][:44]}'))
    # load() 阶段 ImportError
    log = Log()
    v_load_general(VProc('api'), [EP('zv_demo', load_error=ImportError('No module named zv_demo_plugin'))], {}, log)
    rows.append(('entry point load() 抛 ImportError', 'vLLM', f'继续启动，插件缺席：{log[-1][1][:40]}'))
    log = Log()
    s_load_plugins(VProc('main'), [EP('zv_demo', load_error=ImportError('No module named zv_demo_plugin'))], {}, log)
    rows.append(('entry point load() 抛 ImportError', 'SGLang', '继续启动，插件缺席'))
    # 同一进程注册两次同名 OOT op
    registry = {}

    def register_oot():
        assert 'RMSNorm' not in registry, 'Duplicate op name: RMSNorm'
        registry['RMSNorm'] = object
    try:
        v_load_general(VProc('api'), [EP('plugin_a', register_oot), EP('plugin_b', register_oot)], {}, Log())
        rows.append(('两个插件 register_oot 同名', 'vLLM', '继续启动'))
    except AssertionError as e:
        rows.append(('两个插件 register_oot 同名', 'vLLM', f'启动失败：AssertionError: {e}'))
    # endpoint 工厂抛错
    log = Log()

    def bad_factory():
        raise RuntimeError('fastapi missing')
    got = v_load_endpoint([EP('zv_demo_api', bad_factory)], {'VLLM_PLUGINS': 'zv_demo_api'}, ('generate',), log)
    rows.append(('endpoint 工厂抛错', 'vLLM', f'继续启动，路由不挂（{len(got)} 个插件）'))
    return rows


def show_failure(a):
    print('== failure：插件出错时谁会把服务带下去')
    for what, who, res in failure_matrix():
        print(f'   {what:<36}{who:<8}{res}')
    print('   vLLM 的 general 插件跑在每个 worker 里，抛错 = 起不来；SGLang 包了 try，代价是插件静默缺席，要自己看日志')
    print()


# --------------------------------------------------------------------------- 场景：hooks
def hook_trace(order):
    """order：[(type, 名字)] 按注册顺序；返回调用轨迹。"""
    trace = []
    h = Hooks()
    h.targets['run_batch'] = lambda x: trace.append('原函数') or x

    def make(typ, name):
        if typ == 'REPLACE':
            return lambda x: trace.append(f'{name}(REPLACE)') or x
        if typ == 'BEFORE':
            return lambda x: trace.append(f'{name}(BEFORE)')
        if typ == 'AFTER':
            return lambda r, x: trace.append(f'{name}(AFTER)')

        def around(orig, x):
            trace.append(f'{name}(AROUND 进)')
            r = orig(x)
            trace.append(f'{name}(AROUND 出)')
            return r
        return around
    for typ, name in order:
        h.register('run_batch', make(typ, name), typ)
    h.apply()
    h.targets['run_batch'](1)
    return trace


def show_hooks(a):
    print('== hooks：SGLang 同一个 target 上挂多个 hook 时的执行顺序')
    for order in ([('AFTER', 'h1'), ('AROUND', 'h2'), ('BEFORE', 'h3')],
                  [('AROUND', 'timer'), ('REPLACE', 'mine')],
                  [('AFTER', 'h1'), ('REPLACE', 'r1'), ('AROUND', 'h2')]):
        reg = ' → '.join(f'{n}:{t}' for t, n in order)
        print(f'   注册顺序 {reg}')
        print(f'   执行轨迹 {" → ".join(hook_trace(order))}')
    print('   REPLACE 先排到最里层（替换原函数，不再调用原函数），其余按注册顺序一层层往外包，最后注册的最先执行')
    print('   vLLM 没有通用 hook：换算子靠 CustomOp.register_oot（实例化时换类），换模型靠 ModelRegistry，加 worker 方法靠 --worker-extension-cls')
    print()


# --------------------------------------------------------------------------- 场景：bench
def random_prompts(n, in_len, seed, vocab=151643):
    """vllm random 数据集：offset 随机，token = (offset + index + arange) % vocab（datasets.py:748-750）。"""
    rng = random.Random(seed)
    return [tuple((rng.randrange(vocab) + i + k) % vocab for k in range(in_len)) for i in range(n)]


class BlockCache:
    """块级前缀缓存：命中最多 num_tokens - 1；空闲队列头部先被驱逐；请求结束时块倒序入队。"""

    def __init__(self, cap_blocks):
        self.cap = cap_blocks
        self.free = OrderedDict()   # 块哈希 → None；可驱逐，头部最旧
        self.used = 0               # 被 running 请求引用的块数

    def run(self, prompts, out_len, concurrency):
        hit = total = 0
        running = []
        for p in prompts:
            nfull = (len(p) - 1) // BLOCK          # request.py:314-318
            keys = [hash(p[:(i + 1) * BLOCK]) for i in range(len(p) // BLOCK)]
            h = 0
            for k in keys[:nfull]:
                if k in self.free:
                    h += 1
                else:
                    break
            for k in keys[:h]:
                del self.free[k]
            need = (len(p) + out_len + BLOCK - 1) // BLOCK - h
            while self.used + h + need + len(self.free) > self.cap and self.free:
                self.free.popitem(last=False)
            self.used += h + need
            hit += h * BLOCK
            total += len(p)
            own = keys[:h] + keys[h:] + [None] * (need - len(keys[h:]))
            running.append(own)
            while len(running) > concurrency:
                self._finish(running.pop(0))
        while running:
            self._finish(running.pop(0))
        return hit, total

    def _finish(self, blocks):
        self.used -= len(blocks)
        for k in reversed(blocks):                  # single_type_kv_cache_manager.py:599
            if k is not None:
                self.free[k] = None
                self.free.move_to_end(k)


def bench_rerun(n, in_len, out_len, seed2, reset, concurrency=256, cap=sc.KV_CAP // BLOCK):
    cache = BlockCache(cap)
    cache.run(random_prompts(n, in_len, 0), out_len, concurrency)
    if reset:
        cache.free.clear()
    hit, total = cache.run(random_prompts(n, in_len, seed2), out_len, concurrency)
    return hit / total


def show_bench(a):
    n, L, O = a.num_prompts, a.input_len, a.output_len
    print(f'== bench：服务不重启，用同样的参数再跑一遍 vllm bench serve（random {L}/{O}，并发 {a.concurrency}）')
    print(f'   KV 容量 {sc.KV_CAP // BLOCK} 块 = {sc.KV_CAP} token；一轮 prompt + 输出共 {n * (L + O)} token')
    for name, seed2, reset in (('--seed 0 复跑', 0, False), ('--seed 1 复跑', 1, False),
                               ('先 POST /reset_prefix_cache', 0, True)):
        r = bench_rerun(n, L, O, seed2, reset, a.concurrency)
        print(f'   {name:<30}第二轮前缀命中 {r * 100:5.1f}%')
    for nn in (100, 200, 300, 400, 1000):
        print(f'   num-prompts {nn:<5}（{nn * (L + O) / sc.KV_CAP:4.2f}× 容量）同 seed 复跑命中 '
              f'{bench_rerun(nn, L, O, 0, False, a.concurrency) * 100:5.1f}%')
    full = sc.gpu_ms([(L, 0)])
    hit_len = (L - 1) // BLOCK * BLOCK
    tail = sc.gpu_ms([(L - hit_len, hit_len)])
    print(f'   单条 prefill：冷 {full:.1f} ms，命中 {hit_len} token 后只算 {L - hit_len} 个 {tail:.1f} ms')
    print('   /reset_prefix_cache 要 VLLM_SERVER_DEV_MODE=1（envs.py:1473）；SGLang bench 只在 SGLANG_IS_IN_CI 或 --flush-cache 时 flush，'
          '--seed 默认 42')
    print()


SCEN = dict(procs=show_procs, env=show_env, endpoint=show_endpoint, platform=show_platform,
            failure=show_failure, hooks=show_hooks, bench=show_bench)


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', choices=list(SCEN) + ['all'], default='procs')
    ap.add_argument('--tp', type=int, default=4)
    ap.add_argument('--pp', type=int, default=1)
    ap.add_argument('--dp', type=int, default=1)
    ap.add_argument('--method', choices=['spawn', 'fork'], default='spawn',
                    help='vllm serve 默认被改成 spawn；离线 LLM() 在 CUDA 未初始化时是 fork')
    ap.add_argument('--cache-hit', dest='cache_hit', action='store_true', help='modelinfo 缓存已命中')
    ap.add_argument('--num-prompts', dest='num_prompts', type=int, default=1000)
    ap.add_argument('--input-len', dest='input_len', type=int, default=1024)
    ap.add_argument('--output-len', dest='output_len', type=int, default=128)
    ap.add_argument('--concurrency', type=int, default=256)
    return ap


def main():
    a = parser().parse_args()
    for name in (SCEN if a.scenario == 'all' else [a.scenario]):
        SCEN[name](a)


if __name__ == '__main__':
    main()
