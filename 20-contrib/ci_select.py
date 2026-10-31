"""CI 选择逻辑的离线复刻：改了哪些文件，vLLM 的 Buildkite 会跑哪些 step；SGLang 的测试注册怎么被解析和分片。

只依赖标准库。data/ 下是从上游抽出来的快照：
  data/ci_steps.json    vllm-project/ci-infra eb2d628 读 vllm 0d6f92a 的 .buildkite/test_areas/*.yaml，
                        36 个 yaml、254 个 step，字段 file/group/label/key/device/gpus/timeout/optional/deps/amd
  data/vpaths.txt.gz    vllm 0d6f92a 的 git ls-files（8636 个路径），用来找“永远匹配不上”的依赖

对照的源码（ci-infra eb2d628）：
  buildkite/pipeline_generator/buildkite_step.py:1001-1005   _matches_source_dependency：去尾 /，相等或以 "dep/" 开头
  buildkite/pipeline_generator/buildkite_step.py:1544-1561   _source_file_dependencies_match：! 开头是排除项
  buildkite/pipeline_generator/buildkite_step.py:1488-1525   _step_should_run：NOAUTO → only_step_keys → selector →
                                                             image-build → nightly → optional → AMD 原生 → run_all → 依赖匹配
  buildkite/pipeline_generator/step.py:90-106                yaml 顶层有 depends_on 时，把 yaml 自己追加进每个 step 的依赖
  buildkite/pipeline_generator/global_config.py:143-170      _should_run_all：RUN_ALL / TORCH_NIGHTLY / ready-run-all-tests 标签 /
                                                             文件 re.match 命中 run_all_patterns 且不命中 exclude
  vllm .buildkite/ci_config.yaml（0d6f92a）                  run_all_patterns 与 run_all_exclude_patterns
  buildkite/pipeline_generator/pipeline_generator.py:486-501 is_docs_only_change：只有 docs/ recipes/ *.md mkdocs.yaml
  sglang python/sglang/test/ci/ci_register.py（8fabf5e）     AST 解析 register_*_ci(...)；effective_suite；auto_partition（LPT）
  sglang test/README.md:115-129                               新增注册测试的三条棘轮，其中 est_time × GPU 数 ≤ 1200

场景：
  select    几组典型 diff 各会触发哪些 step、多少“超时上限 GPU·分钟”
  dead      依赖列表里拼错或指向已删除文件、永远不会触发的条目
  docsonly  只改文档时整条流水线跳过的条件
  runall    哪些文件一改就全量跑
  sglang    register_cuda_ci 的解析、报错、1200 加权秒棘轮和 LPT 分片

用法：
  python3 ci_select.py --scenario select
  python3 ci_select.py --diff vllm/plugins/__init__.py tests/plugins_tests/test_platform_plugins.py
  python3 ci_select.py --scenario all

简化：AMD 原生 step（device 以 mi 开头）和 amd 镜像 step 只计数不展开；
AMD_NATIVE_RUNTIME_SOURCE_DEPENDENCIES、selector、only_step_keys 都按未设置处理。
"""
import argparse
import ast
import gzip
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')

# vllm .buildkite/ci_config.yaml
RUN_ALL_PATTERNS = ['docker/Dockerfile', 'CMakeLists.txt', 'requirements/common.txt', 'requirements/cuda.txt',
                    'requirements/kv_connectors.txt', 'requirements/build/cuda.txt', 'requirements/test/cuda.txt',
                    'setup.py', 'csrc/', 'cmake/', '.buildkite/check-torch-abi.py']
RUN_ALL_EXCLUDES = ['docker/Dockerfile.', 'csrc/cpu/', 'csrc/rocm/', 'cmake/hipify.py', 'cmake/cpu_extension.cmake']
# 唯一一个顶层 depends_on 为空的 yaml，不会把自己追加进依赖
NO_GROUP_DEPENDS = {'.buildkite/test_areas/rust_frontend_cargo.yaml'}


# --------------------------------------------------------------------------- 数据
def load_steps():
    with open(os.path.join(DATA, 'ci_steps.json')) as f:
        steps = json.load(f)
    for s in steps:                                      # step.py:90-106
        if s['file'] not in NO_GROUP_DEPENDS:
            s['deps'] = list(s['deps']) + [s['file']]
    return steps


def load_paths():
    with gzip.open(os.path.join(DATA, 'vpaths.txt.gz'), 'rt') as f:
        return [x for x in f.read().splitlines() if x]


def is_amd(step):
    return (step['device'] or '').startswith('mi')


# --------------------------------------------------------------------------- vLLM 选择逻辑
def matches(dep, path):
    """buildkite_step.py:1001-1005。"""
    n = dep.rstrip('/')
    if not n:
        return False
    return path == n or path.startswith(n + '/')


def deps_match(deps, diff):
    """buildkite_step.py:1544-1561。"""
    if not deps:
        return False
    inc = [d for d in deps if not d.startswith('!')]
    exc = [d[1:] for d in deps if d.startswith('!')]
    return any(any(matches(i, f) for i in inc) and not any(matches(e, f) for e in exc) for f in diff)


def should_run_all(diff, labels=(), env=None):
    """global_config.py:143-170。re.match 只锚开头，所以 "setup.py" 也会命中 "setup.pyc"。"""
    env = env or {}
    if env.get('RUN_ALL') == '1' or env.get('TORCH_NIGHTLY') == '1' or 'ready-run-all-tests' in labels:
        return True
    for f in diff:
        if any(re.match(p, f) for p in RUN_ALL_PATTERNS) and not any(re.match(e, f) for e in RUN_ALL_EXCLUDES):
            return True
    return False


def is_docs_only(diff):
    """pipeline_generator.py:486-501。空 diff 不算 docs-only。"""
    if not diff:
        return False
    for f in diff:
        if not f:
            continue
        if f.startswith(('docs/', 'recipes/')) or f.endswith('.md') or f == 'mkdocs.yaml':
            continue
        return False
    return True


def step_should_run(step, diff, run_all=False, nightly=False, env=None):
    """buildkite_step.py:1488-1525（selector / only_step_keys / AMD 原生依赖按未设置处理）。"""
    env = env or {}
    if env.get('NOAUTO') == '1':
        return False
    if step['key'] and step['key'].startswith('image-build'):
        return True
    if nightly:
        return True
    if step['optional']:
        return False
    if run_all:
        return True
    return deps_match(step['deps'], diff)


def select(steps, diff, labels=(), env=None):
    """返回 (状态, 被选中的非 AMD step 列表, 被选中的 AMD 原生 step 数, 其中带 amd 镜像的数)。"""
    env = env or {}
    run_all = should_run_all(diff, labels, env)
    if env.get('DOCS_ONLY_DISABLE', '0') == '0' and not run_all and is_docs_only(diff):
        return 'docs-only，整条流水线跳过', [], 0, 0
    nightly = 'ready-run-all-tests' in labels or env.get('NIGHTLY') == '1'
    chosen = [s for s in steps if step_should_run(s, diff, run_all, nightly, env)]
    nv = [s for s in chosen if not is_amd(s)]
    amd_native = sum(1 for s in chosen if is_amd(s))
    mirrored = sum(1 for s in nv if s['amd'])
    return ('run_all' if run_all else '按依赖选择'), nv, amd_native, mirrored


def gpu_minutes(sel):
    """timeout_in_minutes × 卡数，是上限不是实际耗时；没写 timeout 的 step 不计。"""
    return sum((s['timeout'] or 0) * (s['gpus'] or 1) for s in sel)


def dead_deps(steps, paths):
    """依赖条目在仓库里一个文件都匹配不上。rust/ 等目录前缀也按同样规则判断。"""
    out = []
    for s in steps:
        for d in s['deps']:
            if d.startswith('!'):
                continue
            if not any(matches(d, p) for p in paths):
                out.append((s['key'], d))
    return out


def near_miss(dep, paths):
    """猜一个最可能的正确写法：补 .py 或去掉尾部截断。"""
    for cand in (dep + '.py',):
        if cand in paths:
            return cand
    pref = [p for p in paths if p.startswith(dep)]
    return pref[0] + (f' 等 {len(pref)} 个' if len(pref) > 1 else '') if pref else '（仓库中已不存在）'


# --------------------------------------------------------------------------- SGLang：ci_register.py
PARAM_ORDER = ('est_time', 'suite', 'nightly', 'disabled')
KWARG_ONLY = ('stage', 'runner_config')
REGISTER = {'register_cpu_ci': 'cpu', 'register_cuda_ci': 'cuda', 'register_amd_ci': 'amd', 'register_npu_ci': 'npu',
            'register_xpu_ci': 'xpu', 'register_musa_ci': 'musa', 'register_mlx_ci': 'mlx', 'register_mps_ci': 'mps',
            'register_ppu_ci': 'ppu'}
UNSET = object()


def _const(node):
    return node.value if isinstance(node, ast.Constant) else UNSET


def parse_call(call, fn='test.py'):
    """RegistryVisitor._parse_call_args 的分支顺序。"""
    name = call.func.id
    args = {k: UNSET for k in PARAM_ORDER + KWARG_ONLY}
    seen = set()
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise ValueError(f'{fn}: starred arguments are not supported in {name}()')
    if len(call.args) > len(PARAM_ORDER):
        raise ValueError(f'{fn}: too many positional arguments in {name}()')
    for k, a in zip(PARAM_ORDER, call.args):
        seen.add(k)
        args[k] = _const(a)
    for kw in call.keywords:
        if kw.arg is None:
            raise ValueError(f'{fn}: **kwargs are not supported in {name}()')
        if kw.arg not in args:
            raise ValueError(f"{fn}: unknown argument '{kw.arg}' in {name}()")
        if kw.arg in seen:
            raise ValueError(f"{fn}: duplicated argument '{kw.arg}' in {name}()")
        seen.add(kw.arg)
        args[kw.arg] = _const(kw.value)
    if args['est_time'] is UNSET:
        raise ValueError(f'{fn}: est_time is a required constant in {name}()')
    st, rc, su = (args[k] is not UNSET for k in ('stage', 'runner_config', 'suite'))
    if not ((st and rc and not su) or (not st and not rc and su)):
        raise ValueError(f'{fn}: {name}() must specify exactly one of (stage, runner_config) pair or suite; '
                         f'got stage={st}, runner_config={rc}, suite={su}')
    if not isinstance(args['est_time'], (int, float)):
        raise ValueError(f'{fn}: est_time must be a number in {name}()')
    for k in ('suite', 'stage', 'runner_config'):
        if args[k] is not UNSET and not isinstance(args[k], str):
            raise ValueError(f'{fn}: {k} must be a string in {name}()')
    nightly = args['nightly']
    if nightly is UNSET:
        nightly = False
    elif not isinstance(nightly, bool):
        raise ValueError(f'{fn}: nightly must be a boolean in {name}()')
    disabled = None if args['disabled'] is UNSET else args['disabled']
    if disabled is not None and not isinstance(disabled, str):
        raise ValueError(f'{fn}: disabled must be a string in {name}()')
    stage = args['stage'] if st else None
    runner = args['runner_config'] if rc else None
    return dict(backend=REGISTER[name], est_time=float(args['est_time']), stage=stage, runner_config=runner,
                suite=args['suite'] if su else None, nightly=nightly, disabled=disabled,
                effective_suite=f'{stage}-test-{runner}' if st else args['suite'])


def _has_main_with_call(stmt):
    t = stmt.test
    if not (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == '__name__'
            and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq) and len(t.comparators) == 1
            and isinstance(t.comparators[0], ast.Constant) and t.comparators[0].value == '__main__'):
        return False
    return any(isinstance(c, ast.Call) for c in ast.walk(ast.Module(body=stmt.body, type_ignores=[])))


def parse_source(src, fn='test.py'):
    """visit_Module：只看模块顶层的表达式语句，函数里、if 里的注册都不算。返回 (registries, has_main)。"""
    regs, has_main = [], False
    for stmt in ast.parse(src).body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            c = stmt.value
            if isinstance(c.func, ast.Name) and c.func.id in REGISTER:
                regs.append(parse_call(c, fn))
        elif isinstance(stmt, ast.If) and _has_main_with_call(stmt):
            has_main = True
    return regs, has_main


def collect(src, fn='test.py'):
    """collect_tests 的两道 sanity check。"""
    regs, has_main = parse_source(src, fn)
    if not regs:
        raise ValueError(f'No CI registry found in {fn}')
    if any(r['disabled'] is None for r in regs) and not has_main:
        raise ValueError(f'{fn}: missing `if __name__ == "__main__":` entry.')
    return regs


def gpu_count(runner_config):
    m = re.match(r'(\d+)-gpu', runner_config or '')
    return int(m.group(1)) if m else 1


def weighted_cost(reg):
    """test/README.md:126-129：est_time × GPU 数，单个 backend 默认 PR 注册 ≤ 1200。"""
    return reg['est_time'] * gpu_count(reg['runner_config'])


def auto_partition(files, size):
    """ci_register.py auto_partition：按 (-est, 文件名) 排序，依次塞进当前总和最小的分片。files：[(文件名, est)]。"""
    parts = [[] for _ in range(size)]
    sums = [0.0] * size
    for fn, est in sorted(files, key=lambda x: (-x[1], x[0])):
        i = min(range(size), key=sums.__getitem__)
        parts[i].append(fn)
        sums[i] += est
    return parts, sums


# --------------------------------------------------------------------------- 场景
EXAMPLES = [
    ('改插件加载器', ['vllm/plugins/__init__.py']),
    ('只改插件测试', ['tests/plugins_tests/test_platform_plugins.py']),
    ('改调度器', ['vllm/v1/core/sched/scheduler.py']),
    ('改 KV connector', ['vllm/distributed/kv_transfer/kv_connector/v1/base.py']),
    ('改 bench 工具', ['vllm/benchmarks/serve.py']),
    ('改 plugins.yaml', ['.buildkite/test_areas/plugins.yaml']),
    ('改 CPU 后端 kernel', ['csrc/cpu/attention.cpp']),
    ('改 CUDA kernel', ['csrc/attention/paged_attention_v1.cu']),
    ('只改设计文档', ['docs/design/plugin_system.md']),
    ('文档 + 一行代码', ['docs/design/plugin_system.md', 'vllm/envs.py']),
]


def show_select(a, steps):
    print('== select：改哪些文件，Buildkite 跑哪些 step（非 AMD 部分；timeout×卡数 是上限）')
    print(f'   快照：{len(steps)} 个 step，{sum(s["optional"] for s in steps)} 个 optional（PR 默认不跑），'
          f'{sum(is_amd(s) for s in steps)} 个 AMD 原生')
    print(f'   {"改动":<18}{"状态":<22}{"step 数":>7}{"GPU·分钟上限":>14}{"+AMD":>6}  例子')
    for name, diff in EXAMPLES:
        state, sel, amd_n, mir = select(steps, diff)
        eg = ', '.join(s['key'] for s in sel[:3]) + (' …' if len(sel) > 3 else '')
        print(f'   {name:<18}{state:<22}{len(sel):>7}{gpu_minutes(sel):>14}{amd_n + mir:>6}  {eg}')
    wide = sum(1 for s in steps if 'vllm/' in s['deps'] and not s['optional'] and not is_amd(s))
    print(f'   {wide} 个默认 step 直接依赖整个 "vllm/"，vllm/ 下任何改动都会拉起它们；'
          '它们几乎都带 "!vllm/distributed/kv_transfer/"，所以只改 KV connector 时 NVIDIA 侧一个都不跑')
    print('   改 yaml 本身也会触发：step.py:90-106 把 yaml 路径追加进了它每个 step 的依赖')
    print()


def show_diff(a, steps):
    state, sel, amd_n, mir = select(steps, a.diff, a.label)
    print(f'== diff：{" ".join(a.diff)}')
    print(f'   {state}；{len(sel)} 个 step，GPU·分钟上限 {gpu_minutes(sel)}；另有 AMD 原生 {amd_n}、amd 镜像 {mir}')
    for s in sel:
        hit = [d for d in s['deps'] if not d.startswith('!') and any(matches(d, f) for f in a.diff)]
        why = 'run_all' if state == 'run_all' else ('image-build' if not hit else f'命中 {hit[0]}')
        print(f'   {s["key"]:<52}{str(s["device"]):<12}{s["gpus"]}×{str(s["timeout"]):<5}{why}')
    print()


def show_dead(a, steps):
    paths = set(load_paths())
    plist = sorted(paths)
    dead = dead_deps(steps, plist)
    print(f'== dead：依赖条目在 {len(plist)} 个仓库文件里一个也匹配不上（改了对应文件也不会触发这个 step）')
    for key, d in dead:
        print(f'   {key:<38}{d:<40}→ {near_miss(d, plist)}')
    print(f'   共 {len(dead)} 条。matches() 只认“相等”或“以 dep/ 开头”，不做前缀模糊匹配，少写 .py 就永远不命中')
    print('   这类 step 往往还有别的宽依赖（如 vllm/engine/）兜着，所以平时看不出来；只改那个测试文件时才会漏跑')
    print()


def show_docsonly(a, steps):
    print('== docsonly：is_docs_only_change 的判定')
    for diff in ([], ['README.md'], ['docs/design/plugin_system.md', 'mkdocs.yaml'], ['recipes/Qwen3.md'],
                 ['docs/mkdocs/hooks/url_schemes.py'], ['docs/design/plugin_system.md', 'vllm/envs.py'],
                 ['examples/offline_inference/basic.py']):
        print(f'   {str(diff):<56}{"跳过 CI" if is_docs_only(diff) else "照常跑"}')
    print('   docs/ 下的 .py 也算文档（前缀判断在后缀之前）；空 diff 不算 docs-only；'
          'DOCS_ONLY_DISABLE=1、run_all、only_step_keys 任一成立都不跳过')
    print()


def show_runall(a, steps):
    print('== runall：一改就全量跑的文件（re.match 只锚开头）')
    for f in ('setup.py', 'CMakeLists.txt', 'csrc/attention/paged_attention_v1.cu', 'csrc/cpu/attention.cpp',
              'csrc/rocm/attention.cu', 'cmake/utils.cmake', 'cmake/hipify.py', 'docker/Dockerfile',
              'docker/Dockerfile.cpu', 'requirements/cuda.txt', 'requirements/rocm.txt', 'requirements/test/cuda.txt',
              'setup.pyc'):
        print(f'   {f:<44}{"全量" if should_run_all([f]) else "-"}')
    _, sel, _, _ = select(steps, ['csrc/attention/paged_attention_v1.cu'])
    print(f'   全量 = {len(sel)} 个非 optional 非 AMD step，GPU·分钟上限 {gpu_minutes(sel)}；'
          '打 ready-run-all-tests 标签时连 optional 也一起跑（nightly=1）')
    print()


SG_CASES = [
    ('新式写法', 'register_cuda_ci(est_time=300, stage="stage-b", runner_config="1-gpu-large")\n'
                'if __name__ == "__main__":\n    unittest.main()\n'),
    ('老式 suite', 'register_amd_ci(est_time=120, suite="stage-a-test-1-gpu-amd")\n'
                  'if __name__ == "__main__":\n    unittest.main()\n'),
    ('est_time 用变量', 'T = 300\nregister_cuda_ci(est_time=T, stage="stage-b", runner_config="1-gpu-large")\n'),
    ('stage 和 suite 都写', 'register_cuda_ci(est_time=300, stage="stage-b", runner_config="1-gpu-large", '
                          'suite="x")\n'),
    ('nightly="true"', 'register_cuda_ci(300, "nightly-1-gpu", "true")\n'),
    ('**kwargs', 'register_cuda_ci(**CFG)\n'),
    ('注册写在 if 里', 'if True:\n    register_cuda_ci(est_time=60, stage="stage-a", runner_config="1-gpu-small")\n'),
    ('没有 main', 'register_cuda_ci(est_time=60, stage="stage-a", runner_config="1-gpu-small")\n'),
    ('main 里只有 pass', 'register_cuda_ci(est_time=60, stage="stage-a", runner_config="1-gpu-small")\n'
                      'if __name__ == "__main__":\n    pass\n'),
    ('disabled 不要 main', 'register_cuda_ci(est_time=60, stage="stage-a", runner_config="1-gpu-small", '
                         'disabled="flaky #123, until 2026-12-01")\n'),
]


def show_sglang(a, steps):
    print('== sglang：test 文件顶部 register_*_ci(...) 被 AST 解析（不执行）')
    for name, src in SG_CASES:
        try:
            regs = collect(src)
            r = regs[0]
            res = f'{r["backend"]} suite={r["effective_suite"]} est={r["est_time"]:.0f}' + \
                  (f' disabled={r["disabled"]!r}' if r['disabled'] else '')
        except ValueError as e:
            res = '报错：' + str(e).replace('test.py: ', '')
        print(f'   {name:<20}{res}')
    print('-- 1200 加权秒棘轮（est_time × runner_config 里的卡数）')
    for est, rc in ((900, '1-gpu-large'), (400, '4-gpu-h100'), (150, '8-gpu-h200'), (160, '8-gpu-h200')):
        reg = dict(est_time=est, runner_config=rc)
        w = weighted_cost(reg)
        print(f'   est_time={est:<5}{rc:<14}{w:>6.0f}  {"通过" if w <= 1200 else "超限：挪去 nightly，或写 ci-cost-override: 理由"}')
    files = [('test_a.py', 600), ('test_b.py', 500), ('test_c.py', 400), ('test_d.py', 300), ('test_e.py', 300),
             ('test_f.py', 200), ('test_g.py', 100)]
    parts, sums = auto_partition(files, 3)
    print('-- auto_partition（LPT）：7 个文件分 3 片')
    for i, (p, s) in enumerate(zip(parts, sums)):
        print(f'   分片 {i}：{", ".join(p):<36}{s:>5.0f} s')
    print(f'   最长分片 {max(sums):.0f} s，下界 max(最大单文件, 总和/3) = {max(600, sum(e for _, e in files) / 3):.0f} s')
    print('   vLLM 的分配写在 yaml 里（每个 step 一个 label + 依赖），SGLang 写在测试文件里（文件自己声明 suite 和耗时）')
    print()


SCEN = dict(select=show_select, dead=show_dead, docsonly=show_docsonly, runall=show_runall, sglang=show_sglang)


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', choices=list(SCEN) + ['all'], default='select')
    ap.add_argument('--diff', nargs='*', default=None, help='自己给一组改动路径，打印逐个 step 的命中原因')
    ap.add_argument('--label', nargs='*', default=[], help='PR 标签，例如 ready-run-all-tests')
    return ap


def main():
    a = parser().parse_args()
    steps = load_steps()
    if a.diff:
        show_diff(a, steps)
        return
    for name in (SCEN if a.scenario == 'all' else [a.scenario]):
        SCEN[name](a, steps)


if __name__ == '__main__':
    main()
