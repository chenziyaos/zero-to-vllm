"""提 PR 前的本地自查：标题格式、DCO 签名、AI 署名、大改动和文档提示（只依赖标准库）。

对照的规则（vllm 0d6f92a，sglang 8fabf5e）：
  vllm   .github/workflows/pr-title.yml:15-40       ^(\\[[^][]+\\])+ +[^[:space:]].*$；只在非 draft 且带 ready 标签时检查
         docs/contributing/README.md:190-196        DCO：每个 commit 带 Signed-off-by:（git commit -s）
         docs/contributing/README.md:206-228        AI 辅助：人要参与、不要零碎 PR、PR 描述里说明、commit 加 Co-authored-by:
         docs/contributing/README.md:232-258        常用 type / scope 标签，可叠加不加空格
         docs/contributing/README.md:271-287        用户可见改动同一个 PR 改文档；参数 / 配置 / 环境变量写在 help= / docstring / 注释里
         docs/contributing/README.md:309-312        >500 行（不含 kernel/data/config/test）先开 RFC issue，否则打 rfc-required
         docs/contributing/README.md:385-387        无写权限的贡献者最多 6 个 open PR
  sglang .github/workflows/pr-title.yml:20-60       "[Type] scope: title" 或 "[Type] [Hardware] scope: title"；
                                                     8 个 Type；硬件标签全大写，Intel/Apple/Huawei/Ascend 四个例外；
                                                     Revert "原标题"；首尾不能有空白
         .github/workflows/pr-title.test.cjs        合法 / 非法样例（test_pr_lint.py 原样搬过来）

用法：
  python3 pr_lint.py --title "[Bugfix][Scheduler] Fix priority handling"
  python3 pr_lint.py --title "[Fix] scheduler: Fix priority handling" --repo sglang
  python3 pr_lint.py --git origin/main..HEAD           # 在 vllm 仓库里检查当前分支的 commit 与 diff
  python3 pr_lint.py --scenario demo

注意：JS 的 $ 在没有 m 标志时只匹配字符串末尾，Python 的 $ 还会匹配末尾换行之前，所以这里一律用 re.fullmatch。
"""
import argparse
import re
import subprocess
import sys

# --------------------------------------------------------------------------- 标题
V_TITLE = re.compile(r'(\[[^\]\[]+\])+ +\S.*', re.S)          # bash ERE 的 . 也吃换行

S_TYPES = ('Feat', 'Fix', 'Refactor', 'Perf', 'Doc', 'Test', 'CI', 'Chore')
S_TITLE = re.compile(r'\[(' + '|'.join(S_TYPES) + r')\] (?:\[([A-Za-z0-9][A-Za-z0-9._+-]*)\] )?'
                     r'[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*: \S(?:[^\r\n]*\S)?')
S_REVERT = re.compile(r'Revert "\S(?:[^\r\n]*\S)?"')
S_BRANDS = ('Intel', 'Apple', 'Huawei', 'Ascend')
JS_WS = ' \t\n\r\v\f             ' \
        '    　﻿'


def vllm_title(title):
    """返回 (是否通过, 说明)。"""
    if V_TITLE.fullmatch(title):
        tags = re.findall(r'\[([^\]\[]+)\]', title.split(' ', 1)[0])
        return True, f'标签 {tags}'
    return False, 'PR title must start with one or more [Tag] prefixes, followed by a space and a description.'


def sglang_title(title):
    m = S_TITLE.fullmatch(title)
    hw = m.group(2) if m else None
    brand = next((b for b in S_BRANDS if hw and b.lower() == hw.lower()), None)
    hw_ok = not hw or hw == (brand or hw.upper())
    revert = bool(S_REVERT.fullmatch(title))
    if ((m and hw_ok) or revert) and title == title.strip(JS_WS):
        if revert:
            return True, 'Revert 标题'
        return True, f'type={m.group(1)} hardware={hw or "-"} scope={title.split("] ")[-1].split(":")[0]}'
    if m and not hw_ok:
        why = f'硬件标签 [{hw}] 要写成 [{brand or hw.upper()}]'
    elif title != title.strip(JS_WS):
        why = '首尾有空白'
    elif re.match(r'\[[A-Za-z]+\]', title) and not re.match(r'\[(' + '|'.join(S_TYPES) + r')\]', title):
        why = f'Type 只能是 {", ".join(S_TYPES)}'
    elif re.match(r'\[\w+\] [^\[]', title) and ': ' not in title:
        why = '缺 "scope: "'
    else:
        why = '格式应为 "[Type] scope: title" 或 "[Type] [Hardware] scope: title"，字段之间、冒号之后各一个空格'
    return False, why


# --------------------------------------------------------------------------- commit
SIGNOFF = re.compile(r'^Signed-off-by: .+ <[^<>@\s]+@[^<>\s]+>\s*$', re.M)
COAUTHOR = re.compile(r'^(Co-authored-by|Assisted-by|Generated-by): .+$', re.M)


def check_commit(msg, ai_assisted=False):
    """返回问题列表。DCO 要求每个 commit 都有 Signed-off-by。"""
    issues = []
    if not SIGNOFF.search(msg):
        issues.append('缺 Signed-off-by:（git commit -s；已提交的用 git rebase --signoff origin/main）')
    if ai_assisted and not COAUTHOR.search(msg):
        issues.append('AI 辅助生成的改动要加 Co-authored-by: 之类的 trailer，PR 描述里也要写明')
    return issues


# --------------------------------------------------------------------------- diff
def excluded_from_loc(path):
    """>500 行规则里不计的 kernel / data / config / test。仓库没给精确定义，这里按目录和后缀近似。"""
    return (path.startswith(('tests/', 'csrc/', 'benchmarks/', 'examples/', 'docs/', '.buildkite/', '.github/'))
            or path.endswith(('.cu', '.cuh', '.cpp', '.h', '.hpp', '.json', '.yaml', '.yml', '.toml', '.txt',
                              '.md', '.jinja', '.csv')))


USER_VISIBLE = ('vllm/envs.py', 'vllm/engine/arg_utils.py', 'vllm/config/', 'vllm/entrypoints/')


def check_diff(numstat):
    """numstat：[(added, deleted, path)]。返回提示列表。"""
    tips = []
    core = sum(a + d for a, d, p in numstat if not excluded_from_loc(p))
    total = sum(a + d for a, d, _ in numstat)
    if core > 500:
        tips.append(f'非 kernel/data/config/test 改动 {core} 行（总 {total} 行）> 500：先开 RFC issue，否则会被打 rfc-required')
    paths = [p for _, _, p in numstat]
    touched = [p for p in paths if p.startswith(USER_VISIBLE)]
    if touched and not any(p.startswith('docs/') for p in paths):
        tips.append(f'动了用户可见入口 {touched[:3]}：参数 / 配置 / 环境变量写进 help= / docstring / envs.py 注释，'
                    '其余行为变化在同一个 PR 改 docs/')
    if paths and not any(p.startswith('tests/') for p in paths) and any(p.startswith('vllm/') for p in paths):
        tips.append('没有改 tests/：说明哪个测试覆盖了这次改动，或补一个')
    return tips


def git(args):
    return subprocess.run(['git'] + args, check=True, capture_output=True, text=True).stdout


def from_git(rng, ai):
    msgs = [m for m in git(['log', '--format=%H%n%B%x00', rng]).split('\x00') if m.strip()]
    rows = []
    for line in git(['diff', '--numstat', rng.replace('..', '...', 1) if '...' not in rng else rng]).splitlines():
        a, d, p = line.split('\t', 2)
        rows.append((0 if a == '-' else int(a), 0 if d == '-' else int(d), p))
    out = []
    for m in msgs:
        sha, _, body = m.strip().partition('\n')
        for i in check_commit(body, ai):
            out.append(f'{sha[:10]} {i}')
    return out, check_diff(rows), len(msgs)


# --------------------------------------------------------------------------- 演示
DEMO_TITLES = [
    '[Bugfix][Scheduler] Fix priority handling',
    '[Perf][ROCm][Kernel] Optimize fused MoE',
    '[Bugfix] [Scheduler] Fix priority handling',
    '[Bugfix]Fix priority handling',
    'Fix priority handling',
    '[Fix] scheduler: Fix priority handling',
    '[Fix] [AMD] scheduler: Fix priority handling',
    '[Fix] [Amd] scheduler: Fix priority handling',
    '[Bugfix] scheduler: Fix priority handling',
    'Revert "[Fix] scheduler: Fix priority handling"',
]


def demo():
    print('== title：同一个改动，两边的标题规则')
    print(f'   {"标题":<50}{"vLLM":<8}SGLang')
    for t in DEMO_TITLES:
        v, _ = vllm_title(t)
        s, why = sglang_title(t)
        print(f'   {t:<50}{"通过" if v else "不通过":<8}{"通过" if s else "不通过：" + why}')
    print('   vLLM 只要求 [任意标签]+空格+描述，标签之间不加空格是惯例不是检查项；SGLang 的 Type 是封闭集合，scope 必填')
    print('   vLLM 的检查只在 PR 非 draft 且带 ready 标签时触发，draft 阶段标题不合规不会报错')
    print()
    print('== commit：DCO 与 AI 署名')
    for name, msg, ai in (
            ('只有标题行', '[Bugfix] Fix priority handling\n', False),
            ('git commit -s', '[Bugfix] Fix priority handling\n\nSigned-off-by: A Dev <a@example.com>\n', False),
            ('AI 辅助但没署名', '[Bugfix] Fix\n\nSigned-off-by: A Dev <a@example.com>\n', True),
            ('AI 辅助 + 署名', '[Bugfix] Fix\n\nCo-authored-by: Claude\nSigned-off-by: A Dev <a@example.com>\n', True)):
        iss = check_commit(msg, ai)
        print(f'   {name:<18}{"OK" if not iss else "；".join(iss)}')
    print()
    print('== diff：大改动与文档提示')
    for name, rows in (
            ('插件小修 + 测试', [(12, 3, 'vllm/plugins/__init__.py'), (40, 0, 'tests/plugins_tests/test_x.py')]),
            ('新 kernel 900 行', [(820, 0, 'csrc/attention/new_kernel.cu'), (60, 4, 'vllm/_custom_ops.py'),
                                (180, 0, 'tests/kernels/test_new.py')]),
            ('调度器重写', [(430, 260, 'vllm/v1/core/sched/scheduler.py'), (120, 0, 'tests/v1/core/test_sched.py')]),
            ('加环境变量', [(8, 0, 'vllm/envs.py'), (15, 2, 'vllm/v1/worker/gpu_model_runner.py')])):
        tips = check_diff(rows)
        print(f'   {name:<18}{"OK" if not tips else tips[0]}')
        for t in tips[1:]:
            print(f'   {"":<18}{t}')
    print('   另外两条硬约束脚本查不了：无写权限时最多 6 个 open PR；AI 辅助的 PR 描述里要写明')
    print()


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--title')
    ap.add_argument('--repo', choices=['vllm', 'sglang', 'both'], default='both')
    ap.add_argument('--git', metavar='RANGE', help='例如 origin/main..HEAD')
    ap.add_argument('--ai', action='store_true', help='这次改动有 AI 辅助生成的代码')
    ap.add_argument('--scenario', choices=['demo'])
    return ap


def main():
    a = parser().parse_args()
    if a.scenario == 'demo' or not (a.title or a.git):
        demo()
        return 0
    bad = 0
    if a.title is not None:
        for repo, fn in (('vllm', vllm_title), ('sglang', sglang_title)):
            if a.repo in (repo, 'both'):
                ok, why = fn(a.title)
                bad += not ok
                print(f'{repo:<7}{"PASS" if ok else "FAIL"}  {why}')
    if a.git:
        issues, tips, n = from_git(a.git, a.ai)
        print(f'{n} 个 commit')
        for i in issues:
            print('FAIL  ' + i)
        for t in tips:
            print('TIP   ' + t)
        bad += len(issues)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
