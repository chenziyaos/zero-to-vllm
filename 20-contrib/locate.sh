#!/usr/bin/env bash
# 20 章配套：从一行日志、一个参数名、一个行为变化，定位到 vLLM / SGLang 的源码和引入它的 commit。
# 每段相互独立，在 vllm 或 sglang 仓库根目录下执行。版本：vLLM 0d6f92a，SGLang 8fabf5e。
# 依赖：git、rg（ripgrep）；log 段需要能起 vllm serve。
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
KW=${2:-}

case "${1:-help}" in

# ---------------------------------------------------------------- 1. 日志 → 源码
log-debug)
  # 默认日志里只有文件名 [scheduler.py:123]；DEBUG 级别时 formatter 换成相对路径
  # （vllm/logging_utils/formatter.py:18-70：去掉开头的 vllm/，中间层折叠成 ...），同名文件就分得清了
  VLLM_LOGGING_LEVEL=DEBUG vllm serve "$MODEL" 2>&1 | tee serve.log | grep -m 20 -E '\[(v1|core|engine|plugins)/'
  ;;

log-grep)
  # 拿日志正文里一段不含变量的原文去搜。日志用 %s 占位，搜的时候去掉变量部分
  # 例：bash locate.sh log-grep "Only one platform plugin can be activated"
  rg -n --fixed-strings "${KW:?给一段日志原文}" vllm/ python/ 2>/dev/null || true
  ;;

log-proc)
  # 日志前缀 (EngineCore pid=…) / (Worker_TP1 pid=…) 告诉你是哪个进程打的；
  # 同一行出现 7 次通常不是 bug，是 7 个进程各打一次（见 sim_plugins.py --scenario procs）
  grep -oE '^\(?[A-Za-z_]+[A-Za-z0-9_]* pid=[0-9]+\)?' serve.log | sort | uniq -c
  ;;

# ---------------------------------------------------------------- 2. 参数名 → 定义处
arg-help)
  # vllm serve --help=<关键字>：按参数组名或参数名子串过滤（vllm/utils/argparse_utils.py:240-279）
  # 例：--help=max-num-seqs、--help=SchedulerConfig、--help=all
  vllm serve --help="${KW:-max-num-seqs}"
  ;;

arg-src)
  # CLI 参数大多由 dataclass 字段自动生成：--max-num-seqs ↔ SchedulerConfig.max_num_seqs
  # 先找 config 字段与 docstring，再找 arg_utils.py 里的特殊处理
  f=$(echo "${KW:-max-num-seqs}" | tr '-' '_')
  rg -n "^\s+${f}\s*:" vllm/config/
  rg -n "${f}" vllm/engine/arg_utils.py | head -20
  ;;

env-src)
  # 环境变量全部集中在 vllm/envs.py：类型声明 + lambda 解析 + 注释即文档
  rg -n "\"${KW:-VLLM_PLUGINS}\"" vllm/envs.py
  # SGLang 的环境变量在 python/sglang/srt/environ.py（EnvStr / EnvBool …）
  rg -n "${KW:-SGLANG_PLUGINS}" python/sglang/srt/environ.py 2>/dev/null || true
  ;;

# ---------------------------------------------------------------- 3. 行为变化 → commit
pickaxe)
  # git log -S：某个字符串出现次数发生变化的 commit，找“这个参数 / 这句日志是哪次加的、哪次删的”
  git log --oneline -S "${KW:-VLLM_PLUGINS}" -- vllm/ | head -20
  ;;

pickaxe-regex)
  # -G 按正则匹配 diff 行，适合找某个函数签名被改过的 commit
  git log --oneline -G "${KW:-def load_general_plugins}" -- vllm/ | head -20
  ;;

blame)
  # 带 -w 忽略空白、-C 跟踪跨文件移动，否则会停在“格式化 / 搬文件”那个 commit 上
  git blame -w -C -C -L "/def load_general_plugins/,+15" vllm/plugins/__init__.py
  ;;

pr-of)
  # 从 commit 找 PR：vllm 是 squash merge，标题末尾带 (#12345)
  git log -1 --format='%h %s' "${KW:-HEAD}"
  ;;

# ---------------------------------------------------------------- 4. 改动 → 测试和 CI
tests-for)
  # 改了某个模块，谁在测它：先按 import 搜，再看 .buildkite/test_areas/*.yaml 里谁依赖它
  mod=${KW:-vllm/plugins}
  rg -l "$(echo "${mod%.py}" | tr '/' '.')" tests/ | head -20
  rg -n -- "- ${mod}" .buildkite/test_areas/ | head -20
  ;;

ci-steps)
  # 离线算一组改动会触发哪些 Buildkite step（ci_select.py 复刻 ci-infra 的选择逻辑）
  python3 "$(dirname "$0")/ci_select.py" --diff $(git diff --name-only origin/main...HEAD)
  ;;

help|*)
  sed -n '2,4p' "$0"
  grep -E '^[a-z-]+\)' "$0" | tr -d ')' | sed 's/^/  /'
  ;;
esac
