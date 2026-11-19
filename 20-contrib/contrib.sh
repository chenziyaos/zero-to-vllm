#!/usr/bin/env bash
# 20 章配套：从 fork 到合入的命令清单。每段相互独立，按需复制，不要整份直接跑。
# 版本：vLLM 0d6f92a（docs/contributing/README.md、docs/contributing/incremental_build.md、
#       docs/benchmarking/cli.md），SGLang 8fabf5e（docs/developer_guide/contribution_guide.mdx、test/README.md）
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
HERE="$(cd "$(dirname "$0")" && pwd)"

case "${1:-help}" in

# ---------------------------------------------------------------- 环境
vllm-dev)
  # 只改 Python：用预编译 wheel 里的 kernel，不本地编译 CUDA（README.md:43-44）
  git clone https://github.com/<you>/vllm.git && cd vllm
  git remote add upstream https://github.com/vllm-project/vllm.git
  VLLM_USE_PRECOMPILED=1 uv pip install -e .
  vllm download-kernels
  uv pip install "pre-commit>=4.5.1" && pre-commit install      # README.md:84-91
  ;;

vllm-dev-kernel)
  # 改 csrc/：增量编译见 docs/contributing/incremental_build.md；改 csrc/ 的 PR 在 CI 上会触发 run_all
  uv pip install -e . --no-build-isolation
  ;;

sglang-dev)
  git clone https://github.com/<you>/sglang.git && cd sglang
  pip install -e "python"
  pip3 install pre-commit && pre-commit install && pre-commit run --all-files
  ;;

# ---------------------------------------------------------------- 本地验证
vllm-test)
  # 只跑和改动相关的测试；tests/ 全量在单机上跑不完
  pre-commit run                       # 只查暂存的文件；-a 查全部
  pytest -s -v tests/plugins_tests/test_platform_plugins.py
  pytest -s -v tests/plugins_tests/ -k endpoint
  ;;

sglang-test)
  # CPU 单测套件，和 PR CI 的 base-a-test-cpu 是同一组
  python3 test/run_suite.py --hw cpu --suite base-a-test-cpu
  python3 test/registered/core/test_srt_endpoint.py        # 注册测试都能直接当脚本跑
  ;;

bench-ab)
  # 改前改后各跑一次。复跑同一服务会命中上一轮留下的前缀缓存（docs/benchmarking/cli.md:156-161）：
  # 要么换 --seed，要么开 dev mode 后重置缓存，要么重启服务
  VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" &
  for tag in base patch; do
    curl -s -X POST localhost:8000/reset_prefix_cache
    vllm bench serve --model "$MODEL" --dataset-name random --random-input-len 1024 --random-output-len 128 \
      --num-prompts 1000 --max-concurrency 256 --seed 0 --save-result --result-filename "bench_${tag}.json"
  done
  ;;

sglang-bench)
  # SGLang 默认 --seed 42，只在 SGLANG_IS_IN_CI 或 --flush-cache 时 flush 缓存
  python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input-len 1024 \
    --random-output-len 128 --num-prompts 1000 --max-concurrency 256 --flush-cache
  ;;

# ---------------------------------------------------------------- 提交
commit)
  # DCO：每个 commit 都要有 Signed-off-by（README.md:190-196）；AI 辅助要加 trailer 并在 PR 描述里写明（206-228）
  git commit -s -m "[Bugfix][Plugins] Strip whitespace in VLLM_PLUGINS"
  # git commit -s --trailer "Co-authored-by: <AI 工具名>"
  git rebase --signoff origin/main     # 补签已有的 commit
  python3 "$HERE/pr_lint.py" --title "[Bugfix][Plugins] Strip whitespace in VLLM_PLUGINS" --repo vllm
  python3 "$HERE/pr_lint.py" --git origin/main..HEAD
  ;;

# ---------------------------------------------------------------- 触发 CI（PR 评论里发）
vllm-ci)
  cat <<'EOF'
  /ci run             还没 ready 时先跑一轮上游 CI（README.md:345-350）
  /amd-ci run         只跑 AMD
  /ci retry           重跑失败的 job
  /ci cancel          取消
  打 ready 标签        标题检查和默认流水线才会启动；ready-run-all-tests 标签连 optional step 一起跑
EOF
  ;;

sglang-ci)
  cat <<'EOF'
  /tag-and-rerun-ci        新 PR 第一次跑：加 run-ci 标签并在当前 commit 上触发（只加标签不会跑当前 commit）
  /tag-run-ci-label extra  加 run-ci + run-ci-extra，只对之后的 commit 生效
  /run-cpu-ci              只跑 CPU 单测（base-a-test-cpu）
  /rerun-failed-ci         重跑失败 / 跳过 / 取消 / 超时的 workflow，作者本人可用
  /rerun-test -c           重跑本 PR 新增或改动的测试文件
  /rerun-group <group>     按 scripts/ci/rerun_test_groups.json 里的组重跑
EOF
  ;;

help|*)
  sed -n '2,4p' "$0"
  grep -E '^[a-z-]+\)' "$0" | tr -d ')' | sed 's/^/  /'
  ;;
esac
