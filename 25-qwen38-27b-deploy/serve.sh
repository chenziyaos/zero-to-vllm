#!/usr/bin/env bash
# VLLM-25 实战：Qwen3.8-27B 按卡起服务，日志落到 results/<tier>.log
# 模型是 Gated DeltaNet 混合注意力（48 GDN + 16 全注意力），checkpoint 自带 MTP 草稿头。
# 用法：
#   bash serve.sh 5090-nvfp4       # 1×RTX 5090 Inferact NVFP4，eager，32K（上游 recipe 有记录）
#   bash serve.sh 5090x2-fp8       # 2×RTX 5090 TP2 官方 FP8，262K，recipe 示例 MTP k=3
#   bash serve.sh 5090x2-nvfp4     # 2×RTX 5090 TP2 Inferact NVFP4，262K，MTP k=3
#   bash serve.sh h100-fp8         # 1×H100 官方 FP8，32K，主力档
#   bash serve.sh h100-int4        # 1×H100 RedHatAI INT4 W4A16（Marlin），32K
#   bash serve.sh b200-mxfp4       # 1×B200 RedHatAI MXFP4 真 W4A4，262K
#   bash serve.sh h200-bf16        # 1×H200 官方 BF16，262K，长上下文
#   bash serve.sh text-only        # 1×5090 NVFP4 加 --language-model-only，只跑文本
#   bash serve.sh no-mtp           # 2×5090 TP2 FP8，显式关掉 MTP 做对照
#   bash serve.sh sgl              # 只打印 SGLang BF16（h200-bf16）对照命令，不启动
#   bash serve.sh plan <tier>      # 不起服务，只打 deploy_planner 的显存账
# 环境变量：MODEL 覆盖模型名，PORT（默认 8000），OUT（默认 results），GMU（默认 0.90）
# 日志里要找的几行（行号对应 vllm 0d6f92a）：
#   "Model loading took"                    v1/worker/gpu_model_runner.py:5089
#   "Available KV cache memory"             v1/worker/gpu_worker.py:776
#   "GPU KV cache size" / "Maximum concurrency"  v1/core/kv_cache_utils.py:1147–1149
#   "Graph capturing finished"              v1/worker/gpu_model_runner.py:6517
#   "Setting attention block size"          platforms/interface.py:1114（混合层块大小，TP2 FP8 约 1600）
#   "does not have native support for FP4"  model_executor/kernels/linear/nvfp4/marlin.py:35
#   "spec_decode_num_accepted_tokens_total" /metrics（接受率 = 两快照 Δaccepted/Δdraft）
#   "CUDA out of memory"                    32G 单卡抓 CUDA graph 可能 OOM，所以 5090 单卡档默认 eager
set -euo pipefail

PORT=${PORT:-8000}
OUT=${OUT:-results}
GMU=${GMU:-0.90}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

# Qwen3.8 工具调用 parser 是 qwen3_coder（不是 Qwen3-32B 的 hermes）
COMMON=(--served-model-name qwen38 --max-num-seqs 8 --gpu-memory-utilization "$GMU"
        --kv-cache-dtype fp8 --host 127.0.0.1 --port "$PORT"
        --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)

MTP3=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')

serve() {
  local tag=$1 model=$2; shift 2
  echo "== vllm serve ${MODEL:-$model} $* -> $OUT/$tag.log"
  # pipefail 保留服务进程的失败码；完整日志不经筛选，避免漏掉异常栈。
  vllm serve "${MODEL:-$model}" "$@" 2>&1 | tee "$OUT/$tag.log"
}

plan_for() {
  local base=(--scenario mem --gmu "$GMU" --max-num-seqs 8 --kv-dtype fp8)
  case "$1" in
    5090-nvfp4)   python3 "$HERE/deploy_planner.py" "${base[@]}" --gpu rtx5090 --fmt nvfp4 --eager --max-model-len 32768 ;;
    5090x2-fp8)   python3 "$HERE/deploy_planner.py" "${base[@]}" --gpu rtx5090 --n 2 --tp 2 --fmt fp8 --spec 3 ;;
    5090x2-nvfp4) python3 "$HERE/deploy_planner.py" "${base[@]}" --gpu rtx5090 --n 2 --tp 2 --fmt nvfp4 --spec 3 ;;
    h100-fp8)     python3 "$HERE/deploy_planner.py" "${base[@]}" --gpu h100 --fmt fp8 --max-model-len 32768 ;;
    h100-int4)    python3 "$HERE/deploy_planner.py" "${base[@]}" --gpu h100 --fmt int4 --max-model-len 32768 ;;
    b200-mxfp4)   python3 "$HERE/deploy_planner.py" "${base[@]}" --gpu b200 --fmt mxfp4 ;;
    h200-bf16)    python3 "$HERE/deploy_planner.py" "${base[@]}" --gpu h200 --fmt bf16 ;;
    *) echo "未知档位：$1" >&2; exit 2 ;;
  esac
}

case "${1:-}" in
  plan)
    plan_for "${2:?用法：bash serve.sh plan <档位>}" ;;
  5090-nvfp4)
    serve 5090-nvfp4 Inferact/Qwen3.8-27B-NVFP4 "${COMMON[@]}" \
      --tensor-parallel-size 1 --max-model-len 32768 --enforce-eager ;;
  5090x2-fp8)
    serve 5090x2-fp8 Qwen/Qwen3.8-27B-FP8 "${COMMON[@]}" \
      --tensor-parallel-size 2 --max-model-len 262144 "${MTP3[@]}" ;;
  5090x2-nvfp4)
    serve 5090x2-nvfp4 Inferact/Qwen3.8-27B-NVFP4 "${COMMON[@]}" \
      --tensor-parallel-size 2 --max-model-len 262144 "${MTP3[@]}" ;;
  h100-fp8)
    serve h100-fp8 Qwen/Qwen3.8-27B-FP8 "${COMMON[@]}" \
      --tensor-parallel-size 1 --max-model-len 32768 ;;
  h100-int4)
    serve h100-int4 RedHatAI/Qwen3.8-27B-INT4 "${COMMON[@]}" \
      --tensor-parallel-size 1 --max-model-len 32768 ;;
  b200-mxfp4)
    serve b200-mxfp4 RedHatAI/Qwen3.8-27B-MXFP4 "${COMMON[@]}" \
      --tensor-parallel-size 1 --max-model-len 262144 ;;
  h200-bf16)
    serve h200-bf16 Qwen/Qwen3.8-27B "${COMMON[@]}" \
      --tensor-parallel-size 1 --max-model-len 262144 ;;
  text-only)
    serve 5090-textonly Inferact/Qwen3.8-27B-NVFP4 "${COMMON[@]}" \
      --tensor-parallel-size 1 --max-model-len 32768 --enforce-eager --language-model-only ;;
  no-mtp)
    serve 5090x2-fp8-nomtp Qwen/Qwen3.8-27B-FP8 "${COMMON[@]}" \
      --tensor-parallel-size 2 --max-model-len 262144 ;;
  sgl)
    echo "== SGLang 只提供 h200-bf16 的对照命令；打印后需自行核对版本支持再执行"
    python3 "$HERE/commands.py" --profile h200-bf16 --engine sglang
    ;;
  *)
    grep -E '^#   bash serve.sh' "$0" | sed 's/^#   //'
    exit 2 ;;
esac
