#!/usr/bin/env bash
# VLLM-24 实战：Qwen3-32B 按卡起服务，日志落到 results/<tier>.log
# 用法：
#   bash serve.sh h100-bf16-tp2  # 2×H100 BF16 TP2，recipe 第 198 行的命令
#   bash serve.sh h100-fp8       # 1×H100 Qwen3-32B-FP8，数据中心首选
#   bash serve.sh h20-fp8        # 1×H20 96G FP8，KV 池比 H100 多 15 GiB
#   bash serve.sh l40s-fp8       # 1×L40S FP8 权重 + fp8 KV（不加 fp8 KV 装不下一条 40K）
#   bash serve.sh 5090-awq       # 1×RTX 5090 AWQ + fp8 KV，max-num-seqs 32
#   bash serve.sh 4090-awq       # 1×RTX 4090 AWQ + fp8 KV，上下文砍到 16K，max-num-seqs 8，gmu 0.95
#   bash serve.sh 3090x2-awq     # 2×RTX 3090 TP2 AWQ，CUDA graph 只抓 [1,2,4,8]（#40121 的绕法）
#   bash serve.sh eagle3         # 1×H100 FP8 + EAGLE3 k=3（recipe 49–53）
#   bash serve.sh qwen35-27b     # 1×H100 Qwen3.5-27B-FP8，262K 上下文（Qwen3.5 recipe 131–137）
#   bash serve.sh sgl            # 同一张 H100 换 SGLang 起 FP8，对照 KV 池
#   bash serve.sh plan <tier>    # 不起服务，只打 deploy_planner 的显存账
# 环境变量：MODEL 覆盖模型名，PORT（默认 8000），OUT（默认 results），GMU 覆盖 gpu-memory-utilization
# 日志里要找的几行（行号对应 vllm 0d6f92a）：
#   "Model loading took"                    v1/worker/gpu_model_runner.py:5089
#   "Available KV cache memory"             v1/worker/gpu_worker.py:776
#   "GPU KV cache size" / "Maximum concurrency"  v1/core/kv_cache_utils.py:1147–1149
#   "Graph capturing finished"              v1/worker/gpu_model_runner.py:6517
#   "Selected ... for"                      model_executor/kernels/linear/__init__.py:779
#   "Falling back to unoptimized AWQ"       model_executor/layers/quantization/auto_awq.py:327
#   "Using KV cache scaling factor 1.0"     model_executor/layers/quantization/kv_cache.py:159
#   "does not have native support for FP4"  model_executor/kernels/linear/nvfp4/marlin.py:35
#   "Setting attention block size"          platforms/interface.py:1114（Qwen3.5 混合层）
#   "CUDA out of memory ... sampler"        v1/worker/gpu_model_runner.py:5930
set -euo pipefail

PORT=${PORT:-8000}
OUT=${OUT:-results}
HERE=$(cd "$(dirname "$0")" && pwd)
PATTERN='Model loading took|Available KV cache memory|KV cache size|Maximum concurrency|Graph capturing finished|Selected |Using .* for AutoAWQ|Falling back to unoptimized AWQ|KV cache scaling factor|native support for FP4|Setting attention block size|unevenly partitioned|out of memory|Error'
mkdir -p "$OUT"

# Qwen3-32B recipe 30–40：思考内容拆到 reasoning_content，工具调用走 hermes
COMMON=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes)

serve() {
  local tag=$1 model=$2 gmu=$3; shift 3
  echo "== vllm serve ${MODEL:-$model} $*  (gmu ${GMU:-$gmu}) -> $OUT/$tag.log"
  vllm serve "${MODEL:-$model}" --port "$PORT" --gpu-memory-utilization "${GMU:-$gmu}" \
    "$@" 2>&1 | tee "$OUT/$tag.log" | grep -E --line-buffered "$PATTERN" || true
}

case "${1:-}" in
  h100-bf16-tp2)
    serve h100-bf16-tp2 Qwen/Qwen3-32B 0.92 "${COMMON[@]}" --tensor-parallel-size 2 ;;
  h100-fp8)
    serve h100-fp8 Qwen/Qwen3-32B-FP8 0.92 "${COMMON[@]}" ;;
  h20-fp8)
    serve h20-fp8 Qwen/Qwen3-32B-FP8 0.92 "${COMMON[@]}" ;;
  l40s-fp8)
    serve l40s-fp8 Qwen/Qwen3-32B-FP8 0.92 "${COMMON[@]}" \
      --kv-cache-dtype fp8 --max-num-seqs 64 --max-num-batched-tokens 4096 ;;
  5090-awq)
    serve 5090-awq Qwen/Qwen3-32B-AWQ 0.92 "${COMMON[@]}" --quantization awq \
      --kv-cache-dtype fp8 --max-num-seqs 32 --max-num-batched-tokens 4096 ;;
  4090-awq)
    serve 4090-awq Qwen/Qwen3-32B-AWQ 0.95 "${COMMON[@]}" --quantization awq \
      --kv-cache-dtype fp8 --max-model-len 16384 --max-num-seqs 8 --max-num-batched-tokens 2048 ;;
  3090x2-awq)
    serve 3090x2-awq Qwen/Qwen3-32B-AWQ 0.75 "${COMMON[@]}" --quantization awq \
      --tensor-parallel-size 2 --max-num-seqs 32 \
      --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}' ;;
  eagle3)
    serve eagle3 Qwen/Qwen3-32B-FP8 0.92 "${COMMON[@]}" --max-num-seqs 32 \
      --speculative-config '{"model": "RedHatAI/Qwen3-32B-speculator.eagle3", "num_speculative_tokens": 3, "method": "eagle3"}' ;;
  qwen35-27b)
    # Qwen3.5 recipe 33–43：工具调用换 qwen3_coder；--language-model-only 省掉 0.86 GiB 视觉塔
    serve qwen35-27b Qwen/Qwen3.5-27B-FP8 0.92 --max-model-len 262144 --language-model-only \
      --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder ;;
  sgl)
    echo "== sglang ${MODEL:-Qwen/Qwen3-32B-FP8} -> $OUT/sgl.log"
    python -m sglang.launch_server --model-path "${MODEL:-Qwen/Qwen3-32B-FP8}" --port "$PORT" \
      --mem-fraction-static "${GMU:-0.90}" --reasoning-parser qwen3 --tool-call-parser qwen25 \
      2>&1 | tee "$OUT/sgl.log" | grep -E --line-buffered 'KV Cache|max_total_num_tokens|backend|Error' || true ;;
  plan)
    P="python3 $HERE/deploy_planner.py --scenario mem"
    case "${2:-}" in
      h100-bf16-tp2) $P --gpu h100 --n 2 --tp 2 --fmt bf16 ;;
      h100-fp8)      $P --gpu h100 --fmt fp8 ;;
      h20-fp8)       $P --gpu h20 --fmt fp8 ;;
      l40s-fp8)      $P --gpu l40s --fmt fp8 --kv-dtype fp8 --max-num-seqs 64 --batched 4096 ;;
      5090-awq)      $P --gpu rtx5090 --fmt awq --kv-dtype fp8 --max-num-seqs 32 --batched 4096 ;;
      4090-awq)      $P --gpu rtx4090 --fmt awq --kv-dtype fp8 --gmu 0.95 --max-model-len 16384 --max-num-seqs 8 --batched 2048 ;;
      3090x2-awq)    $P --gpu rtx3090 --n 2 --tp 2 --fmt awq --gmu 0.75 --max-num-seqs 32 --batched 8192 --capture 8 ;;
      qwen35-27b)    python3 "$HERE/deploy_planner.py" --scenario qwen35 ;;
      eagle3)        python3 "$HERE/deploy_planner.py" --scenario spec ;;
      *) echo "plan 支持：h100-bf16-tp2 h100-fp8 h20-fp8 l40s-fp8 5090-awq 4090-awq 3090x2-awq qwen35-27b eagle3"; exit 1 ;;
    esac ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac
