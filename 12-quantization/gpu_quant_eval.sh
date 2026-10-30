#!/usr/bin/env bash
# VLLM-12 GPU 实验：同一个模型换量化格式起服务，看 kernel 选择日志、KV 容量、吞吐和精度。
#
# 用法：
#   bash gpu_quant_eval.sh bf16            # 基线：BF16 权重 + BF16 KV
#   bash gpu_quant_eval.sh fp8             # 在线转 FP8（--quantization fp8_per_tensor；fp8 已不做在线量化）
#   bash gpu_quant_eval.sh fp8block        # 在线转 FP8 block 128
#   bash gpu_quant_eval.sh awq|gptq        # 读 AWQ / GPTQ checkpoint（AWQ_MODEL / GPTQ_MODEL）
#   bash gpu_quant_eval.sh nvfp4           # 读 ModelOpt NVFP4 checkpoint（NVFP4_MODEL，sm100 最快，sm90 落 Humming / Marlin）
#   bash gpu_quant_eval.sh kvfp8           # BF16 权重 + --kv-cache-dtype fp8（无 scale 时用 1.0，日志有 WARNING）
#   bash gpu_quant_eval.sh kvtq            # BF16 权重 + --kv-cache-dtype turboquant_k8v4
#   bash gpu_quant_eval.sh backend X       # 在线 FP8 + --linear-backend X，看回退 WARNING
#   bash gpu_quant_eval.sh bench           # 对起好的服务压测（随机 1024/256，并发 64）
#   bash gpu_quant_eval.sh gsm8k           # lm_eval gsm8k 5-shot 对起好的服务测精度（需 pip install lm_eval）
#   bash gpu_quant_eval.sh sgl_fp8|sgl_awq|sgl_kvfp8    # SGLang 对照（需要另装 sglang）
#
# 环境变量：MODEL（默认 Qwen/Qwen2.5-7B-Instruct）、AWQ_MODEL、GPTQ_MODEL、NVFP4_MODEL、OUT、PORT、GMU。
# 日志里要找的几行（行号对应 vllm 0d6f92a）：
#   "Using %s for ..."                             kernels/linear/__init__.py（每种线性层第一次选中 kernel 时打一行）
#   "Using %s for AutoAWQMarlinLinearMethod"       layers/quantization/auto_awq.py:451
#   "Applied linear backend override for %r: %r"   kernels/linear/__init__.py:256
#   "has no kernel for this linear layer type"     kernels/linear/__init__.py:393（--linear-backend 无匹配，回退 auto）
#   "no longer supports online quantization"       layers/quantization/fp8.py:106
#   "is not supported for the current GPU. Minimum capability"   config/vllm.py:945
#   "is not divisible by group_size" / "block_k"   mixed_precision/marlin.py:58、fp8_utils.py:1296
#   "GPU KV cache size: N tokens"                  v1/core/kv_cache_utils.py:1148
#   "Using KV cache scaling factor 1.0"            layers/quantization/kv_cache.py:159（FP8 KV 没有校准 scale）
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
AWQ_MODEL=${AWQ_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}
GPTQ_MODEL=${GPTQ_MODEL:-Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4}
NVFP4_MODEL=${NVFP4_MODEL:-nvidia/Llama-3.1-8B-Instruct-FP4}
OUT=${OUT:-results}
PORT=${PORT:-8000}
GMU=${GMU:-0.90}
PATTERN='Using .* for|linear backend|has no kernel|online quantization|Minimum capability|divisible|KV cache size|Maximum concurrency|kv_cache_dtype|KV cache scaling factor|quantization='

mkdir -p "$OUT"

serve() {
  local tag=$1 model=$2; shift 2
  echo "== vllm serve $model $*"
  vllm serve "$model" --port "$PORT" --gpu-memory-utilization "$GMU" "$@" 2>&1 | \
    tee "$OUT/$tag.log" | grep -E --line-buffered "$PATTERN" || true
}

sgl() {
  local tag=$1 model=$2; shift 2
  echo "== sglang.launch_server --model-path $model $*"
  python -m sglang.launch_server --model-path "$model" --port "$PORT" --mem-fraction-static "$GMU" "$@" 2>&1 | \
    tee "$OUT/$tag.log" | grep -E --line-buffered 'KV Cache|max_total_num_tokens|quant|gemm|backend' || true
}

case "${1:-}" in
  bf16)     serve bf16     "$MODEL" ;;
  fp8)      serve fp8      "$MODEL" --quantization fp8_per_tensor ;;
  fp8block) serve fp8block "$MODEL" --quantization fp8_per_block ;;
  awq)      serve awq      "$AWQ_MODEL" ;;
  gptq)     serve gptq     "$GPTQ_MODEL" ;;
  nvfp4)    serve nvfp4    "$NVFP4_MODEL" ;;
  kvfp8)    serve kvfp8    "$MODEL" --kv-cache-dtype fp8 ;;
  kvtq)     serve kvtq     "$MODEL" --kv-cache-dtype turboquant_k8v4 ;;
  backend)  serve "backend_${2:?backend name}" "$MODEL" --quantization fp8_per_tensor --linear-backend "$2" ;;
  bench)
    vllm bench serve --model "$MODEL" --port "$PORT" --dataset-name random \
      --random-input-len 1024 --random-output-len 256 --num-prompts 1024 --max-concurrency 64 \
      --save-result --result-dir "$OUT" 2>&1 | tee -a "$OUT/bench.log" | \
      grep -E "Output token throughput|Mean TTFT|Mean TPOT|P99" || true ;;
  gsm8k)
    lm_eval --model local-completions \
      --model_args "model=$MODEL,base_url=http://127.0.0.1:$PORT/v1/completions,num_concurrent=32" \
      --tasks gsm8k --num_fewshot 5 2>&1 | tee -a "$OUT/gsm8k.log" | grep -E "exact_match|gsm8k" || true ;;
  sgl_fp8)   sgl sgl_fp8   "$MODEL" --quantization fp8 ;;
  sgl_awq)   sgl sgl_awq   "$AWQ_MODEL" --quantization awq_marlin ;;
  sgl_kvfp8) sgl sgl_kvfp8 "$MODEL" --kv-cache-dtype fp8_e4m3 ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
