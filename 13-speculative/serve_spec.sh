#!/usr/bin/env bash
# VLLM-13 GPU 实验：同一个目标模型，换投机方法 / k / 验证规则起服务，看接受长度、per-position 曲线和吞吐。
#
# 用法：
#   bash serve_spec.sh base               # 不开投机的基线
#   bash serve_spec.sh eagle3 [K]         # EAGLE-3，K 默认 3
#   bash serve_spec.sh ngram [K]          # n-gram（prompt lookup），K 默认 5，prompt_lookup_min 2、max 4
#   bash serve_spec.sh suffix             # SuffixDecoding（需要 pip install arctic-inference）
#   bash serve_spec.sh draft [K]          # 小草稿模型（DRAFT_SMALL，默认 Llama-3.2-1B-Instruct）
#   bash serve_spec.sh dynamic            # EAGLE-3 k=5 + num_speculative_tokens_per_batch_size 按并发降 k
#   bash serve_spec.sh prob               # EAGLE-3 + draft_sample_method=probabilistic
#   bash serve_spec.sh block              # EAGLE-3 + rejection_sample_method=block（强制 MRV2）
#   bash serve_spec.sh fly                # EAGLE-3 k=5 + FLy（有损，日志有 WARNING）
#   bash serve_spec.sh synthetic          # 不看草稿质量，按接受长度 2.5 伪造接受（容量规划用，输出不对）
#   bash serve_spec.sh bench [C]          # 对起好的服务压测，C 并发（默认 1；建议 1/16/64/256 各跑一次）
#   bash serve_spec.sh sgl_eagle3|sgl_adaptive|sgl_ngram|sgl_block   # SGLang 对照（需要另装 sglang）
#
# 环境变量：MODEL（默认 meta-llama/Llama-3.1-8B-Instruct）、DRAFT（EAGLE-3 草稿头）、DRAFT_SMALL、OUT、PORT、GMU。
# 日志里要找的几行（行号对应 vllm 0d6f92a / sglang 8fabf5e）：
#   "SpecDecoding metrics:"                       v1/spec_decode/metrics.py:121（每 10 秒一行：接受长度、per-position、接受率）
#   "Overriding cudagraph_mode from %s to"         config/vllm.py:1096（MRV1 + 动态 k：FULL 降成 PIECEWISE）
#   "dynamic speculative decoding"                 v1/spec_decode/dynamic/utils.py:19（DP>1 时自动关掉动态 k）
#   "FLy verification is a lossy"                  config/speculative.py:2115
#   "Adaptive spec params updated: steps"          sglang srt/speculative/adaptive_spec_params.py:268
#   "Max running requests is reset to 48"          sglang srt/speculative/speculative_hook.py（没写 --max-running-requests 时）
set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
DRAFT=${DRAFT:-RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3}
DRAFT_SMALL=${DRAFT_SMALL:-meta-llama/Llama-3.2-1B-Instruct}
OUT=${OUT:-results}
PORT=${PORT:-8000}
GMU=${GMU:-0.90}
PATTERN='SpecDecoding metrics|Overriding cudagraph_mode|speculative decoding|lossy|KV cache size|Maximum concurrency|cudagraph sizes|Model Runner'

mkdir -p "$OUT"

serve() {
  local tag=$1; shift
  echo "== vllm serve $MODEL $*"
  vllm serve "$MODEL" --port "$PORT" --gpu-memory-utilization "$GMU" "$@" 2>&1 | \
    tee "$OUT/$tag.log" | grep -E --line-buffered "$PATTERN" || true
}

sgl() {
  local tag=$1; shift
  echo "== sglang.launch_server --model-path $MODEL $*"
  python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" --mem-fraction-static "$GMU" "$@" 2>&1 | \
    tee "$OUT/$tag.log" | grep -E --line-buffered 'accept len|Adaptive spec|reset to 48|max_running_requests|KV Cache' || true
}

E3() { echo "{\"method\":\"eagle3\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":${1:-3}${2:-}}"; }

case "${1:-}" in
  base)      serve base ;;
  eagle3)    serve "eagle3_k${2:-3}" -sc "$(E3 "${2:-3}")" ;;
  ngram)     serve "ngram_k${2:-5}" -sc "{\"method\":\"ngram\",\"num_speculative_tokens\":${2:-5},\"prompt_lookup_min\":2,\"prompt_lookup_max\":4}" ;;
  suffix)    serve suffix -sc '{"method":"suffix","num_speculative_tokens":16}' ;;
  draft)     serve "draft_k${2:-4}" -sc "{\"method\":\"draft_model\",\"model\":\"$DRAFT_SMALL\",\"num_speculative_tokens\":${2:-4}}" ;;
  dynamic)   serve dynamic -sc "$(E3 5 ',"num_speculative_tokens_per_batch_size":[[1,16,5],[17,64,3],[65,128,1],[129,1024,0]]')" ;;
  prob)      serve prob -sc "$(E3 3 ',"draft_sample_method":"probabilistic"')" ;;
  block)     VLLM_USE_V2_MODEL_RUNNER=1 serve block -sc "$(E3 3 ',"rejection_sample_method":"block"')" ;;
  fly)       serve fly -sc "$(E3 5 ',"rejection_sample_method":"fly"')" ;;
  synthetic) serve synthetic -sc "$(E3 5 ',"rejection_sample_method":"synthetic","synthetic_acceptance_length":2.5')" ;;
  bench)
    C=${2:-1}
    vllm bench serve --model "$MODEL" --port "$PORT" --dataset-name sharegpt \
      --num-prompts $((C * 8 > 64 ? C * 8 : 64)) --max-concurrency "$C" --temperature 0 \
      --save-result --result-dir "$OUT" 2>&1 | tee -a "$OUT/bench_c$C.log" | \
      grep -E "Output token throughput|Mean TPOT|Median TPOT|Mean TTFT|acceptance" || true ;;
  sgl_eagle3)   sgl sgl_eagle3   --speculative-algorithm EAGLE3 --speculative-draft-model-path "$DRAFT" \
                  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 ;;
  sgl_adaptive) sgl sgl_adaptive --speculative-algorithm EAGLE3 --speculative-draft-model-path "$DRAFT" \
                  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-adaptive ;;
  sgl_ngram)    sgl sgl_ngram    --speculative-algorithm NGRAM ;;
  sgl_block)    sgl sgl_block    --speculative-algorithm EAGLE3 --speculative-draft-model-path "$DRAFT" \
                  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
                  --speculative-use-rejection-sampling --speculative-use-block-verification ;;
  *) sed -n '2,17p' "$0"; exit 1 ;;
esac
