#!/usr/bin/env bash
# VLLM-16 GPU 实验：结构化输出（后端、投机、推理模型）和多 LoRA（max_loras、预分配、调度排队）。单卡即可。
#
# 用法：
#   bash serve_so_lora.sh so [BACKEND]          # 起结构化输出实例，BACKEND = auto（默认）/ xgrammar / guidance / outlines
#   bash serve_so_lora.sh so_spec [K]           # 同上 + ngram 投机 K 步（默认 3），看 k+1 行 bitmask 的代价
#   bash serve_so_lora.sh so_think              # Qwen3-8B + reasoning_parser qwen3：推理段不受约束，</think> 后才上 mask
#   bash serve_so_lora.sh so_bench [DATASET]    # benchmark_serving_structured_output.py：json / json-unique / grammar / regex / choice
#   bash serve_so_lora.sh so_ratio              # structured-output-ratio 0 / 0.5 / 1 三档，对照 sim overlap
#   bash serve_so_lora.sh auto_lock             # 第一个请求带 multipleOf，第二个普通 schema：看 auto 锁后端报错
#   bash serve_so_lora.sh lora [MAX_LORAS] [RANK]   # 起多 LoRA 实例（默认 max_loras=1、max_lora_rank=16）
#   bash serve_so_lora.sh lora_mem              # max_lora_rank 16 / 64 / 256 各起一次，抓 KV cache 可用显存
#   bash serve_so_lora.sh lora_bench [N]        # vllm bench serve，--lora-modules 随机 N 个 adapter（默认 8）
#   bash serve_so_lora.sh lora_hot NAME PATH    # 运行时加载 / 卸载 adapter（需 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1）
#   bash serve_so_lora.sh sgl_so | sgl_lora     # SGLang 对照：--grammar-backend xgrammar；--max-loras-per-batch 8 + drainer
#
# 环境变量：MODEL（默认 Qwen/Qwen3-8B）、LORA_DIR（放 adapter 的目录，每个子目录一个 adapter，名字即目录名）、
#           OUT、GMU、MAXLEN、VLLM_SRC（benchmark 脚本在 vLLM 源码 benchmarks/ 下）。
# 日志里要找的几行（vllm 0d6f92a / sglang 8fabf5e）：
#   "only supports one backend per engine"       structured_output/__init__.py 143–150
#   "Invalid structured outputs"                  语法编译抛异常，请求直接结束
#   "Available KV cache memory"                   worker 启动时；max_lora_rank / max_loras 越大这里越小
#   /metrics: vllm:lora_requests_info{running_lora_adapters=…, waiting_lora_adapters=…, max_lora=…}
#   SGLang："LoRA drain" 相关日志在 lora_drain_wait_threshold > 0 时出现
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-8B}
OUT=${OUT:-results}
GMU=${GMU:-0.90}
MAXLEN=${MAXLEN:-8192}
VLLM_SRC=${VLLM_SRC:-$HOME/vllm}
LORA_DIR=${LORA_DIR:-$HOME/loras}
PORT=8000
mkdir -p "$OUT"

lora_modules() {   # LORA_DIR 下每个子目录 → name=path
  local n=${1:-1000} i=0
  for d in "$LORA_DIR"/*/; do
    [ $i -ge "$n" ] && break
    printf '%s=%s ' "$(basename "$d")" "${d%/}"
    i=$((i + 1))
  done
}

lora_names() {
  local n=${1:-1000}
  ls -1 "$LORA_DIR" | head -n "$n" | tr '\n' ' '
}

case "${1:-}" in
  so)
    BACKEND=${2:-auto}
    vllm serve "$MODEL" --port $PORT --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN" \
      --structured-outputs-config.backend "$BACKEND" 2>&1 | tee "$OUT/so_$BACKEND.log" ;;
  so_spec)
    K=${2:-3}
    vllm serve "$MODEL" --port $PORT --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN" \
      --structured-outputs-config.backend xgrammar \
      --speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":$K,\"prompt_lookup_max\":4}" \
      2>&1 | tee "$OUT/so_spec$K.log" ;;
  so_think)
    vllm serve "$MODEL" --port $PORT --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN" \
      --reasoning-parser qwen3 --structured-outputs-config.backend xgrammar \
      2>&1 | tee "$OUT/so_think.log" ;;
  so_bench)
    DATASET=${2:-json}
    python3 "$VLLM_SRC/benchmarks/benchmark_serving_structured_output.py" --backend vllm --model "$MODEL" \
      --port $PORT --dataset "$DATASET" --num-prompts 500 --request-rate 20 --output-len 256 \
      --save-results --result-dir "$OUT" --result-filename "so_$DATASET.json" ;;
  so_ratio)
    for r in 0 0.5 1; do
      python3 "$VLLM_SRC/benchmarks/benchmark_serving_structured_output.py" --backend vllm --model "$MODEL" \
        --port $PORT --dataset json --num-prompts 500 --request-rate 20 --output-len 256 \
        --structured-output-ratio $r --save-results --result-dir "$OUT" --result-filename "so_ratio_$r.json"
    done ;;
  auto_lock)
    # 1) multipleOf → xgrammar 不支持 → guidance，引擎锁成 guidance
    curl -s localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "{
      \"model\": \"$MODEL\", \"max_tokens\": 32,
      \"messages\": [{\"role\": \"user\", \"content\": \"给一个价格\"}],
      \"response_format\": {\"type\": \"json_schema\", \"json_schema\": {\"name\": \"p\",
        \"schema\": {\"type\": \"object\", \"properties\": {\"price\": {\"type\": \"number\", \"multipleOf\": 0.01}},
                     \"required\": [\"price\"]}}}}" | tee "$OUT/auto_lock_1.json"; echo
    # 2) 普通 schema → xgrammar，与 guidance 冲突
    curl -s localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "{
      \"model\": \"$MODEL\", \"max_tokens\": 32,
      \"messages\": [{\"role\": \"user\", \"content\": \"给一个名字\"}],
      \"response_format\": {\"type\": \"json_schema\", \"json_schema\": {\"name\": \"n\",
        \"schema\": {\"type\": \"object\", \"properties\": {\"name\": {\"type\": \"string\"}}}}}}" \
      | tee "$OUT/auto_lock_2.json"; echo ;;
  lora)
    ML=${2:-1}; RANK=${3:-16}
    # shellcheck disable=SC2046
    vllm serve "$MODEL" --port $PORT --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN" \
      --enable-lora --max-loras "$ML" --max-lora-rank "$RANK" --max-cpu-loras 32 \
      --lora-modules $(lora_modules) 2>&1 | tee "$OUT/lora_ml${ML}_r$RANK.log" ;;
  lora_mem)
    for RANK in 16 64 256; do
      timeout 600 vllm serve "$MODEL" --port $PORT --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN" \
        --enable-lora --max-loras 8 --max-lora-rank $RANK 2>&1 | tee "$OUT/lora_mem_r$RANK.log" &
      pid=$!
      until grep -q "Available KV cache memory" "$OUT/lora_mem_r$RANK.log" 2>/dev/null; do sleep 5; done
      grep -h "Available KV cache memory\|GPU KV cache size" "$OUT/lora_mem_r$RANK.log"
      kill $pid; wait $pid 2>/dev/null || true
    done ;;
  lora_bench)
    N=${2:-8}
    # shellcheck disable=SC2046
    vllm bench serve --backend vllm --model "$MODEL" --port $PORT --dataset-name random \
      --random-input-len 1024 --random-output-len 256 --num-prompts 800 --request-rate 8 \
      --lora-modules $(lora_names "$N") --lora-assignment random \
      --save-result --result-dir "$OUT" --result-filename "lora_bench_$N.json"
    curl -s localhost:$PORT/metrics | grep lora_requests_info ;;
  lora_hot)
    NAME=$2; LPATH=$3
    curl -s -X POST localhost:$PORT/v1/load_lora_adapter -H 'Content-Type: application/json' \
      -d "{\"lora_name\": \"$NAME\", \"lora_path\": \"$LPATH\"}"; echo
    curl -s localhost:$PORT/v1/models | python3 -m json.tool | grep '"id"'
    curl -s -X POST localhost:$PORT/v1/unload_lora_adapter -H 'Content-Type: application/json' \
      -d "{\"lora_name\": \"$NAME\"}"; echo ;;
  sgl_so)
    python3 -m sglang.launch_server --model-path "$MODEL" --port 30000 --mem-fraction-static 0.85 \
      --grammar-backend xgrammar --reasoning-parser qwen3 2>&1 | tee "$OUT/sgl_so.log" ;;
  sgl_lora)
    # shellcheck disable=SC2046
    python3 -m sglang.launch_server --model-path "$MODEL" --port 30000 --mem-fraction-static 0.85 \
      --enable-lora --max-lora-rank 16 --max-loras-per-batch 8 --max-loaded-loras 32 \
      --lora-eviction-policy lru --lora-drain-wait-threshold 2.0 \
      --lora-paths $(lora_modules) 2>&1 | tee "$OUT/sgl_lora.log" ;;
  *)
    sed -n 2,20p "$0"; exit 1 ;;
esac
