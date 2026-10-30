#!/usr/bin/env bash
# VLLM-10 GPU 实验：同一个模型、同一组负载，只换注意力后端，看选择日志、block、CG 模式和延迟。
#
# 用法：
#   bash gpu_backend_bench.sh select     # 每个后端起一次引擎，只抓选择 / block / cudagraph 相关日志
#   bash gpu_backend_bench.sh latency    # vllm bench latency：后端 × batch 1/16/128，输入 1024 / 输出 128
#   bash gpu_backend_bench.sh serve      # vllm bench serve：后端 × 并发 8/64，随机数据集，看 TTFT / TPOT
#   bash gpu_backend_bench.sh blocksize  # 不给 / 给 --block-size 16 / 64 / 128，看自动选择被怎样改写
#   bash gpu_backend_bench.sh kvfp8      # --kv-cache-dtype fp8 时，自动选择会不会换后端
#
# 环境变量：MODEL（默认 Qwen/Qwen3-8B）、OUT、GMU、BACKENDS（空格分隔，覆盖默认后端列表）。
# 日志里要找的几行（行号对应 vllm 0d6f92a）：
#   "Using %s attention backend out of potential backends: [...]"   platforms/cuda.py:513（自动选择）
#   "Using %s backend."                                            platforms/cuda.py:452（用户指定）
#   "Selected backend X is not valid for this configuration. Reason: [...]"  platforms/cuda.py:447
#   "--block-size %d precluded higher-priority backend(s) ..."     platforms/cuda.py:503
#   "Setting kv cache block size to %d for %s backend(s)."         platforms/interface.py:782
#   "setting cudagraph_mode=..." / "not compatible"                config/compilation.py（CG 降级）
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-8B}
OUT=${OUT:-results}
GMU=${GMU:-0.90}
read -r -a LIST <<< "${BACKENDS:-AUTO FLASH_ATTN FLASHINFER TRITON_ATTN FLEX_ATTENTION}"
PATTERN='attention backend|backend\.|not valid for this configuration|precluded|Setting kv cache block size|cudagraph_mode|not compatible|Graph capturing finished|KV cache size'

ab_flag() { [[ "$1" == AUTO ]] && echo "" || echo "--attention-backend $1"; }

run_select() {
  mkdir -p "$OUT/select"
  for b in "${LIST[@]}"; do
    echo "== $b"
    # shellcheck disable=SC2046
    vllm bench latency --model "$MODEL" --gpu-memory-utilization "$GMU" \
      --input-len 8 --output-len 1 --batch-size 1 --num-iters-warmup 0 --num-iters 1 \
      $(ab_flag "$b") 2>&1 | tee "$OUT/select/$b.log" | grep -E "$PATTERN" || true
  done
}

run_latency() {
  mkdir -p "$OUT/latency"
  for b in "${LIST[@]}"; do
    for bs in 1 16 128; do
      tag="$b-bs$bs"
      echo "== $tag"
      # shellcheck disable=SC2046
      vllm bench latency --model "$MODEL" --gpu-memory-utilization "$GMU" \
        --input-len 1024 --output-len 128 --batch-size "$bs" --num-iters-warmup 3 --num-iters 10 \
        $(ab_flag "$b") --output-json "$OUT/latency/$tag.json" 2>&1 | \
        tee "$OUT/latency/$tag.log" | grep -E "Avg latency|attention backend|backend\." || true
    done
  done
  python3 - "$OUT/latency" <<'EOF'
import json, sys, glob, os
print(f"{'后端-batch':<28}{'平均 s':>10}{'p99 s':>10}")
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.json'))):
    d = json.load(open(f))
    print(f"{os.path.basename(f)[:-5]:<28}{d['avg_latency']:>10.3f}{d['percentiles'].get('99', 0):>10.3f}")
EOF
}

run_serve() {
  mkdir -p "$OUT/serve"
  for b in "${LIST[@]}"; do
    echo "== $b：起服务"
    # shellcheck disable=SC2046
    vllm serve "$MODEL" --gpu-memory-utilization "$GMU" $(ab_flag "$b") > "$OUT/serve/$b.server.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 180); do
      curl -sf localhost:8000/health >/dev/null 2>&1 && break
      sleep 2
    done
    grep -E "$PATTERN" "$OUT/serve/$b.server.log" || true
    for c in 8 64; do
      vllm bench serve --model "$MODEL" --dataset-name random --random-input-len 2048 --random-output-len 256 \
        --num-prompts $((c * 8)) --max-concurrency "$c" \
        --save-result --result-dir "$OUT/serve" --result-filename "$b-c$c.json" 2>&1 | \
        grep -E "Output token throughput|Mean TTFT|Mean TPOT|P99 TPOT" || true
    done
    kill "$pid"; wait "$pid" 2>/dev/null || true
  done
}

run_blocksize() {
  mkdir -p "$OUT/blocksize"
  for bs in auto 16 64 128; do
    flag=""; [[ "$bs" != auto ]] && flag="--block-size $bs"
    echo "== block $bs"
    # shellcheck disable=SC2086
    vllm bench latency --model "$MODEL" --gpu-memory-utilization "$GMU" \
      --input-len 8 --output-len 1 --batch-size 1 --num-iters-warmup 0 --num-iters 1 \
      $flag 2>&1 | tee "$OUT/blocksize/$bs.log" | grep -E "$PATTERN|No valid attention backend" || true
  done
}

run_kvfp8() {
  mkdir -p "$OUT/kvfp8"
  for kv in auto fp8 fp8_e5m2; do
    echo "== kv $kv"
    vllm bench latency --model "$MODEL" --gpu-memory-utilization "$GMU" \
      --input-len 8 --output-len 1 --batch-size 1 --num-iters-warmup 0 --num-iters 1 \
      --kv-cache-dtype "$kv" 2>&1 | tee "$OUT/kvfp8/$kv.log" | grep -E "$PATTERN" || true
  done
}

case "${1:-}" in
  select) run_select ;;
  latency) run_latency ;;
  serve) run_serve ;;
  blocksize) run_blocksize ;;
  kvfp8) run_kvfp8 ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
