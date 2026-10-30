#!/usr/bin/env bash
# VLLM-09 GPU 实验：同一个模型、同一组 batch，换编译档位和 cudagraph_mode，看启动时间、显存和单步延迟。
#
# 用法：
#   bash gpu_cudagraph_modes.sh latency    # vllm bench latency：每种模式 × batch 1/8/64/256 的 decode 延迟
#   bash gpu_cudagraph_modes.sh startup    # 只起引擎，抓启动日志里的编译 / 捕获耗时和显存
#   bash gpu_cudagraph_modes.sh cache      # 冷 / 热编译缓存的启动时间对比
#
# 结果写到 results/<组名>/。每个点一份 --output-json 和一份完整日志，日志里要找的几行：
#   "torch.compile took %.2f s in total"                       compilation/monitor.py:20
#   "Directly load the compiled graph(s) for compile range"    compilation/backends.py:311（命中缓存）
#   "Compiling a graph for compile range %s takes %.2f s"      compilation/backends.py:418（没命中）
#   "Profiling CUDA graph memory: PIECEWISE=… FULL=…"          v1/worker/gpu_model_runner.py:6268
#   "Graph capturing finished in %.0f secs, took %.2f GiB"     v1/worker/gpu_model_runner.py:6517
#   "Cudagraph mode … is not compatible …" / "setting cudagraph_mode=…"   降级提示
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-8B}
OUT=${OUT:-results}
IN=${IN:-512}
OUTLEN=${OUTLEN:-128}
GMU=${GMU:-0.90}

# 名字 → 参数。-O 后面跟数字会被 FlexibleArgumentParser 改写成 --optimization-level。
declare -A MODES=(
  [eager]="--enforce-eager"
  [O0]="-O0"
  [O1]="-O1"
  [none]="-cc {\"cudagraph_mode\":\"NONE\"}"
  [piecewise]="-cc {\"cudagraph_mode\":\"PIECEWISE\"}"
  [full_decode_only]="-cc {\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}"
  [full_and_piecewise]="-cc {\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}"
  [full]="-cc {\"cudagraph_mode\":\"FULL\"}"
)
ORDER=(eager O0 O1 none piecewise full_decode_only full_and_piecewise full)

run_latency() {
  mkdir -p "$OUT/latency"
  for m in "${ORDER[@]}"; do
    for bs in 1 8 64 256; do
      tag="$m-bs$bs"
      echo "== $tag"
      # shellcheck disable=SC2086
      vllm bench latency --model "$MODEL" --gpu-memory-utilization "$GMU" \
        --input-len "$IN" --output-len "$OUTLEN" --batch-size "$bs" \
        --num-iters-warmup 5 --num-iters 20 \
        ${MODES[$m]} \
        --output-json "$OUT/latency/$tag.json" 2>&1 | tee "$OUT/latency/$tag.log" | \
        grep -E "Avg latency|torch.compile took|Graph capturing finished|cudagraph_mode=|not compatible" || true
    done
  done
  python3 - "$OUT/latency" <<'EOF'
import json, sys, glob, os
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.json'))):
    d = json.load(open(f))
    rows.append((os.path.basename(f)[:-5], d['avg_latency'], d['percentiles'].get('99')))
print(f"{'模式-batch':<28}{'平均 s':>10}{'p99 s':>10}")
for r in rows:
    print(f"{r[0]:<28}{r[1]:>10.3f}{r[2]:>10.3f}")
EOF
}

run_startup() {
  mkdir -p "$OUT/startup"
  for m in "${ORDER[@]}"; do
    echo "== $m"
    # 只做一次 1 token 的生成：引擎初始化（profile_run → KV 分配 → capture_model）全部走完
    # shellcheck disable=SC2086
    vllm bench latency --model "$MODEL" --gpu-memory-utilization "$GMU" \
      --input-len 8 --output-len 1 --batch-size 1 --num-iters-warmup 0 --num-iters 1 \
      ${MODES[$m]} 2>&1 | tee "$OUT/startup/$m.log" | \
      grep -E "torch.compile took|Compiling a graph|Directly load|Profiling CUDA graph memory|Graph capturing finished|init engine|KV cache size|Available KV cache memory" || true
  done
}

run_cache() {
  mkdir -p "$OUT/cache"
  export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$PWD/.vllm_cache}
  rm -rf "$VLLM_CACHE_ROOT/torch_compile_cache"
  for round in cold warm; do
    echo "== $round（VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT）"
    vllm bench latency --model "$MODEL" --gpu-memory-utilization "$GMU" \
      --input-len 8 --output-len 1 --batch-size 1 --num-iters-warmup 0 --num-iters 1 2>&1 | \
      tee "$OUT/cache/$round.log" | \
      grep -E "Using cache directory|torch.compile took|Directly load|Compiling a graph|init engine" || true
  done
  echo "== 关掉缓存（VLLM_DISABLE_COMPILE_CACHE=1）"
  VLLM_DISABLE_COMPILE_CACHE=1 vllm bench latency --model "$MODEL" --gpu-memory-utilization "$GMU" \
    --input-len 8 --output-len 1 --batch-size 1 --num-iters-warmup 0 --num-iters 1 2>&1 | \
    tee "$OUT/cache/disabled.log" | grep -E "torch.compile took|Compiling a graph|init engine" || true
}

case "${1:-}" in
  latency) run_latency ;;
  startup) run_startup ;;
  cache) run_cache ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
