#!/usr/bin/env bash
# VLLM-03 压测参数组。先在另一个终端起服务：
#   VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-8B --gpu-memory-utilization 0.92
# VLLM_SERVER_DEV_MODE=1 只为打开 /reset_prefix_cache，生产环境不要开。
#
# 用法：
#   bash bench.sh smoke            # 10 条请求确认链路通
#   bash bench.sh rate             # 开环：固定负载，扫 --request-rate
#   bash bench.sh conc             # 闭环：--request-rate inf，扫 --max-concurrency
#   bash bench.sh sharegpt         # ShareGPT 真实长度分布，扫 --request-rate
#   bash bench.sh burst            # 同一速率，扫 --burstiness
#   bash bench.sh sweep            # 用 vllm bench sweep serve 扫服务端参数（会自己起停服务）
# 结果写到 results/<组名>/，每个点一份 --save-detailed JSON，可直接喂给 sim_goodput.py --from-json。
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-8B}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
N=${N:-1000}
SLO=${SLO:-"ttft:1000 tpot:40"}
SHAREGPT=${SHAREGPT:-ShareGPT_V3_unfiltered_cleaned_split.json}
OUT=${OUT:-results}

# 每次压测共用的参数：
#   --ignore-eos      让每条请求都跑满 OSL，output_len 才是你设的那个数
#   --temperature 0   新版不再默认贪心，不写会打一行 WARNING，且采样参数会影响长度
#   --percentile-metrics / --metric-percentiles  默认只报 ttft,tpot,itl 的 p99，补上 e2el 和 p50/p90
#   --num-warmups     默认 0；第一批请求会撞上 CUDA Graph 之外的冷路径
COMMON=(--backend vllm --model "$MODEL" --host "$HOST" --port "$PORT"
        --ignore-eos --temperature 0
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99
        --num-warmups 20 --goodput $SLO
        --save-result --save-detailed)

reset_cache() {
  # random 数据集在固定 --seed 下每次生成同一批 prompt，不清前缀缓存第二轮会虚高
  curl -sf -X POST "http://$HOST:$PORT/reset_prefix_cache" >/dev/null || echo "reset_prefix_cache 不可用（没开 VLLM_SERVER_DEV_MODE？）"
}

snap() {  # 压测前后各抓一次 /metrics，算差值看服务端口径
  curl -sf "http://$HOST:$PORT/metrics" > "$1" || true
}

run() {  # run <组名> <标签> <其余 bench 参数...>
  local group=$1 label=$2; shift 2
  mkdir -p "$OUT/$group"
  reset_cache
  snap "$OUT/$group/$label.before.prom"
  vllm bench serve "${COMMON[@]}" "$@" \
    --label "$label" --result-dir "$OUT/$group" --result-filename "$label.json"
  snap "$OUT/$group/$label.after.prom"
}

case "${1:-}" in
  smoke)
    run smoke n10 --dataset-name random --random-input-len 1024 --random-output-len 128 --num-prompts 10
    ;;
  rate)
    for r in 1 2 4 8 12 16 20 24 32 inf; do
      run rate "r$r" --dataset-name random --random-input-len 1024 --random-output-len 128 \
        --num-prompts "$N" --request-rate "$r"
    done
    ;;
  conc)
    for c in 1 4 16 32 64 128 256 512; do
      run conc "c$c" --dataset-name random --random-input-len 1024 --random-output-len 128 \
        --num-prompts "$N" --request-rate inf --max-concurrency "$c"
    done
    ;;
  sharegpt)
    [ -f "$SHAREGPT" ] || wget -q https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    for r in 4 8 16 32 inf; do
      run sharegpt "r$r" --dataset-name sharegpt --dataset-path "$SHAREGPT" --num-prompts "$N" --request-rate "$r"
    done
    ;;
  burst)
    for b in 0.25 0.5 1 4; do
      run burst "b$b" --dataset-name random --random-input-len 1024 --random-output-len 128 \
        --num-prompts "$N" --request-rate 16 --burstiness "$b"
    done
    ;;
  sweep)
    cat > serve_params.json <<'EOF'
[
  {"_benchmark_name": "b2048",  "max_num_batched_tokens": 2048},
  {"_benchmark_name": "b8192",  "max_num_batched_tokens": 8192},
  {"_benchmark_name": "b16384", "max_num_batched_tokens": 16384}
]
EOF
    cat > bench_params.json <<'EOF'
[
  {"_benchmark_name": "c32",  "max_concurrency": 32},
  {"_benchmark_name": "c128", "max_concurrency": 128}
]
EOF
    vllm bench sweep serve \
      --serve-cmd "vllm serve $MODEL --gpu-memory-utilization 0.92" \
      --bench-cmd "vllm bench serve --backend vllm --model $MODEL --dataset-name random --random-input-len 1024 --random-output-len 128 --num-prompts $N --request-rate inf --ignore-eos --temperature 0" \
      --serve-params serve_params.json --bench-params bench_params.json \
      --num-runs 3 --warmup-num-prompts 50 -o "$OUT/sweep"
    ;;
  *)
    sed -n '2,13p' "$0"; exit 1 ;;
esac
