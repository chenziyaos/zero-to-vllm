#!/usr/bin/env bash
# VLLM-25：Qwen3.8-27B 真机压测。本脚本只请求已经启动的服务，不负责起服务。
#
# 用法：
#   bash bench.sh smoke      # 算术题 + qwen3_coder 工具调用，断言必须通过
#   bash bench.sh recipe     # random 1024/1024，128 条，并发 128 的参考负载
#   bash bench.sh decode     # 输入 1K / 输出 256，并发 1 4 8 16 32
#   bash bench.sh long       # 单并发，按上下文预算预扣输出和模板余量
#   bash bench.sh mtp-snapshot > m1.txt   # 压测前抓一份 /metrics
#   bash bench.sh mtp m1.txt m2.txt       # 用两个快照算 accepted/draft
#
# 环境变量：MODEL（默认 qwen38）、TOKENIZER、PORT、URL、OUT、CONC、NUM、
#           MAX_MODEL_LEN、OUTPUT_LEN、TEMPLATE_MARGIN、LENS。
# 真实吞吐字段以落盘的 bench_*.json 为准；脚本不 grep 输出，也不吞失败。
set -euo pipefail

MODEL=${MODEL:-qwen38}
TOKENIZER=${TOKENIZER:-Qwen/Qwen3.8-27B}
PORT=${PORT:-8000}
OUT=${OUT:-results}
URL=${URL:-http://127.0.0.1:$PORT}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

bench() {
  local tag=$1 inlen=$2 outlen=$3 conc=$4
  local num=${NUM:-${5:-$(( conc * 4 > 8 ? conc * 4 : 8 ))}}
  echo "== in=$inlen out=$outlen conc=$conc num=$num"
  # pipefail + tee：vllm 非零退出会让整条命令失败，完整日志同时落盘。
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --base-url "$URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --dataset-name random --random-input-len "$inlen" --random-output-len "$outlen" --random-range-ratio 0 \
    --num-prompts "$num" --max-concurrency "$conc" --request-rate inf --seed 42 --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --extra-body '{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    --save-result --result-dir "$OUT" --result-filename "bench_${tag}.json" 2>&1 | tee "$OUT/bench_${tag}.log"
}

case "${1:-}" in
  smoke)
    # 17×19=323；工具调用必须解析出 get_weather(city=北京)。失败时 bench_checks 非零退出。
    curl -fsS "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\":\"user\",\"content\":\"17*19 等于几？只回数字\"}],
      \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}
    }" | python3 "$HERE/bench_checks.py" arithmetic
    curl -fsS "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\":\"user\",\"content\":\"调用工具查询北京今天的天气\"}],
      \"tools\": [{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",
        \"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],
      \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}
    }" | python3 "$HERE/bench_checks.py" tool
    ;;
  recipe) bench recipe 1024 1024 128 128 ;;
  decode)
    for c in ${CONC:-"1 4 8 16 32"}; do bench "decode_c$c" 1024 256 "$c"; done ;;
  long)
    # 5090 单卡档 MAX_MODEL_LEN=32768；262K 档调到 262144。预算先扣输出和模板余量。
    max_len=${MAX_MODEL_LEN:-32768}
    output=${OUTPUT_LEN:-16}
    margin=${TEMPLATE_MARGIN:-256}
    for budget in ${LENS:-"8192 16384 32768"}; do
      if (( budget > max_len )); then
        echo "预算 $budget 超过 MAX_MODEL_LEN=$max_len" >&2; exit 2
      fi
      if (( budget <= output + margin )); then
        echo "预算 $budget 不够 output=$output + margin=$margin" >&2; exit 2
      fi
      bench "long_b$budget" "$(( budget - output - margin ))" "$output" 1
    done ;;
  mtp-snapshot)
    curl -fsS "$URL/metrics" | grep -E 'spec_decode_num_(accepted|draft)_tokens_total'
    ;;
  mtp)
    [[ $# -ge 3 ]] || { echo '用法：bash bench.sh mtp BEFORE AFTER' >&2; exit 2; }
    python3 "$HERE/bench_checks.py" mtp "$2" "$3" ;;
  *)
    echo '用法：bash bench.sh {smoke|recipe|decode|long|mtp-snapshot|mtp BEFORE AFTER}' >&2
    exit 2 ;;
esac
