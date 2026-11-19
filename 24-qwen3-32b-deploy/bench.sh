#!/usr/bin/env bash
# VLLM-24 压测：对 serve.sh 起好的服务跑冒烟、recipe 口径、decode 并发扫描、长上下文，结果落 $OUT/bench_*.json。
#
# 用法：
#   bash bench.sh smoke            # 问 17×19，答案里要有 323；再打一条工具调用看 hermes parser
#   bash bench.sh recipe           # recipe 232–243 的口径：random 1024/1024，128 条，不限并发（服务端要加 --no-enable-prefix-caching）
#   bash bench.sh sharegpt         # ShareGPT 200 条、并发 60，和 jarvislabs 的 2×H100 口径对齐（需 SHAREGPT=json 路径）
#   bash bench.sh decode           # 输入 1K / 输出 1K，并发 1 4 8 16 32 64 扫一遍，和正文 decode 估算表对照
#   bash bench.sh long             # 并发 1，输入 16K / 32K / 40K，看 TTFT（4090 档上下文只有 16K，用 LENS 覆盖）
#   bash bench.sh metrics          # 拉 /metrics 里的 EAGLE3 接受数、KV 使用率、prefix 命中
#
# 环境变量：MODEL（默认 Qwen/Qwen3-32B-FP8，要和 serve.sh 的档位一致）、PORT、OUT、CONC、NUM（每档请求数，默认 并发×4，至少 8）、LENS。
# 输出里要看的字段（vllm/benchmarks/serve.py）：
#   "Output token throughput (tok/s)"   和正文估算表的 c1 / c32 对照
#   "Mean TTFT (ms)" / "Mean TPOT (ms)" 1000 / TPOT 就是单流 tok/s
#   /metrics 里 vllm:spec_decode_num_accepted_tokens / vllm:spec_decode_num_drafts 加 1 得平均接受长度
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-32B-FP8}
PORT=${PORT:-8000}
OUT=${OUT:-results}
URL=http://127.0.0.1:$PORT
mkdir -p "$OUT"

bench() {
  local tag=$1 inlen=$2 outlen=$3 conc=$4
  local num=${NUM:-$(( conc * 4 > 8 ? conc * 4 : 8 ))}
  echo "== in=$inlen out=$outlen conc=$conc num=$num"
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --base-url "$URL" --model "$MODEL" \
    --dataset-name random --random-input-len "$inlen" --random-output-len "$outlen" --random-range-ratio 0 \
    --num-prompts "$num" --max-concurrency "$conc" --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir "$OUT" --result-filename "bench_${tag}.json" 2>&1 | \
    grep -E 'throughput|TTFT|TPOT|Successful|Failed' || true
}

case "${1:-}" in
  smoke)
    curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "{
      \"model\": \"$MODEL\", \"max_tokens\": 2048, \"temperature\": 0.6, \"top_p\": 0.95, \"top_k\": 20,
      \"messages\": [{\"role\": \"user\", \"content\": \"17 乘以 19 等于多少？只回答数字。\"}]}" | \
      tee "$OUT/smoke.json" | python3 -c '
import json, sys
m = json.load(sys.stdin)["choices"][0]["message"]
print("reasoning:", (m.get("reasoning_content") or m.get("reasoning") or "")[:120].replace("\n", " "))
print("content  :", m["content"])
sys.exit(0 if "323" in (m["content"] or "") else "!! 答案里没有 323")'
    curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "{
      \"model\": \"$MODEL\", \"max_tokens\": 1024, \"tool_choice\": \"auto\",
      \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"get_weather\",
        \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}],
      \"messages\": [{\"role\": \"user\", \"content\": \"北京今天天气怎么样？\"}]}" | \
      tee "$OUT/smoke_tool.json" | python3 -c '
import json, sys
m = json.load(sys.stdin)["choices"][0]["message"]
calls = m.get("tool_calls") or []
print("tool_calls:", [(c["function"]["name"], c["function"]["arguments"]) for c in calls])
sys.exit(0 if calls else "!! 没解析出工具调用，查 --enable-auto-tool-choice --tool-call-parser")' ;;
  recipe)
    NUM=${NUM:-128} bench recipe 1024 1024 "${CONC:-128}" ;;
  sharegpt)
    : "${SHAREGPT:?先下载 ShareGPT_V3_unfiltered_cleaned_split.json，再 SHAREGPT=路径 bash bench.sh sharegpt}"
    vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
      --base-url "$URL" --model "$MODEL" --dataset-name sharegpt --dataset-path "$SHAREGPT" \
      --num-prompts "${NUM:-200}" --max-concurrency "${CONC:-60}" \
      --percentile-metrics ttft,tpot,itl,e2el \
      --save-result --result-dir "$OUT" --result-filename bench_sharegpt.json 2>&1 | \
      grep -E 'throughput|TTFT|TPOT|Successful|Failed' || true ;;
  decode)
    for c in ${CONC:-1 4 8 16 32 64}; do bench "decode_c$c" 1024 1024 "$c"; done ;;
  long)
    for L in ${LENS:-16384 32768 40000}; do NUM=${NUM:-4} bench "long_$L" "$L" 256 1; done ;;
  metrics)
    curl -s "$URL/metrics" | grep -E '^vllm:(spec_decode_num_accepted_tokens|spec_decode_num_drafts|kv_cache_usage_perc|num_requests_running|prefix_cache_hits)' | \
      grep -v '_created' ;;
  *) sed -n '2,11p' "$0"; exit 1 ;;
esac
