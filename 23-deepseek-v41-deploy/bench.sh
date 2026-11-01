#!/usr/bin/env bash
# VLLM-23 压测：对 serve.sh 起好的服务跑冒烟、decode 并发扫描、长上下文 prefill，结果落 $OUT/bench_*.json。
#
# 用法：
#   bash bench.sh smoke            # 问 17×19，答案里要有 323；再打一条工具调用看 parser
#   bash bench.sh decode           # 输入 8K / 输出 1K，并发 1 4 8 16 32 64 扫一遍（和 yhfgyyf、#46310 的口径对齐）
#   bash bench.sh long             # 并发 1，输入 32K / 128K / 256K，看 TTFT 和 prefill 吞吐
#   bash bench.sh metrics          # 拉 /metrics 里的 DSpark 接受率、KV 使用率
#
# 环境变量：MODEL、PORT、OUT、CONC（覆盖并发列表）、NUM（每档请求数，默认 并发×4，至少 8）。
# 输出里要看的字段（vllm/benchmarks/serve.py）：
#   "Output token throughput (tok/s)"   和正文估算表的 c1 / c16 对照
#   "Mean TTFT (ms)" / "Mean TPOT (ms)"
#   /metrics 里 vllm:spec_decode_num_accepted_tokens / vllm:spec_decode_num_drafts 相除得平均接受长度
set -euo pipefail

MODEL=${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash-1017}
PORT=${PORT:-8000}
OUT=${OUT:-results}
URL=http://127.0.0.1:$PORT
mkdir -p "$OUT"

bench() {
  local tag=$1 inlen=$2 outlen=$3 conc=$4
  local num=${NUM:-$(( conc * 4 > 8 ? conc * 4 : 8 ))}
  echo "== in=$inlen out=$outlen conc=$conc num=$num"
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --base-url "$URL" --model "$MODEL" --tokenizer-mode deepseek_v41 \
    --dataset-name random --random-input-len "$inlen" --random-output-len "$outlen" --random-range-ratio 0 \
    --num-prompts "$num" --max-concurrency "$conc" --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir "$OUT" --result-filename "bench_${tag}.json" 2>&1 | \
    grep -E 'throughput|TTFT|TPOT|Successful|Failed' || true
}

case "${1:-}" in
  smoke)
    curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "{
      \"model\": \"$MODEL\", \"max_tokens\": 2048,
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
sys.exit(0 if calls else "!! 没解析出工具调用，查 --tool-call-parser deepseek_v41")' ;;
  decode)
    for c in ${CONC:-1 4 8 16 32 64}; do bench "decode_c$c" 8192 1024 "$c"; done ;;
  long)
    for L in 32768 131072 262144; do NUM=${NUM:-4} bench "long_$L" "$L" 256 1; done ;;
  metrics)
    curl -s "$URL/metrics" | grep -E '^vllm:(spec_decode_num_accepted_tokens|spec_decode_num_drafts|kv_cache_usage_perc|num_requests_running|prefix_cache_hits)' | \
      grep -v '_created' ;;
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac
