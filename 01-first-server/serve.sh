#!/usr/bin/env bash
# VLLM-01：起一个 Qwen3-8B 服务。验证于 vLLM v0.32.0 / 主干 0d6f92a。
#
# 用法：
#   bash serve.sh                        # 默认 Qwen/Qwen3-8B，端口 8000
#   MODEL=Qwen/Qwen3-0.6B bash serve.sh  # 24 GB 卡上建议先用小模型
#   EXTRA="--gpu-memory-utilization 0.85 --max-model-len 16384" bash serve.sh
#
# 日志同时写到 logs/serve-时间戳.log，给 parse_startup_log.py 用。
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-8B}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
TP=${TP:-1}
EXTRA=${EXTRA:-}

mkdir -p logs
LOG=logs/serve-$(date +%Y%m%d-%H%M%S).log

# 采样默认值取模型 generation_config.json（--generation-config auto 是默认行为）。
# reasoning parser 把 <think> 段拆到 message.reasoning；工具调用需要显式
# --enable-auto-tool-choice，并且必须同时给 --tool-call-parser，否则启动即报错。
# Qwen3 的 JSON 工具调用在主干上由 hermes 解析（qwen3_json 是它的别名）。
set -x
vllm serve "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --served-model-name qwen3 \
  $EXTRA 2>&1 | tee "$LOG"
