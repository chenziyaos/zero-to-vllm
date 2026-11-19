#!/usr/bin/env bash
# 19 章配套：多副本、K8s、sleep mode 与 RL 权重同步的命令清单。
# 每段相互独立，按需复制，不要整份直接跑。
# 版本：vLLM 0d6f92a（docs/serving/data_parallel_deployment.md、docs/deployment/k8s.md、
#       docs/features/sleep_mode.md、docs/training/async_rl.md、docs/training/weight_transfer/*.md、
#       vllm/entrypoints/rl/online/api_router.py），SGLang 8fabf5e（sgl-router/README.md、sglang_for_rl.mdx）
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3-8B}
VLLM=${VLLM:-http://127.0.0.1:8000}
SGL=${SGL:-http://127.0.0.1:30000}
HEAD_IP=${HEAD_IP:-10.0.0.1}
HERE="$(cd "$(dirname "$0")" && pwd)"

case "${1:-help}" in

# ---------------------------------------------------------------- 多副本：vLLM 内置 DP
dp-single)
  # 单机 8 卡：DP=4 × TP=2，一个端口。--max-num-seqs 按 rank 算；
  # --max-num-queued-reqs 按整个服务算，要设成 dp × max-num-seqs + 想要的队列深度
  vllm serve "$MODEL" --data-parallel-size 4 --tensor-parallel-size 2 \
    --max-num-seqs 256 --max-num-queued-reqs 1280
  ;;

dp-head)
  # 两机 DP=4：头节点跑 rank 0、1 和 API server
  vllm serve "$MODEL" --data-parallel-size 4 --data-parallel-size-local 2 \
    --data-parallel-address "$HEAD_IP" --data-parallel-rpc-port 13345 \
    --api-server-count 4      # API server 成瓶颈时横向扩，仍是一个端口
  ;;

dp-worker)
  # 第二台：--headless 不起 API server，rank 2、3
  vllm serve "$MODEL" --headless --data-parallel-size 4 --data-parallel-size-local 2 \
    --data-parallel-start-rank 2 \
    --data-parallel-address "$HEAD_IP" --data-parallel-rpc-port 13345
  ;;

dp-ray)
  # 一条命令拉起所有节点；单个副本跨节点时加 VLLM_RAY_DP_PACK_STRATEGY=span
  vllm serve "$MODEL" --data-parallel-size 4 --data-parallel-size-local 2 \
    --data-parallel-backend=ray
  ;;

dp-hybrid)
  # 每个节点各自起 API server，只往本节点的 rank 派；上游 LB 在节点之间分流。不能和 --headless 同用
  # 节点 0：START=0；节点 1：START=2
  vllm serve "$MODEL" --data-parallel-size 4 --data-parallel-size-local 2 \
    --data-parallel-start-rank "${START:-0}" --data-parallel-hybrid-lb \
    --data-parallel-address "$HEAD_IP" --data-parallel-rpc-port 13345
  ;;

kv-events)
  # 给外部前缀感知路由（llm-d、sgl-router 的 indexer 等）发 KV 事件
  vllm serve "$MODEL" --kv-events-config \
    '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557", "topic": "kv-events"}'
  ;;

# ---------------------------------------------------------------- 多副本：SGLang 路由
sgl-router)
  # 不加 --chat-routing 走旧路径，默认 round_robin；reorg 下默认 power_of_two
  sgl-router --model-id qwen3 --chat-routing reorg --policy cache_aware \
    --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
    --cache-affinity-min-matched-tokens 1024
  ;;

sgl-router-balanced)
  # 首选副本的待算 prefill 超过备选 × 2 且多出 1024 token 才改投
  sgl-router --model-id qwen3 --chat-routing reorg --policy cache_aware \
    --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
    --affinity-mode balanced --affinity-load-factor 2 --affinity-load-gap 1024 \
    --affinity-balanced-by prefill-tokens
  ;;

sgl-router-k8s)
  sgl-router --model-id qwen3 --chat-routing reorg --policy cache_aware \
    --service-discovery --service-discovery-namespace prod --selector app=engines-qwen3
  ;;

# ---------------------------------------------------------------- K8s
k8s-yaml)
  # startupProbe 管启动，liveness 只管卡死；编译缓存和 HF 缓存都放卷上
  cat <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata: {name: qwen3-8b, labels: {app: qwen3-8b}}
spec:
  replicas: 1
  selector: {matchLabels: {app: qwen3-8b}}
  template:
    metadata: {labels: {app: qwen3-8b}}
    spec:
      volumes:
      - name: hf-cache                # 权重：PVC，Pod 重建也在
        persistentVolumeClaim: {claimName: qwen3-8b}
      - name: vllm-cache              # torch_compile_cache：emptyDir 能活过容器重启
        emptyDir: {}
      - name: shm                     # TP 用共享内存
        emptyDir: {medium: Memory, sizeLimit: 8Gi}
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command: ["/bin/sh", "-c"]
        args: ["vllm serve Qwen/Qwen3-8B"]
        env:
        - {name: VLLM_CACHE_ROOT, value: /vllm-cache}
        ports: [{containerPort: 8000}]
        resources: {limits: {nvidia.com/gpu: "1"}}
        volumeMounts:
        - {name: hf-cache, mountPath: /root/.cache/huggingface}
        - {name: vllm-cache, mountPath: /vllm-cache}
        - {name: shm, mountPath: /dev/shm}
        startupProbe:                 # 最长 300 s；成功之前 liveness / readiness 都不跑
          httpGet: {path: /health, port: 8000}
          periodSeconds: 10
          failureThreshold: 30
        livenessProbe:
          httpGet: {path: /health, port: 8000}
          periodSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet: {path: /health, port: 8000}
          periodSeconds: 5
EOF
  ;;

k8s-measure)
  # 先去掉探针测一次真实启动时间（k8s.md 故障排查一节的建议），再定 failureThreshold
  kubectl logs -f deploy/qwen3-8b | grep -E 'Loading weights took|torch.compile takes|Graph capturing finished|Application startup complete'
  kubectl get events --field-selector reason=Unhealthy
  ;;

# ---------------------------------------------------------------- sleep mode（VLLM_SERVER_DEV_MODE=1）
sleep-serve)
  # 开发端点不要暴露给用户
  VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" --enable-sleep-mode
  ;;

sleep-l1)
  curl -s -X POST "$VLLM/sleep?level=1"          # 权重卸到 CPU pinned 内存，丢 KV
  curl -s "$VLLM/is_sleeping"
  curl -s -X POST "$VLLM/wake_up"
  ;;

sleep-l2)
  curl -s -X POST "$VLLM/sleep?level=2"          # 权重和 KV 都丢
  curl -s -X POST "$VLLM/wake_up?tags=weights"   # 先只分配权重显存
  curl -s -X POST "$VLLM/collective_rpc" -H 'Content-Type: application/json' -d '{"method":"reload_weights"}'
  curl -s -X POST "$VLLM/wake_up?tags=kv_cache"
  ;;

sleep-kv-only)
  # 只放 KV：先 level 0 keep 暂停，再释放；保留的请求唤醒后重算
  curl -s -X POST "$VLLM/sleep?level=0&mode=keep"
  curl -s -X POST "$VLLM/release_kv_cache_memory"
  curl -s -X POST "$VLLM/wake_up?tags=kv_cache"
  ;;

# ---------------------------------------------------------------- RL：暂停与权重同步（VLLM_SERVER_DEV_MODE=1）
rl-serve)
  VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" --weight-transfer-config '{"backend": "nccl"}'
  ;;

rl-world)
  curl -s "$VLLM/get_world_size"                 # 推理端 TP × PP × DP；NCCL world_size = 训练端广播 rank 数 + 这个数
  ;;

rl-pause)
  # abort（默认）/ wait 暂停期间新请求 503，客户端要重试；keep 照收照排
  curl -s -X POST "$VLLM/pause?mode=${MODE:-abort}"
  curl -s "$VLLM/is_paused"
  ;;

rl-sync)
  # 训练端是 NCCL rank 0。init 会阻塞到所有 rank 加入，要放后台发；update_weights 也要后台发，同时 rank 0 广播
  # init_info 字段见 weight_transfer/nccl.md「Trainer Without vLLM over HTTP」
  curl -s -X POST "$VLLM/start_weight_update"
  curl -s -X POST "$VLLM/update_weights" -H 'Content-Type: application/json' -d '{"update_info": {}}' &
  # …… 训练端在这里按 names 顺序广播 ……
  wait
  curl -s -X POST "$VLLM/finish_weight_update" -H 'Content-Type: application/json' -d '{"weight_version": "step-42"}'
  curl -s "$VLLM/weight_info"
  curl -s -X POST "$VLLM/resume"
  ;;

rl-version)
  # 只改标签不换权重；生成请求带 return_weight_versions 拿到 [start, end) 版本段
  curl -s -X POST "$VLLM/update_weight_version" -H 'Content-Type: application/json' -d '{"new_version": "v8"}'
  curl -s "$VLLM/v1/completions" -H 'Content-Type: application/json' -d "{
    \"model\": \"$MODEL\", \"prompt\": \"1+1=\", \"max_tokens\": 16,
    \"return_weight_versions\": true, \"return_token_ids\": true}"
  ;;

# ---------------------------------------------------------------- SGLang RL 对照
sgl-rl-serve)
  python -m sglang.launch_server --model-path "$MODEL" --enable-memory-saver
  ;;

sgl-rl)
  curl -s -X POST "$SGL/pause_generation" -H 'Content-Type: application/json' -d '{"mode": "retract"}'
  curl -s -X POST "$SGL/update_weights_from_disk" -H 'Content-Type: application/json' -d "{\"model_path\": \"${CKPT:-/ckpt/step-42}\", \"weight_version\": \"step-42\"}"
  curl -s -X POST "$SGL/continue_generation"
  # 显存让给同卡训练：要求引擎空闲
  curl -s -X POST "$SGL/release_memory_occupation" -H 'Content-Type: application/json' -d '{"tags": ["kv_cache", "weights"]}'
  curl -s -X POST "$SGL/resume_memory_occupation" -H 'Content-Type: application/json' -d '{}'
  ;;

# ---------------------------------------------------------------- 离线推演
sim)
  python3 "$HERE/sim_router.py" --scenario all
  ;;

*)
  grep -E '^[a-z0-9-]+\)' "$0" | tr -d ')' | tr '\n' ' '; echo
  ;;
esac
