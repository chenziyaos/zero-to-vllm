# VLLM-01 第一个服务

运行环境：GPU（serve.sh、smoke_test.py），任意机器（parse_startup_log.py）

验证基线：vLLM v0.32.0 / 主干 0d6f92a，CUDA 13.0，驱动 R580+。

## 文件

| 文件 | 作用 | 需要 GPU |
|---|---|---|
| serve.sh | 起 Qwen3-8B，带 reasoning parser 与工具调用，日志 tee 到 logs/ | 是 |
| smoke_test.py | 六步冒烟：/health、/v1/models、chat 非流式与流式（TTFT、ITL）、前缀缓存、/metrics | 否，但要有服务 |
| parse_startup_log.py | 把启动日志压成一张表，可选和 02-ledger 账本对账 | 否 |
| sample_startup.log | 示意日志：格式逐行取自主干 logger 格式串，数字按 H100 80GB + Qwen3-8B 推算，不是一次真实启动的拷贝 | 否 |

## 跑一遍

```bash
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install vllm --torch-backend=auto
vllm download-kernels          # 不跑这步，Hopper 及以上首启会现场编译 FlashInfer kernel

bash serve.sh                  # 另开一个终端
python3 smoke_test.py
python3 parse_startup_log.py logs/serve-*.log --model qwen3-8b --gpu h100-sxm
```

没有 GPU 也可以先读日志：

```bash
python3 parse_startup_log.py sample_startup.log --model qwen3-8b --gpu h100-sxm
python3 parse_startup_log.py sample_startup.log --json
```

## 期望输出（sample_startup.log）

```text
  版本 / 模型                  0.32.0 / Qwen/Qwen3-8B
  架构                         Qwen3ForCausalLM
  进程                         APIServer(12301), EngineCore(12388)
  max_model_len                40960
  max_num_batched_tokens       8192
  注意力后端                   FLASH_ATTN
  Model Runner                 V2
  权重显存 GiB / 加载 s        15.27 / 11.284317
  torch.compile s（缓存命中）  41.09（否）
  KV 可用 GiB                  54.84
  KV 容量 token                399,312
  max_model_len 下最大并发     9.75
  CUDA Graph 捕获 s / GiB      7 / 0.68
  可回填 --kv-cache-memory     57994617159
  init engine s                59.73
  首行到末行 s                 90
  监听地址                     http://0.0.0.0:8000

  和账本对账（误差 = 日志 / 账本 - 1）
    权重 GiB         日志        15.27  账本        15.26  误差 +0.1%
    KV 可用 GiB      日志        54.84  账本        55.25  误差 -0.7%
    KV token       日志   399,312.00  账本   402,304.00  误差 -0.7%
```

自己的日志误差超过 10% 时，先看三件事：卡上是否有别的进程（Free memory on device 一行），
`--max-num-batched-tokens` 是否和默认值不同（决定 profile 的激活峰值），是否开了 `--enforce-eager`（Graph 显存为 0）。

## 第二次启动

同一个模型、同样参数再起一次，`torch.compile` 一行应变成
`Directly load the compiled graph(s) for compile range (1, 8192) from the cache`，解析结果里“缓存命中”变成“是”。
把日志里 `--kv-cache-memory=` 的数字回填给 `--kv-cache-memory-bytes`，可以跳过显存 profiling：

```bash
EXTRA="--kv-cache-memory-bytes 57994617159" bash serve.sh
```

## 失败时

parse_startup_log.py 识别四类启动失败并给出处理建议，退出码为 1，可以直接挂进部署流水线：

| 键 | 日志特征 | 先做什么 |
|---|---|---|
| err_free_mem | Free memory on device … is less than desired GPU memory utilization | nvidia-smi 找占卡进程，或调低 util |
| err_no_mem | No available memory for the cache blocks | 权重加激活已经吃满，换量化权重或加卡 |
| err_max_len | To serve at least one request with the model's max seq len | 调低 --max-model-len，报错里会给出估算上限 |
| err_ready_timeout | Timed out waiting for engine core processes to start | 预先下载权重，或调大 VLLM_ENGINE_READY_TIMEOUT_S |
