# VLLM-02 推理账本

运行环境：只需要 CPU 上的 Python 3.10+，不依赖 torch 或 vllm。

对应文档：VLLM-02｜上手｜推理账本：gpu_memory_utilization、KV 容量与 Roofline

| 文件 | 做什么 | 对照的源码（vllm 主干 0d6f92a） |
|---|---|---|
| `sim_profile_run.py` | 按 `determine_available_memory` 的顺序复算一次启动，依次是 requested、memory_profiling、CUDA Graph 估计、KV 可用、块数 / token / 并发、util 提示行、`--kv-cache-memory` 建议值。加 `--sglang` 会追加 SGLang 默认 `mem_fraction_static` 的推导 | `v1/worker/utils.py`、`utils/mem_utils.py`、`v1/worker/gpu_worker.py`、`v1/worker/gpu/cudagraph_utils.py`、`config/vllm.py`、`v1/core/kv_cache_planning.py`、`v1/core/kv_cache_utils.py` |
| `sim_ledger.py` | 事前估算：不看日志，只看模型结构和 GPU 规格。输出权重、激活、KV 池、可分配块数（已扣 null block）、日志口径的 token 数，以及按 HBM 带宽算的 decode 下限 | 同上，外加 `common/specs.py` |

## 1. 逐行复算示意日志

```bash
python3 sim_profile_run.py
python3 sim_profile_run.py --log ../01-first-server/sample_startup.log   # 从日志抽数字
python3 sim_profile_run.py --sglang                                      # 加 SGLang 对照
python3 sim_profile_run.py --no-cg-estimate                              # 模拟 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
```

默认输出（H100 80GB SXM + Qwen3-8B BF16，util 0.92，max_num_seqs 1024）：

```
[1] request_memory  (v1/worker/utils.py)
    requested = ceil(85,045,231,616 x 0.92) = 78,241,613,087 B = 72.87 GiB
[2] memory_profiling  (utils/mem_utils.py)
    total_consumed = 启动前 free - profile 后 free = 15.89 GiB（权重 15.27 + 非 torch 0.62）
    transient_peak_headroom = torch 峰值 - torch 常驻 = 1.43 GiB
    non_kv_cache_memory = 17.32 GiB
[3] profile_cudagraph_memory  (v1/worker/gpu/cudagraph_utils.py)
    捕获网格 51 个尺寸：[1, 2, 4, 8, 16, 24] ... [464, 480, 496, 512]  ← 日志 Capturing CUDA graphs ... 51/51
[4] Available KV cache memory  (gpu_worker.py)
    72.87 - 17.32 - 0.71 = 54.84 GiB
[5] 块与 token
    num_blocks = 54.84 GiB // page = 24,957，其中 1 块是 null block，可用 24,956
    并发 = 24,957 / cdiv(40960, 16) = 9.75x
    GPU KV cache size = int(9.7488 x 40960) = 399,312 tokens
[6] CUDA graph util 提示行
    delta = 0.71 / 79.20 = 0.008964；等价旧口径 0.9110，保持旧 KV 需调到 0.9290
[7] --kv-cache-memory 建议值
    贴着 requested：57,994,617,159 B（54.01 GiB）
    贴着整卡空闲：  64,261,364,776 B（59.85 GiB）
[8] SGLang（--sglang）
    reserved = 512 + 12288（激活）+ 128（并行）+ 976（decode 图 256x2 + prefill 图 58x8） = 13,904 MiB
    mem_fraction_static = round((81559 - 13904) / 81559, 3) = 0.830
```

每一行都能在 `../01-first-server/sample_startup.log` 里找到对应的数字。用自己的日志跑 `--log` 时，由于日志里的 GiB 只保留两位小数，反推结果会有 0.01 GiB、几十个 token 的取整偏差，属于正常现象。

## 2. 事前估算

```bash
python3 sim_ledger.py --model qwen3-8b --gpu h100-sxm
python3 sim_ledger.py --total-gib 79.2 --act-gib 1.43 --nontorch-gib 0.62 --graph-gib 0.71   # 用实测值校准
python3 sim_ledger.py --model qwen3-8b --tp 2
python3 sim_ledger.py --model llama3.1-70b --gpu h100-sxm --tp 4 --kv-dtype fp8
python3 sim_ledger.py --model deepseek-v3 --gpu h200 --tp 8 --weight-dtype fp8 --max-model-len 163840
```

| 场景 | KV 可用 | 每卡每 token | 日志口径 token |
|---|---|---|---|
| Qwen3-8B，H100，默认粗估 | 55.43 GiB | 144 KiB | 403,616 |
| Qwen3-8B，H100，实测值校准 | 54.85 GiB | 144 KiB | 399,392（日志 399,312，差 0.02%） |
| Qwen3-8B，2 × H100 | 63.31 GiB | 72 KiB | 921,968 |
| Llama-3.1-70B，4 × H100，FP8 KV | 38.15 GiB | 40 KiB | 1,000,048 |
| DeepSeek-V3，8 × H200，FP8 权重 | 48.61 GiB | 68.6 KiB | 742,672 |

DeepSeek-V3 这一行按 MLA 的 latent 计算。MLA 下每张卡都存一份完整的 latent 副本，所以 TP 不会摊薄每 token 的 KV。稀疏 MLA 的 `fp8_ds_mla` 格式每 token 每层只占 656 B，比这里 BF16 的 1,152 B 更小，第 10 章再细算。

## 3. 两个脚本对不上日志时

1. 先对 `Available KV cache memory`。如果差得多，用 `VLLM_LOGGING_LEVEL=DEBUG` 重启，从 `Memory profiling takes …` 那一行取 torch peak 和 non-torch 两个值，分别填进 `--act-gib` 和 `--nontorch-gib`。
2. KV 对得上而 token 对不上，说明每 token 字节数算错了。依次检查 `--kv-dtype`、模型是否带滑窗或混合层（第 7 章），以及 MLA 后端的块大小和对齐。
3. 日志的 token 数比本脚本多 16 的整数倍：这是因为日志口径不扣 null block。本脚本把两种口径都打印了。
