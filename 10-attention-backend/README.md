# VLLM-10 注意力后端：FlashAttention、FlashInfer、Triton 与 MLA

运行环境：模拟器只要 CPU 和 Python 3 标准库；`gpu_backend_bench.sh` 需要一张能放下 Qwen3-8B 的 GPU 和装好的 vLLM。基线 vllm 主干 0d6f92a，sglang 8fabf5e。

| 文件 | 内容 |
|---|---|
| `sim_backend_select.py` | 复刻 `_get_backend_priorities` 的候选顺序、`validate_configuration` 的逐项校验、`get_attn_backend_cls` 的选择与日志、`--block-size` 挤占告警、首选 block、各后端的 CG 支持级别；MLA 两条计算路径的 FLOPs / 字节；每 token KV；SGLang 的 `get_default_attn_backend` |
| `test_sim_backend_select.py` | 24 个单测：候选顺序、FP8 / sinks / batch invariant 下的回退、用户指定后端、block 规则、CG 支持、MLA 账、SGLang 默认 |
| `gpu_backend_bench.sh` | 真机实验：同一负载切换 FLASH_ATTN / FLASHINFER / TRITON_ATTN / FLEX_ATTENTION；`--block-size` 与 `--kv-cache-dtype` 怎样改变自动选择 |
| `backend_matrix.md` | 选择链路逐行对照、候选顺序、稠密与 MLA 后端能力表、元数据构建差别、SGLang 对照 |

## 用法

```bash
python3 sim_backend_select.py --scenario matrix   # 典型模型 × GPU × KV dtype → 选中谁、block、CG 支持、全部合法候选
python3 sim_backend_select.py --scenario trace    # 每个候选的拒绝原因 + 选择器会打出的日志
python3 sim_backend_select.py --scenario block    # --block-size 怎样把高优先级后端挤掉；各后端首选 block
python3 sim_backend_select.py --scenario cg       # 后端 CG 支持 × 请求模式 × 投机解码 → 实际 cudagraph 模式
python3 sim_backend_select.py --scenario mla      # MLA decode 吸收 vs 解压、prefill 两条路径的 FLOPs
python3 sim_backend_select.py --scenario kv       # 每 token KV 字节：GQA / MLA / 假如 DSV3 存 MHA
python3 sim_backend_select.py --scenario sglang   # 同样的配置 SGLang 默认选谁
python3 -m unittest test_sim_backend_select

bash gpu_backend_bench.sh select                  # 每个后端起一次引擎，只看选择日志
bash gpu_backend_bench.sh latency                 # 后端 × batch 1/16/128 的端到端延迟
bash gpu_backend_bench.sh serve                   # 后端 × 并发 8/64 的 TTFT / TPOT
bash gpu_backend_bench.sh blocksize               # 不给 / 给 --block-size 16/64/128
bash gpu_backend_bench.sh kvfp8                   # kv auto / fp8 / fp8_e5m2 时自动选择的变化
```

模拟器的假设：所有内核都编译进了 wheel，FlashInfer 能下载 trtllm cubin（`artifactory=False` 可关），tokenspeed-mla 没装（`tokenspeed=True` 可开），FlashMLA 可导入。没有建模 mm_prefix 组合后端、稀疏 MLA 细则、DCP / PCP、KV connector。与真机不一致时以真机日志为准，并请对照 `backend_matrix.md` 的行号查源码。

## 期望输出

`--scenario matrix`：

```
配置                                             选中               FA block CG 支持                        合法候选
Qwen3-8B @ A100 80GB SXM                       FLASH_ATTN        2    16 UNIFORM_BATCH                FLASH_ATTN,FLASHINFER,TRITON_ATTN,FLEX_ATTENTION
Qwen3-8B @ L40S                                FLASH_ATTN        2    16 UNIFORM_BATCH                FLASH_ATTN,FLASHINFER,TRITON_ATTN,FLEX_ATTENTION
Qwen3-8B @ H100 SXM                            FLASH_ATTN        3    16 ALWAYS                       FLASH_ATTN,FLASHINFER,TRITON_ATTN,FLEX_ATTENTION
Qwen3-8B @ B200                                FLASHINFER        -    16 UNIFORM_BATCH                FLASHINFER,FLASH_ATTN,TRITON_ATTN,FLEX_ATTENTION
Qwen3-8B @ A100 80GB SXM kv=fp8                FLASHINFER        -    16 UNIFORM_BATCH                FLASHINFER,TRITON_ATTN
Qwen3-8B @ H100 SXM kv=fp8                     FLASH_ATTN        3    16 ALWAYS                       FLASH_ATTN,FLASHINFER,TRITON_ATTN
Qwen3-8B @ B200 kv=fp8                         FLASHINFER        -    16 UNIFORM_BATCH                FLASHINFER,FLASH_ATTN,TRITON_ATTN
Qwen3-8B @ H100 SXM kv=fp8_e5m2                FLASHINFER        -    16 UNIFORM_BATCH                FLASHINFER,TRITON_ATTN
Qwen3-8B @ H100 SXM batch-invariant            FLASH_ATTN        3    16 ALWAYS                       FLASH_ATTN,TRITON_ATTN,FLEX_ATTENTION
gpt-oss-120b @ A100 80GB SXM                   TRITON_ATTN       -    16 ALWAYS                       TRITON_ATTN
gpt-oss-120b @ H100 SXM                        FLASH_ATTN        3    16 ALWAYS                       FLASH_ATTN,FLASHINFER,TRITON_ATTN
gpt-oss-120b @ B200                            FLASHINFER        -    16 UNIFORM_BATCH                FLASHINFER,FLASH_ATTN,TRITON_ATTN
DeepSeek-V3 @ H100 SXM                         FLASH_ATTN_MLA    -    16 UNIFORM_BATCH                FLASH_ATTN_MLA,FLASHMLA,TRITON_MLA
DeepSeek-V3 @ H100 SXM kv=fp8                  FLASHMLA          -    64 UNIFORM_BATCH                FLASHMLA,TRITON_MLA
DeepSeek-V3 @ B200                             FLASHINFER_MLA    -    32 ALWAYS                       FLASHINFER_MLA,CUTLASS_MLA,TRITON_MLA
DeepSeek-V3 @ B200 kv=fp8 +tokenspeed          FLASHINFER_MLA    -    32 ALWAYS                       FLASHINFER_MLA,TOKENSPEED_MLA,CUTLASS_MLA,TRITON_MLA
DeepSeek-V3 @ A100 80GB SXM                    TRITON_MLA        -    16 UNIFORM_BATCH                TRITON_MLA
DeepSeek-V3 @ RTX PRO 6000                     TRITON_MLA        -    16 UNIFORM_BATCH                TRITON_MLA
```

`--scenario trace`：

```
== Qwen3-8B @ A100 80GB SXM kv=fp8
  [0] FLASH_ATTN                   ✗ ['kv_cache_dtype not supported', 'FP8 KV cache requires FA3 on SM90, FA4 with head_size=512 on SM90, or FA4 on SM100']
  [1] FLASHINFER                   ✓
  [2] TRITON_ATTN                  ✓
  [3] FLEX_ATTENTION               ✗ ['kv_cache_dtype not supported']
  [4] TURBOQUANT                   ✗ ['kv_cache_dtype not supported']
  INFO Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'TRITON_ATTN'].

== gpt-oss-120b @ A100 80GB SXM
  [0] FLASH_ATTN                   ✗ ['attention sinks not supported', 'sink not supported on compute capability < 9.0']
  [1] FLASHINFER                   ✗ ['attention sinks not supported']
  [2] TRITON_ATTN                  ✓
  [3] FLEX_ATTENTION               ✗ ['attention sinks not supported']
  [4] TURBOQUANT                   ✗ ['kv_cache_dtype not supported', 'attention sinks not supported', 'sliding window not supported']
  INFO Using TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN'].

== DeepSeek-V3 @ B200
  [0] FLASHINFER_MLA               ✓
  [1] TOKENSPEED_MLA               ✗ ['kv_cache_dtype not supported', 'tokenspeed_mla package is not installed. Install it with: `uv pip install tokenspeed-mla`']
  [2] CUTLASS_MLA                  ✓
  [3] FLASH_ATTN_MLA               ✗ ['compute capability not supported', 'FlashAttention MLA not supported on this device']
  [4] FLASHMLA                     ✗ ['FlashMLA Dense is only supported on Hopper devices.']
  [5] TRITON_MLA                   ✓
  [6] FLASHMLA_SPARSE              ✗ ['non-sparse not supported']
  [7] FLASHINFER_MLA_SPARSE        ✗ ['non-sparse not supported']
  INFO Using FLASHINFER_MLA attention backend out of potential backends: ['FLASHINFER_MLA', 'CUTLASS_MLA', 'TRITON_MLA'].
  INFO Setting kv cache block size to 32 for FLASHINFER_MLA backend(s).

== DeepSeek-V3 @ A100 80GB SXM -ab FLASHMLA
  ValueError: Selected backend AttentionBackendEnum.FLASHMLA is not valid for this configuration. Reason: ['compute capability not supported', 'FlashMLA Dense is only supported on Hopper devices.']

== Qwen3-8B @ H100 SXM kv=fp8_e5m2 -ab FLASHINFER
  INFO Using AttentionBackendEnum.FLASHINFER backend.
```

`--scenario block`：

```
== DeepSeek-V3 @ B200 --block-size 16
  [0] FLASHINFER_MLA               ✗ ['block_size not supported']
  [1] TOKENSPEED_MLA               ✗ ['kv_cache_dtype not supported', 'block_size not supported', 'tokenspeed_mla package is not installed. Install it with: `uv pip install tokenspeed-mla`']
  [2] CUTLASS_MLA                  ✗ ['block_size not supported']
  [3] FLASH_ATTN_MLA               ✗ ['compute capability not supported', 'FlashAttention MLA not supported on this device']
  [4] FLASHMLA                     ✗ ['block_size not supported', 'FlashMLA Dense is only supported on Hopper devices.']
  [5] TRITON_MLA                   ✓
  [6] FLASHMLA_SPARSE              ✗ ['non-sparse not supported']
  [7] FLASHINFER_MLA_SPARSE        ✗ ['non-sparse not supported']
  WARNING --block-size 16 precluded higher-priority backend(s) FLASHINFER_MLA, CUTLASS_MLA. Using TRITON_MLA instead, which may result in reduced performance. Consider removing --block-size to auto-select the optimal block size.
  INFO Using TRITON_MLA attention backend out of potential backends: ['TRITON_MLA'].

== DeepSeek-V3 @ B200 --block-size 64
  [0] FLASHINFER_MLA               ✓
  [1] TOKENSPEED_MLA               ✗ ['kv_cache_dtype not supported', 'tokenspeed_mla package is not installed. Install it with: `uv pip install tokenspeed-mla`']
  [2] CUTLASS_MLA                  ✗ ['block_size not supported']
  [3] FLASH_ATTN_MLA               ✗ ['compute capability not supported', 'FlashAttention MLA not supported on this device']
  [4] FLASHMLA                     ✗ ['FlashMLA Dense is only supported on Hopper devices.']
  [5] TRITON_MLA                   ✓
  [6] FLASHMLA_SPARSE              ✗ ['non-sparse not supported']
  [7] FLASHINFER_MLA_SPARSE        ✗ ['non-sparse not supported']
  INFO Using FLASHINFER_MLA attention backend out of potential backends: ['FLASHINFER_MLA', 'TRITON_MLA'].

== DeepSeek-V3 @ H100 SXM kv=fp8 --block-size 16
  [0] FLASH_ATTN_MLA               ✗ ['kv_cache_dtype not supported']
  [1] FLASHMLA                     ✗ ['block_size not supported']
  [2] FLASHINFER_MLA               ✗ ['block_size not supported', 'compute capability not supported']
  [3] TRITON_MLA                   ✓
  [4] FLASH_ATTN_MLA_SPARSE        ✗ ['non-sparse not supported']
  [5] FLASHMLA_SPARSE              ✗ ['non-sparse not supported']
  [6] FLASHINFER_MLA_SPARSE_SM90   ✗ ['non-sparse not supported']
  WARNING --block-size 16 precluded higher-priority backend(s) FLASHMLA. Using TRITON_MLA instead, which may result in reduced performance. Consider removing --block-size to auto-select the optimal block size.
  INFO Using TRITON_MLA attention backend out of potential backends: ['TRITON_MLA'].

== Qwen3-8B @ B200 --block-size 128
  [0] FLASHINFER                   ✓
  [1] FLASH_ATTN                   ✓
  [2] TRITON_ATTN                  ✓
  [3] FLEX_ATTENTION               ✓
  [4] TURBOQUANT                   ✗ ['kv_cache_dtype not supported']
  INFO Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'FLASH_ATTN', 'TRITON_ATTN', 'FLEX_ATTENTION'].

== Qwen3-8B @ H100 SXM --block-size 8
  [0] FLASH_ATTN                   ✗ ['block_size not supported']
  [1] FLASHINFER                   ✗ ['block_size not supported']
  [2] TRITON_ATTN                  ✗ ['block_size not supported']
  [3] FLEX_ATTENTION               ✗ ['block_size not supported']
  [4] TURBOQUANT                   ✗ ['kv_cache_dtype not supported', 'block_size not supported']
  ValueError: No valid attention backend found

== 不指定 --block-size 时各后端自动取的 block（get_preferred_block_size，默认 16）
  FLASH_ATTN       支持 ['mult:16']                                                  → 16
  FLASHINFER       支持 ['fix:16', 'fix:32', 'fix:64', 'fix:128', 'fix:256', 'fix:512', 'fix:1024'] → 16
  FLASHINFER_MLA   支持 ['fix:32', 'fix:64']                                         → 32
  CUTLASS_MLA      支持 ['fix:128']                                                  → 128
  FLASHMLA         支持 ['fix:64']                                                   → 64
  TRITON_MLA       支持 ['mult:16']                                                  → 16
```

`--scenario cg`：

```
后端                               CG 支持                        默认 F&P               要 FULL               F&P + EAGLE k=3     
FLASH_ATTN (FA2, A100)           UNIFORM_BATCH                FULL_AND_PIECEWISE   FULL_AND_PIECEWISE   FULL_AND_PIECEWISE  
FLASH_ATTN (FA3, H100)           ALWAYS                       FULL_AND_PIECEWISE   FULL                 FULL_AND_PIECEWISE  
FLASHINFER (A100, FA2 decode)    UNIFORM_BATCH                FULL_AND_PIECEWISE   FULL_AND_PIECEWISE   FULL_AND_PIECEWISE  
FLASHINFER (H100, XQA)           UNIFORM_BATCH                FULL_AND_PIECEWISE   FULL_AND_PIECEWISE   FULL_AND_PIECEWISE  
FLASHINFER (H100, 无 cubin)       UNIFORM_SINGLE_TOKEN_DECODE  FULL_AND_PIECEWISE   FULL_AND_PIECEWISE   PIECEWISE           
FLASHINFER (B200, trtllm)        UNIFORM_BATCH                FULL_AND_PIECEWISE   FULL_AND_PIECEWISE   FULL_AND_PIECEWISE  
TRITON_ATTN                      ALWAYS                       FULL_AND_PIECEWISE   FULL                 FULL_AND_PIECEWISE  
FLASHINFER_MLA (B200)            ALWAYS                       FULL_AND_PIECEWISE   FULL                 FULL_AND_PIECEWISE  
FLASH_ATTN_MLA (H100)            UNIFORM_BATCH                FULL_AND_PIECEWISE   FULL_AND_PIECEWISE   FULL_AND_PIECEWISE  
FLASHMLA (H100)                  UNIFORM_BATCH                FULL_AND_PIECEWISE   FULL_AND_PIECEWISE   FULL_AND_PIECEWISE  
CUTLASS_MLA (B200)               UNIFORM_SINGLE_TOKEN_DECODE  FULL_AND_PIECEWISE   FULL_AND_PIECEWISE   PIECEWISE
```

`--scenario mla`：

```
== DeepSeek-V3 单层 decode 一个 token（H=128, nope=128, rope=64, v=128, Lkv=512, BF16）
   ctx 路径              GFLOP   读 KV MiB   FLOP/B
  1024 mqa             0.319       1.12    270.2
  1024 mha            34.444       1.12  29198.2
  1024 mha_cached      0.084      80.00      1.0
  8192 mqa             2.315       9.00    245.3
  8192 mha           275.549       9.00  29198.2
  8192 mha_cached      0.671     640.00      1.0
 32768 mqa             9.160      36.00    242.7
 32768 mha          1102.196      36.00  29198.2
 32768 mha_cached      2.684    2560.00      1.0
  对照 Qwen3-8B（GQA 32/8）ctx 8192：0.134 GFLOP，读 32.00 MiB，4.0 FLOP/B
  A100 80GB SXM BF16 拐点 153 FLOP/B
  H100 SXM BF16 拐点 295 FLOP/B
  H200 BF16 拐点 206 FLOP/B

== prefill chunk：s 个新 token 看 ctx 个 token 的注意力 FLOPs（GFLOP，含 MHA 路径的解压）
  s=    1 ctx= 8192  MHA      275.5  MQA        2.3  MQA/MHA  0.01
  s=   64 ctx= 8192  MHA      317.8  MQA      146.0  MQA/MHA  0.46
  s= 2048 ctx= 2048  MHA      412.3  MQA     1168.2  MQA/MHA  2.83
  s= 8192 ctx= 8192  MHA     5772.4  MQA    18691.7  MQA/MHA  3.24
```

`--scenario kv`：

```
模型               结构                     每 token KV（BF16）          FP8     128K 上下文 GiB
Qwen3-8B         GQA 32/8                        147,456       73,728            18.00
Llama-3.1-8B     GQA 32/8                        131,072       65,536            16.00
Llama-3.1-70B    GQA 64/8                        327,680      163,840            40.00
DeepSeek-V3      MLA 512+64                       70,272       35,136             8.58
DSV3 若存 MHA      128 头 K192+V128               4,997,120    2,498,560           610.00
  MLA 相对 MHA 缩小 71.1 倍
```

`--scenario sglang`：

```
配置                                       SGLang 默认      vLLM 默认         
Qwen3-8B @ A100 80GB SXM                 flashinfer     FLASH_ATTN      
Qwen3-8B @ H100 SXM                      fa3            FLASH_ATTN      
Qwen3-8B @ H100 SXM EAGLE topk=4         flashinfer     FLASH_ATTN      
Qwen3-8B @ B200                          trtllm_mha     FLASHINFER      
Qwen3-8B @ A100 80GB SXM 无 flashinfer    triton         FLASH_ATTN      
gpt-oss-120b @ A100 80GB SXM             triton         TRITON_ATTN     
gpt-oss-120b @ H100 SXM                  fa3            FLASH_ATTN      
DeepSeek-V3 @ H100 SXM                   fa3            FLASH_ATTN_MLA  
DeepSeek-V3 @ B200                       trtllm_mla     FLASHINFER_MLA  
DeepSeek-V3 @ A100 80GB SXM              triton         TRITON_MLA
```
