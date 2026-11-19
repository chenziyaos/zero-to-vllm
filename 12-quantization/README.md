# VLLM-12 量化：FP8、INT4、NVFP4 与 KV cache 量化

运行环境：模拟器只要 CPU 和 Python 3 标准库；`gpu_quant_eval.sh` 需要一张 H100（NVFP4 最好用 B200）和装好的 vLLM，SGLang 对照另装 sglang。基线 vllm 主干 0d6f92a，sglang 8fabf5e。

| 文件 | 内容 |
|---|---|
| `sim_quant_mem.py` | 每种权重格式每参数字节、四个模型在 H100 / H200 上的权重 + KV 余量；22 种 KV dtype 中常用的 13 种每 token 字节（含 TurboQuant 槽位与首尾层保护、MLA 稀疏布局）；`kernels/linear/__init__.py` 候选列表 + is_supported / can_implement 回放（SM75 ~ SM120）；`--linear-backend` 过滤与回退；TP 切分对 group / block 的对齐；FP8 KV 的 ITL 直线与盈亏点；SGLang 参数对照 |
| `test_sim_quant_mem.py` | 14 个单测：字节账、TurboQuant 槽位、首尾层、Humming 只在 SM90 提前、FP8 block / NVFP4 / AWQ 的 kernel 选择、批不变性、backend 回退、Qwen2.5-72B 补齐、FP8 KV 盈亏点 |
| `gpu_quant_eval.sh` | 真机实验：BF16 / 在线 FP8 / FP8 block / AWQ / GPTQ / NVFP4 / FP8 KV / TurboQuant KV / 指定后端九种起法 + bench + gsm8k，和 SGLang 三种对照起法 |
| `quant_matrix.md` | 注册、kernel 选择、形状约束、KV 量化的源码行号，以及与 SGLang 的参数对照 |

## 用法

```bash
python3 sim_quant_mem.py --scenario weights   # 每参数几字节，放不放得下，剩多少 KV
python3 sim_quant_mem.py --scenario kv        # 每种 KV dtype 每 token 几字节
python3 sim_quant_mem.py --scenario kernel    # SM75 ~ SM120 上每种量化落到哪个 kernel
python3 sim_quant_mem.py --scenario tp        # TP 切分后 group / block 对不对得齐
python3 sim_quant_mem.py --scenario backend   # --linear-backend 过滤与回退
python3 sim_quant_mem.py --scenario fp8kv     # FP8 KV 的盈亏点
python3 sim_quant_mem.py --scenario sglang    # 和 SGLang 的参数对照
python3 -m unittest test_sim_quant_mem

bash gpu_quant_eval.sh bf16 && bash gpu_quant_eval.sh bench   # 先基线（两个终端）
bash gpu_quant_eval.sh fp8                                     # 换一种格式再压一次
bash gpu_quant_eval.sh backend marlin                          # 强制 Marlin，看日志里选中的 kernel
```

模拟器的假设：权重只算线性层数据和 scale，embedding / lm_head / MoE gate 保持 BF16，忽略 norm 和 bias；每卡激活 + 非 torch + CUDA Graph 合计记 6 GiB；KV 默认 BF16。kernel 回放假设官方 CUDA 13 wheel、flashinfer / cutlass-dsl / humming-kernels 已装、DeepGEMM 可用，b12x / conch / fbgemm / quark 没装；只回放 CUDA 路径，ROCm / XPU / CPU 不在内。FP8 KV 的 ITL 系数取自 vLLM 2026-04-22 博客。与真机不一致时以真机日志为准。

## 期望输出

`--scenario weights`：

```
== 每参数字节（线性层；embedding / lm_head / gate 保持 BF16）
  bf16         2.00000 B   BF16 原样
  fp8-tensor   1.00000 B   FP8 E4M3，每张量 1 个 FP32 scale
  fp8-block    1.00024 B   FP8 E4M3，128x128 块 1 个 FP32 scale（DeepSeek-V3）
  mxfp8        1.03125 B   MXFP8，每 32 个 1 字节 E8M0 scale
  gptq-int4    0.51562 B   INT4 g128 对称，FP16 scale
  awq-int4     0.51953 B   INT4 g128 非对称，FP16 scale + 4-bit zero point
  nvfp4        0.56250 B   NVFP4 E2M1，每 16 个 1 字节 E4M3 scale + 每张量 FP32
  mxfp4        0.53125 B   MXFP4 E2M1，每 32 个 1 字节 E8M0 scale

模型                     总参数 B       bf16 fp8-tensor  fp8-block      mxfp8  gptq-int4   awq-int4      nvfp4      mxfp4
Llama-3.1-8B             8.0       15.0        8.5        8.5        8.7        5.3        5.3        5.6        5.4
Llama-3.1-70B           70.6      131.4       67.7       67.7       69.7       36.8       37.0       39.8       37.8
Qwen3-235B-A22B        235.1      437.9      220.2      220.2      227.0      114.7      115.5      124.9      118.1
DeepSeek-V3            671.0     1249.9      626.8      626.9      646.2      324.9      327.4      354.2      334.7
（单位 GiB，整模型）

== Llama-3.1-70B @ H100 SXM，util 0.9，每卡杂项 6 GiB，KV BF16
格式           TP     每卡权重 GiB     每卡 KV 余量 GiB     每卡可放 token
bf16          1        131.4            -66.1          放不下
bf16          2         65.7             -0.4          放不下
bf16          4         32.9             32.4        425,020
bf16          8         16.4             48.9      1,280,658
fp8-tensor    1         67.7             -2.4          放不下
fp8-tensor    2         33.8             31.4        206,097
fp8-tensor    4         16.9             48.4        633,916
fp8-tensor    8          8.5             56.8      1,489,554
awq-int4      1         37.0             28.2         92,556
awq-int4      2         18.5             46.8        306,465
awq-int4      4          9.3             56.0        734,284
awq-int4      8          4.6             60.7      1,589,922
nvfp4         1         39.8             25.5         83,580
nvfp4         2         19.9             45.4        297,489
nvfp4         4          9.9             55.3        725,308
nvfp4         8          5.0             60.3      1,580,946

== Qwen3-235B-A22B @ H200，util 0.9，每卡杂项 6 GiB，KV BF16
格式           TP     每卡权重 GiB     每卡 KV 余量 GiB     每卡可放 token
bf16          2        219.0            -99.4          放不下
bf16          4        109.5             10.1        225,076
bf16          8         54.9             64.7      1,443,958
fp8-block     2        110.1              9.5        105,776
fp8-block     4         55.1             64.5      1,439,244
fp8-block     8         27.7             92.0      2,052,065
awq-int4      2         57.8             61.8        689,585
awq-int4      4         29.0             90.7      2,023,052
awq-int4      8         14.5            105.1      2,344,462
nvfp4         2         62.5             57.1        637,401
nvfp4         4         31.3             88.3      1,970,868
nvfp4         8         15.7            103.9      2,318,326

== DeepSeek-V3 @ H200，util 0.9，每卡杂项 6 GiB，KV BF16
格式           TP     每卡权重 GiB     每卡 KV 余量 GiB     每卡可放 token
bf16          8        157.9            -38.3          放不下
fp8-block     8         79.3             40.3        616,387
nvfp4         8         44.9             74.8      1,142,395
mxfp4         8         42.4             77.2      1,179,947

读法：同一张卡上，权重省下来的每个 GiB 都变成 KV；KV 余量翻倍 ≈ 能同时跑的长请求翻倍。
```

`--scenario kv`：

```
== Llama-3.1-8B（32 层，8 个 KV 头，head_dim 128）@ H100 TP1，BF16 权重，KV 余量 50.3 GiB
--kv-cache-dtype           每头每层 B  每 token KB     压缩比     可放 token  说明
auto                          512      128.00   1.00x      412,246  跟随模型 dtype
fp8                           256       64.00   2.00x      824,492  E4M3，scale 默认 1.0，校准后才准
int8_per_token_head           264       66.00   1.94x      799,507  每头每 token 一个 FP32 scale
int4_per_token_head           136       34.00   3.76x    1,551,984  同上，4 bit
nvfp4                         144       36.00   3.56x    1,465,763  FlashInfer 后端，每 16 个一个 E4M3 scale
turboquant_k8v4               196       58.88   2.17x      896,263  首尾各 2 层保持 BF16；docstring：2.6x, +1.17% PPL
turboquant_4bit_nc            134       45.31   2.82x    1,164,523  首尾各 2 层保持 BF16；docstring：3.8x, +2.71%
turboquant_k3v4_nc            118       41.81   3.06x    1,262,002  首尾各 2 层保持 BF16；docstring：~3.5x, +10.63%
turboquant_3bit_nc            102       38.31   3.34x    1,377,291  首尾各 2 层保持 BF16；docstring：4.9x, +20.59%

== TurboQuant 槽位字节（head_dim 128）：key = FP8 时 hd，否则 ceil(hd*b/8)+2；value = ceil(hd*b/8)+4；对齐到偶数
  turboquant_k8v4      K8 V4 NC=False 槽位 196 B / BF16 512 B = 2.61x
  turboquant_4bit_nc   K4 V4 NC=True  槽位 134 B / BF16 512 B = 3.82x
  turboquant_k3v4_nc   K3 V4 NC=True  槽位 118 B / BF16 512 B = 4.34x
  turboquant_3bit_nc   K3 V3 NC=True  槽位 102 B / BF16 512 B = 5.02x
  首尾保护：32 层的稠密模型跳过 [0, 1, 30, 31]；混合模型（Mamba / 线性注意力）返回 []

== MLA（DeepSeek 系）每层每 token 字节：latent 512 + rope 64
  auto            1152 B  x61 层 =   68.6 KB/token  1.00x
  fp8              576 B  x61 层 =   34.3 KB/token  2.00x
  fp8_ds_mla       656 B  x61 层 =   39.1 KB/token  1.76x
  nvfp4_ds_mla     352 B  x61 层 =   21.0 KB/token  3.27x
  fp8_ds_mla / nvfp4_ds_mla 是 DeepSeek-V3.2 稀疏 MLA（FlashMLA sparse）的专用布局，nvfp4_ds_mla 只在 SM100。
```

`--scenario kernel`：

```
== 每种量化在不同 GPU 上落到哪个线性层 kernel（默认环境：官方 CUDA 13 wheel，flashinfer + humming 已装）
-- FP8 静态 per-tensor（ModelOpt FP8）
  T4 / sm75              MarlinFP8ScaledMM（前面跳过 5 个，第一个原因：FlashInferFP8ScaledMM: FlashInfer + >=sm_100）
  A100 / sm80            MarlinFP8ScaledMM（前面跳过 5 个，第一个原因：FlashInferFP8ScaledMM: FlashInfer + >=sm_100）
  L40S / sm89            CutlassFP8ScaledMM（前面跳过 1 个，第一个原因：FlashInferFP8ScaledMM: FlashInfer + >=sm_100）
  H100 / sm90            CutlassFP8ScaledMM（前面跳过 1 个，第一个原因：FlashInferFP8ScaledMM: FlashInfer + >=sm_100）
  B200 / sm100           FlashInferFP8ScaledMM
  RTX PRO 6000 / sm120   FlashInferFP8ScaledMM
-- FP8 per-channel + 动态 per-token（FP8_DYNAMIC）
  T4 / sm75              MarlinFP8ScaledMM（前面跳过 5 个，第一个原因：FlashInferFP8ScaledMM: FlashInfer + >=sm_100）
  A100 / sm80            MarlinFP8ScaledMM（前面跳过 5 个，第一个原因：FlashInferFP8ScaledMM: FlashInfer + >=sm_100）
  L40S / sm89            CutlassFP8ScaledMM（前面跳过 1 个，第一个原因：FlashInferFP8ScaledMM: FlashInfer + >=sm_100）
  H100 / sm90            CutlassFP8ScaledMM（前面跳过 1 个，第一个原因：FlashInferFP8ScaledMM: FlashInfer + >=sm_100）
  B200 / sm100           CutlassFP8ScaledMM（前面跳过 1 个，第一个原因：FlashInferFP8ScaledMM: needs per-tensor scales）
  RTX PRO 6000 / sm120   CutlassFP8ScaledMM（前面跳过 1 个，第一个原因：FlashInferFP8ScaledMM: needs per-tensor scales）
-- FP8 block 128（DeepSeek-V3）
  T4 / sm75              MarlinFP8ScaledMM（前面跳过 4 个，第一个原因：FlashInferFp8DeepGEMMDynamicBlockScaled: needs FlashInfer sm90 blockscale AND DeepGEMM）
  A100 / sm80            MarlinFP8ScaledMM（前面跳过 4 个，第一个原因：FlashInferFp8DeepGEMMDynamicBlockScaled: needs FlashInfer sm90 blockscale AND DeepGEMM）
  L40S / sm89            MarlinFP8ScaledMM（前面跳过 4 个，第一个原因：FlashInferFp8DeepGEMMDynamicBlockScaled: needs FlashInfer sm90 blockscale AND DeepGEMM）
  H100 / sm90            FlashInferFp8DeepGEMMDynamicBlockScaled
  B200 / sm100           DeepGemmFp8BlockScaledMM（前面跳过 1 个，第一个原因：FlashInferFp8DeepGEMMDynamicBlockScaled: needs FlashInfer sm90 blockscale AND DeepGEMM）
  RTX PRO 6000 / sm120   DeepGemmFp8BlockScaledMM（前面跳过 1 个，第一个原因：FlashInferFp8DeepGEMMDynamicBlockScaled: needs FlashInfer sm90 blockscale AND DeepGEMM）
-- AWQ INT4 g128 + zp（BF16 激活）
  T4 / sm75              Marlin（前面跳过 2 个，第一个原因：CutlassW4A8 min capability 90）
  A100 / sm80            Marlin（前面跳过 2 个，第一个原因：CutlassW4A8 min capability 90）
  L40S / sm89            Marlin（前面跳过 2 个，第一个原因：CutlassW4A8 min capability 90）
  H100 / sm90            Machete（前面跳过 1 个，第一个原因：CutlassW4A8: needs sm90 + FP8 activation）
  B200 / sm100           Marlin（前面跳过 2 个，第一个原因：CutlassW4A8: needs sm90 + FP8 activation）
  RTX PRO 6000 / sm120   Marlin（前面跳过 2 个，第一个原因：CutlassW4A8: needs sm90 + FP8 activation）
-- GPTQ INT4 g128（FP16 激活）
  T4 / sm75              Marlin（前面跳过 2 个，第一个原因：CutlassW4A8 min capability 90）
  A100 / sm80            Marlin（前面跳过 2 个，第一个原因：CutlassW4A8 min capability 90）
  L40S / sm89            Marlin（前面跳过 2 个，第一个原因：CutlassW4A8 min capability 90）
  H100 / sm90            Machete（前面跳过 1 个，第一个原因：CutlassW4A8: needs sm90 + FP8 activation）
  B200 / sm100           Marlin（前面跳过 2 个，第一个原因：CutlassW4A8: needs sm90 + FP8 activation）
  RTX PRO 6000 / sm120   Marlin（前面跳过 2 个，第一个原因：CutlassW4A8: needs sm90 + FP8 activation）
-- NVFP4 W4A4（ModelOpt NVFP4）
  T4 / sm75              MarlinNvFp4（前面跳过 5 个，第一个原因：FlashInferCuteDslNvFp4: sm10x + FlashInfer CuTe-DSL）
  A100 / sm80            MarlinNvFp4（前面跳过 5 个，第一个原因：FlashInferCuteDslNvFp4: sm10x + FlashInfer CuTe-DSL）
  L40S / sm89            MarlinNvFp4（前面跳过 5 个，第一个原因：FlashInferCuteDslNvFp4: sm10x + FlashInfer CuTe-DSL）
  H100 / sm90            HummingNvFp4（前面跳过 5 个，第一个原因：FlashInferCuteDslNvFp4: sm10x + FlashInfer CuTe-DSL）
  B200 / sm100           FlashInferCuteDslNvFp4
  RTX PRO 6000 / sm120   FlashInferCutlassNvFp4（前面跳过 1 个，第一个原因：FlashInferCuteDslNvFp4: sm10x + FlashInfer CuTe-DSL）
-- NVFP4 W4A16
  T4 / sm75              MarlinNvFp4
  A100 / sm80            MarlinNvFp4
  L40S / sm89            MarlinNvFp4
  H100 / sm90            HummingNvFp4（前面跳过 1 个，第一个原因：FlashInferCuteDslNvFp4W4A16: sm100/103/12x）
  B200 / sm100           FlashInferCuteDslNvFp4W4A16
  RTX PRO 6000 / sm120   MarlinNvFp4
-- MXFP4 权重 + BF16 激活
  T4 / sm75              MarlinMxFp4（前面跳过 1 个，第一个原因：FlashInferMxFp4: FlashInfer + >=sm_100）
  A100 / sm80            MarlinMxFp4（前面跳过 1 个，第一个原因：FlashInferMxFp4: FlashInfer + >=sm_100）
  L40S / sm89            MarlinMxFp4（前面跳过 1 个，第一个原因：FlashInferMxFp4: FlashInfer + >=sm_100）
  H100 / sm90            HummingMxFp4（前面跳过 1 个，第一个原因：FlashInferMxFp4: FlashInfer + >=sm_100）
  B200 / sm100           MarlinMxFp4（前面跳过 1 个，第一个原因：FlashInferMxFp4: only supports MXFP4 dynamic activation）
  RTX PRO 6000 / sm120   MarlinMxFp4（前面跳过 1 个，第一个原因：FlashInferMxFp4: only supports MXFP4 dynamic activation）
-- MXFP4 权重 + MXFP4 激活
  T4 / sm75              MarlinMxFp4（前面跳过 1 个，第一个原因：FlashInferMxFp4: FlashInfer + >=sm_100）
  A100 / sm80            MarlinMxFp4（前面跳过 1 个，第一个原因：FlashInferMxFp4: FlashInfer + >=sm_100）
  L40S / sm89            MarlinMxFp4（前面跳过 1 个，第一个原因：FlashInferMxFp4: FlashInfer + >=sm_100）
  H100 / sm90            HummingMxFp4（前面跳过 1 个，第一个原因：FlashInferMxFp4: FlashInfer + >=sm_100）
  B200 / sm100           FlashInferMxFp4
  RTX PRO 6000 / sm120   FlashInferMxFp4

== H100 上的三个开关
  FP8 block，默认                             → FlashInferFp8DeepGEMMDynamicBlockScaled
  FP8 block，没有 nvcc（DeepGEMM JIT 不可用）      → CutlassFp8BlockScaledMM
  FP8 block，VLLM_USE_DEEP_GEMM=0           → CutlassFp8BlockScaledMM
  NVFP4 W4A4，默认（humming 已装）               → HummingNvFp4
  NVFP4 W4A4，卸掉 humming-kernels           → MarlinNvFp4
  AWQ 但 N=8200（Machete 要求 N%128）            → Humming
  同上，VLLM_DISABLED_KERNELS=Machete（N=8192）   → Humming

== Blackwell 上 DeepGEMM 的模型黑名单（utils/deep_gemm.py:29）
  B200 model_type=llama             FP8 block → DeepGemmFp8BlockScaledMM
  B200 model_type=qwen3_5_moe_text  FP8 block → CutlassFp8BlockScaledMM
  B200 model_type=kolibri1          FP8 block → DeepGemmFp8BlockScaledMM

== 量化方法自己的最低算力（config/vllm.py:945，不满足直接启动报错）
  FP8 75，ModelOpt FP8 80，ModelOpt NVFP4 75，AWQ 75，GPTQ 60，MXFP4 80，compressed-tensors 70
  V100（sm70）上 AWQ 直接拒绝；GPTQ 能起：Marlin 要 75 跳过，FP16 激活落到 Exllama，BF16 激活落到 TritonW4A16。
  sm70 GPTQ fp16 → Exllama
  sm70 GPTQ bf16 → TritonW4A16
```

`--scenario tp`：

```
== Qwen2.5-72B：down_proj 的 in_features = intermediate_size，按 TP 切 K（RowParallelLinear）
intermediate  TP    每卡 K  K%128   Marlin INT4 g128  FP8 block 128
       29568   1   29568      0               通过           通过
       29568   2   14784     64               拒绝           拒绝
       29568   4    7392     96               拒绝           拒绝
       29568   8    3696    112               拒绝           拒绝
  ↑ Qwen2.5-72B-Instruct（BF16）
       29696   1   29696      0               通过           通过
       29696   2   14848      0               通过           通过
       29696   4    7424      0               通过           通过
       29696   8    3712      0               通过           通过
  ↑ Qwen2.5-72B-Instruct-GPTQ-Int4 / AWQ

  Marlin 的拒绝原因：in_features per partition 3696 is not divisible by group_size = 128.
  FP8 block 的拒绝原因：Weight input_size_per_partition = 3696 is not divisible by weight quantization block_k = 128.
  量化发布版把 29568 补零到 29696 = 232 x 128，TP 2/4/8 每卡都是整数个 group。

== DeepSeek-V3 FP8 block 128：MoE 专家 intermediate 2048、稠密 MLP 18432（validate_fp8_block_shape_moe）
  TP8   专家每卡  256（通过）  稠密每卡  2304（通过）
  TP16  专家每卡  128（通过）  稠密每卡  1152（通过）
  TP32  专家每卡   64（拒绝）  稠密每卡   576（拒绝）
  所以 TP32 跑不了 DeepSeek-V3 的 FP8 block 权重；超过 16 卡要用 DP + EP（专家整块放，不切 intermediate）。

== Llama-3.1-70B（28672）
  TP2   每卡 K  14336  K%128 =   0  通过
  TP4   每卡 K   7168  K%128 =   0  通过
  TP8   每卡 K   3584  K%128 =   0  通过
  TP16  每卡 K   1792  K%128 =   0  通过
```

`--scenario backend`：

```
== --linear-backend 只是过滤候选列表，没有匹配就回退到 auto（不会启动失败）
GPU    --linear-backend     NVFP4 W4A4                     FP8 block                                FP8 channel                    MXFP4 W4A16     
sm100  auto                 FlashInferCuteDslNvFp4         DeepGemmFp8BlockScaledMM                 CutlassFP8ScaledMM             MarlinMxFp4     
sm100  flashinfer_cutedsl   FlashInferCuteDslNvFp4         DeepGemmFp8BlockScaledMM                 CutlassFP8ScaledMM             MarlinMxFp4       （3 条回退 WARNING）
sm100  cutlass              CutlassNvFp4                   CutlassFp8BlockScaledMM                  CutlassFP8ScaledMM             MarlinMxFp4       （1 条回退 WARNING）
sm100  marlin               MarlinNvFp4                    MarlinFP8ScaledMM                        MarlinFP8ScaledMM              MarlinMxFp4     
sm90   auto                 HummingNvFp4                   FlashInferFp8DeepGEMMDynamicBlockScaled  CutlassFP8ScaledMM             HummingMxFp4    
sm90   marlin               MarlinNvFp4                    MarlinFP8ScaledMM                        MarlinFP8ScaledMM              MarlinMxFp4     
sm90   humming              HummingNvFp4                   HummingFP8ScaledMM                       HummingFP8ScaledMM             HummingMxFp4    
sm90   deep_gemm            HummingNvFp4                   DeepGemmFp8BlockScaledMM                 CutlassFP8ScaledMM             HummingMxFp4      （3 条回退 WARNING）
sm90   triton               HummingNvFp4                   TritonFp8BlockScaledMM                   CutlassFP8ScaledMM             HummingMxFp4      （3 条回退 WARNING）

== linear_backend_per_quant 优先于 linear_backend（_get_linear_backend）
  sm100 linear_backend=auto    per_quant=None                         NVFP4 W4A16 → FlashInferCuteDslNvFp4W4A16  FP8 block → DeepGemmFp8BlockScaledMM
  sm100 linear_backend=auto    per_quant={'nvfp4_w4a16': 'humming'}   NVFP4 W4A16 → HummingNvFp4                 FP8 block → DeepGemmFp8BlockScaledMM
  sm100 linear_backend=marlin  per_quant={'nvfp4_w4a16': 'humming'}   NVFP4 W4A16 → HummingNvFp4                 FP8 block → MarlinFP8ScaledMM
  sm120 linear_backend=auto    per_quant=None                         NVFP4 W4A16 → MarlinNvFp4                  FP8 block → DeepGemmFp8BlockScaledMM
  W4A16 在 auto 下是 "强制" 选：sm100/103 有 CuTe-DSL 就用它，非 sm90 一律 Marlin，只有 sm90 走列表（Humming 在 Marlin 前）。

== VLLM_BATCH_INVARIANT=1 时 NVFP4 线性层
  sm100 → CutlassNvFp4
  sm90 → EmulationNvFp4
```

`--scenario fp8kv`：

```
== ITL = slope x context + intercept（单请求 decode，vLLM 博客 2026-04-22 的拟合值）
模型             GPU    slope 比    盈亏点 token       ITL@4k BF16/FP8 ms      ITL@32k BF16/FP8 ms     ITL@128k BF16/FP8 ms
Llama-3.1-8B   H100      54%        7,000        6.62 / 6.68              7.87 / 7.36             12.17 / 9.69      
Llama-3.1-8B   B200      54%        3,623        4.00 / 4.00              4.52 / 4.28              6.29 / 5.23      
gpt-oss-20b    B200      58%       13,333        3.16 / 3.18              3.27 / 3.24              3.62 / 3.44      

== 博客 Table 3（H100）：两级累加修好精度之后，kernel 调优把盈亏点拉回来
  Llama-3.1-8B   v0.10.2   FP8          盈亏点   24,889 token  slope 63% of BF16
  Llama-3.1-8B   v0.19.1   FP8          盈亏点    7,010 token  slope 54% of BF16
  gpt-oss-20b    v0.10.2   FP8          盈亏点  741,565 token  slope 96% of BF16
  gpt-oss-20b    v0.19.1   FP8          盈亏点   22,109 token  slope 80% of BF16
  gpt-oss-20b    v0.19.1   FP8 skip-SW  盈亏点    7,659 token  slope 71% of BF16

  并发 8 的吞吐：Llama-3.1-8B +14.9%（450.3 → 517.5 tok/s，ITL 15.18 → 12.93 ms），gpt-oss-20b +4.8%。
  gemma-4-E2B（head_dim 256）：decode slope 68%，但 prefill TTFT 二次项 6.93e-07 → 1.12e-06（约 1.6x 更慢）。
```

`--scenario sglang`：

```
== 同一份 checkpoint 两边怎么起
  BF16 权重在线转 FP8         vLLM:   --quantization fp8_per_tensor（fp8 已不做在线量化，会报错）
                         SGLang: --quantization fp8（仍然在线量化）
  FP8 checkpoint（DeepSeek-V3） vLLM:   自动识别 quantization_config，不用传
                         SGLang: 同左；传了不兼容的方法会报 does not match
  AWQ / GPTQ             vLLM:   自动；awq / awq_marlin / auto_awq 都映射到 AutoAWQConfig
                         SGLang: awq_marlin / gptq_marlin，awq 走非优化路径会 WARNING
  ModelOpt NVFP4         vLLM:   自动（modelopt_fp4）
                         SGLang: --quantization modelopt_fp4，或 nvfp4_online 在线转
  指定 GEMM 后端             vLLM:   --linear-backend X，或 kernel_config.linear_backend_per_quant
                         SGLang: --fp8-gemm-backend / --fp4-gemm-backend
  KV cache FP8           vLLM:   --kv-cache-dtype fp8（scale 读 checkpoint，没有就 1.0 + WARNING）
                         SGLang: --kv-cache-dtype fp8_e4m3 + --quantization-param-path
  KV cache 4 bit         vLLM:   nvfp4 / int4_per_token_head / turboquant_*
                         SGLang: nvfp4 / fp4_mx_block16

== FP8 GEMM 的 auto 顺序
  vLLM  block：FlashInfer+DeepGEMM 组合（仅 sm90）→ DeepGEMM → CUTLASS → b12x → Marlin → Humming → Triton → torch
  SGLang block：DeepGEMM（ENABLE_JIT_DEEPGEMM）→ Blackwell + FlashInfer → sm120 CUTLASS → aiter → npu → xpu → Triton
  SGLang --fp4-gemm-backend auto：sm100 flashinfer_cutedsl，sm80~sm90 marlin，其它 flashinfer_cutlass

== KV cache dtype 选项个数
  vLLM 22 个（config/cache.py:44）；SGLang 8 个（srt/arg_groups/fields/model.py:197）
```
