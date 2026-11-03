# VLLM-24 实战：Qwen3-32B 部署，从 4090 到 H20

运行环境：`deploy_planner.py` 和 `test_deploy_planner.py` 只依赖 Python 3 标准库，不联网、不需要 GPU，在笔记本上就能把每一档的显存账算出来。`serve.sh` 和 `bench.sh` 要在真机上跑，Qwen3-32B 需要 vLLM 0.8.5 以上（recipe 的 `min_vllm`），Qwen3.5-27B 需要 0.17.0 以上。源码行号对应 vllm 主干 0d6f92a。

## 文件

| 文件 | 内容 |
|---|---|
| `deploy_planner.py` | 部署规划器，场景有 weights / mem / kv / tp / spec / profile / decode / rank / qwen35，算 checkpoint 字节、每卡显存账、KV 池、TP2 和两份单卡的取舍、EAGLE3 开销、源码默认参数、decode 带宽估算，以及 Qwen3.5-27B 的混合层块大小 |
| `test_deploy_planner.py` | 23 个回归测试：字节数钉在 config.json、HF 仓库文件大小和源码默认值上，校准钉在 #22344 的启动日志上（5.68 GiB / 23,264 token / 1.42x） |
| `serve.sh` | 十一个档位的 `vllm serve` 命令（从 1×4090 到 2×H100，加 EAGLE3、Qwen3.5-27B 和 SGLang 对照），日志落 `results/<档位>.log`，并过滤出要看的几行 |
| `bench.sh` | 冒烟（算术题 + 工具调用）、recipe 口径、ShareGPT、decode 并发扫描、长上下文、`/metrics` |

## 用法

```bash
# 不需要 GPU：先算账
python3 deploy_planner.py --scenario all
python3 deploy_planner.py --scenario mem --gpu rtx4090 --fmt awq --kv-dtype fp8 --gmu 0.95 --max-model-len 16384 --max-num-seqs 8 --batched 2048
python3 deploy_planner.py --scenario mem --gpu h100 --n 2 --tp 2 --fmt bf16
python3 -m unittest test_deploy_planner -v
bash serve.sh plan 4090-awq          # 打印 serve.sh 某一档对应的显存账

# 真机：起服务，再压测
bash serve.sh h100-fp8               # 1×H100 FP8，数据中心首选
bash serve.sh 5090-awq               # 1×RTX 5090 AWQ + fp8 KV，消费级首选
bash bench.sh smoke
bash bench.sh recipe                 # 服务端先加 --no-enable-prefix-caching
CONC="1 8 32" bash bench.sh decode
bash bench.sh long
bash bench.sh metrics
```

`serve.sh` 的环境变量：`MODEL`（覆盖档位默认模型）、`OUT`、`PORT`、`GMU`（覆盖档位默认的 gpu-memory-utilization）。`bench.sh` 的 `MODEL` 默认是 `Qwen/Qwen3-32B-FP8`，压别的档位时要改成和服务端一致。

## 口径

- **weights**：checkpoint 字节取 HF 仓库 safetensors 合计。embed 和 lm_head 不共享（`tie_word_embeddings: false`），各 0.778B 参数，所有量化格式都保持 BF16。加载后常驻：AWQ 取 #22344 日志的 18.14 GiB，NVFP4 取 #22783 的 21.27 GiB，BF16 / FP8 按 checkpoint 算。
- **mem**：KV 池 = gmu × 显存 − 常驻 − 其他。「其他」= 非 torch（TP1 0.4 GiB，TP>1 再加 NCCL 0.3）+ max(激活峰值, 采样器预热) + CUDA graph。激活峰值按 batched token 线性缩放，系数由 #22344 反推（eager、gmu 0.95、max-model-len 16384、batched 32768 时激活约 5.80 GiB）；采样器预热按 `seqs × VOCAB × 4 B × 3` 算；CUDA graph 按 51 个尺寸 0.85 GiB 线性缩放。
- **kv**：64 层 × 8 KV 头 × 128 × K/V，BF16 每 token 256 KiB，TP 每翻一倍减半。
- **tp**：两张卡比较 TP2 一份和两份单卡；TP 的 all-reduce 按每层两次、每次 NVLink 10 µs / PCIe 5.0 25 µs / PCIe 4.0 35 µs 估。
- **spec**：EAGLE3 接受长度取 RedHatAI 模型卡；加载后多出的 1.37 GiB 取 #28919 日志。
- **decode**：带宽模型，η = 0.75，每步固定开销 0.5 ms。只用来排序和定量级，**全部是估算**。和公开实测的比值在 0.55–1.03 之间，偏差原因见 `decode` 场景末尾两行。
- **qwen35**：checkpoint 按模块拆来自 HF 仓库的 safetensors 索引；块大小按 `platforms/interface.py` 1050–1125 的规则复算，和 LMCache 文档的 784 对得上。

## 期望输出

### weights

```
-- checkpoint 字节（HF 仓库 safetensors 合计）
   参数 32.76B = 可量化线性层 31.21B + 保持 BF16 的 1.56B（embed、lm_head 各 0.778B，不共享）
   bf16    61.02 GiB    65.5 GB  每参数 16.00 bit   recipe vram_minimum_gb  77（≈ 65.5 × 1.18）
   fp8     31.97 GiB    34.3 GB  每参数  8.38 bit   recipe vram_minimum_gb  39（≈ 34.3 × 1.14）
   awq     18.00 GiB    19.3 GB  每参数  4.72 bit   recipe vram_minimum_gb  20（≈ 19.3 × 1.03）
   nvfp4   19.25 GiB    20.7 GB  每参数  5.05 bit   recipe vram_minimum_gb  22（≈ 20.7 × 1.06）
   加载后常驻：awq 18.14 GiB（#22344），nvfp4 21.27 GiB（#22783），bf16 / fp8 按 checkpoint 算
   recipe 的 vram_minimum_gb：bf16 / fp8 是 checkpoint GB 的 1.14–1.18 倍，awq / nvfp4 只多 3–6%，都只够权重加一点 KV，开不满 40K
```

### mem（1×H100 FP8，默认参数）

```
-- 显存账（每卡）
   H100 80G × 1，TP1，fp8 权重，KV bf16，max-num-seqs 1024，batched 8192
     可用 72.8 GiB = 79.11 × 0.92
     常驻权重 31.97 GiB（DeepGEMM（需 nvcc）或 CUTLASS block FP8）
     其他 2.99 GiB = 非 torch 0.40 + max(激活峰值 1.45, 采样器预热 1.74) + CUDA graph 0.85（51 个尺寸）
     KV 池 37.83 GiB → 154,940 token（256 KiB/token），40960 上下文并发 3.78
-- 和公开日志对账
   #22344 RTX 5090 D AWQ eager gmu 0.95：KV 5.68 GiB（日志 5.68），23,265 token（日志 23,264），并发 1.42（日志 1.42）。这一条是校准点
   LMCache FAQ TP2 H100：306,365 token（FAQ 275,760），并发 7.48（FAQ 6.73）；FAQ 每卡 KV 33.66 GiB，比估算少 3.7，差在 LMCache 自己的缓冲（估算）
   #28919 eagle3 k=5、max-model-len 4096：日志 69,264 token、16.91x，69,263 / 4096 = 16.91，按 64 层是 16.91 GiB，加草稿 1 层是 17.17 GiB，日志 17.08 在两者之间
   recipe 第 226 行说默认用 90% 显存，源码默认是 0.92（config/cache.py:215）
```

### mem（1×RTX 4090 AWQ，fp8 KV、16K、seqs 8、gmu 0.95）

```
-- 显存账（每卡）
   RTX 4090 24G × 1，TP1，awq 权重，KV fp8，max-num-seqs 8，batched 2048
     可用 22.3 GiB = 23.52 × 0.95
     常驻权重 18.14 GiB（AWQ Marlin（--quantization awq 也会落到这里，auto_awq.py:263））
     其他 0.85 GiB = 非 torch 0.40 + max(激活峰值 0.36, 采样器预热 0.01) + CUDA graph 0.08（5 个尺寸）
     KV 池 3.36 GiB → 27,501 token（128 KiB/token），16384 上下文并发 1.68
-- 和公开日志对账
   #22344 RTX 5090 D AWQ eager gmu 0.95：KV 5.68 GiB（日志 5.68），23,265 token（日志 23,264），并发 1.42（日志 1.42）。这一条是校准点
   LMCache FAQ TP2 H100：306,365 token（FAQ 275,760），并发 7.48（FAQ 6.73）；FAQ 每卡 KV 33.66 GiB，比估算少 3.7，差在 LMCache 自己的缓冲（估算）
   #28919 eagle3 k=5、max-model-len 4096：日志 69,264 token、16.91x，69,263 / 4096 = 16.91，按 64 层是 16.91 GiB，加草稿 1 层是 17.17 GiB，日志 17.08 在两者之间
   recipe 第 226 行说默认用 90% 显存，源码默认是 0.92（config/cache.py:215）
```

### tp

```
-- 两张卡怎么用：TP2 一份，还是两份单卡（KV 是每个副本的，c1 / c32 是 decode 带宽估算）
   配置                          副本    常驻/卡    KV/副本    token/副本   40K 并发     c1    c32 合计
   2×H100 TP2 BF16              1    30.5     39.0     319,327     7.80     67      1936
   2×H100 各跑 FP8                2    32.0     37.8     154,940     3.78     70      2016
   2×H100 TP2 FP8               1    16.0     53.5     438,352    10.70    115      3101
   2×L40S TP2 BF16              1    30.5      8.4      68,413     1.67     18       515
   2×L40S 各跑 FP8 + fp8 KV       2    32.0      7.2      58,965     1.44     19       564
   2×4090 TP2 AWQ               1     9.1     10.6      86,697     2.12     55      1359
   2×4090 各跑 AWQ + fp8 KV       2    18.1      2.5      20,628     0.50     38      1099
   TP2 每层两次 all-reduce：NVLink 上每步多 1.3 ms，PCIe 4.0 上多 4.5 ms（估算），4090 / L40S 没有 NVLink
   KV 头 8 个，TP 最多切到 8；TP2 每卡 KV 字节减半，所以 TP2 的 token 数是单卡的两倍多
```

### spec

```
-- EAGLE3（recipe 49–53：RedHatAI/Qwen3-32B-speculator.eagle3，k=3）
   草稿文件 2.91 GiB，加载后多 1.37 GiB（#28919：62.3970 − 61.02），KV 多 1 层：260 KiB/token
   模型卡接受长度（每次验证平均产出 token），后面是比 k−1 多出来的部分：
   coding        k1 1.67 k2 2.06(+0.39) k3 2.29(+0.23) k4 2.39(+0.10) k5 2.47(+0.08) k6 2.50(+0.03) k7 2.53(+0.03)
   math          k1 1.73 k2 2.21(+0.48) k3 2.49(+0.28) k4 2.69(+0.20) k5 2.80(+0.11) k6 2.83(+0.03) k7 3.08(+0.25)
   summarization k1 1.62 k2 1.95(+0.33) k3 2.15(+0.20) k4 2.23(+0.08) k5 2.27(+0.04) k6 2.32(+0.05) k7 2.33(+0.01)
   k=3 之后每多一个草稿只多 0.1 左右，草稿和验证的开销却按 k 涨：recipe 选 k=3 是对的，高并发再降到 1–2
   k=1：CUDA graph 上限 min(seqs × 2 × 2, 512)，seqs 32 时最大 128，抓 19 个尺寸
   k=3：CUDA graph 上限 min(seqs × 4 × 2, 512)，seqs 32 时最大 256，抓 35 个尺寸
   k=5：CUDA graph 上限 min(seqs × 6 × 2, 512)，seqs 32 时最大 384，抓 43 个尺寸
   ngram（prompt_lookup_max / min，speculative.py:531 / 534）：不加载模型、不占显存，适合改写、代码补全这类输入里有原文的负载
```

### profile

```
-- 源码默认值（arg_utils.py 2950–3012，#17885）
   RTX 4090 24G           23.5 GiB → batched  2048，max-num-seqs  256
   RTX 5090 32G           31.6 GiB → batched  2048，max-num-seqs  256
   L40S 48G               44.4 GiB → batched  2048，max-num-seqs  256
   A100 80G               79.2 GiB → batched  2048，max-num-seqs  256   名字里有 a100，被排除在 70 GiB 档外
   H100 80G               79.1 GiB → batched  8192，max-num-seqs 1024
   H20 96G                95.1 GiB → batched  8192，max-num-seqs 1024
   H200 141G             140.4 GiB → batched  8192，max-num-seqs 1024
   B200 180G             178.3 GiB → batched 16384，max-num-seqs 1024
-- CUDA graph 尺寸（config/vllm.py 2465–2490、2588–2610）
   balanced      seqs 1024：51 个尺寸，最大 512，前几个 [1, 2, 4, 8, 16, 24]
   interactivity seqs 1024：76 个尺寸，最大 512，前几个 [1, 2, 3, 4, 5, 6]
   #40121 的绕法只抓 [1, 2, 4, 8]：4 个尺寸
-- 显存提示（gpu_worker.py 780–814）
   H100 80G gmu 0.92，graph 0.85 GiB：等价旧口径 gmu = 0.92 − 0.85/79.11 = 0.9093；想保住 KV 池就调到 0.9307
```

### decode

```
-- decode 带宽估算（η=0.75，上下文 1K；每步读一遍权重加 KV）
   RTX 4090 24G      TP1 awq  bs1  26.6 ms     38 tok/s；bs8  29.1 ms    275 tok/s；bs32  37.6 ms    850 tok/s
   RTX 5090 32G      TP1 awq  bs1  15.2 ms     66 tok/s；bs8  16.6 ms    482 tok/s；bs32  21.4 ms   1496 tok/s
   L40S 48G          TP1 fp8  bs1  53.9 ms     19 tok/s；bs8  56.8 ms    141 tok/s；bs32  66.7 ms    480 tok/s
   A100 80G          TP1 bf16 bs1  43.5 ms     23 tok/s；bs8  44.8 ms    179 tok/s；bs32  49.0 ms    654 tok/s
   H100 80G          TP1 fp8  bs1  14.3 ms     70 tok/s；bs8  15.0 ms    533 tok/s；bs32  17.6 ms   1820 tok/s
   H100 80G          TP2 bf16 bs1  14.9 ms     67 tok/s；bs8  15.2 ms    525 tok/s；bs32  16.5 ms   1936 tok/s
   H20 96G           TP1 fp8  bs1  12.0 ms     83 tok/s；bs8  12.7 ms    632 tok/s；bs32  14.8 ms   2162 tok/s
   H20 96G           TP1 bf16 bs1  22.4 ms     45 tok/s；bs8  23.1 ms    347 tok/s；bs32  25.2 ms   1270 tok/s
   B200 180G         TP1 bf16 bs1  11.5 ms     87 tok/s；bs8  11.8 ms    679 tok/s；bs32  12.9 ms   2490 tok/s
-- 公开实测 / 估算
   Spheron H100 FP8 单流             bs1  实测单流  72.5  估算  70.1  比值 1.03  Spheron 部署博客
   jarvislabs 2×H100 TP2 BF16 c60  bs60 实测单流  48.1  估算  59.9  比值 0.80  ShareGPT，vLLM
   #40121 2×3090 TP2 AWQ c32       bs32 实测单流  22.0  估算  40.0  比值 0.55  graph 只抓到 8
   #27649 PRO 6000 BF16 单流         bs1  实测单流  15.0  估算  20.2  比值 0.74  CUDA 13.0，v0.11.0
   H100 单流对得上（1.03）。#40121 只有 0.55：图只抓到 8，32 并发的批走不了 CUDA graph
   #27649 是 v0.11.0 配 CUDA 13.0 跑 SM120，kernel 还没调好；jarvislabs 的 0.80 里有 ShareGPT 长短不一的 prefill 插队
```

### rank

```
-- 卡配置排行（c1 / c32 是 decode 带宽估算，全部是估算）
   配置                           常驻     KV  KV token     上下文     并发    c1    c32  档位 / 路径
   1×4090 AWQ 默认参数            18.1    1.8        7K     40K   0.18     —      —  不可行：40K 一条都装不下
   1×4090 AWQ fp8 KV 16K      18.1    3.4       28K     16K   1.68    38   1002  入门：砍上下文和并发
   2×3090 TP2 AWQ              9.1    7.0       57K     40K   1.39    53   1281  入门：#40121 的绕法
   1×5090 AWQ fp8 KV          18.1    9.6       79K     40K   1.92    66   1759  消费级首选：AWQ Marlin
   1×5090 NVFP4 fp8 KV        21.3    6.5       53K     40K   1.30    57   1547  消费级可选：原生 FP4
   1×L40S FP8                 32.0    7.2       29K     40K   0.72     —      —  不推荐：FP8 Marlin，KV 不够
   1×L40S FP8 fp8 KV          32.0    7.4       61K     40K   1.49    19    532  可用：FP8 Marlin
   2×L40S TP2 BF16            30.5    8.4       68K     40K   1.67    18    515  可用：PCIe all-reduce
   1×A100 BF16                61.0   10.1       41K     40K   1.01    23    654  勉强：BF16 刚够一条 40K
   1×H100 BF16                61.0    8.8       36K     40K   0.88     —      —  不推荐：KV 不够一条 40K
   1×H100 FP8                 32.0   37.8      155K     40K   3.78    70   1820  数据中心首选：DeepGEMM / CUTLASS
   2×H100 TP2 BF16            30.5   39.0      319K     40K   7.80    67   1936  参照：recipe 第 198 行
   1×H20 FP8                  32.0   52.5      215K     40K   5.25    83   2162  H20 首选：带宽 4.0 TB/s
   1×H20 BF16                 61.0   23.5       96K     40K   2.35    45   1270  H20 可用：精度不让步时
   1×PRO 6000 BF16            61.0   23.4       96K     40K   2.34    20    575  可用：带宽只有 1.8 TB/s
   1×H200 BF16                61.0   65.2      267K     40K   6.52    53   1518  高配：单卡 BF16 开满 40K 六条
```

### qwen35

```
-- Qwen3.5-27B checkpoint 按模块拆（GiB：BF16 / FP8 / GPTQ-Int4）
   mlp                 31.88  15.94   8.29
   linear_attn（GDN）    10.36   5.20  10.36
   self_attn            3.13   1.56   3.13
   lm_head              2.37   2.37   2.37
   embed                2.37   2.37   2.37
   visual               0.86   0.86   0.86
   mtp                  0.79   0.44   0.79
   bf16 合计  51.75 GiB =  55.6 GB，recipe vram_minimum_gb 65
   fp8  合计  28.75 GiB =  30.9 GB，recipe vram_minimum_gb 33
   gptq 合计  28.16 GiB =  30.2 GB，recipe vram_minimum_gb 17
   GPTQ-Int4 只量化了 MLP，GDN 和注意力还是 BF16：28.16 GiB 比 FP8 只小 0.6，recipe 写的 17 GB / 单张 24 GB 装不下
   加 --language-model-only 省掉 visual 0.86，GPTQ 还剩 27.30 GiB，24 GB 卡照样装不下
-- 混合层的块大小（注意力页要 ≥ GDN 状态页）
   TP1 KV bf16 spec 0 对齐 16：GDN 页 3,207,168 B → 块 784 token（v0.19，LMCache 文档写 784）
   TP1 KV bf16 spec 0 对齐 64：GDN 页 3,207,168 B → 块 832 token（主干 + flashinfer GDN prefill）
   TP2 KV fp8  spec 1 对齐 16：GDN 页 1,613,824 B → 块 1584 token（v0.19，TP2 fp8 KV + MTP k=1）
   TP2 KV fp8  spec 1 对齐 64：GDN 页 1,613,824 B → 块 1600 token（主干，同上）
   derek 在 v0.19 上跑 Qwen3.6-27B（TP2 fp8 KV、MTP k=1）日志写 1600，和这里同一量级；3.6 的 config 我们没逐项核对
-- 每条序列的 GDN 状态（48 层，和上下文长度无关）
   mamba_cache_mode=none ：TP1    147 MiB，TP2 MTP1    148 MiB
   mamba_cache_mode=align：TP1    294 MiB，TP2 MTP1    222 MiB
   默认 none（config/cache.py:304），开 --enable-prefix-caching 才走 align，每条多一页
-- KV：16 层全注意力 × 4 KV 头 × 256，bf16 64 KiB/token，fp8 32 KiB/token
   1×H100 FP8：KV 池约 41.0 GiB，32768 上下文每条  2.14 GiB，并发  19.1（估算）
   1×H100 FP8：KV 池约 41.0 GiB，262144 上下文每条 16.14 GiB，并发   2.5（估算）
   同样 1 GiB KV，Qwen3.5-27B 装的 token 是 Qwen3-32B 的 4 倍；262K 上下文单卡 H100 FP8 能开两三条
```
