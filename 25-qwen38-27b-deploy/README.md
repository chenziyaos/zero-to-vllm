# VLLM-25 实战：Qwen3.8-27B 部署，Gated DeltaNet 混合注意力的显存账

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-vllm/blob/main/colab/25-qwen38-27b-deploy.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

运行环境：`deploy_planner.py`、`commands.py`、`bench_checks.py` 和 `test_deploy_planner.py` 只依赖 Python 3 标准库，不联网、不需要 GPU，在笔记本上就能把每一档的显存账算出来。`serve.sh` 和 `bench.sh` 要在真机上跑，recipe 的 `min_vllm_version` 是 0.17.0，端到端验证在 `0.26.1rc1.dev608+g99a10304d`，外挂 DFlash2 草稿要 ≥0.28.0。源码行号对应 vllm 主干 0d6f92a，recipe 指 vllm-project/recipes 仓库 0511d96 的 `models/Qwen/Qwen3.8-27B.yaml`。

## 这个模型和 Qwen3-32B 有什么不一样

Qwen3.8-27B 和 2.4T 旗舰是同一套骨干：64 层里只有 16 层全注意力（每 4 层 1 层，`full_attention_interval=4`），另外 48 层是 Gated DeltaNet（GDN）线性注意力。带来三个部署层面的直接后果：

- **KV 显存只剩四分之一。** 全注意力只有 16 层、4 个 KV 头、头维 256，BF16 每 token 64 KiB；Qwen3-32B 是 64 层 × 8 头 × 128 维 = 256 KiB。262K 上下文一条序列只要 16 GiB（FP8 KV 8 GiB），1M YaRN 也就 61 GiB。
- **每条序列有一份和上下文长度无关的固定状态。** 48 层 GDN，每层存 conv 历史 + 48 个 128×128 的 fp32 时间状态矩阵，TP1 一条序列约 147 MiB，开 prefix caching 翻倍到 294 MiB。262K 长上下文也只多这么点。
- **checkpoint 自带 MTP 草稿头**（`text_config.mtp_num_hidden_layers=1`，顶层 config 没有这个字段）。MTP 默认不开；recipe 的 2×5090 示例显式给 `num_speculative_tokens=3`，不写这个参数草稿就不生效。接受率得从 `/metrics` 的 accepted/draft 两个计数器做前后快照差算，单次吞吐看不出来。

## 文件

| 文件 | 内容 |
|---|---|
| `deploy_planner.py` | 部署规划器，场景有 ckpt / state / block / kv / mem / fit / rank / mtp / decode：六种 checkpoint 字节对账、GDN 状态、混合层块大小、全注意力 KV、单卡显存账、上游 recipe 记录、十档卡型排序、MTP 开销、decode 带宽估算 |
| `commands.py` | 七档 `vllm serve` 命令生成器（TP2 档默认带 MTP k=3，含 SGLang BF16 对照起点）和 `vllm bench serve` 压测命令，只打印不执行 |
| `serve.sh` | 真机启动脚本（1×5090 NVFP4 到 2×5090 TP2、H100、H200、B200），完整日志落 `results/<档位>.log`，失败不吞退出码 |
| `bench.sh` | 冒烟断言（算术题 + qwen3_coder 工具调用）、recipe 口径负载、decode 并发扫描、按上下文预算的长上下文、MTP 前后快照 |
| `bench_checks.py` | 冒烟响应断言和 MTP accepted/draft 计数器差值计算，纯标准库 |
| `test_deploy_planner.py` | 46 个回归测试：常量钉在 config.json、六家 HF 仓库 safetensors 合计、源码默认值、命令生成和计数器口径上 |
| `results/sim.txt` | `--scenario all` 的完整输出，正文和画板里的数都从这里来 |

## 用法

```bash
# 不需要 GPU：先算账
python3 deploy_planner.py --scenario all
python3 deploy_planner.py --scenario mem --gpu rtx5090 --fmt nvfp4 --kv-dtype fp8 --eager --max-model-len 32768 --max-num-seqs 8
python3 deploy_planner.py --scenario mem --gpu rtx5090 --n 2 --tp 2 --fmt fp8 --kv-dtype fp8 --spec 3
python3 deploy_planner.py --scenario rank
python3 deploy_planner.py --scenario fit
python3 -m unittest test_deploy_planner -v
bash serve.sh plan 5090x2-fp8         # 不起服务，只打这一档的显存账
python3 commands.py --profile b200-mxfp4 --mtp 3   # 打印某档的 serve 命令

# 真机：起服务，再压测
bash serve.sh 5090-nvfp4              # 1×5090 Inferact NVFP4，eager，32K
bash serve.sh 5090x2-nvfp4            # 2×5090 TP2 NVFP4，262K，MTP k=3
bash serve.sh h100-fp8                # 1×H100 官方 FP8，32K，主力档
bash bench.sh smoke                   # 算术题 323 + get_weather 工具调用，断言不过即非零退出
bash bench.sh recipe                  # random 1024/1024，128 条，并发 128；服务端先关 prefix caching 对齐口径
CONC="1 4 8 16 32" bash bench.sh decode
MAX_MODEL_LEN=32768 LENS="8192 16384 32768" bash bench.sh long   # 预算先扣输出 16 和模板余量 256
bash bench.sh mtp-snapshot > m1.txt   # 压测前抓 /metrics 计数器
# ... 压测负载 ...
bash bench.sh mtp-snapshot > m2.txt
bash bench.sh mtp m1.txt m2.txt       # Δaccepted/Δdraft：recipe 记录 FP8 0.771 / NVFP4 0.897
```

## 选卡结论（原始字节估算 + 上游 recipe 记录，详见 `rank` / `fit`）

| 配置 | 权重/卡 | 上下文预算 | 结论 |
|---|---|---|---|
| 1×4090 NVFP4 | 24.6 GiB | 0 | 起不来：sm89 没有 Blackwell FP4 W4A4 kernel |
| 1×5090 FP8 | 28.7 GiB | 0 | 起不来：gmu 0.90 下权重超可用，没有 KV 池 |
| 1×5090 NVFP4 eager 32K | 24.6 GiB | 32K 原始预算 2 条 | 上游 recipe 记录 eager、32K 可起；非唯一可行配置 |
| 2×5090 TP2 FP8/NVFP4 | 14.4/12.3 | 262K 原始预算 2/3 条 | 262K 有上游记录，接受率 0.771 / 0.897 |
| 1×H100 FP8 | 28.7 GiB | 262K 原始预算 4 条 | 主力档（启动默认 32K，长上下文按 KV 池放开） |
| 1×H200 BF16 | 51.7 GiB | 262K 原始预算 8 条 | 141G，长上下文主力 |
| 1×B200 MXFP4 | 18.5 GiB | 262K 原始预算 17 条 | 真 W4A4，2026-09-20 verified |

并发数是原始字节估算（gmu 0.90、fp8 KV、max-num-seqs 8，和 `rank` 场景同口径），没复刻混合页池的 padding 和分配器；verified / 上游记录的字样只指 recipe 历史记录，本次没有 GPU 实测。

## 口径

- **ckpt**：六种 checkpoint 的字节取 HF 树上全部 safetensors 合计（2026-11-03 核对）。没有一个量化档是均匀 4 bit——embed / lm_head / 视觉塔 / 部分线性层保持 BF16。Ascend w8a8（Eco-Tech，ModelScope）用 recipe 注释里的 32.1 GB。
- **state / block**：GDN 状态公式钉在 `mamba_utils.py:282`（conv 通道 10240 = 128·16·2 + 128·48）和 fp32 时间状态矩阵上；混合层块大小按 `platforms/interface.py:1050-1125` 的规则复算——一块注意力 KV 必须装下一整页 GDN 状态，所以块是 784 / 1600 token，而不是稠密模型常见的 16。
- **mem**：KV 池 = gmu × 显存 − 常驻权重 − 其他。「其他」沿用 VLLM-24 的系数（非 torch 0.4 GiB、TP>1 加 NCCL 0.3、激活按 batched token、采样器预热按 seqs、CUDA graph 按抓取尺寸数缩放）。这是原始字节估算，没有复刻混合 KV 页池的 padding、分组和共享块分配；运行态差额以启动日志为准。
- **fit**：只列上游 recipe 记录（2×5090 TP2 三档权重/KV token/接受率，1×5090 eager 逐项记录），明确标注「非本次 GPU 实测」，不用反推折减系数去拟合估算器。
- **rank**：估算器只覆盖 TP1/2/4，并发数字固定 gmu 0.90、fp8 KV、max-num-seqs 8 这一档口径。更大的 TP 受 24 个 Q 头、4 个 KV 头和 48 个 GDN v 头的整除约束（例如 TP16 连 KV 头都分不开）；但不能据此推广成「所有多机 TP 都不可用」。
- **mtp**：GDN 状态按草稿位增加（TP1 每个草稿位每层多 20 KiB conv），接受率 BF16 92.2% / FP8 84.8% / NVFP4 81.5–84.8%（模型卡短 prompt）与 2×5090 的 0.771 / 0.897 / 0.788 一起列出，没有统一达标线，必须同一负载前后快照自己算；recipe 里 #51812 / #51674 的修复提示按实际镜像 commit 核验，不替发行版下结论。
- **decode**：带宽模型，η = 0.75，固定开销 0.5 ms，TP all-reduce 按 64 层每层两次小消息估（一次 reduce、一次 post-weight update）。只用来排序和定量级，**全部是估算**；Ascend 950PR 实测单流 64 tok/s（MTP 开）作为量级对照。
- **语言模型档**：`--language-model-only` 不加载 0.86 GiB 视觉塔（Qwen3_5ForConditionalGeneration，多模态）。
