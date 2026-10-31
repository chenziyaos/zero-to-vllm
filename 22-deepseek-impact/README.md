# VLLM-22 前沿：DeepSeek 开源如何改写 vLLM

运行环境：`ds_impact.py` 和 `test_ds_impact.py` 只依赖 Python 3 标准库，不联网、不需要 GPU。`grep` 场景复用 `../21-frontier-month/data/` 下的提交快照（vllm 0d6f92a、sglang 1ab008f5，同一个 30 天窗口），本目录不另存一份。

## 文件

| 文件 | 内容 |
|---|---|
| `ds_impact.py` | 计算器，场景有 kvbytes / eplb / replay / adaptive / components / grep / lines，另有 `--grep` 按标题正则列出两边窗口内的提交 |
| `test_ds_impact.py` | 22 个回归测试：MLA / indexer 字节数、EPLB 移植与 vLLM 单测期望逐位一致、重放公式、自适应验证的 argmax 性质、标题计数 |

## 用法

```bash
python3 ds_impact.py --scenario all
python3 ds_impact.py --scenario kvbytes --swa-dtype mxfp8 --block 256
python3 ds_impact.py --scenario eplb --experts 64 --gpus 16 --redundant 16 --alpha 1.1
python3 ds_impact.py --scenario replay --hit 32768 --new 512
python3 ds_impact.py --scenario adaptive --reqs 64 --budget-knee 64
python3 ds_impact.py --grep "deep_?gemm|mega ?moe"
python3 -m unittest test_ds_impact -v
```

## 口径

- **kvbytes**：MLA 存 `kv_lora_rank=512` 的潜向量和 64 维 RoPE key。vLLM 的 FP8 布局是 512 B NoPE + 4 个 fp32 scale + 64 维 RoPE BF16 = 656 B；indexer K 的 132 / 68 B 来自 `vllm/models/deepseek_v41/attention.py` 的 `_indexer_k_cache_head_dim`；V4.1 SWA 每 token 584 / 528 B 来自同一文件的 `swa_bytes_per_token`。全局 KV 890 B/token 和 V4 的 83.9 / 9.62 GiB 取 vLLM 博客的口径。
- **eplb**：是 `vllm/distributed/eplb/policy/default.py` 的纯 Python 移植，不含 `preserve_intragpu_slots`（那一步只减少迁移量，不改负载）。合成负载用打乱后的 Zipf 分布，`balancedness` 和 `eplb_state.py` 日志同口径：各层 rank 负载均值之和 / 各层 rank 负载最大值之和。
- **replay**：只数 token×层，把每层成本当成一样。真实的层间成本不同（KV-source 层带压缩注意力和 indexer），所以这是上限；博客实测 prefill 计算省 30–40%。
- **adaptive**：是 `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` 里 `get_num_tokens` 的移植。置信度和验证成本曲线是合成的，只用来说明 argmax 的行为，不对应任何实测。
- **grep**：按标题正则计数，一个提交可以命中多项；`dsv4` 用负向前瞻排除了 `4.1`。SGLang 的 `indexer` 命中里有不少是 HiSparse / NSA 的通用改动，不全是 DeepSeek。

## 期望输出

### eplb

```
-- vLLM 单测 tests/distributed/test_eplb_algo.py 的例子：12 专家、4 组、2 节点、8 GPU、16 个物理槽
   层 0：phy2log [5, 6, 5, 7, 8, 4, 3, 4, 10, 9, 10, 2, 0, 1, 11, 1]（与单测期望一致）
   层 1：phy2log [7, 10, 6, 8, 6, 11, 8, 9, 2, 4, 5, 1, 5, 0, 3, 1]（与单测期望一致）
-- 合成负载：64 专家、8 组、2 节点、16 GPU、40 层，Zipf α=0.9
   不做 EPLB（专家按编号平铺）       balancedness 0.30
   EPLB 冗余   0（每卡 4 槽）            balancedness 0.34
   EPLB 冗余  16（每卡 5 槽）            balancedness 0.94
   EPLB 冗余  32（每卡 6 槽）            balancedness 0.96
```

### replay

```
   不重放（新 token 走全部层）                    327680
   decoder 侧重放（层 0–20 全部 token，层 21–39 只算最后 128 个）      174464   53.2%
   encoder 侧重放补算命中末尾 128 个位置（层 0–20）           2688    0.8%
   合计 177152，比不重放少 45.9%
```

### adaptive

```
   平均生存概率（第 k 个草稿被接受的概率）：k1 0.73  k2 0.50  k3 0.32  k4 0.19  k5 0.11
   全部验证：B=160，期望产出 91.1 token，耗时 9.42 ms，9.67 token/ms
   只验 top-B：B=105，期望产出 85.9 token，耗时 8.32 ms，10.32 token/ms
```

### grep

```
   主题              vLLM  SGLang
   deepgemm          21       9
   megamoe           18      17
   deepep             8       9
   flashmla           4       6
   eplb               7       7
   mla               60      29
   engram            12      18
   replay            28      53
   dspark            25      50
   dsv4.1            68     100
   dsv4              30     127
```
