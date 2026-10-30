# VLLM-07 PagedAttention：BlockPool、块表与 KV 分配

运行环境：CPU，只依赖 Python 3 标准库。基线 vllm 主干 0d6f92a，sglang 8fabf5e。

| 文件 | 内容 |
|---|---|
| `sim_block_pool.py` | vLLM V1 的 `KVCacheBlock` / `FreeKVCacheBlockQueue` / `BlockPool` / `allocate_slots` 按源码复刻，加上 worker 侧的块表宽度与 slot_mapping 计算。包含四种显存管理（Orca Max / Pow2 / Oracle、paged）的批大小与浪费对比、块大小扫描、H100 容量估算，以及小池子逐步跟踪 |
| `test_sim_block_pool.py` | 13 个单测，覆盖空闲队列顺序、null block、LIFO / LRU 归还、touch、驱逐、slot_mapping 填充和块表宽度对齐 |
| `trace_paged.md` | 块分配相关代码的逐行对照表（行号对应主干） |

## 用法

```bash
python3 sim_block_pool.py                           # waste：15,696 个槽位、max_len 2048，ShareGPT 分布
python3 sim_block_pool.py --dataset alpaca
python3 sim_block_pool.py --scenario bsweep         # block_size 1 … 256，两个数据集
python3 sim_block_pool.py --watermark 0.05          # 给 waiting 请求留 5% 水位
python3 sim_block_pool.py --scenario h100           # H100 + Qwen3-8B：容量、并发、块表 vs req_to_token
python3 sim_block_pool.py --scenario trace          # 10 块的小池子：分配、释放、复用顺序
python3 -m unittest test_sim_block_pool
```

数据集是合成的。采样按 PagedAttention 论文 Fig.11 的均值（ShareGPT prompt ≈161 / output ≈338，Alpaca ≈19 / ≈58）做对数正态分布（sigma 1.0），总长截断到 max_len。模拟里一步等于每条 running 请求前进 1 个 token，prefill 一步做完。它不估时间，只看显存和批大小。15,696 个槽位对应论文 Table 1 里 OPT-13B / A100-40GB 的 KV 容量。

## 期望输出

`python3 sim_block_pool.py`（ShareGPT，1000 条，prompt 均值 159，output 均值 329）：

```
模式               平均批    峰值     步数       有效       预留       碎片       空闲    抢占
orca-max           7.0     7  47335    23.2%    16.3%    51.3%     9.2%     0
orca-pow2         12.8    27  25691    42.7%    30.1%    21.7%     5.5%     0
orca-oracle       16.8    34  19598    56.0%    39.4%     0.0%     4.6%     0
paged bs=16       27.5    87  11997    91.4%     0.0%     1.3%     7.3%  1538
```

`--dataset alpaca`（prompt 均值 19，output 均值 59）：

```
orca-max           7.0     7   8456     4.1%     3.3%    83.3%     9.3%     0
orca-pow2         50.7   141   1159    29.9%    24.1%    22.6%    23.4%     0
orca-oracle       66.1   195    890    38.9%    31.4%     0.0%    29.7%     0
paged bs=16       88.7   578    663    52.2%     0.0%     4.2%    43.6%   343
```

四列占比的口径：每一步把池子里的槽位分成四类，再对所有步取平均。

- **有效**：已经写入 KV。
- **预留**：已经预留，这条请求以后才会写。
- **碎片**：分出去了，但这条请求永远写不到。对 Orca 是过量预留，对 paged 是末块的空位。
- **空闲**：没有分出去。

`--scenario bsweep`（ShareGPT 节选）：

| block_size | 平均批 | 步数 | 有效 | 碎片 | 抢占 | 块表宽 | 池块数 |
|---|---|---|---|---|---|---|---|
| 1 | 27.8 | 11,829 | 92.7% | 0.0% | 1608 | 2048 | 15,696 |
| 16 | 27.5 | 11,997 | 91.4% | 1.3% | 1538 | 128 | 981 |
| 64 | 26.4 | 12,467 | 88.1% | 5.3% | 1373 | 32 | 245 |
| 128 | 25.1 | 13,127 | 84.0% | 10.1% | 1157 | 16 | 122 |
| 256 | 22.9 | 14,388 | 76.6% | 18.6% | 806 | 8 | 61 |

Alpaca 从 16 扫到 256：平均批从 88.7 降到 44.6，碎片从 4.2% 涨到 52.1%。

`--watermark`（ShareGPT，paged bs=16）：

- 0.01：抢占 971 次，平均批 27.4。
- 0.05：抢占 272 次，平均批 26.8，空闲 9.3%。

`--scenario h100`：

- 可分配 399,296 token，每 token 144 KiB。
- 一条请求打满 40,960 要 2,560 块，日志里的并发是 9.75x。
- 整段预留只能跑 9 条；按块分配时，平均上下文 1,024 / 4,096 / 16,384 分别约能跑 389 / 97 / 24 条。
- vLLM `block_table` int32 [1024, 2560] = 10.0 MiB；SGLang `req_to_token` int32 [4097, 40,964] = 640.2 MiB。

## 读法

- **浪费主要来自预留，而不是分块。** Orca (Max) 一条请求占满 2048 槽，ShareGPT 下只有 23.2% 的槽位存着真实 KV。即使事先知道输出长度（Oracle），预留也占了 39.4%，因为一条请求在生成过程中只用到它预留量的一部分。paged 把预留降到 0，有效占比 91.4%，平均批是 Orca (Max) 的 3.9×。
- **代价是抢占。** 按需分配就可能分不到块。ShareGPT 1000 条里发生了 1538 次抢占（同一条请求可能被抢多次），每次都要重算。watermark 用少量空闲换更少的抢占。
- **block_size 在 16 附近几乎免费。** 从 1 到 16，碎片只涨 1.3 个百分点，块表和池块数都少了 16 倍。超过 64 以后碎片快速上升，短请求为主的 Alpaca 更敏感。这与论文 7.2 节的结论一致。
- **SGLang 的 page_size 1 是另一种取舍。** 它不需要块表，直接用 `req_to_token` 逐 token 存槽位，碎片为 0，前缀能复用到 token 粒度。代价是这张表按 `max_num_reqs × max_context_len` 预分配，在 H100 + 40k 上下文时是 vLLM 块表的 64 倍。
