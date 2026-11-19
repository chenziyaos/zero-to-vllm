# VLLM-13 投机解码：EAGLE、MTP、n-gram 与拒绝采样

运行环境：两个模拟器只需要 CPU 和 Python 3 标准库。`serve_spec.sh` 需要一张 H100 和装好的 vLLM；跑 SuffixDecoding 要另装 `arctic-inference`，跑 SGLang 对照要另装 sglang。源码基线：vllm 主干 0d6f92a，sglang 主干 8fabf5e。

| 文件 | 内容 |
|---|---|
| `sim_spec.py` | 期望接受长度与加速比。H100 + Llama-3.1-8B 上 batch 变大后最优 k 怎么往下掉（理想 roofline 和打折扣两版）。开投机后 CUDA Graph 抓哪些尺寸（逐行移植 `_set_cudagraph_sizes`），每步预留几个 KV 块。n-gram 的 KMP 匹配（逐行移植 `ngram_proposer.py`），链式与树形草稿的回放对比。Dynamic SD 查表和 SGLang adaptive 状态机。两边的指标口径和参数对照 |
| `sim_rejection.py` | 拒绝采样的 Monte Carlo。在 V=4 的马尔可夫链上比较 vLLM 的 standard / block / synthetic / FLy 和 SGLang 的 target-only，看哪条规则无损（TV 距离对比噪声底）、每轮多产出多少 token。另含：用指数噪声 argmax 从残差采样；greedy 草稿与 probabilistic 草稿的接受率对比；SGLang 两个阈值的扫描；FLy 窗口与熵门限的扫描 |
| `test_sim_spec.py` | 12 个单测：闭式解；KMP 对暴力解；Dynamic SD 稠密表与非法配置；CUDA Graph 尺寸个数，以及源码注释里的 563 / 581 / 100；MRV1 取整；SGLang 分档路由与升降档；synthetic 接受率展开 |
| `test_sim_rejection.py` | 5 个单测：无损规则的 TV 低于噪声底的 2.5 倍，有损规则高于 5 倍；block 比 standard 接受得多；FLy 熵门限设到 99（相当于不挂起）时与 standard 逐位相同；指数噪声 argmax 等价于按残差采样 |
| `serve_spec.sh` | 真机实验：base / eagle3 / ngram / suffix / draft / dynamic / prob / block / fly / synthetic 十种起法，外加 bench 压测，以及 SGLang 四种对照起法 |
| `spec_matrix.md` | 配置字段、调度与 KV、CUDA Graph、草稿、验证各部分的源码行号，以及与 SGLang 的参数对照 |

## 用法

```bash
python3 sim_spec.py --scenario theory      # α、k、草稿成本 c → 接受长度、加速比、最优 k
python3 sim_spec.py --scenario batch       # batch 变大后最优 k 怎么往下掉
python3 sim_spec.py --scenario cudagraph   # CUDA Graph 尺寸、覆盖并发、每步预留块数
python3 sim_spec.py --scenario ngram       # KMP 最长后缀匹配；链式 vs 树形
python3 sim_spec.py --scenario dynamic     # vLLM 查表 vs SGLang EMA
python3 sim_spec.py --scenario metrics     # 同一批验证结果，两边各报什么数
python3 sim_spec.py --scenario sglang      # 同一个需求两边怎么配

python3 sim_rejection.py --scenario lossless    # 各规则的 TV 与每轮 token 数
python3 sim_rejection.py --scenario recovered   # 指数噪声 argmax 从残差采样
python3 sim_rejection.py --scenario greedy      # greedy 草稿 vs probabilistic 草稿
python3 sim_rejection.py --scenario threshold   # SGLang 阈值扫描
python3 sim_rejection.py --scenario fly         # FLy 窗口与熵门限扫描

python3 -m unittest test_sim_spec test_sim_rejection   # 17 个单测

bash serve_spec.sh base   && bash serve_spec.sh bench 1   # 先跑基线（起服务和压测各开一个终端）
bash serve_spec.sh eagle3 3                               # 换一种投机方法再压一次
for c in 1 16 64 256; do bash serve_spec.sh bench $c; done
grep -h "SpecDecoding metrics" results/*.log | tail
```

## 模拟器的假设

- **接受模型**：theory 和 batch 场景采用 Leviathan et al. 的 i.i.d. 假设，即每个草稿 token 独立地以概率 α 被接受。
- **batch 场景的两张表**：第一张是理想 roofline，一步时间 = max(字节 / 带宽, FLOP / 峰值算力)。第二张打了折扣：带宽只拿到 80%，算力只拿到 35%，每个草稿步额外加 0.3 ms 固定开销。
- **草稿头规模**：按 EAGLE-3 的形状估算，约 0.40B 参数。
- **未计入的开销**：CPU 调度、kernel launch、拒绝采样本身。
- **n-gram 树形对照**：这是简化模型，把所有历史匹配的后续合并成一棵 trie，不是 SGLang C++ 实现的移植。
- **"无损"的判定**：sim_rejection 检查生成序列前 3 个 token 的联合分布（共 64 种结果）是否与直接从目标模型采样一致，用 TV 距离和有限样本的噪声底对比。

与真机结果不一致时，以真机日志为准。

## 期望输出

`sim_spec.py --scenario theory`（节选）：

```
  α = 0.8
    n-gram（CPU 查表）         c=0.0   k1:1.80  k3:2.95  k5:3.69  k7:4.16   最优 k=16 → 4.89x
    EAGLE / MTP 头          c=0.05  k1:1.71  k3:2.57  k5:2.95  k7:3.08   最优 k=8  → 3.09x
    小草稿模型（1B 配 8B）         c=0.15  k1:1.57  k3:2.04  k5:2.11  k7:2.03   最优 k=5  → 2.11x
    大草稿模型（8B 配 70B）        c=0.25  k1:1.44  k3:1.69  k5:1.64  k7:1.51   最优 k=3  → 1.69x
```

`sim_spec.py --scenario batch`（打折扣那张表，α = 0.6，节选）：

```
  batch  k=0     k=1     k=2     k=3     k=4     k=5     k=6     k=7     最优 k
     1   1.00    1.46    1.64    1.68    1.65    1.59    1.52    1.45    3
    64   1.00    1.50    1.74    1.79    1.50    1.29    1.12    0.99    3
   256   1.00    1.53    1.52    1.26    1.06    0.91    0.79    0.70    1
```

`sim_spec.py --scenario cudagraph`（节选）：

```
  k   qlen  尺寸个数  最大尺寸  MRV1 取整后个数  MRV1 FULL decode 最多覆盖并发   MRV2 uniform decode 图个数  最多覆盖并发
  3   4     51        512       49               128                            49                          128
  7   8     51        512       48               64                             48                          64
```

`sim_rejection.py --scenario lossless`（噪声底 0.0120）：

```
  规则                                  TV        TV/噪声底      每轮 token    理论
  vLLM standard，草稿 probabilistic      0.0093    0.8         2.427       无损
  vLLM standard，草稿 greedy（默认）         0.0100    0.8         1.443       无损
  vLLM block，草稿 probabilistic         0.0124    1.0         2.615       无损
  vLLM synthetic [0.9,0.8,0.7]        0.3281    27.3        3.121       有损
  vLLM fly W=1 熵≥0.3                  0.2822    23.5        3.365       有损
  SGLang target-only 阈值 1/1（默认）       0.0089    0.7         1.448       无损
  SGLang target-only acc=0.5          0.3103    25.8        1.928       有损
  block 比 standard 每轮多产出：草稿 probabilistic +7.7%，草稿 greedy +0.1%。
```
