# VLLM-26 Embedding 与 Rerank 服务

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-vllm/blob/main/colab/26-embedding-rerank.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

运行环境：`sim_pool.py`、`pool_client.py` 和测试只需要 CPU 上的 Python 3 标准库。`serve_pool.sh` 需要一张 GPU 和装好的 vLLM，0.6B 模型 24 GB 显存足够；`rr_qwen` 要用 vLLM 源码里的 jinja 模板，把源码克隆到 `$VLLM_SRC`。源码基线：vllm 主干 0d6f92a，SGLang 对照用主干 8fabf5e。

| 文件 | 内容 |
|---|---|
| `sim_pool.py` | 四个场景。models：六个模型的结构、池化方式、vLLM 对池化模型的 chunked prefill / prefix caching 默认值（移植 `is_chunked_prefill_supported` 和 `is_prefix_caching_supported`），两家的启动参数和接口。ledger：只有 prefill 时一步的时间、每步 token 预算、读权重和算力的分界。rerank：一次查询里 bi-encoder、cross-encoder、decoder reranker 的在线计算量，以及 Qwen3-Reranker 共享前缀能省多少。mrl：Matryoshka 截维对向量库存储的影响，两家对 `dimensions` 的校验和起服时怎么声明 |
| `pool_client.py` | 标准库客户端：embed、search（bi-encoder 召回）、rerank（cross-encoder 精排，vLLM 走 `/rerank`，SGLang 走 `/v1/rerank`）、bench（并发压测，报端到端延迟） |
| `serve_pool.sh` | 真机实验：四个模型的起法、`vllm bench serve` 打 `/v1/embeddings` 和 `/rerank`、客户端演示。没有在 GPU 上实跑过 |
| `data/docs.txt` | 12 段推理服务相关的短文，给 search 和 rerank 当候选 |
| `test_sim_pool.py` | 16 个用例：参数量、vLLM 默认值、启动参数、批预算、rerank token 账、前缀按块复用；客户端对着本地假服务端测请求体和两家响应格式的解析 |

## 用法

```bash
python3 sim_pool.py --scenario models
python3 sim_pool.py --scenario ledger --model qwen3-emb-0.6b --doc-len 512
python3 sim_pool.py --scenario ledger --model bge-m3 --doc-len 512
python3 sim_pool.py --scenario ledger --model qwen3-emb-8b --doc-len 2048
python3 sim_pool.py --scenario rerank --docs 100 --query-len 32 --doc-len 512
python3 sim_pool.py --scenario mrl --vectors 10000000
python3 test_sim_pool.py

# 有 GPU 时（未实测）
bash serve_pool.sh emb                       # 终端 1
bash serve_pool.sh client                    # 终端 2：embed、带 dimensions 的 embed、search、并发压测
bash serve_pool.sh bench_emb 512
bash serve_pool.sh rr_qwen                   # 换 reranker 后
bash serve_pool.sh bench_rr 100              # 压完看 vllm:prefix_cache_hits
```

## 和源码的对应

路径相对 vllm 仓库根目录。

| 本目录 | vllm |
|---|---|
| `vllm_defaults` | `vllm/config/model.py:2200` is_chunked_prefill_supported、`:2246` is_prefix_caching_supported；启动后 `vllm/v1/engine/core.py:274` 遇到 non_causal 的 KV spec 再强制关一遍 |
| `vllm_flags`（Qwen3-Embedding） | `vllm/config/model.py:1297`：有 sentence-transformers 的 modules.json 就判成 pooling runner |
| `vllm_flags`（Qwen3-Reranker） | `docs/models/pooling_models/scoring.md:88` 的 hf_overrides，模板在 `examples/pooling/score/template/qwen3_reranker.jinja` |
| `vllm_batch_budget` | `vllm/engine/arg_utils.py:2995` 起按显存选默认 `max_num_batched_tokens`；`:3202` 不开 chunked prefill 时取 max(max_model_len, 默认值) |
| ledger 里「没有 decode」 | `vllm/v1/core/sched/scheduler.py:154` 池化模型每步不预留采样位；`:2417` 有了 pooler 输出就结束请求 |
| ledger 里「只报 e2el」 | `vllm/benchmarks/serve.py:1856` |
| `scenario_mrl` | `vllm/pooling_params.py:246`：没有 is_matryoshka 就拒绝 dimensions；`vllm/config/model.py:2091` is_matryoshka 读 config |
| 接口列表 | `vllm/entrypoints/pooling/{embed,scoring,classify,pooling}/api_router.py` |
| `serve_pool.sh bench_*` | `vllm/benchmarks/lib/endpoint_request_func.py:1161` openai-embeddings、`:1170` vllm-rerank；`vllm/benchmarks/datasets/datasets.py:1601` random-rerank |

## 期望输出

```
模型                  架构                                  注意力  池化        矩阵乘参数  FLOP/token@512
Qwen3-Embedding-0.6B  Qwen3ForCausalLM                      因果    LAST             440 M          0.94 G
Qwen3-Embedding-4B    Qwen3ForCausalLM                      因果    LAST            3633 M          7.42 G
Qwen3-Embedding-8B    Qwen3ForCausalLM                      因果    LAST            6946 M         14.04 G
bge-m3                XLMRobertaModel                       双向    CLS              302 M          0.65 G
bge-reranker-v2-m3    XLMRobertaForSequenceClassification   双向    CLASSIFY         302 M          0.65 G
Qwen3-Reranker-0.6B   Qwen3ForCausalLM                      因果    CLASSIFY         440 M          0.94 G

vLLM 对池化模型的默认值（vllm/config/model.py:2200、:2246）：
  Qwen3-Embedding-0.6B   chunked prefill 开  prefix caching 开
  Qwen3-Embedding-4B     chunked prefill 开  prefix caching 开
  Qwen3-Embedding-8B     chunked prefill 开  prefix caching 开
  bge-m3                 chunked prefill 关  prefix caching 关
  bge-reranker-v2-m3     chunked prefill 关  prefix caching 关
  Qwen3-Reranker-0.6B    chunked prefill 开  prefix caching 开
  双向注意力不能分块、也不能复用前缀；因果注意力配 LAST 池化两者都能开，配 MEAN / CLS 就不行。
  启动后如果 KV spec 带 non_causal，引擎还会再强制关一遍（vllm/v1/engine/core.py:274）。

启动参数：
  Qwen3-Embedding-0.6B
    vLLM    vllm serve Qwen/Qwen3-Embedding-0.6B --runner pooling
    SGLang  python3 -m sglang.launch_server --model-path Qwen/Qwen3-Embedding-0.6B --is-embedding
  Qwen3-Embedding-4B
    vLLM    vllm serve Qwen/Qwen3-Embedding-4B --runner pooling
    SGLang  python3 -m sglang.launch_server --model-path Qwen/Qwen3-Embedding-4B --is-embedding
  Qwen3-Embedding-8B
    vLLM    vllm serve Qwen/Qwen3-Embedding-8B --runner pooling
    SGLang  python3 -m sglang.launch_server --model-path Qwen/Qwen3-Embedding-8B --is-embedding
  bge-m3
    vLLM    vllm serve BAAI/bge-m3 （不用加，架构本身就是池化模型）
    SGLang  python3 -m sglang.launch_server --model-path BAAI/bge-m3 --attention-backend triton --disable-radix-cache --chunked-prefill-size -1
  bge-reranker-v2-m3
    vLLM    vllm serve BAAI/bge-reranker-v2-m3 （不用加，架构本身就是池化模型）
    SGLang  python3 -m sglang.launch_server --model-path BAAI/bge-reranker-v2-m3 --is-embedding --attention-backend triton --disable-radix-cache --chunked-prefill-size -1
  Qwen3-Reranker-0.6B
    vLLM    vllm serve Qwen/Qwen3-Reranker-0.6B --runner pooling --hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' --chat-template examples/pooling/score/template/qwen3_reranker.jinja
    SGLang  python3 -m sglang.launch_server --model-path Qwen/Qwen3-Reranker-0.6B --disable-radix-cache --chat-template examples/chat_template/qwen3_reranker.jinja

接口：
  vLLM    /v1/embeddings、/v2/embed、/pooling；/score、/v1/score；/rerank、/v1/rerank、/v2/rerank；/classify
  SGLang  /v1/embeddings、/encode；/v1/rerank；/v1/score；/classify、/v1/classify
```

Qwen3-Embedding-0.6B 和 bge-m3 的对照：

```
Qwen3-Embedding-0.6B 在 H100 SXM 上，每条输入 512 token，bf16 权重，MFU 假设 40%
  每 token 0.94 GFLOP（矩阵乘 0.88 + 注意力 0.06）
  每步读权重 0.88 GB（不含词表查表），0.26 ms
  一步里 token 数超过 111 就从读权重受限转成算力受限；没有 decode，几乎每一步都在算力区
  算力上限 421,013 token/s，折合 822 条/s

  vLLM 默认每步 token 预算 8192（chunked prefill 开）
  一步最多放 16 条，一步 19.46 ms；排在第 k 步的请求，延迟约 k × 19.46 ms 加排队
  每条的 KV 56.0 MiB，请求一结束就归还（开着 prefix caching 时留作可复用的缓存块）；KV 池决定的是同一步能放多少条，而不是能挂多少个长对话
  没有 TPOT、ITL；压测只看端到端延迟（vllm bench serve 对池化模型默认只报 e2el，serve.py:1856）

bge-m3 在 H100 SXM 上，每条输入 512 token，bf16 权重，MFU 假设 40%
  每 token 0.65 GFLOP（矩阵乘 0.60 + 注意力 0.05）
  每步读权重 0.60 GB（不含词表查表），0.18 ms
  一步里 token 数超过 109 就从读权重受限转成算力受限；没有 decode，几乎每一步都在算力区
  算力上限 604,605 token/s，折合 1,181 条/s

  vLLM 默认每步 token 预算 8192（chunked prefill 关）
  一步最多放 16 条，一步 13.55 ms；排在第 k 步的请求，延迟约 k × 13.55 ms 加排队
  双向注意力：不分块、不复用前缀，单条输入必须整条放进一步，所以预算至少是 max_model_len
  没有 TPOT、ITL；压测只看端到端延迟（vllm bench serve 对池化模型默认只报 e2el，serve.py:1856）
```

一次查询的 rerank 账：

```
一次查询：query 32 token，候选 100 篇，每篇 512 token；H100 SXM，MFU 假设 40%
做法                                  在线 token     GFLOP   算力下限 ms
bi-encoder：只算 query 向量                   32      28.3         0.072
cross-encoder：bge-reranker-v2-m3         54,800   36050.2        91.128
decoder：Qwen3-Reranker-0.6B              61,800   58821.0       148.688
decoder + 前缀全部命中                    53,880   51282.8       129.633
bi-encoder 的文档向量离线算好：100 篇 × 512 token 只在入库时算一次，查询时只剩 100 次点积。
cross-encoder 每个 (query, 文档) 对都要整段过一遍模型，所以先用向量召回几百篇，再只对前几十篇 rerank。
Qwen3-Reranker 的模板把指令和 query 放在文档前面，同一次查询的 100 个对共享前 93 个 token，按 16 token 一块能复用 80 个。
  这一行是上限：要求前缀块已经算完并留在缓存里，同一步里一起排进去的请求未必互相命中，实际命中率看
  vLLM 的 prefix cache 命中率指标。SGLang 文档给 Qwen3-Reranker 的启动参数带 --disable-radix-cache，这时一个都不复用。
```

截维：

```
10,000,000 个向量的原始存储（不含索引结构）：
  维度   fp32 GB   fp16 GB
  4096     163.8      81.9
  2560     102.4      51.2
  1024      41.0      20.5
   512      20.5      10.2
   256      10.2       5.1
   128       5.1       2.6
截维要模型训练时用过 Matryoshka（MRL），否则召回会明显变差。两家都要求模型 config 先声明支持：
  vLLM    config 里要有 is_matryoshka 或 matryoshka_dimensions，否则请求直接 400（vllm/pooling_params.py:246）。
          Qwen3-Embedding 的 config.json 两个都没有，启动时加 --hf-overrides '{"is_matryoshka": true}'。
  SGLang  同样要 is_matryoshka 或 matryoshka_dimensions（srt/configs/model_config.py:919），否则拒绝请求；
          再校验 1 到 hidden_size、以及在 matryoshka_dimensions 列表里（srt/managers/tokenizer_manager.py:1485）。
          启动时加 --json-model-override-args '{"is_matryoshka": true}'；池化后先截维再做 L2 归一化（srt/layers/pooler.py:258）。
```

## 模拟器的假设

- 参数量按 config.json 逐项累加，只算参与矩阵乘的部分；词表 embedding 是查表，不进每步的读写和 FLOP。embedding 和 rerank 都不需要整张 lm_head（Qwen3-Reranker 在 vLLM 里被改成只取 yes / no 两行）。
- 注意力 FLOP 用 4 × 层数 × 头数 × head_dim × 可见长度，因果取平均可见 (L+1)/2，双向取 L。
- MFU 默认 0.4，是假设值。没有 decode，所以几乎每一步都在算力区，带宽只决定一步的下限。
- Qwen3-Reranker 模板的 token 数（61 / 4 / 9）是用 Qwen3 tokenizer 对默认 instruction 实际数出来的；改了 instruction 要重数。
- 前缀命中那一行是上限：同一步里一起排进去的请求未必互相命中，实际以 prefix cache 命中率指标为准。
