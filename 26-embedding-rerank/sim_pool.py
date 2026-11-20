"""Embedding 和 rerank 服务的账本：只有 prefill、没有 decode 之后，哪些参数还有意义，钱花在哪里。

    python3 sim_pool.py --scenario models                        # 六个模型的结构、池化方式、两家引擎的默认开关和启动参数
    python3 sim_pool.py --scenario ledger --model qwen3-emb-0.6b --doc-len 512
    python3 sim_pool.py --scenario rerank --docs 100 --query-len 32 --doc-len 256
    python3 sim_pool.py --scenario mrl --vectors 10000000         # Matryoshka 截维对向量库存储的影响

只用标准库。模型结构取自各模型 Hugging Face 的 config.json、modules.json 和 1_Pooling/config.json（main 分支，2026-11-19 读取）。
两家引擎的行为取自源码：vLLM 主干 0d6f92a，SGLang 主干 8fabf5e，行号写在用到的地方。
吞吐只按 Roofline 估上限，MFU 是假设值，不是实测。
"""
import argparse
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.specs import GPUS  # noqa: E402


def pad(text, width, right=False):
    """按终端显示宽度补空格：中文字符占两格。"""
    w = sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(text))
    fill = ' ' * max(0, width - w)
    return fill + str(text) if right else str(text) + fill


@dataclass(frozen=True)
class PoolModel:
    name: str
    hf: str
    arch: str
    attn: str            # 'causal'：decoder 因果注意力；'bidirectional'：encoder 双向注意力
    pooling: str         # LAST / CLS / CLASSIFY（cross-encoder 输出一个分数）
    layers: int
    hidden: int
    inter: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    max_len: int
    gated_mlp: bool      # SwiGLU 三个矩阵；BERT 类两层 MLP 两个矩阵

    @property
    def matmul_params(self):
        """参与矩阵乘的参数量（不含词表 embedding 查表；embedding 和 rerank 都不需要整张 lm_head）。"""
        attn = self.hidden * self.heads * self.head_dim * 2 + self.hidden * self.kv_heads * self.head_dim * 2
        mlp = (3 if self.gated_mlp else 2) * self.hidden * self.inter
        return self.layers * (attn + mlp)

    @property
    def weight_bytes(self):
        """每步要从显存读的 bf16 权重。词表 embedding 是按行查表，只读用到的几行，不计。"""
        return 2 * self.matmul_params

    def flops_per_token(self, seq_len):
        """一条长 seq_len 的输入，平均每个 token 的前向 FLOP。注意力项：因果平均看到 (L+1)/2 个 key，双向看到 L 个。"""
        seen = (seq_len + 1) / 2 if self.attn == 'causal' else seq_len
        return 2 * self.matmul_params + 4 * self.layers * self.heads * self.head_dim * seen


MODELS = {
    # Qwen/Qwen3-Embedding-0.6B：Qwen3ForCausalLM，1_Pooling pooling_mode_lasttoken=true，modules.json 带 Normalize
    'qwen3-emb-0.6b': PoolModel('Qwen3-Embedding-0.6B', 'Qwen/Qwen3-Embedding-0.6B', 'Qwen3ForCausalLM', 'causal',
                                'LAST', 28, 1024, 3072, 16, 8, 128, 151669, 32768, True),
    # Qwen/Qwen3-Embedding-4B
    'qwen3-emb-4b': PoolModel('Qwen3-Embedding-4B', 'Qwen/Qwen3-Embedding-4B', 'Qwen3ForCausalLM', 'causal',
                              'LAST', 36, 2560, 9728, 32, 8, 128, 151665, 40960, True),
    # Qwen/Qwen3-Embedding-8B
    'qwen3-emb-8b': PoolModel('Qwen3-Embedding-8B', 'Qwen/Qwen3-Embedding-8B', 'Qwen3ForCausalLM', 'causal',
                              'LAST', 36, 4096, 12288, 32, 8, 128, 151665, 40960, True),
    # BAAI/bge-m3：XLMRobertaModel，1_Pooling pooling_mode_cls_token=true；max_position_embeddings 8194，可用 8192
    'bge-m3': PoolModel('bge-m3', 'BAAI/bge-m3', 'XLMRobertaModel', 'bidirectional',
                        'CLS', 24, 1024, 4096, 16, 16, 64, 250002, 8192, False),
    # BAAI/bge-reranker-v2-m3：XLMRobertaForSequenceClassification，cross-encoder
    'bge-reranker-v2-m3': PoolModel('bge-reranker-v2-m3', 'BAAI/bge-reranker-v2-m3',
                                    'XLMRobertaForSequenceClassification', 'bidirectional',
                                    'CLASSIFY', 24, 1024, 4096, 16, 16, 64, 250002, 8192, False),
    # Qwen/Qwen3-Reranker-0.6B：config 写的是 Qwen3ForCausalLM，打分用 yes / no 两个 token 的 logit
    'qwen3-reranker-0.6b': PoolModel('Qwen3-Reranker-0.6B', 'Qwen/Qwen3-Reranker-0.6B', 'Qwen3ForCausalLM', 'causal',
                                     'CLASSIFY', 28, 1024, 3072, 16, 8, 128, 151669, 40960, True),
}

# Qwen3-Reranker 官方模板（vLLM examples/pooling/score/template/qwen3_reranker.jinja，SGLang
# examples/chat_template/qwen3_reranker.jinja）用 Qwen3 tokenizer 数出来的 token 数，instruction 用默认值。
# query 之前的部分 61 个，query 和 document 之间 4 个，document 之后 9 个。
QWEN3_RERANK_PRE, QWEN3_RERANK_MID, QWEN3_RERANK_SUF = 61, 4, 9
# XLM-R 句对：<s> query </s></s> document </s>，4 个特殊 token
XLMR_PAIR_SPECIAL = 4
VLLM_BLOCK = 16


def vllm_defaults(m):
    """移植 vllm/config/model.py:2200 is_chunked_prefill_supported 和 :2246 is_prefix_caching_supported 的池化分支。"""
    if m.attn == 'bidirectional':
        return False, False
    # 因果注意力：MEAN / CLS 池化不支持，LAST（以及 cross-encoder 取最后一个 token）支持
    ok = m.pooling not in ('MEAN', 'CLS')
    return ok, ok


def vllm_flags(key, m):
    if key == 'qwen3-reranker-0.6b':
        return ("--runner pooling --hf-overrides '{\"architectures\": [\"Qwen3ForSequenceClassification\"], "
                "\"classifier_from_token\": [\"no\", \"yes\"], \"is_original_qwen3_reranker\": true}' "
                "--chat-template examples/pooling/score/template/qwen3_reranker.jinja")
    if m.arch == 'Qwen3ForCausalLM':
        # 有 sentence-transformers 的 modules.json，自动判成 pooling（vllm/config/model.py:1297）；显式写更稳
        return '--runner pooling'
    return '（不用加，架构本身就是池化模型）'


def sglang_flags(key, m):
    """SGLang 文档 docs/docs/supported-models/{embedding,rerank}_models.mdx 给的启动参数。"""
    if key == 'qwen3-reranker-0.6b':
        # 不能加 --is-embedding：要用生成模式算 yes / no 的 logprob（rerank_models.mdx:117）
        return '--disable-radix-cache --chat-template examples/chat_template/qwen3_reranker.jinja'
    if m.arch == 'Qwen3ForCausalLM':
        return '--is-embedding'
    # XLM-R 系：架构不在生成模型名单里，自动走 embedding 模式（configs/model_config.py:2398）；
    # 文档要求 BGE 系列用 triton 或 torch_native 注意力后端。reranker 的命令照抄 rerank_models.mdx，
    # 它显式写了 --is-embedding 并手动关 radix cache 和 chunked prefill；bge-m3 沿用同样的关法
    flags = '--attention-backend triton --disable-radix-cache --chunked-prefill-size -1'
    return ('--is-embedding ' + flags) if m.pooling == 'CLASSIFY' else flags


def scenario_models(a):
    print(pad('模型', 22) + pad('架构', 38) + pad('注意力', 8) + pad('池化', 10) + pad('矩阵乘参数', 12, True)
          + pad('FLOP/token@512', 16, True))
    for k, m in MODELS.items():
        f = m.flops_per_token(512)
        print(pad(m.name, 22) + pad(m.arch, 38) + pad('因果' if m.attn == 'causal' else '双向', 8) + pad(m.pooling, 10)
              + pad(f'{m.matmul_params / 1e6:.0f} M', 12, True) + pad(f'{f / 1e9:.2f} G', 16, True))
    print()
    print('vLLM 对池化模型的默认值（vllm/config/model.py:2200、:2246）：')
    for k, m in MODELS.items():
        cp, pc = vllm_defaults(m)
        print(f'  {m.name:22} chunked prefill {"开" if cp else "关"}  prefix caching {"开" if pc else "关"}')
    print('  双向注意力不能分块、也不能复用前缀；因果注意力配 LAST 池化两者都能开，配 MEAN / CLS 就不行。')
    print('  启动后如果 KV spec 带 non_causal，引擎还会再强制关一遍（vllm/v1/engine/core.py:274）。')
    print()
    print('启动参数：')
    for k, m in MODELS.items():
        print(f'  {m.name}')
        print(f'    vLLM    vllm serve {m.hf} {vllm_flags(k, m)}')
        print(f'    SGLang  python3 -m sglang.launch_server --model-path {m.hf} {sglang_flags(k, m)}')
    print()
    print('接口：')
    print('  vLLM    /v1/embeddings、/v2/embed、/pooling；/score、/v1/score；/rerank、/v1/rerank、/v2/rerank；/classify')
    print('  SGLang  /v1/embeddings、/encode；/v1/rerank；/v1/score；/classify、/v1/classify')


def step_time(m, gpu, tokens, seq_len, mfu):
    """一步 prefill 的时间下限：读一遍权重（bf16）和按 MFU 折算的算力，取大者。"""
    t_mem = m.weight_bytes / (gpu.bw_tbs * 1e12)
    t_cmp = tokens * m.flops_per_token(seq_len) / (mfu * gpu.bf16_tflops * 1e12)
    return max(t_mem, t_cmp), t_mem, t_cmp


def vllm_batch_budget(m, gpu):
    """vllm/engine/arg_utils.py:2995 起：显存 >= 70 GiB 且不是 A100，API server 默认 8192；否则 2048。
    不开 chunked prefill 时取 max(max_model_len, 默认值)（:3202）。"""
    base = 8192 if gpu.mem_gb >= 70 and 'A100' not in gpu.name else 2048
    chunked, _ = vllm_defaults(m)
    return base if chunked else max(m.max_len, base)


def scenario_ledger(a):
    m, gpu = MODELS[a.model], GPUS[a.gpu]
    f = m.flops_per_token(a.doc_len)
    peak = a.mfu * gpu.bf16_tflops * 1e12
    budget = vllm_batch_budget(m, gpu)
    print(f'{m.name} 在 {gpu.name} 上，每条输入 {a.doc_len} token，bf16 权重，MFU 假设 {a.mfu:.0%}')
    print(f'  每 token {f / 1e9:.2f} GFLOP（矩阵乘 {2 * m.matmul_params / 1e9:.2f} + 注意力 {(f - 2 * m.matmul_params) / 1e9:.2f}）')
    print(f'  每步读权重 {m.weight_bytes / 1e9:.2f} GB（不含词表查表），{m.weight_bytes / (gpu.bw_tbs * 1e12) * 1e3:.2f} ms')
    knee = m.weight_bytes / (gpu.bw_tbs * 1e12) * peak / f
    print(f'  一步里 token 数超过 {knee:,.0f} 就从读权重受限转成算力受限；没有 decode，几乎每一步都在算力区')
    print(f'  算力上限 {peak / f:,.0f} token/s，折合 {peak / f / a.doc_len:,.0f} 条/s')
    print()
    print(f'  vLLM 默认每步 token 预算 {budget}（chunked prefill {"开" if vllm_defaults(m)[0] else "关"}）')
    per_step = max(1, budget // a.doc_len)
    t, _, _ = step_time(m, gpu, per_step * a.doc_len, a.doc_len, a.mfu)
    print(f'  一步最多放 {per_step} 条，一步 {t * 1e3:.2f} ms；排在第 k 步的请求，延迟约 k × {t * 1e3:.2f} ms 加排队')
    if a.doc_len > budget:
        print(f'  输入比预算长：{"会被切成多块跨步算" if vllm_defaults(m)[0] else "chunked prefill 关着，单条超过预算会排不进去"}')
    if m.attn == 'causal':
        kv = 2 * m.layers * m.kv_heads * m.head_dim * 2 * a.doc_len
        print(f'  每条的 KV {kv / 2**20:.1f} MiB，请求一结束就归还（开着 prefix caching 时留作可复用的缓存块）；KV 池决定的是同一步能放多少条，而不是能挂多少个长对话')
    else:
        print('  双向注意力：不分块、不复用前缀，单条输入必须整条放进一步，所以预算至少是 max_model_len')
    print('  没有 TPOT、ITL；压测只看端到端延迟（vllm bench serve 对池化模型默认只报 e2el，serve.py:1856）')


def rerank_tokens(kind, n, q, d):
    if kind == 'bi-encoder':
        return q
    if kind == 'cross-encoder':
        return n * (q + d + XLMR_PAIR_SPECIAL)
    if kind == 'decoder':
        return n * (QWEN3_RERANK_PRE + q + QWEN3_RERANK_MID + d + QWEN3_RERANK_SUF)
    if kind == 'decoder+prefix':
        full = QWEN3_RERANK_PRE + q + QWEN3_RERANK_MID + d + QWEN3_RERANK_SUF
        hit = (QWEN3_RERANK_PRE + q) // VLLM_BLOCK * VLLM_BLOCK
        return full + (n - 1) * (full - hit)
    raise ValueError(kind)


def scenario_rerank(a):
    gpu = GPUS[a.gpu]
    n, q, d = a.docs, a.query_len, a.doc_len
    rows = [
        ('bi-encoder：只算 query 向量', 'qwen3-emb-0.6b', 'bi-encoder', q),
        ('cross-encoder：bge-reranker-v2-m3', 'bge-reranker-v2-m3', 'cross-encoder', q + d + XLMR_PAIR_SPECIAL),
        ('decoder：Qwen3-Reranker-0.6B', 'qwen3-reranker-0.6b', 'decoder',
         QWEN3_RERANK_PRE + q + QWEN3_RERANK_MID + d + QWEN3_RERANK_SUF),
        ('decoder + 前缀全部命中', 'qwen3-reranker-0.6b', 'decoder+prefix',
         QWEN3_RERANK_PRE + q + QWEN3_RERANK_MID + d + QWEN3_RERANK_SUF),
    ]
    print(f'一次查询：query {q} token，候选 {n} 篇，每篇 {d} token；{gpu.name}，MFU 假设 {a.mfu:.0%}')
    print(pad('做法', 36) + pad('在线 token', 12, True) + pad('GFLOP', 10, True) + pad('算力下限 ms', 14, True))
    for label, key, kind, seq in rows:
        m = MODELS[key]
        tok = rerank_tokens(kind, n, q, d)
        fl = tok * m.flops_per_token(seq)
        ms = fl / (a.mfu * gpu.bf16_tflops * 1e12) * 1e3
        print(pad(label, 36) + pad(f'{tok:,}', 12, True) + pad(f'{fl / 1e9:.1f}', 10, True) + pad(f'{ms:.3f}', 14, True))
    print(f'bi-encoder 的文档向量离线算好：{n} 篇 × {d} token 只在入库时算一次，查询时只剩 {n} 次点积。')
    print('cross-encoder 每个 (query, 文档) 对都要整段过一遍模型，所以先用向量召回几百篇，再只对前几十篇 rerank。')
    hit = (QWEN3_RERANK_PRE + q) // VLLM_BLOCK * VLLM_BLOCK
    print(f'Qwen3-Reranker 的模板把指令和 query 放在文档前面，同一次查询的 {n} 个对共享前 {QWEN3_RERANK_PRE + q} 个 token，'
          f'按 {VLLM_BLOCK} token 一块能复用 {hit} 个。')
    print('  这一行是上限：要求前缀块已经算完并留在缓存里，同一步里一起排进去的请求未必互相命中，实际命中率看')
    print('  vLLM 的 prefix cache 命中率指标。SGLang 文档给 Qwen3-Reranker 的启动参数带 --disable-radix-cache，这时一个都不复用。')


def scenario_mrl(a):
    print(f'{a.vectors:,} 个向量的原始存储（不含索引结构）：')
    print(pad('维度', 6, True) + pad('fp32 GB', 10, True) + pad('fp16 GB', 10, True))
    for dim in (4096, 2560, 1024, 512, 256, 128):
        print(f'{dim:6}{a.vectors * dim * 4 / 1e9:10.1f}{a.vectors * dim * 2 / 1e9:10.1f}')
    print('截维要模型训练时用过 Matryoshka（MRL），否则召回会明显变差。两家都要求模型 config 先声明支持：')
    print('  vLLM    config 里要有 is_matryoshka 或 matryoshka_dimensions，否则请求直接 400（vllm/pooling_params.py:246）。')
    print("          Qwen3-Embedding 的 config.json 两个都没有，启动时加 --hf-overrides '{\"is_matryoshka\": true}'。")
    print('  SGLang  同样要 is_matryoshka 或 matryoshka_dimensions（srt/configs/model_config.py:919），否则拒绝请求；')
    print('          再校验 1 到 hidden_size、以及在 matryoshka_dimensions 列表里（srt/managers/tokenizer_manager.py:1485）。')
    print("          启动时加 --json-model-override-args '{\"is_matryoshka\": true}'；池化后先截维再做 L2 归一化（srt/layers/pooler.py:258）。")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scenario', choices=['models', 'ledger', 'rerank', 'mrl'], default='models')
    ap.add_argument('--model', choices=list(MODELS), default='qwen3-emb-0.6b')
    ap.add_argument('--gpu', choices=list(GPUS), default='h100-sxm')
    ap.add_argument('--mfu', type=float, default=0.4, help='假设的算力利用率，默认 0.4')
    ap.add_argument('--doc-len', type=int, default=512)
    ap.add_argument('--query-len', type=int, default=32)
    ap.add_argument('--docs', type=int, default=100)
    ap.add_argument('--vectors', type=int, default=10_000_000)
    a = ap.parse_args(argv)
    {'models': scenario_models, 'ledger': scenario_ledger, 'rerank': scenario_rerank, 'mrl': scenario_mrl}[a.scenario](a)


if __name__ == '__main__':
    main()
