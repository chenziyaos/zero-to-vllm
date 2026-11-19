"""模型结构和 GPU 规格表，全系列练习共用。

数字取自各模型 Hugging Face config.json 和厂商公开规格页（稠密 BF16，不含稀疏加速）。
改动或新增条目时，在注释里写清出处。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    name: str
    params_b: float          # 总参数量，十亿
    active_b: float          # 每 token 激活参数量，十亿（稠密模型等于总参数）
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    mla_kv_rank: int = 0     # MLA 的压缩 KV 维度，非 MLA 为 0
    mla_rope_dim: int = 0    # MLA 解耦 RoPE 的维度

    def kv_bytes_per_token(self, dtype_bytes: float = 2) -> float:
        """每个 token 在所有层上占用的 KV cache 字节数。"""
        if self.mla_kv_rank:
            # MLA 每层只存压缩后的 latent 和 RoPE key，K、V 共用
            return self.layers * (self.mla_kv_rank + self.mla_rope_dim) * dtype_bytes
        return 2 * self.layers * self.kv_heads * self.head_dim * dtype_bytes


MODELS = {
    # Qwen/Qwen3-0.6B config.json
    'qwen3-0.6b': Model('Qwen3-0.6B', 0.6, 0.6, 28, 1024, 16, 8, 128, 151936),
    # Qwen/Qwen3-8B config.json；按 config 逐层累加为 8,190,735,360 个参数
    'qwen3-8b': Model('Qwen3-8B', 8.19, 8.19, 36, 4096, 32, 8, 128, 151936),
    # meta-llama/Llama-3.1-8B config.json
    'llama3.1-8b': Model('Llama-3.1-8B', 8.0, 8.0, 32, 4096, 32, 8, 128, 128256),
    # meta-llama/Llama-3.1-70B config.json
    'llama3.1-70b': Model('Llama-3.1-70B', 70.6, 70.6, 80, 8192, 64, 8, 128, 128256),
    # deepseek-ai/DeepSeek-V3 config.json：kv_lora_rank 512，qk_rope_head_dim 64
    'deepseek-v3': Model('DeepSeek-V3', 671, 37, 61, 7168, 128, 128, 128, 129280,
                         mla_kv_rank=512, mla_rope_dim=64),
}


@dataclass(frozen=True)
class GPU:
    name: str
    mem_gb: float
    bw_tbs: float            # HBM 带宽，TB/s
    bf16_tflops: float       # 稠密 BF16 Tensor Core 峰值
    fp8_tflops: float

    @property
    def ridge(self) -> float:
        """BF16 Roofline 拐点：每字节需要多少 FLOP 才能吃满算力。"""
        return self.bf16_tflops / self.bw_tbs


GPUS = {
    'a100-80g': GPU('A100 80GB SXM', 80, 2.039, 312, 0),
    'h100-sxm': GPU('H100 SXM', 80, 3.35, 989, 1979),
    'h200': GPU('H200', 141, 4.8, 989, 1979),
    'l40s': GPU('L40S', 48, 0.864, 362, 733),
    'rtx4090': GPU('RTX 4090', 24, 1.008, 165, 330),
}
