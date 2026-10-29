# VLLM-08 前缀缓存与混合 KV 管理

运行环境：CPU

计划内容：

- sim_prefix_cache.py：块哈希链、命中查找、LRU 驱逐
- sim_hybrid_kv.py：全注意力与滑窗、Mamba 层混合时的 KV 组管理
- 对照 SGLang RadixAttention 的同一组请求命中率

对应飞书文档发布后，本目录补齐代码和期望输出。
