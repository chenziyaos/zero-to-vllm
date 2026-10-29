# VLLM-02 推理账本

运行环境：CPU

计划内容：

- sim_ledger.py：权重、KV 池、可放 token 数、decode 带宽下限，对照启动日志的 GPU KV cache size
- sim_profile_run.py：复现 determine_available_memory 的扣减顺序（权重、峰值激活、非 torch 显存）

对应飞书文档发布后，本目录补齐代码和期望输出。
