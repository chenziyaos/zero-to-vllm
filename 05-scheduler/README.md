# VLLM-05 调度器

运行环境：CPU

计划内容：

- sim_scheduler.py：token 预算、num_computed_tokens 追赶模型、running 优先、KV 不足时抢占
- test_sim_scheduler.py：与主干行为对齐的边界用例

对应飞书文档发布后，本目录补齐代码和期望输出。
