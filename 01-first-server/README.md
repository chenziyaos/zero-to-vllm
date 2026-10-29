# VLLM-01 第一个服务

运行环境：GPU

计划内容：

- serve.sh：起 Qwen3-8B，日志落 logs/，给后续解析用
- smoke_test.py：/health、/v1/models、chat 非流式和流式，测 TTFT 与平均 ITL
- parse_startup_log.py：从启动日志抽出 KV cache 容量、最大并发、CUDA Graph 捕获耗时

对应飞书文档发布后，本目录补齐代码和期望输出。
