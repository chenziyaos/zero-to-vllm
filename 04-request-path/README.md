# VLLM-04 请求旅程

运行环境：CPU

计划内容：

- sim_processes.py：给定 DP、PP、TP 和 --api-server-count，列出进程拓扑与通道（ZMQ、共享内存 MessageQueue）
- trace_request.md：一次 chat 请求在 API server、EngineCore、worker 之间的调用链与源码行号

对应飞书文档发布后，本目录补齐代码和期望输出。
