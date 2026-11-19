# VLLM-19 集群与 RL：路由、K8s 部署、sleep mode 与权重同步

运行环境：`sim_router.py` 和测试只需要 Python 3 标准库，在 CPU 上就能跑；`cluster.sh` 里的命令需要 GPU、vLLM（0d6f92a）或 SGLang（8fabf5e），K8s 那几段还需要 kubectl。

## 文件

| 文件 | 内容 |
|---|---|
| `sim_router.py` | 离线推演，场景有 route / hot / stale / probe / sleep / nccl / pause。路由打分照抄 `vllm/v1/engine/core_client.py:1789-1826`，前缀感知照抄 `sgl-router/src/policies/cache_aware.rs` 的 prefer / balanced 两种模式 |
| `test_sim_router.py` | 15 个回归测试，覆盖 KV 罚分的门槛、本地计数与扫描起点轮转、过期快照覆盖、各策略的命中率和倾斜、探针被杀的最晚时间、编译缓存卷能救回 CrashLoop、wait 比 keep 慢 |
| `cluster.sh` | 命令清单，按段落复制使用：vLLM DP（单机、多机 headless、ray、hybrid LB）、KV 事件、sgl-router、K8s Deployment 和探针、sleep level 0/1/2、RL 的 pause / 权重同步 / 版本标签，以及对应的 SGLang 命令 |
| `cluster_matrix.md` | vLLM 与 SGLang 对照表，每一行都标了源码行号或文档出处 |

## 用法

```bash
python3 sim_router.py --scenario all
python3 sim_router.py --scenario route --replicas 8 --rate 64
python3 sim_router.py --scenario hot
python3 -m unittest test_sim_router -v
bash cluster.sh            # 列出所有段落
bash cluster.sh k8s-yaml   # 打印 Deployment YAML
```

## 模拟器的假设

- **route / hot**：每台副本是 Qwen3-8B on H100，prefill 和 decode 的速度沿用 01、09 章实测的量级，KV 容量 399k token，前缀缓存按 LRU 淘汰。负载是 128 个共享前缀，按 Zipf 0.8 分布，合计 709k token，比单台副本的容量大。vLLM 打分每 100 ms 拿一次快照。prefer 和 balanced 的门槛取 sgl-router 默认值：命中至少 1024 token，factor 2，gap 1024。
- **stale**：64 条请求挤在同一个快照窗口里。“每 6 条被覆盖”模拟的是 coordinator 推来的快照比本地计数还旧的情况。
- **probe**：时间线来自 01 章（冷启动 84 s，编译缓存命中 43 s）。70B 只把读权重的时间按比例放大，其余阶段照搬 8B。kubelet 的判定按 k8s 文档：连续 failureThreshold 次失败才杀，重启退避从 10 s 起翻倍，封顶 300 s。
- **sleep**：用 vLLM 博客里 Phi-3-vision 的唤醒时间反推带宽，结果只当量级看。
- **nccl**：只算字节除以链路带宽，作为下限，不含 NCCL 建组和 packed 拷贝的开销。
- **pause**：64 条 rollout 的剩余长度随机生成，decode 步长 13.2 ms，重算按 01 章的 prefill 吞吐估。

## 期望输出

### route

```
策略                        前缀命中   TTFT p50   TTFT p99   TPOT p99      请求倾斜   prefill 倾斜    实算 prefill
轮询                       66.6%       85ms      857ms     68.1ms      1.00         1.03        3.78M
vLLM 内置打分                67.9%       90ms      807ms     68.3ms      1.03         1.04        3.64M
P2C                      67.3%      101ms      756ms     64.9ms      1.01         1.04        3.71M
前缀感知 prefer              89.3%       23ms      535ms     25.2ms      1.21         1.06        1.21M
前缀感知 balanced            85.8%       24ms      683ms     41.5ms      1.03         1.03        1.61M
```

`--replicas 8 --rate 64` 下：轮询命中 67.7%，实算 prefill 7.37M；prefer 命中 92.3%，实算 1.76M（少 76%），请求倾斜 1.49；balanced 命中 90.6%，倾斜 1.06。

### hot

```
轮询                       78.2%       28ms      507ms     37.7ms      1.00         1.04        2.56M
前缀感知 prefer              89.9%       60ms      381ms     55.5ms      2.32         1.29        1.19M
前缀感知 balanced            87.8%       25ms      471ms     25.6ms      1.03         1.16        1.44M
```

### stale

```
写法                                                  快照不变                  每 6 条被过期快照覆盖
只看快照                                       [64, 0, 0, 0]                 [64, 0, 0, 0]
+ 本地 waiting += client_count            [16, 16, 16, 16]              [21, 21, 11, 11]
+ 扫描起点轮转                                [16, 16, 16, 16]              [16, 16, 16, 16]
+ inflight 下限（当前主干）                     [16, 16, 16, 16]              [16, 16, 16, 16]
```

快照间隔 0 / 100 / 1000 ms 时，稳态 TTFT p50 分别是 78 / 90 / 151 ms。

### probe

```
                                            doc-liveness          default-3x10         startup-30x10
8B，编译缓存在容器层                          CrashLoop（重启 9 次）     CrashLoop（重启 9 次）                  84 s
8B，VLLM_CACHE_ROOT 挂 emptyDir              133 s（重启 1）     CrashLoop（重启 9 次）                  84 s
8B，无 PVC 每次拉权重 2 Gb/s                CrashLoop（重启 9 次）     CrashLoop（重启 9 次）                 150 s
70B TP4，编译缓存在卷上                      CrashLoop（重启 9 次）     CrashLoop（重启 9 次）                  97 s
```

### sleep

```
     重启进程，无编译缓存                               84.4 s
     重启进程，编译缓存命中                              43.3 s
     sleep level 1 → wake_up                   1.6 s
     sleep level 2 → wake + reload_weights     5.0 s
```

### nccl

```
Qwen3-8B BF16              16GB          16                    0.05 s                    0.33 s                    1.31 s
Llama-3.1-70B BF16        141GB         132                    0.40 s                    2.82 s                   11.30 s
DeepSeek-V3 FP8           671GB         625                    1.92 s                   13.42 s                   53.68 s
```

### pause

```
vLLM abort（默认）                     0.0 s  在飞请求带部分结果返回             503            1  重投要把 prompt + 已生成共 152621 token 重新 prefill
vLLM wait                         52.3 s  等最长的一条再跑 3950 步         503            1  无
vLLM keep + clear_cache            0.0 s  冻结在队列里                  照收照排           2  resume 后重算 152621 token 的 KV，约 5.1 s
```
