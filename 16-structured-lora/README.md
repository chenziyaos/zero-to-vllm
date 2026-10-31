# VLLM-16 结构化输出与多 LoRA

运行环境：模拟器只需要 CPU 和 Python 3 标准库。`serve_so_lora.sh` 需要一张 24 GB 以上的卡（默认 Qwen3-8B），装好 vLLM，把 vLLM 源码克隆到 `$VLLM_SRC`（结构化输出压测脚本在 `benchmarks/` 下）；多 LoRA 实验要在 `$LORA_DIR` 下放若干 PEFT adapter（每个子目录一个）。SGLang 对照另装 sglang。源码基线：vllm 主干 0d6f92a，sglang 主干 8fabf5e。

| 文件 | 内容 |
|---|---|
| `sim_so_lora.py` | 十一个场景。mask：bitmask 的列数、预分配与每步 H2D。overlap：XGrammar 各优化档位下填 mask 与 GPU 前向串行 / 重叠的 TPOT，128 并行阈值。compile：编译线程、缓存命中、vLLM 正则 5 s 超时 vs SGLang 轮询上限。spec：逐行移植 `StructuredOutputManager`，复现官方测试的 14 种行模式、trim 长度、推理段屏蔽隐式结束符。auto：13 种 schema 走哪个后端，以及引擎锁后端。jump：jump-forward 的论文收益与两家都没接上的现状。loramem：rank × max_loras × TP × fully_sharded 的预分配显存。punica：Loop / Gather-BMM / SGMV 与 AWS 的 kernel 调优。sched：Zipf 负载下 vLLM max_loras=1/4/8、SGLang 8 槽、drainer 的排队时间。prefix：换路径、改名、多机时前缀缓存的 key。sglang：对照表 |
| `test_sim_so_lora.py` | 26 个单测：列数 4748 / 4008；投机放大 mask 字节；`grammar_invalid_drafts`；两种线程数公式；并行阈值与投机串行；SGLang 超时；14 个 FLOW_CASES、trim、正则终止、R 行、并行路径 R→C、bitmask 后语法状态复原；auto 13 例、嵌套 64 边界、Tekken+Lark；LoRA 显存线性于 rank 和槽数、fully_sharded 只在 TP>1 时省；`validate_lora_batch` 计 base、pinned 预留；drainer 选剩余最少者、1.2 容差、跑空解除、未满不动；vLLM max_loras=1 堵第二个 adapter；SGLang base 占槽；vLLM key 带 path；uuid5 稳定 uuid4 不稳定 |
| `serve_so_lora.sh` | 真机实验：四种后端、ngram 投机、Qwen3 推理段、结构化输出压测（五种数据集、比例扫描）、auto 锁后端复现；多 LoRA 起服、三档 rank 的 KV 可用显存、随机 adapter 压测与 `lora_requests_info`、运行时加载卸载；SGLang 对照 |
| `so_lora_matrix.md` | 结构化输出后端、编译与 bitmask、多 LoRA 调度与显存的源码行号，以及与 SGLang 的参数对照 |

## 用法

```bash
python3 sim_so_lora.py --scenario mask       # bitmask 账本
python3 sim_so_lora.py --scenario overlap    # 填 mask 与前向重叠
python3 sim_so_lora.py --scenario compile    # 编译与超时
python3 sim_so_lora.py --scenario spec       # 投机解码下的语法推进（复现官方测试）
python3 sim_so_lora.py --scenario auto       # auto 后端选择
python3 sim_so_lora.py --scenario jump       # jump-forward
python3 sim_so_lora.py --scenario loramem    # LoRA 预分配显存
python3 sim_so_lora.py --scenario punica     # 三种 LoRA 算子
python3 sim_so_lora.py --scenario sched      # 多 LoRA 调度排队（约 1 秒）
python3 sim_so_lora.py --scenario prefix     # 前缀缓存隔离
python3 sim_so_lora.py --scenario sglang     # SGLang 对照

python3 -m unittest test_sim_so_lora         # 26 个单测

# 结构化输出：两个终端
bash serve_so_lora.sh so xgrammar            # 终端 1
bash serve_so_lora.sh so_bench json          # 终端 2；再换 json-unique 看每请求新 schema 的编译排队
bash serve_so_lora.sh so_ratio               # 结构化比例 0 / 0.5 / 1

# auto 锁后端：用 auto 起服，第一个请求带 multipleOf
bash serve_so_lora.sh so auto
bash serve_so_lora.sh auto_lock              # 第二个请求应报 only supports one backend per engine

# 多 LoRA：max_loras 1 vs 8
bash serve_so_lora.sh lora 1 16              # 终端 1
bash serve_so_lora.sh lora_bench 8           # 终端 2；再用 lora 8 16 起服重跑，对比 TTFT p99
bash serve_so_lora.sh lora_mem               # rank 16 / 64 / 256 的 KV cache 可用显存
```

## 模拟器的假设

- **语法**：玩具实现，有限个 JSON 字符串（前缀可接受即可接受）和两条正则。只用来复现 `tests/v1/structured_output/test_structured_output_manager.py` 的行模式，不代表 xgrammar 的 PDA。
- **overlap**：GPU 单步按 XGrammar 论文 Table 2（MLC-LLM、Llama-3.1-8B）bs1 6.2 ms、bs16 9.0 ms 线性外推；每行 mask 成本取论文 Table 3 各档位；并行按 cpu=16。H2D 有效带宽假设 25 GB/s。
- **loramem**：只算 qkv / o / gate_up / down，bf16；列并行 A 是全量输入、B 按 TP 切；行并行 A 按 TP 切、B 全量（fully_sharded 时也切）。
- **punica**：SGMV 只有论文 Fig 8 / Fig 9 的端点是实测，中间线性插值；Loop 下限按每个不同模型 37 µs 估。
- **sched**：每步固定 20 ms、每个 running 请求出 1 个 token，max_num_seqs=64，不计 token 预算和 LoRA 个数对算子的影响；drainer 阈值 2 s（100 步）。
- **prefix**：块哈希用 sha256 演示，只看 extra key 是否不同。

与真机结果不一致时，以真机日志为准。

## 期望输出

`sim_so_lora.py --scenario sched`（节选）：

```
  配置                             排队 p50     p99      最长    没排上 |  冷门 adapter p99 base p99
  vLLM max_loras=1（默认）            3.24s  116.5s  119.5s    317 |          117.8s     0.5s
  vLLM max_loras=8                0.00s   22.5s   27.6s      0 |           27.6s     0.0s
  SGLang 8 槽（含 base）              0.00s   38.1s   43.7s      0 |           43.7s     0.0s
  SGLang 8 槽 + drainer 2 s        2.94s   16.3s   21.3s      0 |           18.6s    12.6s
```

`sim_so_lora.py --scenario auto`（节选）：

```
  price: multipleOf 0.01             fast      guidance  number + multipleOf
  patternProperties + properties     fast      outlines  patternProperties + properties；有 patternProperties
  multipleOf + 慢 tokenizer           slow      reject    number + multipleOf；guidance 不支持慢 tokenizer
  嵌套 65 层                            fast      reject    嵌套超过 64 层
```

`sim_so_lora.py --scenario loramem`（节选）：

```
  Qwen3-8B        16         1   1      83MiB              83MiB
  Qwen3-8B        16         8   1     666MiB             666MiB
  Qwen3-8B       256         8   1   10656MiB           10656MiB
  Qwen3-8B       256         8   4    5688MiB            2664MiB
```
