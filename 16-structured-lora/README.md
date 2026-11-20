# VLLM-16 结构化输出与多 LoRA

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-vllm/blob/main/colab/16-structured-lora.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

运行环境：模拟器和 `check_adapter.py` 只需要 CPU 和 Python 3 标准库，`train_lora.py` 要装 `requirements-lora.txt`。`serve_so_lora.sh` 需要一张 24 GB 以上的卡（默认 Qwen3-8B），装好 vLLM，把 vLLM 源码克隆到 `$VLLM_SRC`（结构化输出压测脚本在 `benchmarks/` 下）；多 LoRA 实验要在 `$LORA_DIR` 下放若干 PEFT adapter（每个子目录一个）。SGLang 对照另装 sglang。源码基线：vllm 主干 0d6f92a，sglang 主干 8fabf5e。

| 文件 | 内容 |
|---|---|
| `sim_so_lora.py` | 十一个场景。mask：bitmask 的列数、预分配与每步 H2D。overlap：XGrammar 各优化档位下填 mask 与 GPU 前向串行 / 重叠的 TPOT，128 并行阈值。compile：编译线程、缓存命中、vLLM 正则 5 s 超时 vs SGLang 轮询上限。spec：逐行移植 `StructuredOutputManager`，复现官方测试的 14 种行模式、trim 长度、推理段屏蔽隐式结束符。auto：13 种 schema 走哪个后端，以及引擎锁后端。jump：jump-forward 的论文收益与两家都没接上的现状。loramem：rank × max_loras × TP × fully_sharded 的预分配显存。punica：Loop / Gather-BMM / SGMV 与 AWS 的 kernel 调优。sched：Zipf 负载下 vLLM max_loras=1/4/8、SGLang 8 槽、drainer 的排队时间。prefix：换路径、改名、多机时前缀缓存的 key。sglang：对照表 |
| `test_sim_so_lora.py` | 26 个单测：列数 4748 / 4008；投机放大 mask 字节；`grammar_invalid_drafts`；两种线程数公式；并行阈值与投机串行；SGLang 超时；14 个 FLOW_CASES、trim、正则终止、R 行、并行路径 R→C、bitmask 后语法状态复原；auto 13 例、嵌套 64 边界、Tekken+Lark；LoRA 显存线性于 rank 和槽数、fully_sharded 只在 TP>1 时省；`validate_lora_batch` 计 base、pinned 预留；drainer 选剩余最少者、1.2 容差、跑空解除、未满不动；vLLM max_loras=1 堵第二个 adapter；SGLang base 占槽；vLLM key 带 path；uuid5 稳定 uuid4 不稳定 |
| `serve_so_lora.sh` | 真机实验：四种后端、ngram 投机、Qwen3 推理段、结构化输出压测（五种数据集、比例扫描）、auto 锁后端复现；多 LoRA 起服、三档 rank 的 KV 可用显存、随机 adapter 压测与 `lora_requests_info`、运行时加载卸载；SGLang 对照 |
| `train_lora.py` | 用 PEFT 在 `data/city_country.jsonl` 上训一个 LoRA adapter（城市 → 一行 JSON），只对 completion 算 loss，打印留出样本训练前后的输出；默认用随机权重的小 Llama 只验证流程，`--base Qwen/Qwen3-0.6B` 训真模型 |
| `check_adapter.py` | 上线前检查：只读 `adapter_config.json` 和 safetensors 头部，对照 vLLM 和 SGLang 加载时的校验（rank 上限、DoRA、bias、modules_to_save、加词表、目标模块），给出两边的启动参数；只用标准库 |
| `test_lora_adapter.py` | 13 个用例：12 个测 `check_adapter`，1 个在装了 torch / transformers / peft 时用小模型训一遍再检查 |
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

## 自己训一个 LoRA adapter

`serve_so_lora.sh` 和下面的模拟器都假设手里已经有 adapter。这一节从零训一个，再检查它能不能被两边加载。训练环境和推理环境分开装：

```bash
pip install -r ../requirements-lora.txt                     # torch、transformers、peft，CPU 就行
python3 train_lora.py                                       # 随机权重的小 Llama，几秒，只验证流程
python3 train_lora.py --base Qwen/Qwen3-0.6B --steps 80 --lr 1e-3 --out loras/qwen-city-r8
python3 train_lora.py --base Qwen/Qwen3-0.6B --rank 64 --out loras/qwen-city-r64   # 换 rank，给多 LoRA 实验凑不同档位
python3 check_adapter.py loras/*                            # 上线前检查，只要标准库
python3 test_lora_adapter.py
```

任务是让模型把「城市：巴黎」答成一行 JSON。底座不会自发输出这个格式，训完的 adapter 会，所以加不加 adapter 一眼就能看出差别。下面是 Qwen3-0.6B 在 Apple M4 Pro 的 CPU 上实跑的输出（float32，batch 8，80 步）：

```
底座 Qwen/Qwen3-0.6B，LoRA r=8 alpha=16 模块 ['q_proj', 'k_proj', 'v_proj', 'o_proj']
可训练参数 2,293,760 / 总参数 598,343,680（0.383%）
  step    0  loss 2.3322
  step   13  loss 0.0050
  step   26  loss 0.0008
  step   39  loss 0.0013
  step   52  loss 0.0008
  step   65  loss 0.0029
  step   78  loss 0.0014
  step   79  loss 0.0005
训练 80 步，用时 73.3 s，adapter 存到 loras/qwen-city-r8
留出的 4 条：loss 2.338 → 0.057，完全答对 0% → 75%
探针 '城市：巴西利亚'
  训练前 '时间：2023年'
  训练后 '{"city": "巴西利亚", "country": "巴西"}'
```

留出的 4 个城市里答对 3 个。可训练参数只占 0.38%，adapter 文件 9.2 MB。`check_adapter.py` 对这个目录的检查结果：

```
== qwen-city-r8
  底座      Qwen/Qwen3-0.6B
  r / alpha 8 / 16
  声明模块  ['v_proj', 'k_proj', 'o_proj', 'q_proj']
  实际模块  ['k_proj', 'o_proj', 'q_proj', 'v_proj']（224 个张量）
  vLLM    可以加载
  SGLang  可以加载
  vLLM    --enable-lora --max-lora-rank 8 --lora-modules qwen-city-r8=/绝对路径/loras/qwen-city-r8
  SGLang  --enable-lora --max-lora-rank 8 --lora-target-modules k_proj o_proj q_proj v_proj --lora-paths qwen-city-r8=/绝对路径/loras/qwen-city-r8
```

几个容易踩的点，检查脚本都会报出来：

- vLLM 的 `--max-lora-rank` 只能取 1、8、16、32、64、128、256、320、512，默认 16（`vllm/config/lora.py:27`）。rank 64 的 adapter 不改这个参数会加载失败。
- SGLang 不传 `--max-lora-rank` 时，取启动时所有 adapter 里最大的 r（`lora/lora_manager.py:896`）；之后热加载更大 rank 的 adapter 会报 incompatible。
- 两边都不支持 DoRA。vLLM 的 `modules_to_save` 只接受 `classifier` / `score`，所以把 `lm_head` 或 `embed_tokens` 也一起训的 adapter 上不了 vLLM。
- 训练时往词表里加了 token（目录里有 `added_tokens.json`、id 超出底座词表），SGLang 拒绝加载。
- SGLang 的 `--lora-target-modules` 会把 q/k/v_proj 归一成 `qkv_proj`、gate/up_proj 归一成 `gate_up_proj`；只写 `qkv_proj` 而 adapter 还训了 `o_proj`，同样加载失败。

把训好的几个目录放进 `$LORA_DIR`，就能接着跑 `serve_so_lora.sh lora`、`lora_bench` 和 `sgl_lora`。GPU 上的加载和压测本节没有实测。

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
