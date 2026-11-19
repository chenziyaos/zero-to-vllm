# VLLM-23 实战：DeepSeek-V4.1-Flash 部署，消费级显卡与 H20

运行环境：`deploy_planner.py` 和 `test_deploy_planner.py` 只依赖 Python 3 标准库，不联网、不需要 GPU，在笔记本上就能把每一档的显存账算出来。`serve.sh` 和 `bench.sh` 要在真机上跑，需要 vLLM 0.31.0 以上（recipe 的 `min_vllm`），SM120（RTX PRO 6000 / 5090）还需要带 flashinfer-ai/flashinfer#4380 的 FlashInfer。源码行号对应 vllm 主干 0d6f92a。

## 文件

| 文件 | 内容 |
|---|---|
| `deploy_planner.py` | 部署规划器，场景有 ckpt / mem / kv / indexer / host / decode / rank，算 checkpoint 拆分、每卡常驻、indexer 预留、KV 池、Engram 主机内存和 decode 带宽估算 |
| `test_deploy_planner.py` | 22 个回归测试：字节数钉在源码常量和 recipe 注释上，两条校准钉在 H200 recipe 实测和 commdata2338 的 4×RTX PRO 6000 实测上 |
| `serve.sh` | 十个档位的 `vllm serve` 命令（含 SM89 分叉的 docker 命令和 SGLang 对照），日志落 `results/<档位>.log`，并过滤出要看的几行 |
| `bench.sh` | 冒烟（算术题 + 工具调用）、decode 并发扫描、长上下文 prefill、`/metrics` 里的 DSpark 接受率 |

## 用法

```bash
# 不需要 GPU：先算账
python3 deploy_planner.py --scenario all
python3 deploy_planner.py --scenario mem --gpu h20_141g --n 8 --tp 8 --engram gpu --max-num-seqs 128 --batched 8192
python3 deploy_planner.py --scenario host --gpu rtxpro6000 --n 4 --tp 4 --thp
python3 -m unittest test_deploy_planner -v
bash serve.sh plan h20x8            # 打印 serve.sh 某一档对应的显存账

# 真机：起服务，再压测
bash serve.sh pro6000x4             # 4×RTX PRO 6000，消费级首选
bash serve.sh h20e_tp8              # 8×H20-3e 141G，数据中心首选
bash bench.sh smoke
bash bench.sh decode
CONC="1 16" bash bench.sh decode
bash bench.sh long
bash bench.sh metrics
```

`serve.sh` 的环境变量：`MODEL`、`OUT`、`PORT`、`GMU`（覆盖档位默认值）、`ADAPTIVE`（SM120 上自适应验证报错时设 `false`）、`SPEC`（DSpark 草稿数，默认 5）、`TEXT_ONLY=1`（加 `--language-model-only`）、`FORK_IMAGE`、`HF_HOME`。

## 口径

- **ckpt**：五块的参数量和 GiB 取 recipe 的 Memory 表，合计 475.3 GiB；`vram_minimum_gb` 是 511 GB × 1.2，recipe 写 614。Engram 行数 768,022,850、每行 256 字节 FP8，加 scale 是 188.83 GiB，这张表可以整张搬到主机内存（`--engram-config '{"cpu_offload":true}'`）。
- **mem**：常驻 = 静态切分 × 系数。系数有两档：SM120 的 1.037 由 commdata2338 的 74.29 GiB/卡反推，SM90 的 1.134 由 recipe H200 的 81.2 GiB 反推。预留 = indexer logits + 「其他」，「其他」也是这两条实测校准出来的残差，gather workspace 已经含在里面，所以不单列。KV 池 = gmu × 显存 − 常驻 − Engram − 预留。
- **kv**：只有第 2、8、14、20 层分配全局 KV，每 token 2.5 条。博客的 890 B/token 是 SM100 的 nvfp4 格式；SM90 / SM120 / SM89 用 `fp8_ds_mla`，是 (584 + 132) × 2.5 = 1790 B。`--kv-cache-dtype fp8` 在这个模型上是 `fp8_ds_mla` 的别名（`sparse_mla.py:112`）。
- **indexer**：v0.31.0 / v0.32.0 固定 512 MiB；recipe 注释按 batched × len × 2 B 估；主干 #62860 之后是 `max(512 MiB, min(seqs, batched) × (1 + spec) × round_up(len, align) × 4 B)`。排行默认按主干算，`--version` 可切换。
- **host**：每 rank 私有的 pinned 内存按 2 的幂取整，188.83 GiB 的表实际占 264 GiB；`use_thp` 按实际大小分；DP > 1 且 Engram 在主机时默认走 `/dev/shm` 共享，整机只有一份。
- **decode**：带宽模型，η = 0.6，均匀路由，DSpark 每步接受 4.2 个草稿（每步 5.2 token）。只用来排序和定量级，**全部是估算**。和公开实测的比值在 0.39–1.19 之间，偏差原因见 `decode` 场景末尾三行。H20 上唯一的公开点是 SGLang（#46310，比值 0.39），H20 的 c1 / c16 估算按上限读。
- **rank**：c1 / c16 列是 decode 估算；放不下的配置打「—」。

## 期望输出

### mem（4×RTX PRO 6000，默认参数）

```
-- 显存账（每卡）
   RTX PRO 6000 96G × 4，TP4 DP1，Engram 在主机
     可用 88.9 GiB = 95.6 × gmu
     常驻权重 74.3 GiB（静态 71.6 × 系数 1.037）
     预留 1.2 GiB = indexer logits 0.50 + 其他 0.65（其他是两条实测校准出的残差；gather workspace 2.06 不单列）
     KV 池 13.5 GiB → 8.08M token（1790 B/token），1048576 上下文并发 7.69
     主机内存 ≥ 312 GiB，建议配 384 GB
-- 和 recipe 对账
   H200 TP4 Engram 卸载：常驻 81.2（recipe 实测 81.2），KV 38.5 GiB（recipe 38.5），23.1M token（recipe 约 20M）
   H100 TP8 静态常驻 35.8 GiB（recipe 写约 36，是静态账）
   8×H200 TP4×2：按 TP×DP 摊是每 rank 23.6 GiB、每副本 94.4 GiB（recipe 注释 23.6 / 94）；主干默认 dp_shared_memory，整机一份 188.8 GiB 在 /dev/shm
```

### host

```
-- Engram 放主机：日志写的大小和实际 pinned 的大小
   1 卡 TP1：1 份，每 rank 私有 pinned，每份日志 188.83 GiB，实占  264.0 GiB；合计 188.83 → 264.0 GiB
   4 卡 TP4：4 份，每 rank 私有 pinned，每份日志  47.21 GiB，实占   66.0 GiB；合计 188.83 → 264.0 GiB
   8 卡 TP8：8 份，每 rank 私有 pinned，每份日志  23.60 GiB，实占   33.0 GiB；合计 188.83 → 264.0 GiB
   8 卡 TP4：4 份，DP 副本共享 /dev/shm，每份日志  47.21 GiB，实占   47.2 GiB；合计 188.83 → 188.8 GiB
   私有 pinned 按 2 的幂取整（engram.py:1122 注释）；加 use_thp 走大页，按实际大小
```

### decode

```
-- decode 带宽估算（η=0.6，均匀路由，接受 4.2 个草稿 → 每步 5.2 token）
   RTX PRO 6000 96G      ×4 卸载   0  bs1   12.3 ms/步     424 tok/s；bs8   41.8 ms/步     995 tok/s；bs16   58.5 ms/步    1421 tok/s
   RTX PRO 6000 96G      ×1 卸载 214  bs1  456.3 ms/步      11 tok/s；bs8 2574.0 ms/步      16 tok/s；bs16 3772.0 ms/步      22 tok/s
   RTX 5090 32G          ×8 卸载   9  bs1   25.8 ms/步     201 tok/s；bs8  124.7 ms/步     334 tok/s；bs16  180.6 ms/步     461 tok/s
   RTX 4090 48G（改装）      ×8 卸载   0  bs1   12.1 ms/步     428 tok/s；bs8   38.4 ms/步    1083 tok/s；bs16   53.3 ms/步    1561 tok/s
   H20 96G               ×8 卸载   0  bs1    4.7 ms/步    1118 tok/s；bs8   11.3 ms/步    3691 tok/s；bs16   15.0 ms/步    5540 tok/s
   H20-3e 141G           ×8 卸载   0  bs1    4.3 ms/步    1198 tok/s；bs8    9.9 ms/步    4219 tok/s；bs16   13.0 ms/步    6409 tok/s
-- 公开实测 / 估算（估算按各自的接受长度重算）
   yhfgyyf 4×PRO 6000 分叉 8K c1           bs1  接受长度 5.21  实测   267.0  估算     425  比值 0.63  sm89 分叉 README
   commdata2338 4×PRO 6000 prose bs1     bs1  接受长度 3.60  实测   252.0  估算     294  比值 0.86  Kraken + B12X
   commdata2338 4×PRO 6000 prose bs8     bs8  接受长度 3.60  实测   822.0  估算     689  比值 1.19  Kraken + B12X
   commdata2338 4×PRO 6000 prose bs16    bs16 接受长度 3.60  实测  1150.0  估算     984  比值 1.17  Kraken + B12X
   devin-lai 8×5090 c1                   bs1  接受长度 3.60  实测    74.8  估算     139  比值 0.54  主线 vLLM + Marlin
   devin-lai 8×5090 c8                   bs8  接受长度 3.60  实测   168.5  估算     231  比值 0.73  主线 vLLM + Marlin
   sglang#46310 8×H20-3e c8（TPOT 换算）     bs8  接受长度 4.60  实测  1438.8  估算    3732  比值 0.39  SGLang，非 vLLM
```

### rank

```
   配置                               常驻     KV  KV token      上下文     并发    主机内存    c1 估算    c16 估算  档位 / 路径
   1×PRO 6000 + 768G 内存           83.2    3.6     2.18M     128K   16.4    768G       11        22  入门：vLLM UVA 卸载；更快的是 KTransformers
   8×4090 24G                     40.6      ✗     0.00M      32K    0.0    384G        —         —  不可行：常驻 40 GiB/卡 > 24
   8×4090 48G（改装）                 40.6    5.3     3.17M     256K   12.0    384G      428      1561  可用：只有 SM89 分叉
   8×5090                         28.1    0.9     0.55M     128K    4.1    512G      201       461  可用：每卡卸 9 GiB 专家
   4×PRO 6000                     74.3   13.5     8.08M    1024K    7.7    384G      424      1421  消费级首选：主线 vLLM + FlashInfer #4380
   8×PRO 6000                     60.7   25.1    15.03M    1024K   14.3    128G      640      2660  消费级高配：Engram 可留显存
   8×H20 96G Engram 主机            40.6   36.9    22.13M    1024K   21.1    384G     1118      5540  数据中心可用：套 H100 模板
   8×H20 96G Engram 显存            64.2   14.3     8.57M     256K   32.5    128G     1118      5540  数据中心可用：不依赖主机内存
   8×H20-3e TP8 Engram 显存         64.2   53.0    31.79M    1024K   30.3    128G     1198      6409  数据中心首选：一台机一个副本
   8×H20-3e TP4×2 Engram 主机       81.2   37.5    22.49M    1024K   21.4    384G      884      7184  数据中心高吞吐：套 H200 模板
```
