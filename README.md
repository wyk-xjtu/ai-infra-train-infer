# AI Infra Train Infer

面向 AI Infra 岗位的训练、推理、RL 一体化原型框架。项目重点不是调用现成分布式库跑通脚本，而是手写并验证训练系统中的关键机制：Tensor Parallel、ZeRO optimizer sharding、FlashAttention 接入、LoRA/full SFT、GRPO 训练闭环、KV Cache、Prefix Cache、Continuous Batching、权重同步和训推编排。

当前训练侧已经完成主要功能验证；推理和 RL 侧保留可运行的轻量实现，待后续开发。

## 核心能力

| 方向 | 实现内容 |
| --- | --- |
| Training | Full SFT、LoRA SFT、GRPO loss、gradient accumulation、AMP/bf16、checkpoint、eval、metrics artifact |
| Parallelism | 自研 Megatron-style TP、DP、TP+DP group 管理、通信/计算 overlap 原型 |
| Optimizer Sharding | Mini-ZeRO Stage 1/2/3：optimizer state sharding、gradient sharding、parameter sharding 原型 |
| Attention | standard attention、PyTorch SDPA、FlashAttention 后端切换 |
| Inference | eager、mock、KV cache path、Paged KV Cache、Continuous Batching、Prefix Cache、CUDA Graph runner 原型 |
| RL | GSM8K rule reward、rollout -> reward -> GRPO train 的闭环接口 |
| Orchestration | Colocate 训推共置、Disaggregated 训推分离、weight sync、artifact/log/summary 输出 |
| Profiling | MFU、tokens/s、step time、显存分解、通信耗时、schedule timeline |

## 项目结构

```text
ai-infra-train-infer/
├── configs/                  # SFT/RL/inference/multi-GPU/benchmark 配置
├── docs/                     # 结果摘要
├── scripts/                  # 训练、推理 benchmark、显存 profiling、实验矩阵脚本
├── src/
│   ├── data/                 # GSM8K 数据加载与 DP shard
│   ├── distributed/          # TP、ZeRO、LoRA、通信原语、stream overlap
│   ├── engines/              # TrainEngine / InferWorker
│   ├── inference/            # KV Cache、scheduler、prefix cache、model runner
│   ├── orchestrator/         # Colocate / Disaggregated 编排
│   ├── profiling/            # MFU、通信、timeline profiling
│   ├── reward/               # GSM8K rule reward
│   ├── transfer/             # IPC / NCCL 权重同步原型
│   └── utils/                # logger、metrics、artifact、memory profiler
└── tests/                    # TP、ZeRO、训推编排、权重同步测试
```

## 训练结果摘要

详细表格见 [docs/training_results_summary.md](docs/training_results_summary.md)。下面只保留核心指标。

### 正确性

| 场景 | 结果 |
| --- | --- |
| Full SFT single GPU | loss 从约 `2.09` 降到 `0.57`，standard / FlashAttention 曲线一致 |
| Full SFT DP + ZeRO-1 | DP2/DP4/DP8 均稳定收敛|
| LoRA single / DP / TP | single、DP2/4/8、TP2/4/8 loss 均稳定下降 |
| 32B LoRA + TP8 |  loss 稳定下降，单 rank current allocated 约 `12.78GB` |

### 显存和吞吐

| 指标 | 结果 |
| --- | --- |
| FlashAttention memory | Full DP8 peak memory `75.05GB -> 63.13GB`，LoRA single `85.33GB -> 67.24GB` |
| LoRA TP memory scaling | current allocated `8.98GB(single) -> 5.55GB(TP2) -> 3.33GB(TP4) -> 2.24GB(TP8)` |
| LoRA TP throughput | FlashAttention 下 `10880 tokens/s(single) -> 19928 tokens/s(TP8)` |
| DP global throughput speedup | Full Flash DP8 约 `6.70x`，LoRA Flash DP8 约 `5.17x` |
| TP end-to-end speedup | LoRA Flash TP8 吞吐约 `1.83x`，step-time speedup 约 `2.01x` |

### 32B MFU

MFU 只报告 32B 实验；小模型实验只用于展示正确性、显存和吞吐趋势。

| 场景 | avg MFU | max MFU | current GB | peak GB |
| --- | ---: | ---: | ---: | ---: |
| 32B LoRA TP8 standard | `21.59%` | `41.23%` | `12.78` | `23.78` |
| 32B LoRA TP8 flash | `22.07%` | `41.31%` | `12.78` | `23.78` |
| 32B Full TP8 standard | `33.82%` | `62.07%` | `54.71` | `62.62` |
| 32B Full TP8 flash | `33.40%` | `64.60%` | `54.71` | `61.67` |

## 运行方式

训练实验采用 config-first 的方式管理：模型路径、并行策略、数据集、训练步数、LoRA、ZeRO、checkpoint 和 profiling 都写在 YAML 里。`scripts/run_colocate.py` 支持命令行参数覆盖 YAML 字段，但推荐只在临时调试时使用 override，正式实验统一新增或复制 `configs/*.yaml`。

建议使用 Python 3.10 或 3.11。Windows 本地开发可安装轻量依赖：

```bash
pip install -r requirements-windows.txt
```

Linux/CUDA 环境安装完整依赖：

```bash
pip install -r requirements.txt
```

模型建议放在项目根目录的 `models/` 下，例如：

```bash
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B
```

最小单卡 SFT smoke test：

```bash
python scripts/run_colocate.py --config configs/minimal_sft.yaml
```

常用配置：

| config | 用途 |
| --- | --- |
| `configs/minimal_sft.yaml` | 单卡最小 SFT 验证，默认 Qwen3-0.6B、8 条 GSM8K、1 次 iteration |


多卡实验也通过 config 控制，例如：

```bash
torchrun --nproc_per_node=8 scripts/run_colocate.py \
  --config configs/experiments/......yaml（需自己修改）
```



## 设计取舍

- TP/PP/ZeRO/LoRA/调度器均为简化版自研实现，优先体现系统机制和可读性。
- 训练结果主要证明正确性、显存效率、吞吐扩展和大模型可运行性；生产级容错、弹性训练和大规模集群调度暂未实现。
- 推理侧实现覆盖 KV Cache、Prefix Cache、Continuous Batching、CUDA Graph 等核心概念，`custom` 后端用于本地验证，vLLM/NCCL/Ray 路径面向 Linux CUDA 环境。
- `outputs/`、`models/`、checkpoint 和缓存目录不纳入仓库；公开展示使用 `docs/training_results_summary.md` 中的结果摘要。

## 测试

```bash
pytest tests
```

重点测试覆盖：

- Tensor Parallel layer 与参考实现对齐
- ZeRO optimizer shard 行为
- Colocate 训推状态机
- IPC weight update protocol mock
