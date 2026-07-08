# AI Infra Train Infer

面向 AI Infra 岗位的训练、推理、RL 一体化原型框架。项目重点是手写并验证训练系统中的关键机制：Tensor Parallel、Pipeline Parallel、ZeRO、FlashAttention、LoRA/full SFT、GRPO 训练闭环、KV Cache、Prefix Cache、Continuous Batching、权重同步和训推编排。

当前训练侧已经覆盖单卡、DP、TP、PP、ZeRO-1/2/3、FlashAttention、LoRA/full SFT 和 32B LoRA/Full TP 实验；推理和 RL 侧保留轻量可运行实现，用于展示训推一体系统的模块边界和数据流，后续将深入开发。

## 核心能力

| 方向 | 实现内容 |
| --- | --- |
| Training | Full SFT、LoRA SFT、GRPO loss、gradient accumulation、AMP/bf16、checkpoint、eval、metrics artifact |
| Parallelism | Megatron-style TP、DP、PP、TP/DP/PP 3D process group、1F1B pipeline schedule |
| Pipeline Parallel | stage-aware layer split、P2P activation/gradient send-recv、micro-batch warmup/steady/cooldown |
| Optimizer Sharding | Mini-ZeRO Stage 1/2/3：optimizer state sharding、gradient sharding、parameter sharding 原型 |
| Attention | standard attention、PyTorch SDPA、FlashAttention 后端切换 |
| Inference | eager、mock、KV cache path、Paged KV Cache、Continuous Batching、Prefix Cache、CUDA Graph runner 原型 |
| RL | GSM8K rule reward、rollout -> reward -> GRPO train 的闭环接口 |
| Orchestration | Colocate 训推共置、Disaggregated 训推分离、weight sync、artifact/log/summary 输出 |
| Profiling | MFU、tokens/s、step time、显存分解、通信耗时、schedule timeline |

## 项目结构

```text
ai-infra-train-infer/
├── configs/                  # SFT/RL/inference/multi-GPU/benchmark/PP 配置
├── docs/                     # 设计说明和代码审查记录
├── scripts/                  # 训练、推理 benchmark、显存 profiling、实验矩阵脚本
├── src/
│   ├── data/                 # GSM8K / SFT 数据加载与 DP shard
│   ├── distributed/          # TP、PP、ZeRO、LoRA、通信原语、stream overlap
│   ├── engines/              # TrainEngine / InferWorker
│   ├── inference/            # KV Cache、scheduler、prefix cache、model runner
│   ├── orchestrator/         # Colocate / Disaggregated 编排
│   ├── profiling/            # MFU、通信、timeline profiling
│   ├── reward/               # GSM8K rule reward
│   ├── transfer/             # IPC / NCCL 权重同步原型
│   └── utils/                # logger、metrics、artifact、memory profiler
└── tests/                    # TP、PP、ZeRO、训推编排、权重同步测试
```



当前限制：PP 主要支持 SFT，GRPO 仍要求 `pp_size=1`；PP 与 ZeRO 组合当前限制为 `zero_stage = 1`。相关测试包括 `tests/test_pipeline_parallel.py` 和 `tests/test_pipeline_scheduler.py`。

## 训练结果摘要

完整结果表格见 [docs/training_results_summary.md](docs/training_results_summary.md)。下面是适合放在项目首页的核心指标。小模型实验用于证明正确性、显存和吞吐趋势；MFU 只报告 32B 实验。

### 正确性

| 场景 | 结果 |
| --- | --- |
| Full SFT single GPU | loss 从约 `2.09` 降到 `0.57`，standard / FlashAttention 曲线一致 |
| Full SFT DP + ZeRO-1 | DP2/DP4/DP8 均稳定收敛 |
| LoRA single / DP / TP | single、DP2/4/8、TP2/4/8 loss 均稳定下降 |
| 32B LoRA + TP8 | loss 稳定下降，单 rank current allocated 约 `12.78GB` |

### 显存和吞吐

| 指标 | 结果 |
| --- | --- |
| FlashAttention memory | Full DP8 peak memory `75.05GB -> 63.13GB`，LoRA single `85.33GB -> 67.24GB` |
| LoRA TP memory scaling | current allocated `8.98GB(single) -> 5.55GB(TP2) -> 3.33GB(TP4) -> 2.24GB(TP8)` |
| LoRA TP throughput | FlashAttention 下 `10880 tokens/s(single) -> 19928 tokens/s(TP8)` |
| DP global throughput speedup | Full Flash DP8 约 `6.70x`，LoRA Flash DP8 约 `5.17x` |
| TP end-to-end speedup | LoRA Flash TP8 吞吐约 `1.83x`，step-time speedup 约 `2.01x` |

### 32B MFU

| 场景 | avg MFU | max MFU | current GB | peak GB |
| --- | ---: | ---: | ---: | ---: |
| 32B LoRA TP8 standard | `21.59%` | `41.23%` | `12.78` | `23.78` |
| 32B LoRA TP8 flash | `22.07%` | `41.31%` | `12.78` | `23.78` |
| 32B Full TP8 standard | `33.82%` | `62.07%` | `54.71` | `62.62` |
| 32B Full TP8 flash | `33.40%` | `64.60%` | `54.71` | `61.67` |

## 运行方式

训练实验采用 config-first 的方式管理：模型路径、并行策略、数据集、训练步数、LoRA、ZeRO、checkpoint 和 profiling 都写在 YAML 里。`scripts/run_colocate.py` 支持命令行参数覆盖 YAML 字段，但推荐只在临时调试时使用 override。

安装依赖：

```bash
pip install -r requirements-windows.txt
```

Linux/CUDA + vLLM/Ray 环境使用完整依赖：

```bash
pip install -r requirements.txt
```

模型建议放在项目根目录的 `models/` 下：

```bash
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B
```

单卡 SFT：

```bash
python scripts/run_colocate.py --config configs/sft.yaml
```

PP smoke test：

```bash
torchrun --nproc_per_node=2 scripts/run_colocate.py \
  --config configs/training/pp_smoke_pp2.yaml
```

PP + TP / PP + DP / PP + ZeRO-1 组合配置：

```text
configs/training/pp_combo_pptp.yaml
configs/training/pp_combo_ppdp.yaml
configs/training/pp_combo_ppzero.yaml
```

多卡对比实验：

```bash
torchrun --nproc_per_node=8 scripts/run_colocate.py \
  --config configs/experiments/duibi/lora_flash_tp8.yaml
```

推理 benchmark：

```bash
python scripts/benchmark_inference.py --mode mock --num-prompts 32 --batch-size 8
```

## 设计取舍

- TP/PP/ZeRO/LoRA/调度器均为简化版自研实现，优先体现系统机制和可读性。
- PP 当前定位是 SFT 训练侧的 1F1B MVP，已覆盖 process group、stage split、P2P 和 scheduler；GRPO + PP 暂未接入。
- 训练结果主要证明正确性、显存效率、吞吐扩展和大模型可运行性；生产级容错、弹性训练和大规模集群调度暂未实现。
- 推理侧实现覆盖 KV Cache、Prefix Cache、Continuous Batching、CUDA Graph 等核心概念，目前仅支持简单推理，尚未进行深入修改。
- `outputs/`、`models/`、checkpoint 和缓存目录不纳入仓库。

## 测试

```bash
pytest tests
```

重点测试覆盖：

- Tensor Parallel layer 与参考实现对齐
- Pipeline process group、P2P 通信、1F1B schedule
- ZeRO optimizer shard 行为
- Colocate 训推状态机
- IPC weight update protocol mock
