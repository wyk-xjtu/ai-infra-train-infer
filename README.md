# AI Infra Train Infer

面向大模型训练、推理与强化学习场景的基础设施项目。项目覆盖模型训练、分布式并行、自研推理引擎、训推编排、权重同步、性能分析和实验产物管理，可用于验证单机多卡环境下的系统设计与优化方案。

项目包含三条核心链路：

```text
训练链路（SFT）
Dataset -> Prompt 构造与 Tokenize -> Forward / SFT Loss -> Backward
        -> Gradient Sync / Optimizer Step -> Checkpoint / Evaluation

RL 链路（GRPO）
Dataset -> Rollout（每个 Prompt 生成 K 条 Response）-> Rule Reward
        -> Advantage 计算 -> C3PO++ 切分与打包（可选）-> Policy Log-prob
        -> IcePop 过滤（可选）-> GRPO Loss / Backward / Optimizer Step
        -> Weight Sync -> 下一轮 Rollout

推理链路
Request -> Tokenize -> Continuous Batching -> Prefix / Radix Cache 匹配
        -> Prefill -> Paged KV Cache -> Decode -> Sampling -> Output
```

## 功能概览

### 分布式训练

- 支持 Full SFT 和 LoRA SFT。
- Data Parallel：按 DP rank 切分数据并同步梯度。
- Tensor Parallel：实现 Column Parallel Linear、Row Parallel Linear、并行 Attention、并行 MLP 和 Transformer Layer。
- 支持 GQA/MQA、RoPE、HF checkpoint 到 TP 模型的权重映射与分片加载。
- Pipeline Parallel：支持按层切分 stage、P2P activation/gradient 通信和 1F1B 调度。
- 1F1B 调度包含 warmup、steady 和 cooldown 阶段，并支持 micro-batch。
- 支持 TP、DP、PP 三维进程组及组合配置。
- ZeRO Stage 1：optimizer state sharding。
- ZeRO Stage 2：optimizer state 与 gradient sharding、reduce-scatter gradient。
- ZeRO Stage 3：parameter、gradient 与 optimizer state sharding，并提供训练和评估阶段的参数聚合。
- 支持 LoRA 与 Tensor Parallel 组合，覆盖 column-parallel 和 row-parallel 线性层。
- 支持 standard attention、PyTorch SDPA 和 FlashAttention 后端切换。
- 支持 FP32、FP16、BF16、AMP、gradient accumulation、gradient clipping 和 gradient checkpointing。
- 支持 cosine、linear、constant 学习率调度及 warmup，以及 checkpoint 保存、恢复和 SFT loss 评估。
- 支持独立 CUDA Stream 管理及通信、计算、权重传输之间的同步与重叠。

### 强化学习

- 实现基于 GRPO 的强化学习闭环：rollout、reward、advantage、policy update 和 weight sync。
- 每个 prompt 支持生成 K 条 response，并按组内奖励计算标准化 relative advantage。
- 使用当前策略、行为策略和参考模型的 token log-prob，计算 clipped policy loss 与 KL penalty。
- 提供 GSM8K 规则奖励，支持答案抽取、格式检查、正确性判定和批量奖励计算。
- 支持 IcePop 训推偏差过滤，根据 rollout policy 与 training policy 的序列级 log-prob 差异调整样本 advantage，并记录过滤比例等诊断指标。
- 支持 C3PO++ response chunking 与 FFD packing；同一 response 的 chunk 共享 advantage，多个 sub-batch 完成后统一更新参数。
- 记录 reward、iteration accuracy、cumulative accuracy、policy loss、KL divergence 和 response 长度等指标。

### 推理引擎

- 支持 `eager`、`kv_cache` 和 `mock` 三种运行模式。
- 实现 Paged KV Cache、物理 block pool、sequence block table 和 KV Cache 生命周期管理。
- 支持 FP16 与 FP8 KV Cache 存储。
- 实现 Continuous Batching，支持 token budget、请求状态管理、chunked prefill 和请求抢占。
- 支持普通 Prefix Cache、K-sample prompt 前缀复用和基于 Radix Tree 的前缀缓存。
- 实现 Paged Attention 与 Partitioned Paged Attention；长上下文 decode 可按 partition 计算后合并结果。
- 支持异步 Prefill/Decode 调度，使用独立 CUDA Stream 组织执行。
- 支持 CUDA Graph capture/replay，并在权重更新后刷新相关缓存状态。
- 支持 draft model speculative decoding，包含候选 token 生成、目标模型校验和接受率统计。
- 支持 temperature、top-p、batch generation、JSONL 批量输入输出和交互式推理。
- 提供独立推理入口和 eager、KV Cache、mock 多模式 benchmark。

### 训推编排

- Colocate 模式：训练与推理共享设备资源，在同一闭环中完成 rollout、reward、training 和 weight update。
- Disaggregated 模式：训练 worker 与推理 worker 分离部署，支持流水执行和权重版本管理。
- 推理后端支持项目内置推理引擎与 vLLM 服务接入。
- 支持同步与异步训推流水线，可配置队列长度、最大权重陈旧度和异步任务管理。
- 支持训练、推理、奖励计算、休眠、权重同步和唤醒等阶段的状态管理与耗时统计。
- 支持 config snapshot、metrics JSONL、run summary、固定 prompt 对比、loss curve 和 rank-aware artifact 输出。

### 权重同步

- 支持 CUDA IPC 权重传输、分块推送、checksum 校验、版本更新和失败回滚。
- 支持 NCCL broadcast/receive 权重同步。
- 支持通过 Ray Object Store 发布和获取权重版本。
- 提供统一的 Weight Transfer Manager，用于选择 IPC、NCCL 或 Ray 传输路径。
- 支持 SwiftSync LoRA 增量同步：计算相邻版本 LoRA delta，仅传输发生变化的参数。
- SwiftSync 提供双缓冲 LoRA 状态，在 shadow buffer 应用增量后切换 active buffer。
- 支持异步 delta/full sync、定期全量校准、版本号与传输统计。

## 项目结构

```text
ai-infra-train-infer/
├── configs/
│   ├── training/              # SFT、RL、Colocate、Disaggregated、PP 配置
│   ├── inference/             # 独立推理配置
│   ├── benchmark/             # 功能对比与消融实验配置
│   └── experiments/           # DP、TP、ZeRO、FlashAttention 等实验矩阵
├── docs/                      # 已整理的实验结果
├── scripts/                   # 训练、推理、benchmark、数据准备与 profiling 入口
├── src/
│   ├── data/                  # 数据加载、prompt 构造与 DP shard
│   ├── distributed/           # TP、PP、ZeRO、LoRA、通信与 CUDA Stream
│   ├── engines/               # 训练引擎、推理 worker、IcePop、C3PO++
│   ├── inference/             # KV Cache、调度器、Attention、推测解码
│   ├── orchestrator/          # Colocate 与 Disaggregated 编排
│   ├── profiling/             # MFU、通信和 timeline profiling
│   ├── reward/                # GSM8K 规则奖励
│   ├── transfer/              # IPC、NCCL、Ray、SwiftSync 权重同步
│   └── utils/                 # 日志、指标、评估和实验产物管理
└── tests/                     # 分布式、推理、编排与权重同步测试
```

## 训练结果摘要

完整实验表格见 [docs/training_results_summary.md](docs/training_results_summary.md)。小模型实验用于验证正确性、显存和吞吐趋势，MFU 数据来自 32B 实验。

### 正确性

| 场景 | 结果 |
| --- | --- |
| Full SFT single GPU | loss 从约 `2.09` 降到 `0.57`，standard / FlashAttention 曲线一致 |
| Full SFT DP + ZeRO-1 | DP2/DP4/DP8 均稳定收敛 |
| LoRA single / DP / TP | single、DP2/4/8、TP2/4/8 loss 均稳定下降 |
| 32B LoRA + TP8 | loss 稳定下降，单 rank current allocated 约 `12.78GB` |

### 显存与吞吐

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

## 快速开始

### 安装依赖

Windows 环境：

```bash
pip install -r requirements-windows.txt
```

Linux/CUDA 环境：

```bash
pip install -r requirements.txt
```

### 准备模型与数据

模型默认放在项目根目录的 `models/` 下：

```bash
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B
```

离线准备 Hugging Face 数据集：

```bash
python scripts/prepare_data.py \
  --dataset yahma/alpaca-cleaned \
  --split train \
  --output ./data/alpaca-cleaned
```

### 运行训练

单卡 SFT：

```bash
python scripts/run_colocate.py \
  --config configs/training/sft.yaml \
  --dp-size 1
```

8 卡 GRPO（TP=2，DP=4）：

```bash
torchrun --nproc_per_node=8 scripts/run_colocate.py \
  --config configs/training/rl.yaml
```

2 卡 Pipeline Parallel smoke test：

```bash
torchrun --nproc_per_node=2 scripts/run_colocate.py \
  --config configs/training/pp_smoke_pp2.yaml
```

PP + TP、PP + DP、PP + ZeRO-1 组合配置：

```text
configs/training/pp_combo_pptp.yaml
configs/training/pp_combo_ppdp.yaml
configs/training/pp_combo_ppzero.yaml
```

8 卡对比实验示例：

```bash
torchrun --nproc_per_node=8 scripts/run_colocate.py \
  --config configs/experiments/duibi/lora_flash_tp8.yaml
```

### 运行推理

单条 prompt：

```bash
python scripts/run_inference.py \
  --model ./models/Qwen3-0.6B \
  --mode eager \
  --prompt "Solve: 18 + 24 =" \
  --max-tokens 128
```

使用独立推理配置：

```bash
python scripts/run_inference.py \
  --model ./models/Qwen3-0.6B \
  --config configs/inference/standalone.yaml
```

推理 benchmark：

```bash
python scripts/benchmark_inference.py \
  --model_path ./models/Qwen3-0.6B \
  --mode all \
  --num_prompts 32 \
  --batch_size 8
```

### 性能分析

```bash
python scripts/profile_memory.py \
  --config configs/experiments/duibi/lora_flash_single.yaml
```


## 测试

```bash
pytest tests
```

测试覆盖：

- Tensor Parallel 层与参考实现的数值对齐。
- Pipeline process group、P2P 通信和 1F1B schedule。
- ZeRO optimizer shard 行为与 3D parallel gradient norm。
- Colocate 训练闭环和异步训推状态管理。
- Paged KV Cache、独立推理入口和推理调度。
- IPC weight update、SwiftSync delta 与双缓冲切换。
- IcePop 过滤逻辑、C3PO++ 切分打包及二者的组合流程。

## 产物管理

运行产生的 `outputs/`、`models/`、checkpoint 和缓存目录通过 `.gitignore` 管理。仓库保留源代码、配置、测试以及已经整理并确认的实验结果。

## 更新记录

### [2026.7.20] 新增

- 新增 IcePop 训推 log-prob 偏差过滤，支持阈值控制、最大过滤比例和诊断指标。
- 新增 C3PO++ response chunking 与 FFD packing，支持跨 chunk advantage 继承和延迟 optimizer update。
- 新增 SwiftSync LoRA delta 权重同步，支持双缓冲、异步传输、版本统计和周期性 full sync。
- 新增异步训推流水线，支持 rollout、training、weight sync 任务编排及 staleness/queue 配置。
- 新增 FP8 KV Cache、Partitioned Paged Attention、Radix Attention Cache 和 speculative decoding。
- 新增独立推理入口、批量 JSONL/交互式推理，以及针对新增模块的 benchmark 与消融配置。
- 新增 iteration/cumulative accuracy、rank-aware artifact 和新增功能的单元测试、组合测试。
