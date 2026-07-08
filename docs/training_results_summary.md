# Training Results and Infra Metrics Summary

本摘要记录了本项目训练框架验证有效的实验结果。小模型实验主要用于证明功能正确性、显存效率和并行扩展趋势；MFU 只报告 32B 模型，小模型 MFU 不作为项目指标。显存峰值包含 PyTorch caching allocator 的影响，因此建议同时看 `current_allocated` 和 `peak_allocated`。

## 1. 训练正确性证据

### 单卡 Full SFT

单卡 Full SFT 能稳定收敛，说明基础训练 loop、loss、optimizer、gradient accumulation、checkpoint 前后的状态流转是可用的。

| case | attention | init loss | final loss | min loss | avg step time | avg tokens/s | current GB | peak GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_single | standard | 2.0986 | 0.5730 | 0.4206 | 0.3235s | 6219 | 24.36 | 41.65 |
| full_single | flash | 2.0895 | 0.5728 | 0.4252 | 0.3129s | 6393 | 24.36 | 40.58 |

结论：standard attention 和 FlashAttention 的 loss 曲线基本一致，FlashAttention 在单卡场景下略微提升吞吐并降低峰值显存。

### 多卡 DP + ZeRO-1 Full SFT

DP/ZeRO-1 下 Full SFT 也能稳定下降，说明数据并行、梯度同步、ZeRO-1 optimizer state sharding 和多进程训练路径是正确的。

| case | attention | init loss | final loss | min loss | avg step time | avg tokens/s | current GB | peak GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_dp2 | standard | 2.0986 | 0.5302 | 0.2736 | 0.4482s | 4344 | 64.53 | 81.84 |
| full_dp2 | flash | 2.0895 | 0.5321 | 0.3020 | 0.4393s | 4239 | 64.53 | 76.59 |
| full_dp4 | standard | 1.9611 | 0.3672 | 0.2404 | 0.7025s | 5099 | 48.42 | 83.06 |
| full_dp4 | flash | 1.9624 | 0.3603 | 0.2383 | 0.6544s | 5673 | 48.43 | 71.16 |
| full_dp8 | standard | 1.9611 | 0.1674 | 0.0648 | 0.7354s | 4926 | 40.38 | 75.05 |
| full_dp8 | flash | 1.9624 | 0.1735 | 0.0667 | 0.6776s | 5357 | 40.38 | 63.13 |

结论：DP2/DP4/DP8 都能收敛；FlashAttention 在 DP4/DP8 下收益更明显，DP4 峰值显存从 83.06GB 降到 71.16GB，DP8 从 75.05GB 降到 63.13GB，同时吞吐分别提升到 5673 和 5357 tokens/s。

### LoRA 单卡、多卡 DP 和 TP

LoRA 场景覆盖了冻结 base model、只训练 adapter 参数、参数过滤、optimizer 参数组构建、DP/TP 下 adapter 参数同步等路径。结果整体稳定下降。

| case | attention | parallel | final loss | min loss | avg step time | avg tokens/s | current GB | peak GB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lora_single | standard | 1 GPU | 1.0252 | 0.6307 | 0.7968s | 10355 | 8.98 | 85.33 |
| lora_single | flash | 1 GPU | 1.0237 | 0.6280 | 0.7550s | 10880 | 8.98 | 67.24 |
| lora_dp2 | flash | DP2 + ZeRO-1 | 1.0596 | 0.6240 | 0.8551s | 8615 | 9.81 | 68.06 |
| lora_dp4 | flash | DP4 + ZeRO-1 | 0.8385 | 0.7745 | 0.9262s | 8254 | 9.54 | 61.07 |
| lora_dp8 | flash | DP8 + ZeRO-1 | 0.7326 | 0.7326 | 1.0042s | 7038 | 9.42 | 60.94 |
| lora_tp2 | flash | TP2 | 1.0259 | 0.6311 | 0.5444s | 14158 | 5.55 | 44.31 |
| lora_tp4 | flash | TP4 | 1.0294 | 0.6351 | 0.4208s | 17434 | 3.33 | 32.27 |
| lora_tp8 | flash | TP8 | 1.0319 | 0.6410 | 0.3759s | 19928 | 2.24 | 26.40 |

结论：LoRA 在 single / DP / TP 下 loss 都保持同量级下降；TP 扩展带来明显的 per-rank 显存下降，`current_allocated` 从单卡 8.98GB 降到 TP8 的 2.24GB，同时吞吐从 10880 tokens/s 提升到 19928 tokens/s。

### 32B LoRA + TP8

32B LoRA + TP8 是最能体现大模型可训练性的实验：在 8 卡 TP 下，LoRA 训练显存保持在较低水平，同时 loss 正常下降。

| case | attention | final loss | min loss | avg step time | avg tokens/s | avg MFU | max MFU | static GB | current GB | peak GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32b_lora_tp8 | standard | 0.9857 | 0.7642 | 0.7598s | 1945 | 21.59% | 41.23% | 11.79 | 12.78 | 23.78 |
| 32b_lora_tp8 | flash | 0.9697 | 0.7476 | 0.6816s | 1988 | 22.07% | 41.31% | 11.79 | 12.78 | 23.78 |

结论：32B LoRA + TP8 能在单 rank 约 12.78GB current allocated、23.78GB peak allocated 下运行，证明框架具备大模型参数切分、adapter 训练和多卡通信组合能力。32B LoRA 的平均 MFU 约 21.6%-22.1%，最大 MFU 约 41.3%，可作为当前项目里最有参考价值的 MFU 指标。

### 32B Full TP8 的性能参考

32B Full TP8 历史实验更适合作为大模型 TP 路径的吞吐、显存和 MFU 参考。

| case | attention | avg step time | avg tokens/s | avg MFU | max MFU | static GB | current GB | peak GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32b_full_tp8 | standard | 0.6695s | 2041 | 33.82% | 62.07% | 27.29 | 54.71 | 62.62 |
| 32b_full_tp8 | flash | 0.7375s | 2015 | 33.40% | 64.60% | 27.29 | 54.71 | 61.67 |

结论：32B Full TP8 的 current allocated 约 54.71GB，peak allocated 约 61.67-62.62GB；平均 MFU 约 33.4%-33.8%，最大 MFU 可到 62%-65%。这是本摘要中用于展示计算利用率的主要 MFU 口径。

## 2. Infra 指标总结

### FlashAttention 收益

FlashAttention 的收益主要体现在峰值显存和部分场景吞吐上：

| scenario | standard peak GB | flash peak GB | peak reduction | standard tokens/s | flash tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_single | 41.65 | 40.58 | 2.6% | 6219 | 6393 |
| full_dp4 | 83.06 | 71.16 | 14.3% | 5099 | 5673 |
| full_dp8 | 75.05 | 63.13 | 15.9% | 4926 | 5357 |
| lora_single | 85.33 | 67.24 | 21.2% | 10355 | 10880 |
| lora_dp8 | 73.79 | 60.94 | 17.4% | 6433 | 7038 |
| lora_tp8 | 28.61 | 26.40 | 7.7% | 20120 | 19928 |

结论：FlashAttention 在长序列或多卡 DP 场景下对峰值显存最有价值；TP8 LoRA 下显存也有下降，但由于通信和小算子开销占比更高，吞吐收益不一定单调。

### Tensor Parallel 显存和吞吐扩展

LoRA + TP 的数据最适合作为 TP 实现正确性与 infra 指标证据。

| case | static GB | current GB | peak GB | avg tokens/s |
| --- | ---: | ---: | ---: | ---: |
| lora_single_flash | 8.84 | 8.98 | 67.24 | 10880 |
| lora_tp2_flash | 4.93 | 5.55 | 44.31 | 14158 |
| lora_tp4_flash | 2.91 | 3.33 | 32.27 | 17434 |
| lora_tp8_flash | 1.92 | 2.24 | 26.40 | 19928 |

结论：TP 扩展符合预期，per-rank 参数/激活相关显存持续下降，吞吐随 TP degree 增大提升，但 TP4 到 TP8 的边际收益下降，符合通信开销上升后的常见特征。

### 多卡加速比

加速比需要区分 DP 和 TP 的口径：DP 是复制模型、切分数据，global throughput 可按 `rank0 tokens/s × dp_size` 估算；TP 是切分模型，同一个 batch 被多个 TP rank 协同计算，不能把 rank0 tokens/s 再乘以 TP size，因此 TP 加速比直接用端到端 tokens/s 或 step time 相对单卡计算。

#### DP global throughput speedup

| case | single baseline | dp degree | rank0 tokens/s | estimated global tokens/s | speedup vs single |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_flash_dp2 | 6393 | 2 | 4239 | 8478 | 1.33x |
| full_flash_dp4 | 6393 | 4 | 5673 | 22692 | 3.55x |
| full_flash_dp8 | 6393 | 8 | 5357 | 42856 | 6.70x |
| lora_flash_dp2 | 10880 | 2 | 8615 | 17230 | 1.58x |
| lora_flash_dp4 | 10880 | 4 | 8254 | 33016 | 3.03x |
| lora_flash_dp8 | 10880 | 8 | 7038 | 56304 | 5.17x |

结论：DP 的 global throughput 随卡数提升明显，但 scaling efficiency 会下降，主要来自梯度同步、optimizer step、数据加载和小模型场景下非 GEMM 开销占比上升。Full Flash DP8 达到约 6.70x global throughput speedup，LoRA Flash DP8 达到约 5.17x。

#### TP end-to-end speedup

| case | single baseline tokens/s | tp degree | tp tokens/s | throughput speedup | single step time | tp step time | step-time speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lora_flash_tp2 | 10880 | 2 | 14158 | 1.30x | 0.7550s | 0.5444s | 1.39x |
| lora_flash_tp4 | 10880 | 4 | 17434 | 1.60x | 0.7550s | 0.4208s | 1.79x |
| lora_flash_tp8 | 10880 | 8 | 19928 | 1.83x | 0.7550s | 0.3759s | 2.01x |

结论：TP 的主要收益同时体现在显存下降和端到端 step time 缩短。LoRA Flash TP8 相比单卡吞吐提升约 1.83x，step time 提升约 2.01x；由于 TP 会引入 all-reduce/all-gather 等通信，速度不会按 TP degree 线性增长，这是符合预期的。

### Data Parallel + ZeRO-1

DP + ZeRO-1 证明了多进程数据并行、梯度同步和优化器状态切分路径。

| case | current GB | peak GB | avg tokens/s | final loss |
| --- | ---: | ---: | ---: | ---: |
| full_dp2_flash | 64.53 | 76.59 | 4239 | 0.5321 |
| full_dp4_flash | 48.43 | 71.16 | 5673 | 0.3603 |
| full_dp8_flash | 40.38 | 63.13 | 5357 | 0.1735 |
| lora_dp2_flash | 9.81 | 68.06 | 8615 | 1.0596 |
| lora_dp4_flash | 9.54 | 61.07 | 8254 | 0.8385 |
| lora_dp8_flash | 9.42 | 60.94 | 7038 | 0.7326 |

结论：Full SFT 在 DP4/DP8 下 current allocated 明显低于 DP2，体现 ZeRO-1 对 optimizer state 的分摊效果；LoRA 场景由于可训练参数较少，current allocated 本身较低，ZeRO-1 的显存收益相对不如 Full SFT 明显，但仍验证了 DP/ZeRO-1 组合路径。

## 3. 总结

本项目已完成训练侧的核心 infra 能力验证，包括单卡训练、DP、多卡 TP、FlashAttention、LoRA 和 ZeRO-1。实验中 Full SFT 单卡 loss 从约 2.09 降到 0.57；DP + ZeRO-1 在 2/4/8 卡下均能稳定收敛，DP8 + FlashAttention 最终 loss 约 0.17。LoRA 覆盖 single / DP / TP 多种并行组合，TP8 + FlashAttention 下 current allocated 显存从单卡 8.98GB 降到 2.24GB，吞吐从 10880 tokens/s 提升到 19928 tokens/s，端到端吞吐加速约 1.83x，step-time speedup 约 2.01x。DP global throughput 也随卡数提升，Full Flash DP8 约 6.70x，LoRA Flash DP8 约 5.17x。32B LoRA + TP8 能在单 rank 约 12.78GB current allocated、23.78GB peak allocated 下稳定训练，证明框架具备大模型切分训练能力。

FlashAttention 在多卡 DP 和 LoRA 场景中显著降低峰值显存：Full DP8 peak memory 从 75.05GB 降到 63.13GB，LoRA single 从 85.33GB 降到 67.24GB，LoRA DP8 从 73.79GB 降到 60.94GB。整体看，训练结果同时覆盖了正确性、显存效率、吞吐、并行扩展性和大模型可运行性几个 AI infra 关键维度。

