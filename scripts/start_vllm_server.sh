#!/bin/bash
# 启动vLLM推理服务（Colocate模式用）
# 用法:
#   bash scripts/start_vllm_server.sh                    # 默认0.6B单卡
#   bash scripts/start_vllm_server.sh Qwen/Qwen3-4B 2   # 4B模型TP=2
#   bash scripts/start_vllm_server.sh Qwen/Qwen3-0.6B 1 8000 0.85
# 参数:
#   $1 - 模型名称 (默认: Qwen/Qwen3-0.6B)
#   $2 - TP大小 (默认: 1)
#   $3 - 端口 (默认: 8000)
#   $4 - GPU显存利用率 (默认: 0.85)

MODEL=${1:-"Qwen/Qwen3-0.6B"}
TP_SIZE=${2:-1}
PORT=${3:-8000}
GPU_UTIL=${4:-0.85}

echo "============================================="
echo "  Starting vLLM Inference Server"
echo "============================================="
echo "  Model: $MODEL"
echo "  TP Size: $TP_SIZE"
echo "  Port: $PORT"
echo "  GPU Utilization: $GPU_UTIL"
echo "  Sleep Mode: enabled"
echo "============================================="

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT" \
    --enable-sleep-mode \
    --trust-remote-code \
    --dtype float16
