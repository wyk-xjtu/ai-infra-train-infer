#!/bin/bash
# =============================================================================
# 对比实验全量运行脚本
# 实验矩阵: 7并行策略 × 2 Attention后端 × 2训练模式 = 28组实验
# 指标采集: 显存(peak/breakdown) / MFU / 吞吐 / 通信 / 训练loss
#
# 用法:
#   bash scripts/run_duibi_all.sh          # 运行全部28组
#   bash scripts/run_duibi_all.sh single   # 只运行单卡实验
#   bash scripts/run_duibi_all.sh dp2      # 只运行2卡DP实验
#   bash scripts/run_duibi_all.sh tp4      # 只运行4卡TP实验
# =============================================================================

# 不使用 set -e，改为手动检查关键操作的退出码
# set -e 会在 for 循环/函数调用中导致意外退出
set -o pipefail

cd /root/aikesi/task_p/train-infer || { echo "[FATAL] Cannot cd to project dir"; exit 1; }
source /root/aikesi/miniconda3/bin/activate infer || { echo "[FATAL] Cannot activate infer env"; exit 1; }
export HF_ENDPOINT=https://hf-mirror.com

# 清理可能残留的 CUBLAS_WORKSPACE_CONFIG（避免 deterministic=false 时 SDPA 崩溃）
unset CUBLAS_WORKSPACE_CONFIG 2>/dev/null || true

CONFIG_DIR="configs/experiments/duibi"
LOG_DIR="outputs/duibi_7.7/logs"
mkdir -p "$LOG_DIR"

FILTER="${1:-all}"  # 可选过滤: single, dp1, dp2, dp4, dp8, tp2, tp4, tp8, all

# 检测可用GPU数量
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
echo "[INFO] Detected $NUM_GPUS GPU(s)"

TOTAL_RUN=0
TOTAL_FAIL=0
TOTAL_SKIP=0

run_experiment() {
    local config="$1"
    local nproc="$2"
    local name
    name=$(basename "$config" .yaml)

    # 过滤逻辑
    if [[ "$FILTER" != "all" ]]; then
        if [[ "$name" != *"_${FILTER}" && "$name" != *"_${FILTER}" ]]; then
            return 0
        fi
    fi

    # 检查GPU数量是否满足
    if [[ "$nproc" -gt "$NUM_GPUS" ]]; then
        echo "  [SKIP] $name: requires $nproc GPUs, only $NUM_GPUS available"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return 0
    fi

    echo ""
    echo "==========================================================="
    echo " Running: $name"
    echo " Config:  $config"
    echo " GPUs:    $nproc"
    echo " Time:    $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==========================================================="

    local cmd
    if [[ "$nproc" -eq 1 ]]; then
        cmd="python scripts/run_colocate.py --config $config --inference-backend custom"
    else
        cmd="torchrun --nproc_per_node=$nproc scripts/run_colocate.py --config $config --inference-backend custom"
    fi

    # 使用临时文件捕获退出码，避免 pipe 掩盖失败
    local exit_code
    $cmd 2>&1 | tee "${LOG_DIR}/${name}.log"
    exit_code=${PIPESTATUS[0]}

    if [[ "$exit_code" -eq 0 ]]; then
        echo "[OK] $name completed successfully (exit=$exit_code)"
        TOTAL_RUN=$((TOTAL_RUN + 1))
    else
        echo "[FAIL] $name failed with exit code $exit_code"
        echo "$name (exit=$exit_code)" >> "${LOG_DIR}/failed_experiments.txt"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
    echo ""
    return 0  # 始终返回0，不让单个实验失败中断整个流程
}

echo "============================================================="
echo " 对比实验批量运行"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 过滤条件: $FILTER"
echo " 可用GPU:  $NUM_GPUS"
echo "============================================================="

# ==================== 单卡实验 (tp=1, dp=1) ====================
echo ""
echo ">>> [Phase 1/4] 单卡实验"
phase1_count=0
# 匹配两种命名: *_single.yaml 和 *_dp1.yaml
for config in "${CONFIG_DIR}"/*_single.yaml "${CONFIG_DIR}"/*_dp1.yaml; do
    [ -f "$config" ] || continue
    run_experiment "$config" 1
    phase1_count=$((phase1_count + 1))
done
echo "[Phase 1] 找到 $phase1_count 个配置文件"

# ==================== 2卡实验 (dp2/tp2) ====================
echo ""
echo ">>> [Phase 2/4] 2卡实验 (dp2 + tp2)"
phase2_count=0
for config in "${CONFIG_DIR}"/*_dp2.yaml "${CONFIG_DIR}"/*_tp2.yaml; do
    [ -f "$config" ] || continue
    run_experiment "$config" 2
    phase2_count=$((phase2_count + 1))
done
echo "[Phase 2] 找到 $phase2_count 个配置文件"

# ==================== 4卡实验 (dp4/tp4) ====================
echo ""
echo ">>> [Phase 3/4] 4卡实验 (dp4 + tp4)"
phase3_count=0
for config in "${CONFIG_DIR}"/*_dp4.yaml "${CONFIG_DIR}"/*_tp4.yaml; do
    [ -f "$config" ] || continue
    run_experiment "$config" 4
    phase3_count=$((phase3_count + 1))
done
echo "[Phase 3] 找到 $phase3_count 个配置文件"

# ==================== 8卡实验 (dp8/tp8) ====================
echo ""
echo ">>> [Phase 4/4] 8卡实验 (dp8 + tp8)"
phase4_count=0
for config in "${CONFIG_DIR}"/*_dp8.yaml "${CONFIG_DIR}"/*_tp8.yaml; do
    [ -f "$config" ] || continue
    run_experiment "$config" 8
    phase4_count=$((phase4_count + 1))
done
echo "[Phase 4] 找到 $phase4_count 个配置文件"

# ==================== 汇总 ====================
echo ""
echo "============================================================="
echo " 全部实验完成"
echo " 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 结果目录: outputs/duibi_7.7/"
echo " 日志目录: ${LOG_DIR}/"
echo "-------------------------------------------------------------"
echo " 统计: 成功=$TOTAL_RUN  失败=$TOTAL_FAIL  跳过=$TOTAL_SKIP"
echo " 配置: Phase1=${phase1_count} Phase2=${phase2_count} Phase3=${phase3_count} Phase4=${phase4_count}"
echo "============================================================="

if [ -f "${LOG_DIR}/failed_experiments.txt" ]; then
    echo ""
    echo "[WARNING] 以下实验运行失败:"
    cat "${LOG_DIR}/failed_experiments.txt"
fi
