#!/bin/bash
# Bubble measurement: sweep num_micro_batches for pp=2
# Measures throughput at different micro-batch counts

cd /root/aikesi/task_p/train-infer
source /root/aikesi/miniconda3/bin/activate infer
export HF_ENDPOINT=https://hf-mirror.com

echo "=== PP=2 Bubble Measurement: num_micro_batches sweep ==="
echo "Theory: bubble_ratio = (pp-1)/num_micro_batches"
echo ""

for NMB in 2 4 8; do
    OUTDIR="outputs/pp_bubble_nmb${NMB}"
    # batch_size must be divisible by num_micro_batches * dp_size
    BATCH=$((NMB * 1))  # dp=1, so batch = nmb * 1
    
    echo "--- num_micro_batches=$NMB, batch_size=$BATCH ---"
    echo "  Theory bubble: (2-1)/$NMB = $(python -c "print(f'{(2-1)/$NMB:.3f}')")"
    
    torchrun --nproc_per_node=2 scripts/run_colocate.py \
        --config configs/training/pp_equiv_pp2.yaml \
        --batch-size $BATCH \
        --iterations 6 2>&1 | grep -E "(step_time|tokens/s|Performance|MFU)" | head -2
    echo ""
done
