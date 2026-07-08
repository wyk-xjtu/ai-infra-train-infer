"""Print training loss curve from TensorBoard events file."""
import sys, os
sys.path.insert(0, os.getcwd())
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logdir = r"outputs/sft/qwen3_0_6b/tensorboard"
ea = EventAccumulator(logdir)
ea.Reload()

tags = ea.Tags().get("scalars", [])
print(f"Available tags: {tags}\n")

if "train/loss" in tags:
    events = ea.Scalars("train/loss")
    print("Iteration |   Loss   |  Step")
    print("----------|----------|------")
    for e in events:
        print(f"  {e.step:>5}    | {e.value:.4f} |  {e.step}")
else:
    print("No train/loss found")
