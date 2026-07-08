"""Download a Hugging Face dataset and save it for offline training.

Examples:
    python scripts/prepare_data.py --dataset yahma/alpaca-cleaned --split train --output data/alpaca-cleaned
    python scripts/prepare_data.py --dataset openai/gsm8k --config main --split train --output data/gsm8k-main
"""
import argparse
import json
import os
from datetime import datetime, timezone


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare an offline dataset directory for train-infer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name")
    parser.add_argument("--config", default=None, help="Optional dataset config name")
    parser.add_argument("--split", default="train", help="Dataset split to save")
    parser.add_argument("--output", required=True, help="Output directory for save_to_disk")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap")
    parser.add_argument(
        "--required-field",
        action="append",
        default=[],
        help="Field that must exist in each sampled row. Can be passed multiple times.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from datasets import load_dataset

    if args.config:
        dataset = load_dataset(args.dataset, args.config, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    missing = []
    if args.required_field:
        sample_count = min(20, len(dataset))
        for field in args.required_field:
            for idx in range(sample_count):
                if field not in dataset[idx]:
                    missing.append(field)
                    break
    if missing:
        fields = ", ".join(sorted(set(missing)))
        raise ValueError(f"Dataset is missing required field(s): {fields}")

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    dataset.save_to_disk(output)

    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "rows": len(dataset),
        "output": output,
        "required_fields": args.required_field,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(output, "train_infer_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
