"""Measures training and inference throughput on the GPU, to size the training budget
before the protocol is fixed. Uses ~200 images of the train split only; the weights are
thrown away and no metric is computed.

    python scripts/benchmark.py fasterrcnn_resnet50_fpn --batch-size 4

Method: the same ``--images`` images (half positive, half negative, seed 0, no
augmentation) are run through 1 warm-up training pass (cuDNN autotuning, worker start,
not timed), 1 timed training pass, then 1 timed inference pass. Throughputs are images
per second of the timed passes, data loading included (4 workers).

The optional switches (``--cudnn-benchmark``, ``--channels-last``, ``--fp32``,
``--trainable-layers``) test one optimisation at a time against the default setting.
The rows of ``runs/benchmark.csv`` were produced by the version of this script in commit
abfc675 (same method, defaults only, fewer columns); new rows go to ``--out``.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cxr.data import DetectionDataset, collate, load_boxes, load_images  # noqa: E402
from cxr.models import build_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--cudnn-benchmark", action="store_true", help="torch.backends.cudnn.benchmark = True")
    parser.add_argument("--channels-last", action="store_true", help="model weights in channels_last memory format")
    parser.add_argument("--fp32", action="store_true", help="no autocast / GradScaler (float32 throughout)")
    parser.add_argument("--trainable-layers", type=int, default=None,
                        help="trainable_backbone_layers passed to torchvision (default: torchvision's, 3)")
    parser.add_argument("--label", default="", help="free text stored with the row")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "benchmark_optim.csv")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    rows = [r for r in load_images() if r["split"] == "train"]
    rng = np.random.default_rng(0)
    # Same mix as the planned sampler: half positive, half negative images
    pos = [r["case_id"] for r in rows if r["target"] == "1"]
    neg = [r["case_id"] for r in rows if r["target"] == "0"]
    cases = list(rng.choice(pos, args.images // 2, replace=False)) + list(rng.choice(neg, args.images // 2, replace=False))
    loader = torch.utils.data.DataLoader(DetectionDataset(cases, load_boxes()), batch_size=args.batch_size,
                                         shuffle=True, num_workers=4, collate_fn=collate, persistent_workers=True)

    device = torch.device("cuda")
    extra = {} if args.trainable_layers is None else {"trainable_backbone_layers": args.trainable_layers}
    model = build_model(args.architecture, size=args.size, **extra).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=0.005, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=not args.fp32)
    steps = 0

    def train_pass():
        nonlocal steps
        model.train()
        for images, targets, _ in loader:
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", dtype=torch.float16, enabled=not args.fp32):
                loss = sum(model(images, targets).values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            steps += 1

    def infer_pass():
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=not args.fp32):
            for images, _, _ in loader:
                model([im.to(device, non_blocking=True) for im in images])

    torch.cuda.reset_peak_memory_stats()
    train_pass()  # warm-up (cuDNN autotuning, worker start)
    warmup_steps = steps
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_pass()
    torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    infer_pass()
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - t0
    train_ips, infer_ips = len(cases) / train_s, len(cases) / infer_s

    n_train = len(rows)
    n_balanced = 2 * len(pos)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {
        "architecture": args.architecture, "size": args.size, "batch_size": args.batch_size,
        "amp_fp16": not args.fp32, "cudnn_benchmark": args.cudnn_benchmark, "channels_last": args.channels_last,
        "trainable_backbone_layers": "default" if args.trainable_layers is None else args.trainable_layers,
        "trainable_params_m": round(trainable / 1e6, 2),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "images_per_pass": len(cases), "warmup_steps": warmup_steps, "timed_train_steps": steps - warmup_steps,
        "timed_train_s": round(train_s, 1), "timed_infer_s": round(infer_s, 1),
        "train_images_per_s": round(train_ips, 2), "inference_images_per_s": round(infer_ips, 2),
        "train_split_images": n_train, "balanced_epoch_images": n_balanced,
        "minutes_per_balanced_epoch": round(n_balanced / train_ips / 60, 1),
        "minutes_per_full_epoch": round(n_train / train_ips / 60, 1),
        "label": args.label,
    }
    print(json.dumps(result, indent=2))
    out = args.out
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != list(result):
            sys.exit(f"{out}: existing header differs from this script's columns; choose another --out")
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result), lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(result)


if __name__ == "__main__":
    main()
