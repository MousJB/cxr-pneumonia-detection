"""Measures training and inference throughput on the GPU, to size the training budget
before the protocol is fixed. Uses ~200 images of the train split only; the weights are
thrown away and no metric is computed.

    python scripts/benchmark.py fasterrcnn_resnet50_fpn --batch-size 4
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    rows = [r for r in load_images() if r["split"] == "train"]
    rng = np.random.default_rng(0)
    # Same mix as the planned sampler: half positive, half negative images
    pos = [r["case_id"] for r in rows if r["target"] == "1"]
    neg = [r["case_id"] for r in rows if r["target"] == "0"]
    cases = list(rng.choice(pos, args.images // 2, replace=False)) + list(rng.choice(neg, args.images // 2, replace=False))
    loader = torch.utils.data.DataLoader(DetectionDataset(cases, load_boxes()), batch_size=args.batch_size,
                                         shuffle=True, num_workers=4, collate_fn=collate, persistent_workers=True)

    device = torch.device("cuda")
    model = build_model(args.architecture, size=args.size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()

    def train_pass():
        model.train()
        for images, targets, _ in loader:
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = sum(model(images, targets).values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    def infer_pass():
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for images, _, _ in loader:
                model([im.to(device, non_blocking=True) for im in images])

    torch.cuda.reset_peak_memory_stats()
    train_pass()  # warm-up (cuDNN autotuning, worker start)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_pass()
    torch.cuda.synchronize()
    train_ips = len(cases) / (time.perf_counter() - t0)
    t0 = time.perf_counter()
    infer_pass()
    torch.cuda.synchronize()
    infer_ips = len(cases) / (time.perf_counter() - t0)

    n_train = len(rows)
    n_balanced = 2 * len(pos)
    result = {
        "architecture": args.architecture, "size": args.size, "batch_size": args.batch_size,
        "gpu": torch.cuda.get_device_name(), "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "train_images_per_s": round(train_ips, 2), "inference_images_per_s": round(infer_ips, 2),
        "train_split_images": n_train, "balanced_epoch_images": n_balanced,
        "minutes_per_balanced_epoch": round(n_balanced / train_ips / 60, 1),
        "minutes_per_full_epoch": round(n_train / train_ips / 60, 1),
    }
    print(json.dumps(result, indent=2))
    out = Path(__file__).resolve().parents[1] / "runs" / "benchmark.csv"
    out.parent.mkdir(exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result), lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(result)


if __name__ == "__main__":
    main()
