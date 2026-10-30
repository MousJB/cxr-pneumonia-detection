"""Controlled comparison of float32 and float16 autocast (AMP) training on the GPU.

Not a training run: the weights are thrown away and no detection metric is computed.
Only training images are used for the training steps; ``--mode epoch`` also times one
inference pass over the validation images (no label, no metric).

``--mode steps`` (repeated, one process per repetition):
    fixed seed for the initial weights, fixed image order (half with opacity, half
    without, augmentation of cxr/data.py with a fixed seed), one data-loader pass:
    ``--warmup`` steps not timed, then ``--steps`` timed steps, then a timed inference
    pass over ``--infer-images`` images. Per step: loss and its 4 components, gradient
    norm (after unscaling with AMP), skipped optimiser steps. Before/after snapshots
    check that the parameters are updated and stay finite.
``--mode epoch``:
    one complete balanced epoch (all training positives + as many negatives, seeded),
    a complete validation inference pass, and the time to save a checkpoint.

During timed regions ``nvidia-smi`` is sampled every 500 ms (utilisation, memory,
SM clock, temperature, power). CUDA is synchronised before and after every timed
region. One JSON object per run is appended to ``--out``.

    python scripts/precision_benchmark.py --precision fp32 --mode steps --repeat 1
"""

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cxr.data import TRAIN_TRANSFORM, DetectionDataset, collate, load_boxes, load_images  # noqa: E402
from cxr.models import build_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261030
LOSS_KEYS = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")


class GpuSampler:
    """``nvidia-smi`` sampled in the background; summary of the samples."""

    FIELDS = ("utilization.gpu", "memory.used", "clocks.sm", "temperature.gpu", "power.draw")

    def __init__(self):
        self.samples = []
        self.proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={','.join(self.FIELDS)}", "--format=csv,noheader,nounits", "-lms", "500"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.samples.append([float(v) for v in line.split(",")])
            except ValueError:
                pass

    def stop(self) -> dict:
        self.proc.terminate()
        self.thread.join(timeout=5)
        if not self.samples:
            return {"samples": 0}
        a = np.array(self.samples)
        out = {"samples": len(a)}
        for i, name in enumerate(("util_pct", "mem_used_mib", "sm_clock_mhz", "temp_c", "power_w")):
            out[name] = {"mean": round(float(a[:, i].mean()), 1), "min": round(float(a[:, i].min()), 1),
                         "max": round(float(a[:, i].max()), 1)}
        return out


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("fp32", "amp"), required=True)
    parser.add_argument("--mode", choices=("steps", "epoch"), default="steps")
    parser.add_argument("--architecture", default="fasterrcnn_resnet50_fpn")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--infer-images", type=int, default=400)
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="repetition number, stored with the result")
    parser.add_argument("--profile", action="store_true",
                        help="profile the timed steps (CUDA kernels) to runs/profile_*.txt; timings not comparable")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "precision_benchmark.jsonl")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    amp = args.precision == "amp"
    device = torch.device("cuda")
    images = load_images()
    train = [r for r in images if r["split"] == "train"]
    pos = sorted(r["case_id"] for r in train if r["target"] == "1")
    neg = sorted(r["case_id"] for r in train if r["target"] == "0")
    rng = np.random.default_rng(SEED)
    if args.mode == "steps":
        half = (args.warmup + args.steps) * args.batch_size // 2
        cases = list(rng.choice(pos, half, replace=False)) + list(rng.choice(neg, half, replace=False))
    else:
        cases = pos + list(rng.choice(neg, len(pos), replace=False))
    cases = [cases[i] for i in rng.permutation(len(cases))]  # fixed order, identical in every run
    boxes = load_boxes()
    loader = torch.utils.data.DataLoader(DetectionDataset(cases, boxes, transform=TRAIN_TRANSFORM, seed=SEED),
                                         batch_size=args.batch_size, shuffle=False, num_workers=4,
                                         collate_fn=collate, pin_memory=False)

    torch.manual_seed(SEED)  # identical new head in every run
    model = build_model(args.architecture, size=args.size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=amp)
    watched = {"head": model.roi_heads.box_predictor.cls_score.weight}
    if hasattr(model.backbone.body, "layer4"):  # ResNet
        watched["backbone_layer4"] = model.backbone.body.layer4[0].conv1.weight
    else:  # MobileNetV3: last trainable convolution of the backbone
        watched["backbone_last_conv"] = [p for p in model.backbone.body.parameters()
                                         if p.requires_grad and p.ndim == 4][-1]
    snapshot0 = {k: v.detach().clone() for k, v in watched.items()}
    torch.cuda.reset_peak_memory_stats()

    losses, components, grad_norms, data_wait = [], [], [], []
    timing = {}

    def step(batch):
        imgs, targets, _ = batch
        imgs = [im.to(device, non_blocking=True) for im in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # no-op when disabled; true gradients in both modes
        grad_norms.append(torch.nn.utils.clip_grad_norm_(params, float("inf")).detach())
        scaler.step(optimizer)  # skipped by GradScaler if a gradient is inf/NaN
        scaler.update()
        losses.append(loss.detach().float())
        components.append(torch.stack([loss_dict[k].detach().float() for k in LOSS_KEYS]))

    model.train()
    it = iter(loader)
    n_warm = args.warmup if args.mode == "steps" else 0  # epoch mode: the whole epoch is timed
    for _ in range(n_warm):
        step(next(it))
    snapshot_warm = {k: v.detach().clone() for k, v in watched.items()}
    sampler = GpuSampler()
    profiler = None
    if args.profile:  # separate runs: profiling slows the steps down
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA])
        profiler.__enter__()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_timed = 0
    while args.mode == "epoch" or n_timed < args.steps:
        t_wait = time.perf_counter()
        batch = next(it, None)
        if batch is None:
            break
        data_wait.append(time.perf_counter() - t_wait)  # time blocked waiting for the data workers
        step(batch)
        n_timed += 1
    torch.cuda.synchronize()
    timing["train_s"] = time.perf_counter() - t0
    if profiler is not None:
        profiler.__exit__(None, None, None)
        table = profiler.key_averages().table(sort_by="cuda_time_total", row_limit=25)
        suffix = "_cudnnbench" if args.cudnn_benchmark else ""
        out = args.out.with_name(f"profile_{args.precision}_{args.size}{suffix}.txt")
        out.parent.mkdir(exist_ok=True)
        out.write_text(table, encoding="utf-8")
    timing["train_images"] = n_timed * args.batch_size if args.mode == "steps" else len(cases)
    gpu_train = sampler.stop()
    if args.mode == "steps":
        infer_cases = cases[: args.infer_images]
    else:
        infer_cases = [r["case_id"] for r in images if r["split"] == "val"]

    # Inference: training images (steps mode) or the whole validation split (epoch mode)
    infer_loader = torch.utils.data.DataLoader(DetectionDataset(infer_cases, boxes), batch_size=args.batch_size,
                                               shuffle=False, num_workers=4, collate_fn=collate)
    model.eval()

    def infer():
        n_boxes = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            for imgs, _, _ in infer_loader:
                out = model([im.to(device, non_blocking=True) for im in imgs])
                n_boxes += sum(len(o["boxes"]) for o in out)
        return n_boxes

    sampler = GpuSampler()
    timing["infer_s"], n_pred_boxes = timed(infer)
    gpu_infer = sampler.stop()
    timing["infer_images"] = len(infer_cases)

    checkpoint_s = None
    if args.mode == "epoch":
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_s, _ = timed(lambda: torch.save({"model": model.state_dict(),
                                                        "optimizer": optimizer.state_dict()}, Path(tmp) / "c.pt"))

    loss_t = torch.stack(losses).cpu().numpy()
    comp_t = torch.stack(components).cpu().numpy()
    gn = torch.stack(grad_norms).cpu().numpy()
    finite_params = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    update = {k: {"warmup": float((snapshot_warm[k] - snapshot0[k]).norm()),
                  "timed": float((v.detach() - snapshot_warm[k]).norm())} for k, v in watched.items()}
    result = {
        "mode": args.mode, "precision": args.precision, "repeat": args.repeat, "architecture": args.architecture,
        "size": args.size, "batch_size": args.batch_size, "cudnn_benchmark": args.cudnn_benchmark,
        "augmentation": "TRAIN_TRANSFORM", "workers": 4, "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
        "warmup_steps": n_warm, "timed_steps": len(losses) - n_warm,
        "train_images_timed": timing["train_images"], "train_s": round(timing["train_s"], 2),
        "train_images_per_s": round(timing["train_images"] / timing["train_s"], 3),
        "step_s_mean": round(timing["train_s"] / (len(losses) - n_warm), 4),
        "data_wait_s_total": round(float(np.sum(data_wait)), 2),
        "infer_images": timing["infer_images"], "infer_s": round(timing["infer_s"], 2),
        "infer_images_per_s": round(timing["infer_images"] / timing["infer_s"], 3),
        "predicted_boxes": n_pred_boxes, "checkpoint_save_s": None if checkpoint_s is None else round(checkpoint_s, 2),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        "gpu_train": gpu_train, "gpu_infer": gpu_infer,
        "loss_finite_all": bool(np.isfinite(loss_t).all()),
        "grad_nonfinite_steps": int((~np.isfinite(gn)).sum()),  # = optimiser steps skipped by GradScaler
        "grad_norm_median": round(float(np.nanmedian(gn[np.isfinite(gn)])), 4),
        "loss_first10_mean": round(float(loss_t[:10].mean()), 4),
        "loss_last50_mean": round(float(loss_t[-50:].mean()), 4),
        "components_last50_mean": dict(zip(LOSS_KEYS, np.round(comp_t[-50:].mean(0), 4).tolist())),
        "loss_curve_every10": np.round(loss_t[::10], 4).tolist(),
        "final_grad_scale": float(scaler.get_scale()) if amp else None,
        "param_update_norm": update, "params_finite": finite_params,
    }
    print(json.dumps(result, indent=1))
    args.out.parent.mkdir(exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
