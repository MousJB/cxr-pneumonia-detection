"""Final training of one pre-registered run (PROTOCOL.md §6). Writes to ``runs/<run id>/``:

- ``log.jsonl``: one header line (commit, hashes of PROTOCOL.md and config.toml, versions,
  settings), then one line per epoch (losses, costs, validation metric for every τ);
- ``val_predictions_epoch<N>.npz``: every validation box of every epoch;
- ``best.pt``: weights of the kept epoch; ``last.pt``: full state to resume after a crash.

    python scripts/train.py A          # ResNet-50 FPN at 512 px
    python scripts/train.py B          # ResNet-50 FPN at 384 px
    python scripts/train.py B --resume # continue from runs/<id>/last.pt
"""

import argparse
import ctypes
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cxr.data import TRAIN_TRANSFORM, DetectionDataset, collate, load_boxes, load_images  # noqa: E402
from cxr.models import build_model  # noqa: E402
from cxr.training import (  # noqa: E402
    BalancedEpochSampler,
    EarlyStopping,
    EpochIndexedDataset,
    above,
    predict,
    select_tau,
    warmup_factor,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"A": ("frcnn-r50-fpn-512", 512), "B": ("frcnn-r50-fpn-384", 384)}
ARCHITECTURE = "fasterrcnn_resnet50_fpn"
SEED = 20261030
LR, MOMENTUM, WEIGHT_DECAY, BATCH_SIZE = 0.005, 0.9, 1e-4, 4
WARMUP_STEPS, MAX_EPOCHS, PATIENCE = 500, 10, 3
AMP = False  # PROTOCOL.md §5.3: float32 (float16 autocast: NaN loss and slower on the GTX 1650)
CUDNN_BENCHMARK = False  # PROTOCOL.md §5.4
LOSS_KEYS = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def keep_awake():
    """Windows: no sleep while this process runs (reset automatically when it exits)."""
    if platform.system() == "Windows":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM_REQUIRED


def log(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record)[:400], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", choices=sorted(RUNS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None, help="smoke test only: steps per epoch")
    parser.add_argument("--max-val", type=int, default=None, help="smoke test only: validation images")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    smoke = args.max_steps is not None or args.max_val is not None
    run_id, size = RUNS[args.run]
    out = args.out or ROOT / "runs" / (run_id + ("-smoke" if smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    if log_path.exists() and not args.resume:
        sys.exit(f"{log_path} exists: use --resume, or remove it for a new run")
    keep_awake()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    device = torch.device("cuda")

    images = load_images()
    boxes = load_boxes()
    train_rows = [r for r in images if r["split"] == "train"]
    val_cases = [r["case_id"] for r in images if r["split"] == "val"][: args.max_val]
    val_truth = {c: [b for b in boxes.get(c, [])] for c in val_cases}
    train_cases = [r["case_id"] for r in train_rows]
    sampler = BalancedEpochSampler([int(r["target"]) for r in train_rows], SEED)
    train_set = EpochIndexedDataset(DetectionDataset(train_cases, boxes, transform=TRAIN_TRANSFORM, seed=SEED))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4,
                                               collate_fn=collate, persistent_workers=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(DetectionDataset(val_cases, boxes), batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2, collate_fn=collate)

    model = build_model(ARCHITECTURE, size=size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=1)
    scaler = torch.amp.GradScaler(enabled=AMP)
    stopper = EarlyStopping(PATIENCE)
    start_epoch, global_step, best_tau = 0, 0, None

    if args.resume and (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)  # RNG states must stay on the CPU
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        plateau.load_state_dict(state["plateau"])
        scaler.load_state_dict(state["scaler"])
        stopper.load_state_dict(state["stopper"])
        start_epoch, global_step, best_tau = state["epoch"] + 1, state["global_step"], state["best_tau"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        log(log_path, {"event": "resume", "from_epoch": start_epoch, "time": datetime.now(UTC).isoformat()})
    else:
        log(log_path, {
            "event": "start", "run": args.run, "run_id": run_id, "smoke_test": smoke,
            "time": datetime.now(UTC).isoformat(),
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": git("status", "--porcelain", "--untracked-files=no"),
            "protocol_sha256": sha256(ROOT / "PROTOCOL.md"), "config_sha256": sha256(ROOT / "config.toml"),
            "split_sha256": sha256(ROOT / "metadata" / "split.csv"), "command": sys.argv,
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
            "settings": {"architecture": ARCHITECTURE, "size": size, "seed": SEED, "lr": LR, "momentum": MOMENTUM,
                         "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "warmup_steps": WARMUP_STEPS,
                         "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "amp_fp16": AMP,
                         "cudnn_benchmark": CUDNN_BENCHMARK, "trainable_params": sum(p.numel() for p in params),
                         "train_images": len(train_cases), "epoch_images": len(sampler),
                         "val_images": len(val_cases)},
        })

    run_t0 = time.perf_counter()
    for epoch in range(start_epoch, MAX_EPOCHS):
        if stopper.bad_epochs >= PATIENCE:  # resumed after the run had already stopped
            break
        sampler.set_epoch(epoch)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        sums = torch.zeros(len(LOSS_KEYS), device=device)
        n_steps = n_images = skipped = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for imgs, targets, _ in train_loader:
            if args.max_steps is not None and n_steps >= args.max_steps:
                break
            if global_step < WARMUP_STEPS:
                for g in optimizer.param_groups:
                    g["lr"] = LR * warmup_factor(global_step, WARMUP_STEPS)
            elif global_step == WARMUP_STEPS:
                for g in optimizer.param_groups:
                    g["lr"] = LR
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                log(log_path, {"event": "non_finite_loss", "epoch": epoch, "step": global_step,
                               "losses": {k: float(v) for k, v in loss_dict.items()}})
                sys.exit("non-finite loss: run stopped (PROTOCOL.md §6)")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale() if AMP else None
            scaler.step(optimizer)
            scaler.update()
            if AMP and scaler.get_scale() < scale_before:
                skipped += 1
            sums += torch.stack([loss_dict[k].detach().float() for k in LOSS_KEYS])
            n_steps += 1
            n_images += len(imgs)
            global_step += 1
        torch.cuda.synchronize()
        train_s = time.perf_counter() - t0
        train_peak = (torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        preds = predict(model, val_loader, device, amp=AMP)
        torch.cuda.synchronize()
        val_s = time.perf_counter() - t0
        tau, metric, per_tau = select_tau(preds, val_truth)
        np.savez_compressed(out / f"val_predictions_epoch{epoch + 1}.npz",
                            cases=np.array(list(preds)),
                            boxes=np.array([preds[c][0] for c in preds], dtype=object),
                            scores=np.array([preds[c][1] for c in preds], dtype=object))
        improved, stop = stopper.step(metric, epoch + 1)
        lr_used = optimizer.param_groups[0]["lr"]
        plateau.step(metric)
        if improved:
            best_tau = tau
            torch.save(model.state_dict(), out / "best.pt")
        means = (sums / max(n_steps, 1)).cpu().numpy()
        log(log_path, {
            "event": "epoch", "epoch": epoch + 1, "lr": lr_used, "lr_next": optimizer.param_groups[0]["lr"],
            "global_step": global_step, "steps": n_steps, "skipped_steps": skipped, "images": n_images,
            "train_loss": float(means.sum()), "train_losses": dict(zip(LOSS_KEYS, means.round(5).tolist())),
            "train_s": round(train_s, 1), "train_images_per_s": round(n_images / train_s, 3),
            "val_s": round(val_s, 1), "val_images_per_s": round(len(val_cases) / val_s, 3),
            "train_peak_allocated_gb": round(train_peak[0], 3), "train_peak_reserved_gb": round(train_peak[1], 3),
            "val_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "val_metric": metric, "val_tau": tau, "val_metric_per_tau": per_tau,
            "val_images_with_box_at_tau": int(sum(len(b) > 0 for b, _ in above(preds, tau).values())),
            "improved": improved, "best_epoch": stopper.best_epoch, "best_metric": stopper.best,
            "bad_epochs": stopper.bad_epochs, "time": datetime.now(UTC).isoformat(),
        })
        t0 = time.perf_counter()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "plateau": plateau.state_dict(),
                    "scaler": scaler.state_dict(), "stopper": stopper.state_dict(), "epoch": epoch,
                    "global_step": global_step, "best_tau": best_tau, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state()}, out / "last.pt")
        log(log_path, {"event": "checkpoint", "epoch": epoch + 1, "save_s": round(time.perf_counter() - t0, 1)})
        if stop:
            break

    converged = stopper.best_epoch is not None and stopper.best_epoch < MAX_EPOCHS
    log(log_path, {
        "event": "end", "best_epoch": stopper.best_epoch, "best_val_metric": stopper.best, "best_tau": best_tau,
        "stopped_early": stopper.bad_epochs >= PATIENCE, "converged_within_budget": converged,
        "best_weights_sha256": sha256(out / "best.pt"), "session_s": round(time.perf_counter() - run_t0, 1),
        "time": datetime.now(UTC).isoformat(),
    })


if __name__ == "__main__":
    main()
