"""Model selection on validation (PROTOCOL.md §8) and the single test evaluation (§9, §10).

    python scripts/evaluate.py select   # validation only -> results/validation_selection.{json,md}
    python scripts/evaluate.py test     # once: needs the committed selection; writes results/test/

``test`` refuses to run if ``results/validation_selection.json`` is not committed unchanged,
or if ``results/test/`` already exists: the test split is evaluated once.
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cxr.data import DetectionDataset, collate, load_boxes, load_images  # noqa: E402
from cxr.evaluation import (  # noqa: E402
    FIELDS,
    image_stats,
    match_boxes,
    non_inferiority,
    paired_bootstrap,
    ratios,
    selection_rule,
)
from cxr.models import build_model  # noqa: E402
from cxr.training import predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUNS = {"A": ("frcnn-r50-fpn-512", 512), "B": ("frcnn-r50-fpn-384", 384)}
SEED, RESAMPLES = 20261030, 2000
SIZE_TERCILES = (11_503.0, 22_064.0)  # px², training boxes (PROTOCOL.md §7.3, §10)
MEDVISION_FILES = {"A": "preds_r50_512.csv", "B": "preds_r50_384.csv"}
RUNS_SUFFIX = ""  # "-smoke": rehearsal on the smoke-test runs (see main)


def run_dir(run_id: str) -> Path:
    return ROOT / "runs" / (run_id + RUNS_SUFFIX)


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def read_log(run_id: str) -> list[dict]:
    path = run_dir(run_id) / "log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_summary(run_id: str) -> dict:
    """Best epoch, τ and measured costs of one training run (PROTOCOL.md §7.5)."""
    log = read_log(run_id)
    end = [r for r in log if r["event"] == "end"]
    if not end:
        sys.exit(f"{run_id}: training not finished")
    epochs = [r for r in log if r["event"] == "epoch"]
    saves = [r["save_s"] for r in log if r["event"] == "checkpoint"]
    return {
        **{k: end[-1][k] for k in ("best_epoch", "best_val_metric", "best_tau", "stopped_early",
                                   "converged_within_budget", "best_weights_sha256")},
        "epochs_run": len(epochs), "optimizer_steps": sum(r["steps"] for r in epochs),
        "skipped_steps": sum(r["skipped_steps"] for r in epochs), "images_processed": sum(r["images"] for r in epochs),
        "train_hours": sum(r["train_s"] for r in epochs) / 3600, "val_hours": sum(r["val_s"] for r in epochs) / 3600,
        "checkpoint_hours": sum(saves) / 3600,
        "gpu_hours": (sum(r["train_s"] + r["val_s"] for r in epochs) + sum(saves)) / 3600,
        "train_minutes_per_epoch": float(np.mean([r["train_s"] for r in epochs]) / 60),
        "val_minutes_per_epoch": float(np.mean([r["val_s"] for r in epochs]) / 60),
        "train_images_per_s": float(np.mean([r["train_images_per_s"] for r in epochs])),
        "val_images_per_s": float(np.mean([r["val_images_per_s"] for r in epochs])),
        "peak_train_allocated_gb": max(r["train_peak_allocated_gb"] for r in epochs),
        "peak_train_reserved_gb": max(r["train_peak_reserved_gb"] for r in epochs),
        "val_metric_per_epoch": [r["val_metric"] for r in epochs], "tau_per_epoch": [r["val_tau"] for r in epochs],
        "train_loss_per_epoch": [r["train_loss"] for r in epochs], "lr_per_epoch": [r["lr"] for r in epochs],
    }


def load_val_predictions(run_id: str, epoch: int) -> dict:
    z = np.load(run_dir(run_id) / f"val_predictions_epoch{epoch}.npz", allow_pickle=True)
    return {c: (np.asarray(b, float).reshape(-1, 4), np.asarray(s, float)) for c, b, s in
            zip(z["cases"], z["boxes"], z["scores"])}


def fmt(m: dict) -> str:
    return f"{m['value']:.4f} [{m['ci'][0]:.4f}, {m['ci'][1]:.4f}]"


def metrics_table(boot: dict, models: list[str]) -> list[str]:
    names = list(boot["models"][models[0]])
    lines = ["| Metric | " + " | ".join(models) + " | " + " | ".join(boot["differences"]) + " |",
             "|---|" + "---|" * (len(models) + len(boot["differences"]))]
    for k in names:
        lines.append(f"| {k} | " + " | ".join(fmt(boot["models"][m][k]) for m in models) + " | "
                     + " | ".join(fmt(d[k]) for d in boot["differences"].values()) + " |")
    return lines


def cmd_select():
    images = load_images()
    boxes = load_boxes()
    val = [r for r in images if r["split"] == "val"]
    truth = {r["case_id"]: boxes.get(r["case_id"], []) for r in val}
    patients = {r["case_id"]: r["nih_patient"] for r in val}
    summaries, preds, stats = {}, {}, {}
    for name, (run_id, _) in RUNS.items():
        summaries[name] = run_summary(run_id)
        preds[name] = load_val_predictions(run_id, summaries[name]["best_epoch"])
        if set(preds[name]) != set(truth):
            if not RUNS_SUFFIX:
                sys.exit(f"{run_id}: the validation predictions do not cover the validation split")
            truth = {c: truth[c] for c in preds[name]}  # rehearsal: smoke runs saw a subset
            patients = {c: patients[c] for c in truth}
    for name in RUNS:
        stats[name] = image_stats(preds[name], truth, summaries[name]["best_tau"])
    boot = paired_bootstrap(stats, patients, RESAMPLES, SEED)
    rule = selection_rule(boot["differences"]["B - A"])
    result = {"split": "val", "runs": {k: RUNS[k][0] for k in RUNS}, "training": summaries,
              "bootstrap": boot, "selection_rule": rule, "selected": rule["selected"],
              "selected_run_id": RUNS[rule["selected"]][0]}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "validation_selection.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    for run_id, _ in RUNS.values():  # training logs are published (runs/ is not versioned)
        shutil.copy(run_dir(run_id) / "log.jsonl", RESULTS / f"training_log_{run_id}.jsonl")
    md = ["# Validation results and model selection (PROTOCOL.md §8)", "",
          (f"Validation split: {boot['images']} images, {boot['patients']} patients. 95 % CI: percentile "
           f"bootstrap over patients, {RESAMPLES} paired resamples, seed {SEED}. Each model at its own τ."), "",
          "| Run | Best epoch / run | τ | Stopped early | Best within budget | GPU h | Train img/s | Peak GB |",
          "|---|---|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        md.append(f"| {name} ({RUNS[name][0]}) | {s['best_epoch']} / {s['epochs_run']} | {s['best_tau']} | "
                  f"{s['stopped_early']} | {s['converged_within_budget']} | {s['gpu_hours']:.2f} | "
                  f"{s['train_images_per_s']:.2f} | {s['peak_train_allocated_gb']:.2f} |")
    md += ["", *metrics_table(boot, list(RUNS)), "", "Selection rule:", ""]
    md += [f"- {k}: {v}" for k, v in rule.items()]
    md += ["", f"**Selected model: {rule['selected']} ({RUNS[rule['selected']][0]})**", ""]
    (RESULTS / "validation_selection.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def medvision_score(s: float, tau: float) -> float:
    return 0.5 * s / tau if s < tau else 0.5 + 0.5 * (s - tau) / (1 - tau)


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def subgroup_tables(stats: dict, rows: list[dict], preds: dict, truth: dict, tau: float) -> dict:
    """PROTOCOL.md §10 (point estimates and counts)."""
    def age_group(a):
        a = int(a)
        return "0-20" if a <= 20 else "21-40" if a <= 40 else "41-60" if a <= 60 else "61-80" if a <= 80 else ">80"

    keys = {"view": lambda r: r["view"], "sex": lambda r: r["sex"], "age": lambda r: age_group(r["age"]),
            "n_boxes": lambda r: r["n_boxes"]}
    out = {}
    for name, key in keys.items():
        groups = defaultdict(list)
        for r in rows:
            groups[key(r)].append(r["case_id"])
        out[name] = {}
        for g, cases in sorted(groups.items()):
            m = ratios(np.array([[stats[c][f] for f in FIELDS] for c in cases], float).sum(0))
            out[name][g] = {"images": len(cases), **{k: float(v) for k, v in m.items()}}
    # recall by opacity size (terciles of training boxes), each ground-truth box at IoU > 0.4
    size_hits = defaultdict(lambda: [0, 0])
    for case, gt in truth.items():
        b, s = preds[case]
        keep = s >= tau
        gt = np.asarray(gt, float).reshape(-1, 4)
        if len(gt) == 0:
            continue
        matched, _ = match_boxes(b[keep], s[keep], gt)
        for j, (w, h) in enumerate(gt[:, 2:]):
            area = w * h
            label = "small" if area < SIZE_TERCILES[0] else "medium" if area < SIZE_TERCILES[1] else "large"
            size_hits[label][0] += 1
            size_hits[label][1] += int(j in matched)
    out["opacity_size_box_recall"] = {k: {"boxes": v[0], "recall": v[1] / v[0]} for k, v in size_hits.items()}
    # false positives among images without opacity, by RSNA detailed class
    fp = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["target"] == "0":
            fp[r["detailed_class"]][0] += 1
            fp[r["detailed_class"]][1] += stats[r["case_id"]]["pred_pos"]
    out["false_positive_rate_by_detailed_class"] = {k: {"images": v[0], "fp_rate": v[1] / v[0]} for k, v in fp.items()}
    return out


def cmd_test(dry_run: Path | None = None):
    """``dry_run``: rehearse the whole pipeline on the VALIDATION split into that folder (no guard,
    config.toml copied there); the test split is not read."""
    sel_path = RESULTS / "validation_selection.json"
    if dry_run is None and (git("ls-files", "--error-unmatch", str(sel_path)).returncode != 0
                            or git("diff", "--quiet", "HEAD", "--", str(sel_path)).returncode != 0):
        sys.exit("results/validation_selection.json must be committed unchanged before the test evaluation")
    base = ROOT if dry_run is None else dry_run
    split = "test" if dry_run is None else "val"
    out = RESULTS / "test" if dry_run is None else dry_run / "test"
    if out.exists():
        sys.exit(f"{out} exists: the test split has already been evaluated (PROTOCOL.md §9)")
    if dry_run is not None:
        dry_run.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "config.toml", base / "config.toml")
    selection = json.loads(sel_path.read_text(encoding="utf-8"))
    out.mkdir(parents=True)
    (out / "STARTED").write_text(git("rev-parse", "HEAD").stdout.strip() + "\n", encoding="utf-8")

    images = load_images()
    boxes = load_boxes()
    test_rows = [r for r in images if r["split"] == split]
    cases = [r["case_id"] for r in test_rows]
    truth = {c: boxes.get(c, []) for c in cases}
    patients = {r["case_id"]: r["nih_patient"] for r in test_rows}
    device = torch.device("cuda")
    loader = torch.utils.data.DataLoader(DetectionDataset(cases, boxes), batch_size=4, shuffle=False, num_workers=2,
                                         collate_fn=collate)
    preds, stats, taus = {}, {}, {}
    for name, (run_id, size) in RUNS.items():
        taus[name] = selection["training"][name]["best_tau"]
        model = build_model("fasterrcnn_resnet50_fpn", size=size, pretrained=False).to(device)
        weights = run_dir(run_id) / "best.pt"
        sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        if sha != selection["training"][name]["best_weights_sha256"]:
            sys.exit(f"{weights}: SHA-256 differs from the one recorded at the end of training")
        model.load_state_dict(torch.load(weights, map_location=device))
        preds[name] = predict(model, loader, device)
        stats[name] = image_stats(preds[name], truth, taus[name])
        write_csv(out / f"predictions_{run_id}.csv", ["case_id", "x", "y", "width", "height", "score"],
                  [[c, *np.round(b, 2).tolist(), round(float(s), 5)] for c in cases
                   for b, s in zip(*preds[name][c])])
        del model
        torch.cuda.empty_cache()

    boot = paired_bootstrap(stats, patients, RESAMPLES, SEED)
    d = boot["differences"]["B - A"]["rsna_metric"]
    primary = {"metric": "competition metric, B - A", "value": d["value"], "ci": d["ci"], "margin": 0.02,
               "state": non_inferiority(d["ci"], 0.02)}

    # MedVision QA files (PROTOCOL.md §7.4) and reports
    mv = base / "medvision"
    write_csv(mv / "ground_truth.csv", ["case_id", "true_label", "patient_id", "meta_view", "meta_sex", "meta_age"],
              [[r["case_id"], "opacity" if truth[r["case_id"]] else "no_opacity", r["nih_patient"], r["view"],
                r["sex"], r["age"]] for r in test_rows])
    for name in RUNS:
        rows = []
        for c in cases:
            s = stats[name][c]["max_score"]
            p = medvision_score(s, taus[name])
            rows.append([c, "opacity" if s >= taus[name] else "no_opacity", f"{1 - p:.6f}", f"{p:.6f}"])
        write_csv(mv / MEDVISION_FILES[name], ["case_id", "predicted_label", "score_no_opacity", "score_opacity"], rows)
    config = (base / "config.toml").read_text(encoding="utf-8")
    commit = git("rev-parse", "HEAD").stdout.strip()
    for name in RUNS:  # metadata only (PROTOCOL.md: criteria unchanged)
        block = re.search(rf"\[versions\.{name}\].*?(?=\n\[)", config, re.DOTALL).group(0)
        new = re.sub(r'code_commit = ""[^\n]*', f'code_commit = "{commit}"', block)
        new = re.sub(r'weights_sha256 = ""[^\n]*',
                     f'weights_sha256 = "{selection["training"][name]["best_weights_sha256"]}"', new)
        config = config.replace(block, new)
    (base / "config.toml").write_text(config, encoding="utf-8")
    reports = base / "reports"
    reports.mkdir(exist_ok=True)
    medvision = {}
    for lang in ("en", "ja"):
        r = subprocess.run([sys.executable, "-m", "medvision_qa", str(base / "config.toml"), "-o",
                            str(reports / f"vvr_{lang}.html"), "--lang", lang],
                           cwd=ROOT, capture_output=True, text=True, check=False)
        medvision[lang] = {"returncode": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}

    selected = selection["selected"]
    analysis = {name: subgroup_tables(stats[name], test_rows, preds[name], truth, taus[name]) for name in RUNS}
    result = {"split": "test", "selected": selected, "selected_run_id": RUNS[selected][0], "tau": taus,
              "bootstrap": boot, "primary_non_inferiority": primary, "medvision_qa": medvision,
              "error_analysis": analysis, "code_commit": commit}
    (out / "test_results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    title = "Test results — single evaluation" if dry_run is None else "DRY RUN on the validation split"
    md = [f"# {title} (PROTOCOL.md §9, §10)", "",
          (f"Evaluated once at commit `{commit}`. Test split: {boot['images']} images, {boot['patients']} patients. "
           f"95 % CI: percentile bootstrap over patients, {RESAMPLES} paired resamples, seed {SEED}. "
           f"τ from validation: A = {taus['A']}, B = {taus['B']}. Selected model (validation, before the test): "
           f"**{selected} ({RUNS[selected][0]})**."), "",
          "## Primary: competition metric, non-inferiority of B (384 px) vs A (512 px), margin 0.02", "",
          f"Δ = {d['value']:.4f}, 95 % CI [{d['ci'][0]:.4f}, {d['ci'][1]:.4f}] → **{primary['state']}**", "",
          "## All metrics", "", *metrics_table(boot, list(RUNS)), "",
          "## MedVision QA (C1–C3)", "", "```", medvision["en"]["stdout"] or medvision["en"]["stderr"], "```", "",
          f"## Error analysis of the selected model ({selected}), descriptive", ""]
    a = analysis[selected]
    for name in ("view", "sex", "age", "n_boxes"):
        md += [f"### By {name}", "", "| Group | Images | Competition metric | Sensitivity | Specificity |",
               "|---|---|---|---|---|"]
        md += [f"| {g} | {v['images']} | {v['rsna_metric']:.4f} | {v['sensitivity']:.4f} | {v['specificity']:.4f} |"
               for g, v in a[name].items()]
        md.append("")
    md += ["### Box recall by opacity size (training terciles)", "", "| Size | Boxes | Recall |", "|---|---|---|"]
    md += [f"| {k} | {v['boxes']} | {v['recall']:.4f} |" for k, v in sorted(a["opacity_size_box_recall"].items())]
    md += ["", "### False-positive rate among images without opacity, by RSNA class", "",
           "| Class | Images | FP rate |", "|---|---|---|"]
    fp = a["false_positive_rate_by_detailed_class"]
    md += [f"| {k} | {v['images']} | {v['fp_rate']:.4f} |" for k, v in fp.items()]
    (out / "test_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "STARTED").unlink()
    print("\n".join(md))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # the tables contain τ, Δ (Windows consoles default to cp1252)
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("select", "test"))
    parser.add_argument("--dry-run-on-val", type=Path, default=None,
                        help="test command only: rehearse on the validation split into this folder")
    parser.add_argument("--rehearsal", type=Path, default=None,
                        help="read runs/<id>-smoke and write the selection to this folder (pipeline test only)")
    args = parser.parse_args()
    global RESULTS, RUNS_SUFFIX
    if args.rehearsal is not None:
        RESULTS, RUNS_SUFFIX = args.rehearsal, "-smoke"
        if args.command == "test" and args.dry_run_on_val is None:
            sys.exit("--rehearsal with test requires --dry-run-on-val")
    if args.command == "select":
        cmd_select()
    else:
        cmd_test(args.dry_run_on_val)


if __name__ == "__main__":
    main()
