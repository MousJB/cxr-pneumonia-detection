"""config.toml is a valid MedVision QA v0.1 configuration and agrees with PROTOCOL.md.

The CSV files it points to are only written in Stage 5, so the configuration is copied to
a temporary folder next to synthetic CSV files in the expected format. The real
MedVision QA loader and evaluation are used (pinned in requirements.txt).
"""

import csv
import json
import re
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pytest

medvision_config = pytest.importorskip("medvision_qa.config")
medvision_evaluation = pytest.importorskip("medvision_qa.evaluation")

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.toml"


def medvision_score(max_box_score: float, tau: float) -> float:
    """Image score of PROTOCOL.md §7.4: increasing, piecewise linear, tau ↦ 0.5."""
    s = max_box_score
    return 0.5 * s / tau if s < tau else 0.5 + 0.5 * (s - tau) / (1 - tau)


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def staged_config(tmp_path):
    """config.toml copied next to synthetic ground truth and predictions."""
    shutil.copy(CONFIG, tmp_path / "config.toml")
    raw = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    n = 120
    truth = ["opacity" if i % 4 == 0 else "no_opacity" for i in range(n)]
    write_csv(tmp_path / raw["data"]["ground_truth_file"], ["case_id", "true_label", "patient_id"],
              [[f"c{i:03d}", truth[i], f"p{i // 2:03d}"] for i in range(n)])
    for version, tau in (("A", 0.35), ("B", 0.5)):
        rows = []
        for i in range(n):
            box = float(np.clip(rng.normal(0.6 if truth[i] == "opacity" else 0.25, 0.2), 0, 1))
            if i == 1:
                box = tau  # exactly at the threshold: tie at 0.5, must be accepted as "opacity"
            p = medvision_score(box, tau)
            rows.append([f"c{i:03d}", "opacity" if box >= tau else "no_opacity", f"{1 - p:.6f}", f"{p:.6f}"])
        write_csv(tmp_path / raw["versions"][version]["predictions_file"],
                  ["case_id", "predicted_label", "score_no_opacity", "score_opacity"], rows)
    return tmp_path / "config.toml"


def test_config_loads_with_medvision_validator(staged_config):
    cfg = medvision_config.load_config(staged_config)
    assert cfg.classes.order == ("no_opacity", "opacity")
    assert cfg.classes.positive == ("opacity",)
    assert set(cfg.versions) == {"A", "B"}
    assert [(c.id, c.kind, c.metric, c.margin) for c in cfg.criteria] == [
        ("C1", "non_inferiority", "auroc", 0.02),
        ("C2", "non_inferiority", "sensitivity", 0.05),
        ("C3", "non_inferiority", "specificity", 0.05),
    ]
    assert (cfg.confidence_level, cfg.bootstrap_resamples, cfg.seed) == (0.95, 2000, 20261030)
    assert cfg.manifest_sha256 == "9484bad013e24091845ac15072cc6e9ce7bb4ecc9b22c83f59b0259883b12af7"
    # Metadata are empty until the test evaluation fills them from the training logs: never guessed.
    selection = ROOT / "results" / "validation_selection.json"
    for name, v in cfg.versions.items():
        if v.code_commit == "" and v.weights_sha256 == "":
            continue
        assert re.fullmatch(r"[0-9a-f]{40}", v.code_commit)
        assert v.weights_sha256 == json.loads(selection.read_text(encoding="utf-8"))["training"][name][
            "best_weights_sha256"]


def test_full_evaluation_runs_on_synthetic_csv(staged_config):
    """The score mapping of the protocol produces files MedVision QA accepts end to end."""
    result = medvision_evaluation.run_evaluation(staged_config)
    assert result.decision.state in ("SATISFIED", "NOT_SATISFIED", "INCONCLUSIVE")


def test_score_mapping_keeps_order_and_threshold():
    tau = 0.3
    scores = np.linspace(0, 1, 101)
    mapped = np.array([medvision_score(s, tau) for s in scores])
    assert np.all(np.diff(mapped) > 0)  # strictly increasing: AUROC unchanged
    assert medvision_score(tau, tau) == 0.5
    assert all((m >= 0.5) == (s >= tau) for s, m in zip(scores, mapped))  # argmax = decision at tau
    assert medvision_score(0.0, tau) == 0.0 and medvision_score(1.0, tau) == 1.0


def test_metadata_hashes_match_files():
    import hashlib

    raw = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = hashlib.sha256((ROOT / "metadata" / "raw_manifest.csv").read_bytes()).hexdigest()
    split = hashlib.sha256((ROOT / "metadata" / "split.csv").read_bytes()).hexdigest()
    assert raw["data"]["manifest_sha256"] == manifest
    assert split in raw["data"]["description"]


@pytest.mark.parametrize("name", ["PROTOCOL.md", "PROTOCOL.ja.md"])
def test_protocol_states_the_config_values(name):
    """The pre-registered values appear in both language versions of the protocol."""
    text = (ROOT / name).read_text(encoding="utf-8")
    for value in ("20261030", "2,000", "0.02", "0.05", "frcnn-r50-fpn-512", "frcnn-r50-fpn-384",
                  "9ec285f5399c198c1d0cd23f1e5b3a9e9b99cee3ab18bccd473115fd4cb7040c"):
        assert value in text, f"{name} does not mention {value}"  # 2,000: bootstrap_resamples in prose
    assert not re.search(r"\bXX\b|TBD", text), f"{name} still has a placeholder"
