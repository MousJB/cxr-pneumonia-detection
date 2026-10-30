"""Prepares the RSNA training images: DICOM → 512 px PNG, boxes rescaled, split by real
patient, duplicate audit (see PROTOCOL.md).

- The RSNA ``patientId`` identifies an **image**. The real patient comes from RSNA's public
  mapping to the original NIH ChestX-ray14 images (``img_id`` = ``<NIH patient>_<n>.png``),
  downloaded from rsna.org and checked image by image through the DICOM ``SOPInstanceUID``.
- Split: by NIH patient, stratified on "the patient has at least one positive image",
  70 % train, 10 % validation, 20 % test, seed 20261030. No patient is in two sets.
- Duplicate audit: exact (SHA-256 of the decoded pixels) and near-duplicates (64-bit pHash,
  Hamming distance ≤ 4) with ``medvision_qa.duplicates``; pairs that cross sets are listed.

Outputs: ``data/png512/<patientId>.png``, ``data/images.csv`` (one row per image, with
DICOM metadata), ``data/boxes_512.csv``; versioned: ``metadata/split.csv`` (identifiers
and set only) and ``metadata/data_audit.json`` (aggregate counts).

    python scripts/prepare_rsna.py
"""

import csv
import hashlib
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

from medvision_qa.duplicates import exact_duplicate_groups, near_duplicate_pairs, phash

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PNG_DIR = ROOT / "data" / "png512"
METADATA = ROOT / "metadata"

SEED = 20261030
SIZE = 512
SPLIT_FRACTIONS = (("train", 0.7), ("val", 0.1))  # the rest: test
NEAR_DUPLICATE_DISTANCE = 4
MAPPING_URL = "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json"
MAPPING = ROOT / "data" / "nih_mapping" / "pneumonia-challenge-dataset-mappings_2018.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_mapping() -> dict[str, dict]:
    if not MAPPING.exists():
        MAPPING.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MAPPING_URL, MAPPING)
    entries = json.loads(MAPPING.read_text(encoding="utf-8"))
    by_id = {e["subset_img_id"]: e for e in entries}
    if len(by_id) != len(entries):
        sys.exit("mapping: duplicate subset_img_id")
    return by_id


def convert(patient_id: str) -> dict:
    """Reads one DICOM, writes the 512 px PNG, returns metadata and fingerprints."""
    ds = pydicom.dcmread(RAW / "stage_2_train_images" / f"{patient_id}.dcm")
    pixels = ds.pixel_array
    if pixels.dtype != np.uint8 or pixels.ndim != 2 or ds.PhotometricInterpretation != "MONOCHROME2":
        raise ValueError(f"{patient_id}: unexpected pixel format {pixels.dtype} {pixels.shape} "
                         f"{ds.PhotometricInterpretation}")
    image = Image.fromarray(pixels)
    small = image.resize((SIZE, SIZE), Image.Resampling.BOX)
    small.save(PNG_DIR / f"{patient_id}.png")
    tiny = np.asarray(image.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    return {
        "patient_id_tag": str(ds.PatientID),
        "sop_instance_uid": str(ds.SOPInstanceUID),
        "view": str(ds.get("ViewPosition", "")),
        "sex": str(ds.get("PatientSex", "")),
        "age": str(ds.get("PatientAge", "")),
        "rows": int(ds.Rows),
        "columns": int(ds.Columns),
        "pixel_sha256": sha256_bytes(pixels.tobytes() + str(pixels.shape).encode()),
        "phash": phash(tiny),
    }


def split_patients(patient_positive: dict[str, bool]) -> dict[str, str]:
    rng = np.random.default_rng(SEED)
    split = {}
    for stratum in (False, True):
        patients = sorted(p for p, pos in patient_positive.items() if pos == stratum)
        patients = [patients[i] for i in rng.permutation(len(patients))]
        start = 0
        for name, fraction in SPLIT_FRACTIONS:
            n = round(fraction * len(patients))
            for p in patients[start:start + n]:
                split[p] = name
            start += n
        for p in patients[start:]:
            split[p] = "test"
    return split


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    labels = read_rows(RAW / "stage_2_train_labels.csv")
    detailed = {r["patientId"]: r["class"] for r in read_rows(RAW / "stage_2_detailed_class_info.csv")}
    mapping = load_mapping()
    ids = sorted({r["patientId"] for r in labels})
    boxes = defaultdict(list)
    for r in labels:
        if r["Target"] == "1":
            boxes[r["patientId"]].append(tuple(float(r[k]) for k in ("x", "y", "width", "height")))

    PNG_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(convert, ids, chunksize=64))

    rows, problems = [], defaultdict(list)
    for pid, res in zip(ids, results):
        entry = mapping.get(pid)
        if entry is None:
            problems["not_in_mapping"].append(pid)
            continue
        if res["sop_instance_uid"] != entry["SOPInstanceUID"]:
            problems["sop_instance_uid_mismatch"].append(pid)
        if res["patient_id_tag"] != pid:
            problems["dicom_patient_id_differs_from_file_name"].append(pid)
        if (res["rows"], res["columns"]) != (1024, 1024):
            problems["not_1024x1024"].append(pid)
        rows.append({
            "case_id": pid,
            "nih_image": entry["img_id"],
            "nih_patient": entry["img_id"].split("_")[0],
            "target": int(pid in boxes),
            "n_boxes": len(boxes[pid]),
            "detailed_class": detailed[pid],
            "nih_labels": "|".join(entry["orig_labels"]),
            **{k: res[k] for k in ("view", "sex", "age", "rows", "columns", "pixel_sha256")},
        })
    if any(problems.values()):
        sys.exit("join problems: " + json.dumps({k: [len(v), v[:3]] for k, v in problems.items()}))

    patient_positive = defaultdict(bool)
    for r in rows:
        patient_positive[r["nih_patient"]] |= bool(r["target"])
    split = split_patients(patient_positive)
    for r in rows:
        r["split"] = split[r["nih_patient"]]

    scale = SIZE / 1024
    box_rows = [
        {"case_id": pid, "x": x * scale, "y": y * scale, "width": w * scale, "height": h * scale}
        for pid in ids for (x, y, w, h) in boxes[pid]
    ]
    write_csv(ROOT / "data" / "images.csv", list(rows[0]), rows)
    write_csv(ROOT / "data" / "boxes_512.csv", ["case_id", "x", "y", "width", "height"], box_rows)
    METADATA.mkdir(exist_ok=True)
    write_csv(METADATA / "split.csv", ["case_id", "nih_patient", "split"], rows)

    # Duplicate audit
    digests = [r["pixel_sha256"] for r in rows]
    hashes = [res["phash"] for res in results]
    exact = exact_duplicate_groups(digests)
    near = near_duplicate_pairs(hashes, max_distance=NEAR_DUPLICATE_DISTANCE, block=1024)

    def describe(i: int, j: int, d: int | None = None) -> dict:
        a, b = rows[i], rows[j]
        out = {"a": a["case_id"], "b": b["case_id"], "same_patient": a["nih_patient"] == b["nih_patient"],
               "splits": f'{a["split"]}/{b["split"]}', "targets": f'{a["target"]}/{b["target"]}'}
        if d is not None:
            out["distance"] = d
        return out

    exact_pairs = [describe(g[0], k) for g in exact for k in g[1:]]
    near_described = [describe(i, j, d) for i, j, d in near]
    ages = [int(r["age"]) for r in rows if r["age"].isdigit()]
    splits_of_patient = defaultdict(set)
    for r in rows:
        splits_of_patient[r["nih_patient"]].add(r["split"])
    audit = {
        "n_images": len(rows),
        "patients_in_several_splits": sum(len(s) > 1 for s in splits_of_patient.values()),
        "n_nih_patients": len(patient_positive),
        "images_per_patient": dict(sorted(Counter(Counter(r["nih_patient"] for r in rows).values()).items())),
        "patients_per_split": dict(sorted(Counter(split.values()).items())),
        "positive_patients_per_split": dict(sorted(Counter(split[p] for p, pos in patient_positive.items() if pos).items())),
        "images_per_split": dict(sorted(Counter(r["split"] for r in rows).items())),
        "positive_images_per_split": dict(sorted(Counter(r["split"] for r in rows if r["target"]).items())),
        "boxes_per_split": dict(sorted(Counter(r["split"] for r in rows for _ in range(r["n_boxes"])).items())),
        "detailed_class_per_split": dict(sorted(Counter(f'{r["split"]}/{r["detailed_class"]}' for r in rows).items())),
        "view_per_split": dict(sorted(Counter(f'{r["split"]}/{r["view"]}' for r in rows).items())),
        "sex_per_split": dict(sorted(Counter(f'{r["split"]}/{r["sex"]}' for r in rows).items())),
        "age": {"min": min(ages), "max": max(ages), "median": float(np.median(ages)),
                "over_100": sum(a > 100 for a in ages), "not_numeric": len(rows) - len(ages)},
        "exact_pixel_duplicate_pairs": exact_pairs,
        "near_duplicate_threshold": NEAR_DUPLICATE_DISTANCE,
        "near_duplicate_pairs_total": len(near),
        "near_duplicate_pairs_same_patient": sum(p["same_patient"] for p in near_described),
        "near_duplicate_pairs_across_splits": [p for p in near_described if p["splits"].split("/")[0] != p["splits"].split("/")[1]],
        "mapping_sha256": sha256_bytes(MAPPING.read_bytes()),
        "split_csv_sha256": sha256_bytes((METADATA / "split.csv").read_bytes()),
    }
    (METADATA / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n")
    for key, value in audit.items():
        if not isinstance(value, list):
            print(f"{key}: {value}")
    print(f"exact duplicate pairs: {len(exact_pairs)}; near-duplicate pairs across splits: "
          f"{len(audit['near_duplicate_pairs_across_splits'])}")


if __name__ == "__main__":
    main()
