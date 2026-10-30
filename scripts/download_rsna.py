"""Downloads the RSNA Pneumonia Detection Challenge files from Kaggle and checks them.

Requires a Kaggle account that has accepted the competition rules, and Kaggle API
credentials (``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME``/``KAGGLE_KEY``
environment variables). The rules forbid redistributing the images: everything stays in
``data/``, which git ignores.

After the download, the script checks the volumes (number of DICOM files, label rows,
consistency between the label tables and the image folder) and writes a manifest
(relative path, size, SHA-256 of every file) to ``metadata/raw_manifest.csv`` (file
names and hashes only, no image content); its own SHA-256 is printed and stored in
``metadata/raw_summary.json``.

    python scripts/download_rsna.py            # download (if needed) + checks + manifest
    python scripts/download_rsna.py --no-download   # checks + manifest only

If the archive was downloaded by hand from the Kaggle website and extracted into
``data/raw``, ``--verify-zip data/rsna-pneumonia-detection-challenge.zip`` also checks every
extracted file against the size and CRC-32 stored in the archive.
"""

import argparse
import csv
import hashlib
import json
import sys
import zipfile
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "rsna-pneumonia-detection-challenge"
RAW = ROOT / "data" / "raw"
METADATA = ROOT / "metadata"

TRAIN_IMAGES = "stage_2_train_images"
TEST_IMAGES = "stage_2_test_images"
LABELS = "stage_2_train_labels.csv"
CLASS_INFO = "stage_2_detailed_class_info.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download() -> None:
    import kagglehub

    RAW.mkdir(parents=True, exist_ok=True)
    path = kagglehub.competition_download(COMPETITION, output_dir=str(RAW))
    print(f"competition files in: {path}")


def crc32_file(path: Path) -> int:
    crc = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            crc = zlib.crc32(chunk, crc)
    return crc


def verify_zip(archive: Path) -> dict:
    """Checks that data/raw holds exactly the archive members, with the same size and CRC-32."""
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
    mismatched = []
    for i, m in enumerate(members, 1):
        target = RAW / m.filename
        if not target.is_file() or target.stat().st_size != m.file_size or crc32_file(target) != m.CRC:
            mismatched.append(m.filename)
        if i % 5000 == 0:
            print(f"  checked {i}/{len(members)} archive members", flush=True)
    extra = sorted(
        {p.relative_to(RAW).as_posix() for p in RAW.rglob("*") if p.is_file()}
        - {m.filename for m in members}
    )
    if mismatched or extra:
        sys.exit(f"archive check failed: {len(mismatched)} missing or different "
                 f"(e.g. {mismatched[:3]}), {len(extra)} extra (e.g. {extra[:3]})")
    return {"archive": archive.name, "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256_file(archive), "members_checked": len(members)}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_volumes() -> dict:
    for name in (TRAIN_IMAGES, TEST_IMAGES, LABELS, CLASS_INFO):
        if not (RAW / name).exists():
            sys.exit(f"missing in {RAW}: {name}")

    train_ids = {p.stem for p in (RAW / TRAIN_IMAGES).glob("*.dcm")}
    test_ids = {p.stem for p in (RAW / TEST_IMAGES).glob("*.dcm")}
    labels = read_rows(RAW / LABELS)
    class_info = read_rows(RAW / CLASS_INFO)

    label_ids = {r["patientId"] for r in labels}
    positive_ids = {r["patientId"] for r in labels if r["Target"] == "1"}
    boxes = [r for r in labels if r["Target"] == "1"]
    if any(not r["x"] for r in boxes) or any(r["x"] for r in labels if r["Target"] == "0"):
        sys.exit("inconsistent rows: Target=1 without a box, or Target=0 with a box")

    info_by_id: dict[str, set[str]] = {}
    for r in class_info:
        info_by_id.setdefault(r["patientId"], set()).add(r["class"])
    conflicting = sorted(i for i, classes in info_by_id.items() if len(classes) > 1)
    image_class = {i: next(iter(c)) for i, c in info_by_id.items() if len(c) == 1}

    problems = {
        "label_ids_without_dicom": sorted(label_ids - train_ids),
        "dicom_without_label": sorted(train_ids - label_ids),
        "class_info_ids_not_in_labels": sorted(set(info_by_id) - label_ids),
        "class_info_conflicting_ids": conflicting,
        "positive_but_class_not_lung_opacity": sorted(
            i for i in positive_ids if image_class.get(i) != "Lung Opacity"
        ),
        "lung_opacity_but_negative": sorted(
            i for i, c in image_class.items() if c == "Lung Opacity" and i not in positive_ids
        ),
        "train_test_overlap": sorted(train_ids & test_ids),
    }
    summary = {
        "competition": COMPETITION,
        "train_dicom_files": len(train_ids),
        "test_dicom_files": len(test_ids),
        "label_rows": len(labels),
        "labelled_images": len(label_ids),
        "positive_images": len(positive_ids),
        "negative_images": len(label_ids - positive_ids),
        "boxes": len(boxes),
        "boxes_per_positive_image": dict(
            sorted(Counter(Counter(r["patientId"] for r in boxes).values()).items())
        ),
        "detailed_class_counts": dict(sorted(Counter(image_class.values()).items())),
        "problems": {k: len(v) for k, v in problems.items()},
    }
    for name, ids in problems.items():
        if ids:
            print(f"WARNING {name}: {len(ids)} (e.g. {ids[:3]})")
    return summary


def write_manifest() -> str:
    files = sorted(
        p for p in RAW.rglob("*")
        if p.is_file() and not p.name.startswith(".") and ".kaggle" not in p.parts
    )
    METADATA.mkdir(exist_ok=True)
    target = METADATA / "raw_manifest.csv"
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["path", "bytes", "sha256"])
        for i, p in enumerate(files, 1):
            writer.writerow([p.relative_to(RAW).as_posix(), p.stat().st_size, sha256_file(p)])
            if i % 5000 == 0:
                print(f"  hashed {i}/{len(files)} files", flush=True)
    return sha256_file(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-download", action="store_true", help="only check data/raw")
    parser.add_argument("--verify-zip", type=Path, help="archive downloaded by hand, to check against")
    args = parser.parse_args()
    if not args.no_download:
        download()
    archive = verify_zip(args.verify_zip) if args.verify_zip else None
    summary = check_volumes()
    if archive:
        summary["archive_check"] = archive
    summary["manifest_sha256"] = write_manifest()
    METADATA.mkdir(exist_ok=True)
    (METADATA / "raw_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
