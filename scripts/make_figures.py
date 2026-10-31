"""Aggregate figures of the README from the committed results (no image data, no dependency).

    python scripts/make_figures.py   # -> results/figures/*.svg
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"A": "#2a78d6", "B": "#eb6834"}  # categorical slots 1 and 2
LABEL = {"A": "A, 512 px", "B": "B, 384 px"}
FONT = "font-family='system-ui, -apple-system, Segoe UI, sans-serif'"


def text(x, y, s, size=12, color=INK_2, anchor="start", weight="normal"):
    return (f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' fill='{color}' "
            f"text-anchor='{anchor}' font-weight='{weight}'>{s}</text>")


def svg(width, height, title, body):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
            f"viewBox='0 0 {width} {height}' role='img' aria-label='{title}'>"
            f"<rect width='100%' height='100%' fill='{SURFACE}'/>" + text(16, 26, title, 15, INK, weight="600")
            + "".join(body) + "</svg>\n")


def validation_curves(training: dict) -> str:
    w, h, left, right, top, bottom = 640, 340, 56, 110, 52, 44
    lo, hi = 0.18, 0.24
    x = lambda e: left + (e - 1) / 9 * (w - left - right)  # noqa: E731
    y = lambda v: top + (hi - v) / (hi - lo) * (h - top - bottom)  # noqa: E731
    body = []
    for v in (0.18, 0.20, 0.22, 0.24):
        body += [f"<line x1='{left}' x2='{w - right}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke='{GRID}'/>",
                 text(left - 8, y(v) + 4, f"{v:.2f}", 11, anchor="end")]
    for e in range(1, 11):
        body.append(text(x(e), h - bottom + 18, str(e), 11, anchor="middle"))
    body.append(text((left + w - right) / 2, h - 8, "Epoch", 12, anchor="middle"))
    body.append(text(16, 44, "Validation competition metric per epoch (one run per model)", 12))
    for name in ("A", "B"):
        vals = training[name]["val_metric_per_epoch"]
        pts = " ".join(f"{x(i + 1):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        body.append(f"<polyline points='{pts}' fill='none' stroke='{COLOR[name]}' stroke-width='2'/>")
        for i, v in enumerate(vals):
            best = i + 1 == training[name]["best_epoch"]
            body.append(f"<circle cx='{x(i + 1):.1f}' cy='{y(v):.1f}' r='{5 if best else 4}' fill='{COLOR[name]}' "
                        f"stroke='{SURFACE}' stroke-width='2'><title>{LABEL[name]}, epoch {i + 1}: {v:.4f}"
                        f"{' (best)' if best else ''}</title></circle>")
        b = training[name]["best_epoch"]
        body.append(text(x(10) + 10, y(vals[-1]) + 4 + (8 if name == "A" else -8), LABEL[name], 12, INK))
        body.append(text(x(b), y(vals[b - 1]) + (20 if name == "A" else -10), f"best {vals[b - 1]:.3f}", 11,
                         anchor="middle"))
    body += legend(w)
    return svg(w, h, "Training: validation metric", body)


def legend(w):
    out = []
    for j, name in enumerate(("A", "B")):
        lx = w - 200 + j * 95
        out += [f"<rect x='{lx}' y='16' width='12' height='12' rx='2' fill='{COLOR[name]}'/>",
                text(lx + 18, 26, LABEL[name], 12, INK)]
    return out


def test_differences(boot: dict) -> str:
    rows = [("Competition metric", "rsna_metric", 0.02), ("Image AUROC", "auroc_image", 0.02),
            ("Sensitivity", "sensitivity", 0.05), ("Specificity", "specificity", 0.05),
            ("Box recall", "box_recall", None), ("Box precision", "box_precision", None),
            ("Small-opacity recall", "small_box_recall", None)]
    w, left, right, top, step = 640, 170, 30, 70, 34
    h = top + step * len(rows) + 46
    lo, hi = -0.06, 0.06
    x = lambda v: left + (v - lo) / (hi - lo) * (w - left - right)  # noqa: E731
    d = boot["differences"]["B - A"]
    body = [text(16, 44, "Test split, B − A with 95 % CI (patient bootstrap); | = pre-registered margin", 12)]
    for v in (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06):
        body += [f"<line x1='{x(v):.1f}' x2='{x(v):.1f}' y1='{top - 10}' y2='{h - 40}' "
                 f"stroke='{INK_2 if v == 0 else GRID}'/>",
                 text(x(v), h - 24, f"{v:+.2f}".replace("+0.00", "0"), 11, anchor="middle")]
    body.append(text((left + w - right) / 2, h - 6, "Difference B − A (positive = B higher)", 12, anchor="middle"))
    for i, (label, key, margin) in enumerate(rows):
        cy = top + step * i + 8
        m = d[key]
        body.append(text(left - 12, cy + 4, label, 12, INK, anchor="end"))
        if margin is not None:
            body.append(f"<line x1='{x(-margin):.1f}' x2='{x(-margin):.1f}' y1='{cy - 9}' y2='{cy + 9}' "
                        f"stroke='{INK}' stroke-width='2'/>")
        body.append(f"<line x1='{x(m['ci'][0]):.1f}' x2='{x(m['ci'][1]):.1f}' y1='{cy}' y2='{cy}' "
                    f"stroke='{COLOR['B']}' stroke-width='2' stroke-linecap='round'/>")
        body.append(f"<circle cx='{x(m['value']):.1f}' cy='{cy}' r='5' fill='{COLOR['B']}' stroke='{SURFACE}' "
                    f"stroke-width='2'><title>{label}: {m['value']:+.4f} [{m['ci'][0]:+.4f}, {m['ci'][1]:+.4f}]"
                    f"</title></circle>")
    return svg(w, h, "Test: B (384 px) minus A (512 px)", body)


def recall_by_size(analysis: dict) -> str:
    sizes = ("small", "medium", "large")
    w, h, left, top, bottom = 640, 300, 56, 70, 44
    y = lambda v: top + (1 - v) * (h - top - bottom)  # noqa: E731
    group, bar = (w - left - 30) / 3, 40
    body = [text(16, 44, "Test split, box recall at IoU > 0.4 by opacity size (terciles of training boxes)", 12)]
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        body += [f"<line x1='{left}' x2='{w - 30}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke='{GRID}'/>",
                 text(left - 8, y(v) + 4, f"{v:.2f}", 11, anchor="end")]
    for i, size in enumerate(sizes):
        cx = left + group * (i + 0.5)
        body.append(text(cx, h - bottom + 18, f"{size} ({analysis['A']['opacity_size_box_recall'][size]['boxes']} "
                                               "boxes)", 12, anchor="middle"))
        for j, name in enumerate(("A", "B")):
            r = analysis[name]["opacity_size_box_recall"][size]["recall"]
            bx = cx - bar - 1 + j * (bar + 2)
            body.append(f"<path d='M{bx:.1f},{y(0):.1f} V{y(r) + 4:.1f} q0,-4 4,-4 H{bx + bar - 4:.1f} q4,0 4,4 "
                        f"V{y(0):.1f} Z' fill='{COLOR[name]}'><title>{LABEL[name]}, {size}: {r:.3f}</title></path>")
            body.append(text(bx + bar / 2, y(r) - 6, f"{r:.2f}", 11, INK, anchor="middle"))
    body += legend(w)
    return svg(w, h, "Test: recall by opacity size", body)


def main() -> None:
    selection = json.loads((ROOT / "results" / "validation_selection.json").read_text(encoding="utf-8"))
    test = json.loads((ROOT / "results" / "test" / "test_results.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    figures = {"validation_curves.svg": validation_curves(selection["training"]),
               "test_differences.svg": test_differences(test["bootstrap"]),
               "test_recall_by_size.svg": recall_by_size(test["error_analysis"])}
    for name, content in figures.items():
        (OUT / name).write_text(content, encoding="utf-8", newline="\n")
        print(OUT / name)


if __name__ == "__main__":
    main()
