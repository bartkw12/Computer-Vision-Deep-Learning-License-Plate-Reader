"""Draw the README figures from the files in ``results/``.

Usage::

    python scripts/make_figures.py           # writes assets/*.png

Like the tables (``scripts/render_readme_tables.py``), every figure is drawn
from a results file, never from typed-in numbers. A figure whose inputs do not
exist yet is skipped with a log line.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")  # files only; no display needed
import matplotlib.pyplot as plt

from alpr.evaluation.images import load_openalpr
from alpr.geometry import crop

logger = logging.getLogger("figures")

# Chart chrome and the first three categorical slots of the reference palette
# (the dataviz guide's default, validated for up to three series at once).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")

#: The readers compared side by side, in fixed color order.
READERS = (
    ("easyocr", "none", "EasyOCR"),
    ("easyocr", "otsu_inv", "EasyOCR + Otsu (v1)"),
    ("fast-plate-ocr:cct-s-v2-global-model", "none", "fast-plate-ocr"),
)

_FORMAT_LABELS = {
    "off": "no format",
    "strict": "exact match",
    "lookalike-1": "fix ≤ 1",
    "lookalike-2": "fix ≤ 2",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _variant(stills: Mapping[str, Any], ocr: str, preprocess: str) -> Mapping[str, Any] | None:
    for variant in stills["variants"]:
        if (variant["ocr"], variant["preprocess"]) == (ocr, preprocess):
            return variant
    return None


def _featured(stills: Mapping[str, Any]) -> list[tuple[str, str, Mapping[str, Any]]]:
    """``(label, color, variant)`` for each featured reader present in the results."""
    found = []
    for (ocr, preprocess, label), color in zip(READERS, SERIES, strict=True):
        variant = _variant(stills, ocr, preprocess)
        if variant is not None:
            found.append((label, color, variant))
    return found


def ocr_variants(stills: Mapping[str, Any], out: Path) -> None:
    """Exact match with 95% intervals for every OCR variant, best at the top."""
    variants = sorted(stills["variants"], key=lambda v: v["overall"]["accuracy"])
    labels = [
        f"{v['ocr'].replace(':cct-', ' cct-').replace('-global-model', '')}  ·  {v['preprocess']}"
        + ("  (v1)" if (v["ocr"], v["preprocess"]) == ("easyocr", "otsu_inv") else "")
        for v in variants
    ]
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(variants) + 1.4))
    for row, variant in enumerate(variants):
        score = variant["overall"]
        ax.plot([100 * score["ci_low"], 100 * score["ci_high"]], [row, row], color=SERIES[0], lw=2)
        ax.plot(100 * score["accuracy"], row, "o", color=SERIES[0], ms=7, mec=SURFACE, mew=1.5)
        ax.annotate(
            f"{100 * score['accuracy']:.1f}%",
            (100 * score["ci_high"], row),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
            fontsize=9,
        )
    ax.set_yticks(range(len(variants)), labels)
    ax.set_xlim(0, 105)
    ax.set_xlabel("exact match, % of plates (line: 95% Wilson interval)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Reading {stills['plates']} labeled OpenALPR plate crops")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _lines_by_category(
    readers: list[tuple[str, str, Mapping[str, Any]]],
    categories: list[str],
    values: list[list[float]],
    *,
    title: str,
    xlabel: str,
    out: Path,
) -> None:
    """One line per reader over ordered categories, direct-labeled at the right end."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for (label, color, _), series in zip(readers, values, strict=True):
        ax.plot(
            categories, series, "-o", color=color, lw=2, ms=7, mec=SURFACE, mew=1.5, label=label
        )
        ax.annotate(
            f"{label}  {series[-1]:.0f}%",
            (len(categories) - 1, series[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
            fontsize=9,
        )
    ax.set_ylim(0, 105)
    ax.set_ylabel("exact match, %")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.3, len(categories) - 0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(readers), fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.subplots_adjust(right=0.7)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def accuracy_by_height(stills: Mapping[str, Any], out: Path) -> None:
    """Exact match per plate-height bin for the featured readers."""
    readers = _featured(stills)
    bins = list(readers[0][2]["by_plate_height"])
    counts = [readers[0][2]["by_plate_height"][b]["n"] for b in bins]
    values = [[100 * v["by_plate_height"][b]["accuracy"] for b in bins] for _, _, v in readers]
    _lines_by_category(
        readers,
        [f"{b}\n(n={n})" for b, n in zip(bins, counts, strict=True)],
        values,
        title="Exact match by plate height",
        xlabel="plate height in the image",
        out=out,
    )


def format_correction(stills: Mapping[str, Any], out: Path) -> None:
    """Exact match on the Brazilian plates under each format rule."""
    readers = _featured(stills)
    modes = [row["mode"] for row in readers[0][2]["format_correction_br"]]
    values = [
        [
            100 * next(r for r in v["format_correction_br"] if r["mode"] == m)["accuracy"]
            for m in modes
        ]
        for _, _, v in readers
    ]
    plates = readers[0][2]["format_correction_br"][0]["n"]
    _lines_by_category(
        readers,
        [_FORMAT_LABELS.get(m, m) for m in modes],
        values,
        title=f"Format-aware correction, {plates} Brazilian plates",
        xlabel="what is done with the plate format LLLDDDD",
        out=out,
    )


def failure_gallery(
    records: Mapping[str, list[Mapping[str, Any]]],
    openalpr: Path,
    out: Path,
    reader: str = "fast-plate-ocr:cct-s-v2-global-model|none",
    count: int = 12,
) -> None:
    """Plate crops the best reader got wrong, with what it read."""
    boxes = {(i.region, i.path.name): i for i in load_openalpr(openalpr)}
    misses = [r for r in records[reader] if r["prediction"] != r["truth"]]
    name = reader.split("|")[0].split(":")[0]
    failures = misses[:count]
    if not failures:
        logger.info("no failures to show for %s", reader)
        return
    columns = 3
    rows = -(-len(failures) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(9, 1.45 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, record in zip(axes.flat, failures, strict=False):
        item = boxes[(record["region"], record["image"])]
        image = cv2.imread(str(item.path))
        plate = cv2.cvtColor(crop(image, item.box), cv2.COLOR_BGR2RGB)
        ax.imshow(plate)
        ax.set_title(
            f"true {record['truth']}   read {record['prediction'] or '(nothing)'}",
            fontsize=9,
            fontweight="normal",
            color=INK_SECONDARY,
            loc="center",
        )
    fig.suptitle(
        f"{len(failures)} of the {len(misses)} plates {name} misread",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    """Draw every figure whose inputs exist."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--records", type=Path, default=Path("outputs/ablation_ocr_stills_records.json")
    )
    parser.add_argument("--openalpr", type=Path, default=Path("data/openalpr"))
    parser.add_argument("--out", type=Path, default=Path("assets"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _style()
    args.out.mkdir(parents=True, exist_ok=True)

    stills_path = args.results / "ablation_ocr_stills.json"
    if stills_path.exists():
        stills = json.loads(stills_path.read_text())
        for name, draw in (
            ("ocr_variants.png", ocr_variants),
            ("accuracy_by_plate_height.png", accuracy_by_height),
            ("format_correction_br.png", format_correction),
        ):
            draw(stills, args.out / name)
            logger.info("-> %s", args.out / name)
    else:
        logger.info("skipping OCR figures: %s does not exist", stills_path)

    if args.records.exists() and args.openalpr.exists():
        failure_gallery(
            json.loads(args.records.read_text()), args.openalpr, args.out / "failures.png"
        )
        logger.info("-> %s", args.out / "failures.png")
    else:
        logger.info("skipping failure gallery: needs %s and %s", args.records, args.openalpr)


if __name__ == "__main__":
    main()
