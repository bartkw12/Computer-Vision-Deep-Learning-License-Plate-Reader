"""Render the README's result tables from the files in ``results/``.

Usage::

    python scripts/render_readme_tables.py           # rewrite README.md in place
    python scripts/render_readme_tables.py --check   # exit 1 if README.md is stale

No number in the README is typed by hand. Each table sits between two markers::

    <!-- results:NAME -->
    ...generated...
    <!-- /results:NAME -->

and is regenerated from the results files listed for ``NAME`` below. While
none of those files exists yet, the block says which command produces them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Results = Mapping[str, Any]

BLOCK = re.compile(
    r"(?P<open><!-- results:(?P<name>[\w-]+) -->\n)"
    r"(?P<body>.*?)"
    r"(?P<close><!-- /results:(?P=name) -->)",
    re.DOTALL,
)

#: (backend, preprocessing) of the readers singled out in tables.
EASYOCR = ("easyocr", "none")
V1_READER = ("easyocr", "otsu_inv")  # what the first version of this project ran
PLATE_READER = ("fast-plate-ocr:cct-s-v2-global-model", "none")

STILLS = "ablation_ocr_stills.json"
LOOKALIKES = "lookalike_evidence.json"
TRAINING = "detector_training_original.json"
LEAKAGE = "dataset_leakage.json"
#: ``alpr eval-images --plate-weights ...`` output, one file per reader.
ENDTOEND_FILES = ("endtoend_stills.json", "endtoend_stills_easyocr.json")
DETECTOR_FILES = {
    "detector_original_test.json": ("original", "test"),
    "detector_original_val.json": ("original", "val"),
    "detector_reproduced_test.json": ("reproduced", "test"),
    "detector_reproduced_val.json": ("reproduced", "val"),
}


# --- formatting -------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _with_ci(score: Mapping[str, Any]) -> str:
    return f"{_pct(score['accuracy'])} ({100 * score['ci_low']:.0f}-{100 * score['ci_high']:.0f})"


def _table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    """A Markdown table; ``align`` has one ``l``/``r`` per column (default: first left)."""
    align = align or "l" + "r" * (len(header) - 1)
    rule = ["---" if a == "l" else "---:" for a in align]
    lines = [header, rule, *rows]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)


def _reader_name(variant: Mapping[str, Any]) -> str:
    return f"`{variant['ocr']}`"


def _find_variant(stills: Results, key: tuple[str, str]) -> Mapping[str, Any] | None:
    for variant in stills["variants"]:
        if (variant["ocr"], variant["preprocess"]) == key:
            return variant
    return None


def featured(stills: Results) -> list[tuple[str, Mapping[str, Any]]]:
    """The readers compared side by side, with column labels."""
    labels = {EASYOCR: "EasyOCR", V1_READER: "EasyOCR + Otsu (v1)", PLATE_READER: "fast-plate-ocr"}
    found = [(label, _find_variant(stills, key)) for key, label in labels.items()]
    return [(label, variant) for label, variant in found if variant is not None]


# --- renderers --------------------------------------------------------------------


def render_headline(files: Mapping[str, Results]) -> str:
    """The top-of-README summary: best reader on stills, detector on test."""
    lines = []
    stills = files.get(STILLS)
    if stills:
        best = max(stills["variants"], key=lambda v: v["overall"]["accuracy"])
        v1 = _find_variant(stills, V1_READER)
        line = (
            f"- **Plate reading** on {stills['plates']} labeled OpenALPR plates: "
            f"{_reader_name(best)} reads **{_with_ci(best['overall'])}** exactly"
        )
        if v1:
            line += f", against {_with_ci(v1['overall'])} for the first version's EasyOCR + Otsu"
        lines.append(line + ". 95% Wilson intervals in brackets.")
    test = files.get("detector_original_test.json")
    if test:
        lines.append(
            f"- **Plate detection** on the held-out test split ({test['images']} images): "
            f"mAP50 **{test['mAP50']:.3f}**, mAP50-95 {test['mAP50-95']:.3f}."
        )
    return "\n".join(lines)


def render_ocr_stills(files: Mapping[str, Results]) -> str:
    """Every OCR variant on the labeled plate boxes (ablations A1, A2)."""
    stills = files[STILLS]
    regions = sorted(stills["variants"][0]["by_region"])
    counts = ", ".join(
        f"{region.upper()} {stills['variants'][0]['by_region'][region]['n']}" for region in regions
    )
    best = max(v["overall"]["accuracy"] for v in stills["variants"])
    rows = []
    for variant in stills["variants"]:
        overall = variant["overall"]
        exact = _with_ci(overall)
        if overall["accuracy"] == best:
            exact = f"**{exact}**"
        preprocess = f"`{variant['preprocess']}`"
        if (variant["ocr"], variant["preprocess"]) == V1_READER:
            preprocess += " (v1)"
        rows.append(
            [
                _reader_name(variant),
                preprocess,
                exact,
                f"{overall['cer']:.3f}",
                *(_pct(variant["by_region"][region]["accuracy"]) for region in regions),
                f"{variant['ms_per_plate']:.0f}",
            ]
        )
    header = ["Reader", "Preprocessing", "Exact match (95% CI)", "CER"]
    header += [region.upper() for region in regions] + ["ms / plate"]
    return (
        f"{stills['plates']} plates ({counts}), read from their labeled boxes, so the detector "
        "plays no part. Exact match compares the whole normalized serial; CER is the "
        "character error rate. Latency is on "
        f"{stills['environment'].get('device', 'cpu')} ({stills['environment']['cpu']}).\n\n"
        + _table(header, rows, "ll" + "r" * (len(header) - 2))
    )


_FORMAT_MODES = {
    "off": "no format: raw text",
    "strict": "exact `LLLDDDD` match only",
    "lookalike-1": "look-alike correction, ≤ 1 character",
    "lookalike-2": "look-alike correction, ≤ 2 characters",
}


def render_format_br(files: Mapping[str, Results]) -> str:
    """Format-aware correction on the Brazilian plates (ablation A3)."""
    readers = featured(files[STILLS])
    modes = [row["mode"] for row in readers[0][1]["format_correction_br"]]
    rows = []
    for mode in modes:
        cells = []
        for _, variant in readers:
            row = next(r for r in variant["format_correction_br"] if r["mode"] == mode)
            cells.append(f"{_pct(row['accuracy'])} ({row['emitted']}/{row['n']})")
        rows.append([_FORMAT_MODES.get(mode, mode), *cells])
    plates = readers[0][1]["format_correction_br"][0]["n"]
    return (
        f"Exact match on the {plates} Brazilian plates, which all share the format `LLLDDDD`. "
        "In brackets: how many plates got a read at all; a rejected read counts as wrong.\n\n"
        + _table(["Format rule", *(label for label, _ in readers)], rows)
    )


def render_plate_height(files: Mapping[str, Results]) -> str:
    """Exact match by plate height in pixels."""
    readers = featured(files[STILLS])
    bins = list(readers[0][1]["by_plate_height"])
    rows = []
    for name in bins:
        n = readers[0][1]["by_plate_height"][name]["n"]
        cells = [_pct(variant["by_plate_height"][name]["accuracy"]) for _, variant in readers]
        rows.append([name, str(n), *cells])
    return _table(["Plate height", "Plates", *(label for label, _ in readers)], rows)


def render_confusions(files: Mapping[str, Results], top: int = 8) -> str:
    """The commonest character substitutions per reader."""
    readers = featured(files[STILLS])
    columns = [
        [f"`{c['truth']}` → `{c['predicted']}` ({c['count']})" for c in v["top_confusions"][:top]]
        for _, v in readers
    ]
    depth = max(len(column) for column in columns)
    rows = [
        [str(i + 1), *(column[i] if i < len(column) else "" for column in columns)]
        for i in range(depth)
    ]
    return (
        f"Most frequent character substitutions, true → read (count), over all "
        f"{files[STILLS]['plates']} plates.\n\n"
        + _table(["#", *(label for label, _ in readers)], rows, "r" + "l" * len(readers))
    )


def _evidence_rows(evidence: list[Mapping[str, Any]], min_count: int) -> list[list[str]]:
    return [
        [
            f"`{row['read']}`",
            f"`{row['top_truth']}`",
            f"{row['count']}/{row['of']}",
            _pct(row["share"]),
            "yes" if row["accepted"] else "no",
        ]
        for row in evidence
        if row["of"] >= min_count
    ]


def render_lookalikes(files: Mapping[str, Results]) -> str:
    """The measured evidence behind the look-alike correction tables."""
    evidence = files[LOOKALIKES]
    method = evidence["method"]
    header = ["Read", "Most often truly", "Count", "Share", "Corrected?"]
    rule = (
        f"Evidence: {', '.join(r.upper() for r in method['evidence_regions'])} plates, "
        f"{len(method['readers'])} readers pooled; {method['held_out_region'].upper()} held out. "
        f"A correction is used when the character was seen at least {method['min_count']} times "
        f"in the wrong class and one answer accounts for at least {_pct(method['min_share'])}."
    )
    return (
        rule
        + "\n\n**A letter where a digit belongs**\n\n"
        + _table(header, _evidence_rows(evidence["digit_evidence"], method["min_count"]), "llrrl")
        + "\n\n**A digit where a letter belongs** (Ontario passenger series: no G, I, O, Q, U)\n\n"
        + _table(
            header,
            _evidence_rows(evidence["letter_evidence"]["GIOQU"], method["min_count"]),
            "llrrl",
        )
    )


_HYPERPARAMETERS = ("model", "epochs", "batch", "imgsz", "optimizer", "lr0", "momentum")
_HYPERPARAMETERS += ("weight_decay", "seed")


def render_training(files: Mapping[str, Results]) -> str:
    """What the original training run used, recovered from its files."""
    run = files[TRAINING]
    args = run["args"]
    rows = [[f"`{key}`", f"`{args[key]}`"] for key in _HYPERPARAMETERS if key in args]
    checkpoint = run.get("checkpoint", {})
    facts = [f"{run['epochs_run']} epochs"]
    if "training_hours" in run:
        facts.append(f"{run['training_hours']:.1f} h of training")
    if checkpoint.get("ultralytics_version"):
        facts.append(f"Ultralytics {checkpoint['ultralytics_version']}")
    if checkpoint.get("date"):
        facts.append(f"checkpoint saved {checkpoint['date'][:10]}")
    text = "Recovered from the run's `args.yaml`, `results.csv` and checkpoint: "
    text += ", ".join(facts) + ".\n\n" + _table(["Argument", "Value"], rows, "ll")
    differences = run.get("config_differences", [])
    if differences:
        text += "\n\n`configs/train_plate_detector.yaml` differs from the run in: " + ", ".join(
            f"`{d['key']}` (run `{d['run']}`, config `{d['config']}`)" for d in differences
        )
    return text


def render_detector(files: Mapping[str, Results]) -> str:
    """Detector precision, recall and mAP per weights and split."""
    rows = []
    for name, (weights, split) in DETECTOR_FILES.items():
        metrics = files.get(name)
        if metrics is None:
            continue
        label = f"{weights}, **{split}**" if split == "test" else f"{weights}, {split}"
        rows.append(
            [
                label,
                str(metrics["images"]),
                str(metrics["instances"]),
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['mAP50']:.3f}",
                f"{metrics['mAP50-95']:.3f}",
            ]
        )
    header = ["Weights, split", "Images", "Plates", "Precision", "Recall", "mAP50", "mAP50-95"]
    note = (
        "The validation split chose the best checkpoint during training, so its numbers are "
        "optimistic; the test split was never seen.\n\n"
    )
    return note + _table(header, rows)


def render_leakage(files: Mapping[str, Results]) -> str:
    """Near-duplicate images shared between dataset splits."""
    report = files[LEAKAGE]
    thresholds = [str(t) for t in report["thresholds"]]
    rows = []
    for name, pair in report["pairs"].items():
        later, earlier = name.split("_vs_")
        cells = [
            f"{pair['within'][t]['count']} ({_pct(pair['within'][t]['share'])})"
            if t in pair["within"]
            else "-"
            for t in thresholds
        ]
        rows.append([f"{later} vs {earlier}", str(pair["queries"]), *cells])
    sizes = ", ".join(f"{split} {n}" for split, n in report["split_sizes"].items())
    header = ["Split pair", "Images", *(f"≤ {t} bits" for t in thresholds)]
    return (
        f"Split sizes: {sizes}. Each image of the later split is matched to its nearest "
        f"image in the earlier one by a 64-bit difference hash; the columns count images whose "
        f"nearest neighbour differs in at most that many bits.\n\n" + _table(header, rows)
    )


def render_endtoend(files: Mapping[str, Results]) -> str:
    """Detector plus reader on the benchmark photos, per reader."""
    first = next(iter(files.values()))["end_to_end"]
    detection = first["detection"]
    regions = list(detection["by_region"])
    recall = ", ".join(
        f"{region.upper()} {_pct(detection['by_region'][region]['recall'])}" for region in regions
    )
    overall = detection["overall"]
    text = (
        f"The detector finds **{_pct(overall['recall'])}** of the {overall['n']} labeled plates "
        f"at IoU ≥ {first['iou_threshold']} ({recall}); found plates overlap their label with a "
        f"mean IoU of {overall['mean_iou_when_detected']:.2f}. Reading accuracy counts a missed "
        "plate as wrong, next to the same reader on the labeled boxes.\n\n"
    )
    rows = []
    for report in files.values():
        reading = report["end_to_end"]["reading"]
        rows.append(
            [
                f"`{report['recognition']['ocr']}`",
                _with_ci(report["recognition"]["overall"]),
                _with_ci(reading["overall"]),
                f"{reading['overall']['cer']:.3f}",
                *(_pct(reading["by_region"][region]["accuracy"]) for region in regions),
            ]
        )
    header = ["Reader", "Labeled boxes", "End to end (95% CI)", "CER"]
    header += [region.upper() for region in regions]
    return text + _table(header, rows)


# --- registry ---------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """One generated README section.

    Attributes:
        files: Results files the section is built from; it renders once any exists.
        command: How to produce them, shown while none exists.
        render: Builds the Markdown from the files that exist. ``None`` while no
            renderer has been written; the block must then still be pending.
    """

    files: tuple[str, ...]
    command: str
    render: Callable[[Mapping[str, Results]], str] | None


_STILLS_CMD = "python scripts/run_ablations.py stills"
BLOCKS: dict[str, Block] = {
    "headline": Block((STILLS, "detector_original_test.json"), _STILLS_CMD, render_headline),
    "ocr-stills": Block((STILLS,), _STILLS_CMD, render_ocr_stills),
    "format-br": Block((STILLS,), _STILLS_CMD, render_format_br),
    "plate-height": Block((STILLS,), _STILLS_CMD, render_plate_height),
    "confusions": Block((STILLS,), _STILLS_CMD, render_confusions),
    "lookalikes": Block((LOOKALIKES,), "python scripts/analyze_confusions.py", render_lookalikes),
    "training-original": Block(
        (TRAINING,),
        "python scripts/recover_training_run.py --weights weights/plate_yolo11n_original.pt",
        render_training,
    ),
    "detector": Block(
        tuple(DETECTOR_FILES),
        "alpr val --weights weights/plate_yolo11n_original.pt --split test "
        "--out results/detector_original_test.json",
        render_detector,
    ),
    "leakage": Block((LEAKAGE,), "python scripts/check_leakage.py", render_leakage),
    "endtoend-stills": Block(
        ENDTOEND_FILES,
        "alpr eval-images --ocr fast-plate-ocr --plate-weights weights/plate_yolo11n_original.pt "
        "--out results/endtoend_stills.json",
        render_endtoend,
    ),
    "video": Block(
        ("video_eval.json",),
        "alpr run VIDEO, alpr label-sheet, then alpr eval-video --out results/video_eval.json",
        None,
    ),
    "video-ablations": Block(
        ("ablation_video_replay.json", "ablation_video_grid.json"),
        "python scripts/run_ablations.py video-replay / video-grid",
        None,
    ),
    "runtime": Block(("runtime.json",), "alpr run VIDEO (its timing.json)", None),
}


def _pending(block: Block) -> str:
    return f"> _Pending: produced by `{block.command}`._"


def render_block(name: str, results_dir: Path) -> str:
    """Build one block's Markdown from whichever of its results files exist."""
    block = BLOCKS[name]
    files = {
        file: json.loads((results_dir / file).read_text())
        for file in block.files
        if (results_dir / file).exists()
    }
    if not files:
        return _pending(block)
    if block.render is None:
        raise SystemExit(
            f"results/{next(iter(files))} exists but block {name!r} has no renderer yet"
        )
    return block.render(files)


def render_readme(text: str, results_dir: Path) -> str:
    """Return ``text`` with every results block regenerated."""

    def replace(match: re.Match[str]) -> str:
        name = match["name"]
        if name not in BLOCKS:
            raise SystemExit(f"README has a results block {name!r} that nothing renders")
        body = render_block(name, results_dir).rstrip("\n")
        return f"{match['open']}{body}\n{match['close']}"

    return BLOCK.sub(replace, text)


def main(argv: list[str] | None = None) -> int:
    """Render README.md in place, or check that it is current."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--check", action="store_true", help="only check; exit 1 if stale")
    args = parser.parse_args(argv)

    current = args.readme.read_text()
    rendered = render_readme(current, args.results)
    if args.check:
        if rendered != current:
            stale = [
                m["name"]
                for m, n in zip(BLOCK.finditer(current), BLOCK.finditer(rendered), strict=True)
                if m["body"] != n["body"]
            ]
            print(
                f"{args.readme} is out of date in: {', '.join(stale)}. "
                "Run python scripts/render_readme_tables.py",
                file=sys.stderr,
            )
            return 1
        print(f"{args.readme} matches {args.results}/")
        return 0
    args.readme.write_text(rendered)
    print(f"rendered {len(BLOCK.findall(rendered))} blocks into {args.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
