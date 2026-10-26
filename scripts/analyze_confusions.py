"""Derive the look-alike correction tables from measured OCR confusions.

Usage::

    python scripts/run_ablations.py stills      # writes the per-image records
    python scripts/analyze_confusions.py

Format-aware correction replaces a character of the wrong class (a letter
where the plate format demands a digit, or the reverse) with the character it
was most likely mistaken for. Which character that is should be measured, not
assumed, so this script counts what OCR actually confused with what.

Method:

* Evidence comes from the US and EU plates of the OpenALPR benchmark only. The
  Brazilian plates are held out, so the effect of the derived tables can be
  tested on plates that played no part in building them.
* Two readers are pooled, one per OCR family, so the tables are not tuned to
  a single model's habits.
* A correction ``read -> truth`` is accepted only if the read character was
  seen at least ``MIN_COUNT`` times in the wrong class and ``truth`` accounts
  for at least ``MIN_SHARE`` of those cases. Anything less clear-cut is left
  out: the read is then rejected rather than "corrected" into a guess.
* For a series that does not issue certain letters, the same counts are taken
  over issued letters only, and the unused letters themselves get a correction
  where one issued letter clearly dominates.

The result is written to ``results/lookalike_evidence.json``. A test checks
that the tables in ``alpr.plate_format`` equal the derived ones.
"""

from __future__ import annotations

import argparse
import json
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from alpr.evaluation.metrics import align
from alpr.provenance import environment_info
from alpr.runs import write_json

DEV_REGIONS = ("us", "eu")
READERS = ("easyocr|none", "fast-plate-ocr:cct-xs-v2-global-model|none")
MIN_COUNT = 5
MIN_SHARE = 0.6

#: Sets of unused letters for which letter tables are derived: none, and the
#: three Ontario passenger series (see alpr.plate_format).
UNUSED_LETTER_SETS = ("", "GIQ", "GIQU", "GIOQU")


def count_confusions(records: dict[str, list[dict[str, Any]]]) -> dict[str, Counter[str]]:
    """Count substitutions as ``read -> Counter(truth)`` over the dev plates."""
    confusions: dict[str, Counter[str]] = defaultdict(Counter)
    for reader in READERS:
        for record in records[reader]:
            if record["region"] not in DEV_REGIONS or not record["prediction"]:
                continue
            for truth, read in align(record["truth"], record["prediction"]):
                if truth and read and truth != read:
                    confusions[read][truth] += 1
    return confusions


def derive(
    confusions: dict[str, Counter[str]], read_chars: str, valid_truths: str
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Apply the acceptance rule to every character in ``read_chars``.

    Args:
        confusions: Output of :func:`count_confusions`.
        read_chars: Characters that are invalid at the position in question.
        valid_truths: Characters that are valid there; only confusions with
            one of these as the truth are counted.

    Returns:
        ``(accepted, evidence)``: the accepted ``read -> truth`` corrections,
        and one evidence row per read character that was observed at all.
    """
    accepted, evidence = {}, []
    for read in read_chars:
        counts = Counter({t: n for t, n in confusions.get(read, {}).items() if t in valid_truths})
        total = sum(counts.values())
        if total == 0:
            continue
        truth, count = counts.most_common(1)[0]
        ok = total >= MIN_COUNT and count / total >= MIN_SHARE
        if ok:
            accepted[read] = truth
        evidence.append(
            {
                "read": read,
                "top_truth": truth,
                "count": count,
                "of": total,
                "share": round(count / total, 3),
                "accepted": ok,
                "all": dict(counts.most_common()),
            }
        )
    evidence.sort(key=lambda row: -row["of"])
    return accepted, evidence


def main() -> None:
    """Derive the tables and write the evidence file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--records", type=Path, default=Path("outputs/ablation_ocr_stills_records.json")
    )
    parser.add_argument("--out", type=Path, default=Path("results/lookalike_evidence.json"))
    args = parser.parse_args()

    confusions = count_confusions(json.loads(args.records.read_text()))
    letters, digits = string.ascii_uppercase, string.digits

    digit_fixes, digit_evidence = derive(confusions, letters, digits)
    report: dict[str, Any] = {
        "method": {
            "evidence_regions": list(DEV_REGIONS),
            "held_out_region": "br",
            "readers": list(READERS),
            "min_count": MIN_COUNT,
            "min_share": MIN_SHARE,
        },
        "digit_fixes": dict(sorted(digit_fixes.items())),
        "digit_evidence": digit_evidence,
        "letter_fixes": {},
        "letter_evidence": {},
    }
    print(f"digit positions:  {report['digit_fixes']}")
    for unused in UNUSED_LETTER_SETS:
        issued = "".join(c for c in letters if c not in unused)
        fixes, evidence = derive(confusions, digits + unused, issued)
        report["letter_fixes"][unused] = dict(sorted(fixes.items()))
        report["letter_evidence"][unused] = evidence
        label = unused or "none"
        print(f"letter positions, unused letters {label:6s}: {report['letter_fixes'][unused]}")

    rejected = [row for row in digit_evidence if not row["accepted"] and row["of"] >= MIN_COUNT]
    print("seen often enough but too ambiguous to correct (digit positions):")
    for row in rejected:
        print(f"   {row['read']} -> {row['all']}")
    report["environment"] = environment_info()
    write_json(args.out, report)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
