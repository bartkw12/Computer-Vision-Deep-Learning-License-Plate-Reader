"""Assembling a plate serial from the text fragments a scene-text OCR returns.

A general-purpose OCR engine reports every piece of text it finds on a plate
crop separately: the serial, often split in two at the gap or the crown, plus
the province name above it and the slogan below. The serial is printed much
larger than everything else, which is what singles it out here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from alpr.plate_format import normalize
from alpr.types import PlateRead


@dataclass(frozen=True, slots=True)
class Fragment:
    """One piece of text found on a crop.

    Attributes:
        text: Recognized text.
        confidence: Recognition confidence in ``[0, 1]``.
        x: Left edge of the text box.
        y: Vertical center of the text box.
        height: Height of the text box.
    """

    text: str
    confidence: float
    x: float
    y: float
    height: float


def merge_fragments(
    fragments: Sequence[Fragment], *, min_relative_height: float = 0.6
) -> PlateRead | None:
    """Join the fragments that make up the serial into one read.

    Fragments much shorter than the tallest one are discarded as small print.
    The rest are ordered into rows, top to bottom, then left to right within a
    row, and concatenated.

    Args:
        fragments: Text fragments from one plate crop.
        min_relative_height: A fragment is kept if it is at least this tall
            relative to the tallest fragment.

    Returns:
        The combined read with a length-weighted mean confidence, or ``None``
        if no fragment contains a letter or digit.
    """
    usable = [(f, normalize(f.text)) for f in fragments]
    usable = [(f, text) for f, text in usable if text]
    if not usable:
        return None

    tallest = max(f.height for f, _ in usable)
    usable = [(f, text) for f, text in usable if f.height >= min_relative_height * tallest]

    # Group into rows: a fragment starts a new row if its center sits more than
    # half a character height below the row above.
    rows: list[list[tuple[Fragment, str]]] = []
    for item in sorted(usable, key=lambda pair: pair[0].y):
        if rows and item[0].y - rows[-1][-1][0].y <= 0.5 * tallest:
            rows[-1].append(item)
        else:
            rows.append([item])

    ordered = [item for row in rows for item in sorted(row, key=lambda pair: pair[0].x)]
    text = "".join(part for _, part in ordered)
    confidence = sum(f.confidence * len(part) for f, part in ordered) / len(text)
    return PlateRead(text, float(confidence))
