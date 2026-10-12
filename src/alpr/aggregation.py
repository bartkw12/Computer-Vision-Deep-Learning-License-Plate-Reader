"""Combining the per-frame reads of one track into a single plate string.

A vehicle is usually visible for dozens of frames, and each frame yields an
independent, noisy read of the same plate. Picking the single most confident
read throws that redundancy away; voting uses it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

#: One per-frame read: ``(text, ocr_confidence)``.
Read = tuple[str, float]

_MIN_WEIGHT = 1e-6


@dataclass(frozen=True, slots=True)
class Consensus:
    """Outcome of aggregating a track's reads.

    Attributes:
        text: The agreed plate string.
        confidence: Score in ``[0, 1]``. For the voting strategies it is the
            share of vote weight the winner received; for ``max_conf`` it is
            the OCR confidence of the chosen read.
        n_reads: Number of reads that were aggregated.
    """

    text: str
    confidence: float
    n_reads: int


def _weight(confidence: float) -> float:
    return max(float(confidence), _MIN_WEIGHT)


def _heaviest(weights: dict[str, float]) -> str:
    # Ties go to the alphabetically first key so results are reproducible.
    return min(weights, key=lambda key: (-weights[key], key))


def max_confidence(reads: Sequence[Read]) -> Consensus | None:
    """Return the single read with the highest OCR confidence.

    This is what the first version of this project did, kept as a baseline.
    """
    if not reads:
        return None
    text, confidence = min(reads, key=lambda read: (-read[1], read[0]))
    return Consensus(text, float(confidence), len(reads))


def string_vote(reads: Sequence[Read]) -> Consensus | None:
    """Vote between whole strings, each read weighted by its confidence."""
    if not reads:
        return None
    weights: dict[str, float] = defaultdict(float)
    for text, confidence in reads:
        weights[text] += _weight(confidence)
    winner = _heaviest(weights)
    return Consensus(winner, weights[winner] / sum(weights.values()), len(reads))


def char_vote(reads: Sequence[Read]) -> Consensus | None:
    """Vote character by character, each read weighted by its confidence.

    Reads are first reduced to the most strongly supported length, then every
    position is decided on its own. The result can be a string that no single
    frame produced, which is the point: two reads that each get a different
    character wrong can still agree on the right plate.
    """
    if not reads:
        return None
    length_weights: dict[int, float] = defaultdict(float)
    for text, confidence in reads:
        length_weights[len(text)] += _weight(confidence)
    length = min(length_weights, key=lambda n: (-length_weights[n], n))
    if length == 0:
        return None

    chars: list[str] = []
    shares: list[float] = []
    for position in range(length):
        weights: dict[str, float] = defaultdict(float)
        for text, confidence in reads:
            if len(text) == length:
                weights[text[position]] += _weight(confidence)
        winner = _heaviest(weights)
        chars.append(winner)
        shares.append(weights[winner] / sum(weights.values()))
    return Consensus("".join(chars), sum(shares) / length, len(reads))


STRATEGIES: dict[str, Callable[[Sequence[Read]], Consensus | None]] = {
    "max_conf": max_confidence,
    "string_vote": string_vote,
    "char_vote": char_vote,
}


def aggregate(reads: Sequence[Read], strategy: str = "string_vote") -> Consensus | None:
    """Reduce a track's reads to one plate string.

    Args:
        reads: ``(text, confidence)`` pairs, one per frame that was read.
        strategy: ``"max_conf"``, ``"string_vote"`` or ``"char_vote"``.

    Returns:
        The consensus, or ``None`` if there are no reads.

    Raises:
        ValueError: If the strategy is unknown.
    """
    try:
        return STRATEGIES[strategy](reads)
    except KeyError:
        raise ValueError(
            f"unknown aggregation strategy {strategy!r}; choose from {sorted(STRATEGIES)}"
        ) from None
