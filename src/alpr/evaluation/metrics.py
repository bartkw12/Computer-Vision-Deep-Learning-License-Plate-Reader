"""Metrics for plate reading: exact match, character error rate, intervals."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import comb
from statistics import NormalDist


def levenshtein(a: str, b: str) -> int:
    """Minimum number of single-character edits turning ``a`` into ``b``."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        previous = current
    return previous[-1]


def align(truth: str, prediction: str) -> list[tuple[str, str]]:
    """Align two strings along a minimum-edit path.

    Returns:
        ``(truth_char, predicted_char)`` pairs in order. A missed character
        has an empty prediction; an inserted one has an empty truth.
    """
    rows, cols = len(truth) + 1, len(prediction) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost[i][j] = min(
                cost[i - 1][j - 1] + (truth[i - 1] != prediction[j - 1]),
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
            )

    pairs: list[tuple[str, str]] = []
    i, j = len(truth), len(prediction)
    while i > 0 or j > 0:
        diagonal = cost[i - 1][j - 1] + (truth[i - 1] != prediction[j - 1]) if i and j else None
        if diagonal is not None and cost[i][j] == diagonal:
            pairs.append((truth[i - 1], prediction[j - 1]))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((truth[i - 1], ""))
            i -= 1
        else:
            pairs.append(("", prediction[j - 1]))
            j -= 1
    return pairs[::-1]


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Unlike the normal approximation it behaves sensibly for small samples and
    for proportions near 0 or 1, which is the regime of a hand-labeled video.

    Returns:
        ``(low, high)``. With no observations the interval is ``(0, 1)``.
    """
    if total == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z * ((proportion * (1 - proportion) / total + z**2 / (4 * total**2)) ** 0.5) / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def mcnemar_exact(only_a_correct: int, only_b_correct: int) -> float:
    """Exact two-sided McNemar test for two readers scored on the same plates.

    Only the plates on which the readers disagree carry information. Under the
    null hypothesis that both are equally accurate, each such plate is equally
    likely to favor either one, so the count favoring one is Binomial(n, 0.5).

    Args:
        only_a_correct: Plates the first reader got right and the second wrong.
        only_b_correct: Plates the second reader got right and the first wrong.

    Returns:
        The two-sided p-value; 1.0 when the readers never disagree.
    """
    disagreements = only_a_correct + only_b_correct
    if disagreements == 0:
        return 1.0
    smaller = min(only_a_correct, only_b_correct)
    tail = sum(comb(disagreements, k) for k in range(smaller + 1)) / 2**disagreements
    return min(1.0, 2 * tail)


def compare_paired(
    predictions_a: Sequence[str | None], predictions_b: Sequence[str | None], truths: Sequence[str]
) -> dict[str, float | int]:
    """Compare two readers plate by plate.

    Returns:
        How many plates only the first got right, how many only the second
        did, and the McNemar p-value for the difference.
    """
    only_a = only_b = 0
    for a, b, truth in zip(predictions_a, predictions_b, truths, strict=True):
        only_a += a == truth and b != truth
        only_b += b == truth and a != truth
    return {
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "p_value": mcnemar_exact(only_a, only_b),
    }


@dataclass(frozen=True, slots=True)
class ReadingScore:
    """Accuracy of a set of plate reads against ground truth.

    Attributes:
        n: Number of ground-truth plates.
        exact: How many were read exactly right.
        accuracy: ``exact / n``.
        ci_low: Lower end of the 95% Wilson interval on ``accuracy``.
        ci_high: Upper end of that interval.
        cer: Character error rate: total edit distance over total ground-truth
            characters. A plate that was not read at all counts as fully wrong.
    """

    n: int
    exact: int
    accuracy: float
    ci_low: float
    ci_high: float
    cer: float

    def as_dict(self) -> dict[str, float | int]:
        """Return the fields as a plain dictionary for JSON output."""
        return asdict(self)


def score_readings(predictions: Sequence[str | None], truths: Sequence[str]) -> ReadingScore:
    """Score predicted plate strings against ground truth.

    Args:
        predictions: One prediction per ground-truth plate; ``None`` where the
            system produced no read.
        truths: The ground-truth strings, already normalized.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(predictions) != len(truths):
        raise ValueError("predictions and truths must have the same length")
    exact = sum(p == t for p, t in zip(predictions, truths, strict=True))
    edits = sum(levenshtein(p or "", t) for p, t in zip(predictions, truths, strict=True))
    characters = sum(len(t) for t in truths)
    low, high = wilson_interval(exact, len(truths))
    return ReadingScore(
        n=len(truths),
        exact=exact,
        accuracy=exact / len(truths) if truths else 0.0,
        ci_low=low,
        ci_high=high,
        cer=edits / characters if characters else 0.0,
    )


def substitution_counts(
    predictions: Sequence[str | None], truths: Sequence[str]
) -> Counter[tuple[str, str]]:
    """Count which characters were mistaken for which.

    Returns:
        Counter keyed by ``(true_char, predicted_char)`` over all substituted
        characters. Insertions and deletions are not included.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for prediction, truth in zip(predictions, truths, strict=True):
        for true_char, predicted_char in align(truth, prediction or ""):
            if true_char and predicted_char and true_char != predicted_char:
                counts[true_char, predicted_char] += 1
    return counts
