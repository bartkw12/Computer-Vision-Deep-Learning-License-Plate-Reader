from collections import Counter

import pytest

from alpr.evaluation.metrics import (
    align,
    compare_paired,
    levenshtein,
    mcnemar_exact,
    score_readings,
    substitution_counts,
    wilson_interval,
)


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
        ("BXYZ123", "BXYZ123", 0),
        ("BXYZ123", "8XYZ128", 2),
        ("", "ABC", 3),
        ("ABC", "", 3),
        ("", "", 0),
    ],
)
def test_levenshtein(a, b, distance):
    assert levenshtein(a, b) == distance
    assert levenshtein(b, a) == distance


def test_align_substitution():
    assert align("BXYZ123", "8XYZ123") == [
        ("B", "8"), ("X", "X"), ("Y", "Y"), ("Z", "Z"), ("1", "1"), ("2", "2"), ("3", "3"),
    ]  # fmt: skip


def test_align_missed_and_extra_characters():
    assert align("AB1", "AB") == [("A", "A"), ("B", "B"), ("1", "")]
    assert align("AB", "XAB") == [("", "X"), ("A", "A"), ("B", "B")]
    assert align("", "") == []


@pytest.mark.parametrize(("truth", "prediction"), [("BXYZ123", "XYZ1238"), ("ABCD", "DCBA")])
def test_alignment_cost_equals_edit_distance(truth, prediction):
    edits = sum(t != p for t, p in align(truth, prediction))
    assert edits == levenshtein(truth, prediction)


def test_wilson_interval_known_values():
    low, high = wilson_interval(8, 10)
    assert (low, high) == pytest.approx((0.4902, 0.9433), abs=1e-4)
    # Closed forms at the extremes: z^2 / (n + z^2) and n / (n + z^2).
    assert wilson_interval(0, 10) == pytest.approx((0.0, 0.27753), abs=1e-5)
    assert wilson_interval(10, 10) == pytest.approx((0.72247, 1.0), abs=1e-5)


def test_wilson_interval_narrows_with_more_data():
    small = wilson_interval(8, 10)
    large = wilson_interval(800, 1000)
    assert large[1] - large[0] < small[1] - small[0]
    assert large[0] < 0.8 < large[1]


def test_wilson_interval_without_data_is_uninformative():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_score_readings():
    score = score_readings(["BXYZ123", "BXYZ128", None], ["BXYZ123", "BXYZ123", "ABCD456"])
    assert (score.n, score.exact) == (3, 1)
    assert score.accuracy == pytest.approx(1 / 3)
    # 0 edits + 1 edit + a missed plate counted as 7 edits, over 21 characters.
    assert score.cer == pytest.approx(8 / 21)
    assert score.ci_low < score.accuracy < score.ci_high
    assert score.as_dict()["exact"] == 1


def test_score_readings_empty_and_mismatched():
    empty = score_readings([], [])
    assert (empty.n, empty.accuracy, empty.cer) == (0, 0.0, 0.0)
    with pytest.raises(ValueError, match="same length"):
        score_readings(["A"], [])


def test_substitution_counts():
    counts = substitution_counts(
        ["8XYZ123", "BXYZ128", "8XYZ12", None], ["BXYZ123", "BXYZ123", "BXYZ123", "ABCD456"]
    )
    assert counts == Counter({("B", "8"): 2, ("3", "8"): 1})


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    # All 10 disagreements favor one reader: 2 * 0.5**10.
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    # 1 of 9: P(X <= 1) = (1 + 9) / 512, doubled.
    assert mcnemar_exact(8, 1) == pytest.approx(2 * 10 / 512)
    assert mcnemar_exact(1, 8) == mcnemar_exact(8, 1)


def test_mcnemar_needs_enough_disagreements_to_be_significant():
    assert mcnemar_exact(3, 0) > 0.05
    assert mcnemar_exact(6, 0) < 0.05


def test_compare_paired_counts_disagreements_only():
    truths = ["A1", "B2", "C3", "D4", "E5"]
    first = ["A1", "B2", "XX", None, "E5"]
    second = ["A1", "XX", "C3", "D4", "XX"]
    result = compare_paired(first, second, truths)
    assert (result["only_a_correct"], result["only_b_correct"]) == (2, 2)
    assert result["p_value"] == 1.0
    with pytest.raises(ValueError):
        compare_paired(first, second[:-1], truths)
