import json
from pathlib import Path

import pytest

from alpr.plate_format import (
    DIGIT_FIXES,
    FOUR_LETTERS_THREE_DIGITS,
    LETTER_FIXES,
    ONTARIO_PASSENGER,
    PlateFormat,
    fit_plate,
    get_formats,
    normalize,
)

ONTARIO = get_formats("ontario")
EXTENDED = get_formats("ontario-extended")
EVIDENCE = Path(__file__).parents[1] / "results" / "lookalike_evidence.json"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("bxyz-123", "BXYZ123"), (" BXYZ 123 ", "BXYZ123"), ("B.X/Y_Z:1 2 3", "BXYZ123"), ("", "")],
)
def test_normalize_keeps_only_upper_alphanumerics(raw, expected):
    assert normalize(raw) == expected


def test_clean_read_needs_no_substitution():
    fit = fit_plate("BXYZ 123", ONTARIO)
    assert fit is not None
    assert (fit.text, fit.substitutions, fit.format_name) == ("BXYZ123", 0, "ontario-passenger")


@pytest.mark.parametrize(
    ("ocr_text", "expected"),
    [
        ("BXYZ 1O3", "BXYZ103"),  # O read for a digit
        ("BXYZ 12S", "BXYZ125"),  # S read for 5
        ("4XYZ 123", "AXYZ123"),  # 4 read for A
    ],
)
def test_lookalikes_are_corrected_not_rejected(ocr_text, expected):
    """Regression: the original code rejected every one of these.

    It demanded an exact ``[A-Z]{4}[0-9]{3}`` match before calling its
    correction function, so the correction could never change anything.
    """
    fit = fit_plate(ocr_text, ONTARIO)
    assert fit is not None
    assert fit.text == expected
    assert fit.substitutions == 1


@pytest.mark.parametrize(
    "ocr_text",
    [
        # 8 at a letter position: seen only 3 times in the evidence, too rare to trust.
        "8XYZ 123",
        # 0 at a letter position: a D in 7 of 12 issued-letter cases, short of the 60% bar.
        "B0YZ 123",
    ],
)
def test_lookalikes_without_clear_evidence_are_rejected(ocr_text):
    """The obvious-looking fixes the measurement did not support are not guessed."""
    assert fit_plate(ocr_text, ONTARIO) is None


def test_serial_is_found_inside_surrounding_text():
    assert fit_plate("ONTARIO BXYZ 123", ONTARIO).text == "BXYZ123"
    assert fit_plate("BXYZ123YOURSTODISCOVER", ONTARIO).text == "BXYZ123"


@pytest.mark.parametrize("text", ["YOURS TO DISCOVER", "ONTARIO", "TORONTO", "DEALER", "", "AB1"])
def test_non_plate_text_is_rejected(text):
    assert fit_plate(text, ONTARIO) is None
    assert fit_plate(text, EXTENDED) is None


def test_exact_window_beats_a_corrected_one():
    # "OBXYZ12" could be forced into shape with fixes, "BXYZ123" needs none.
    assert fit_plate("OBXYZ123", ONTARIO).text == "BXYZ123"


def test_substitution_cap():
    assert fit_plate("4XYZ1O3", ONTARIO, max_substitutions=2).text == "AXYZ103"
    assert fit_plate("4XYZ1O3", ONTARIO, max_substitutions=1) is None
    # All three digits read as letters: no genuine digit left, so not a plate.
    assert fit_plate("BXYZIZS", ONTARIO, max_substitutions=2) is None


def test_strict_mode_accepts_exact_matches_only():
    assert fit_plate("BXYZ123", ONTARIO, max_substitutions=0).text == "BXYZ123"
    assert fit_plate("B0YZ123", ONTARIO, max_substitutions=0) is None


def test_unused_letters_are_invalid_for_ontario_passenger():
    # None of O, I, U has a measured single look-alike, so each read is dropped.
    assert fit_plate("BOYZ123", ONTARIO) is None
    assert fit_plate("BIYZ123", ONTARIO) is None
    assert fit_plate("BUYZ123", ONTARIO) is None


def test_plain_4l3d_format_knows_nothing_about_unused_letters():
    plain = get_formats("4L3D")
    assert fit_plate("BOYZ123", plain).substitutions == 0
    # 0 at a letter position was an O only 58% of the time: not enough to correct.
    assert fit_plate("B0YZ123", plain) is None
    assert fit_plate("B5YZ123", plain).text == "BSYZ123"


def test_tables_match_measured_evidence():
    """The correction tables are the ones ``scripts/analyze_confusions.py`` derived."""
    evidence = json.loads(EVIDENCE.read_text())
    assert dict(DIGIT_FIXES) == evidence["digit_fixes"]
    assert {key: dict(fixes) for key, fixes in LETTER_FIXES.items()} == evidence["letter_fixes"]


@pytest.mark.parametrize(
    ("text", "expected_format"),
    [
        ("BXYZ 123", "ontario-passenger"),
        ("123 BXY", "ontario-passenger-1986"),
        ("BXY 123", "ontario-passenger-1973"),
        ("AB 12345", "ontario-commercial"),
        ("123 4AB", "ontario-commercial-1996"),
    ],
)
def test_extended_set_recognizes_older_and_commercial_series(text, expected_format):
    fit = fit_plate(text, EXTENDED)
    assert fit is not None
    assert fit.format_name == expected_format
    assert fit.substitutions == 0


def test_longest_pattern_wins_a_tie():
    # "BXYZ123" also contains "XYZ123", a valid 1973-series serial.
    assert fit_plate("BXYZ123", EXTENDED).format_name == "ontario-passenger"


def test_any_format_accepts_plausible_lengths_unchanged():
    anything = get_formats("any")
    assert fit_plate("yg9x-2g", anything).text == "YG9X2G"
    assert fit_plate("AB", anything) is None
    assert fit_plate("ABCDEFGHIJ", anything) is None


def test_fit_window_requires_matching_length():
    assert ONTARIO_PASSENGER.fit_window("BXYZ12", 2) is None
    assert FOUR_LETTERS_THREE_DIGITS.fit_window("BXYZ123", 0).text == "BXYZ123"


def test_unknown_format_name():
    with pytest.raises(ValueError, match="unknown plate format"):
        get_formats("narnia")


def test_format_definition_is_validated():
    with pytest.raises(ValueError, match="pattern"):
        PlateFormat("bad", "LLX")
    with pytest.raises(ValueError, match="pattern"):
        PlateFormat("empty", "")
    with pytest.raises(ValueError, match="letter_fixes"):
        PlateFormat("bad", "LD", unused_letters=frozenset("O"), letter_fixes={"0": "O"})
    with pytest.raises(ValueError, match="digit_fixes"):
        PlateFormat("bad", "LD", digit_fixes={"O": "X"})
