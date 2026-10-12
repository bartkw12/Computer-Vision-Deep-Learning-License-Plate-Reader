"""Plate text normalization, plate formats, and format-aware OCR correction.

A plate serial has structure: Ontario's current passenger series is four
letters followed by three digits. That structure is strong evidence about what
the OCR *should* have read, and it is used here in two ways:

* to locate the serial inside a longer OCR string (province name, slogan and
  sticker text often get glued on), and
* to repair look-alike mistakes, where a letter shows up at a digit position
  or the other way round (``O`` for ``0``, ``S`` for ``5``, ``4`` for ``A``).

The order of those steps matters. A candidate is accepted if every character
is either already of the right class *or* has a known look-alike of the right
class, and only then are the look-alikes substituted. Validating strictly
first and correcting afterwards would leave nothing to correct.

Which characters count as look-alikes is measured, not assumed. The tables
below are derived by ``scripts/analyze_confusions.py`` from the confusions two
OCR models made on the US and EU plates of the OpenALPR benchmark, and a
correction is listed only where one answer clearly dominates. Some pairs that
look obvious did not survive: an ``L`` at a digit position turned out to be a
``4`` far more often than a ``1``, and a ``Z`` was a ``7`` or a ``2`` too
evenly to correct either way. A character without an entry is not guessed at;
the read is rejected and other frames of the same vehicle decide.
``results/lookalike_evidence.json`` holds the counts, and a test keeps these
tables equal to it.

Ontario serial formats and their unused letters follow the tables in
Wikipedia, "Vehicle registration plates of Ontario" (retrieved 2026-10-09).
"""

from __future__ import annotations

import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

LETTERS = frozenset(string.ascii_uppercase)
DIGITS = frozenset(string.digits)

_NON_ALPHANUMERIC = re.compile(r"[^A-Z0-9]")

#: At a digit position: the letter OCR produced -> the digit it stood for.
DIGIT_FIXES: Mapping[str, str] = {
    "A": "4",
    "B": "8",
    "D": "0",
    "G": "6",
    "I": "1",
    "L": "4",
    "O": "0",
    "S": "5",
    "T": "1",
}

#: At a letter position: the character OCR produced -> the letter it stood
#: for. Keyed by the letters a series does not issue, sorted and joined,
#: because which letter a character most likely stood for depends on which
#: letters are possible at all.
LETTER_FIXES: Mapping[str, Mapping[str, str]] = {
    "": {"4": "A", "5": "S"},
    "GIQ": {"4": "A", "5": "S"},
    "GIQU": {"0": "O", "4": "A", "5": "S"},
    "GIOQU": {"4": "A", "5": "S"},
}


def letter_fixes_for(unused_letters: frozenset[str]) -> Mapping[str, str]:
    """Return the letter-position corrections for a series.

    Args:
        unused_letters: Letters the series never issues.

    Returns:
        The table derived for exactly that set of unused letters if there is
        one. Otherwise the general table, without any correction that would
        produce a letter the series does not issue.
    """
    key = "".join(sorted(unused_letters))
    if key in LETTER_FIXES:
        return LETTER_FIXES[key]
    return {
        read: letter for read, letter in LETTER_FIXES[""].items() if letter not in unused_letters
    }


def normalize(text: str) -> str:
    """Upper-case ``text`` and drop everything that is not a letter or digit."""
    return _NON_ALPHANUMERIC.sub("", text.upper())


@dataclass(frozen=True, slots=True)
class FormatFit:
    """A plate serial found in OCR text.

    Attributes:
        text: The serial, with look-alike characters corrected.
        substitutions: How many characters had to be corrected.
        format_name: Name of the format that matched.
    """

    text: str
    substitutions: int
    format_name: str


@dataclass(frozen=True)
class PlateFormat:
    """The structure of one plate series.

    Attributes:
        name: Identifier used in logs and results.
        pattern: One character per position, ``L`` for a letter and ``D`` for
            a digit, e.g. ``"LLLLDDD"``.
        unused_letters: Letters this series never issues. They are invalid at
            letter positions.
        letter_fixes: Look-alike corrections applied at letter positions. By
            default the measured table for this series' unused letters.
        digit_fixes: Look-alike corrections applied at digit positions. By
            default the measured table.

    Raises:
        ValueError: If the pattern contains anything but ``L`` and ``D``, or a
            fix maps to a character that is not valid at its position.
    """

    name: str
    pattern: str
    unused_letters: frozenset[str] = frozenset()
    letter_fixes: Mapping[str, str] | None = None
    digit_fixes: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.pattern or set(self.pattern) - {"L", "D"}:
            raise ValueError(f"pattern must be a non-empty string of 'L' and 'D': {self.pattern!r}")
        # The dataclass is frozen, so defaults that depend on other fields are
        # filled in through object.__setattr__.
        if self.letter_fixes is None:
            object.__setattr__(self, "letter_fixes", letter_fixes_for(self.unused_letters))
        if self.digit_fixes is None:
            object.__setattr__(self, "digit_fixes", DIGIT_FIXES)
        bad_letters = [v for v in self.letter_fixes.values() if not self._is_valid_letter(v)]
        if bad_letters:
            raise ValueError(f"{self.name}: letter_fixes map to invalid letters {bad_letters}")
        bad_digits = [v for v in self.digit_fixes.values() if v not in DIGITS]
        if bad_digits:
            raise ValueError(f"{self.name}: digit_fixes map to non-digits {bad_digits}")

    def _is_valid_letter(self, char: str) -> bool:
        return char in LETTERS and char not in self.unused_letters

    def fit_window(self, window: str, max_substitutions: int) -> FormatFit | None:
        """Fit a string of exactly ``len(pattern)`` characters to this format.

        Args:
            window: Normalized text, the same length as the pattern.
            max_substitutions: Reject the fit if more characters than this
                would need correcting. ``0`` demands an exact match.

        Returns:
            The corrected serial, or ``None`` if the window does not fit.
        """
        if len(window) != len(self.pattern):
            return None
        corrected: list[str] = []
        substitutions = 0
        for char, kind in zip(window, self.pattern, strict=True):
            if kind == "L":
                valid, fixes = self._is_valid_letter(char), self.letter_fixes
            else:
                valid, fixes = char in DIGITS, self.digit_fixes
            if valid:
                corrected.append(char)
                continue
            substitutions += 1
            if char not in fixes or substitutions > max_substitutions:
                return None
            corrected.append(fixes[char])
        return FormatFit("".join(corrected), substitutions, self.name)

    def fit(self, text: str, max_substitutions: int) -> FormatFit | None:
        """Find the best-fitting serial anywhere inside ``text``.

        Every window of the pattern's length is tried. The one needing the
        fewest corrections wins; the leftmost wins a tie.
        """
        best: FormatFit | None = None
        for start in range(len(text) - len(self.pattern) + 1):
            candidate = self.fit_window(text[start : start + len(self.pattern)], max_substitutions)
            if candidate and (best is None or candidate.substitutions < best.substitutions):
                best = candidate
        return best


#: Ontario passenger series issued since 1997, e.g. ``BXYZ 123``. It never
#: uses G, I, O, Q or U.
ONTARIO_PASSENGER = PlateFormat("ontario-passenger", "LLLLDDD", frozenset("GIOQU"))

#: The same shape with no knowledge of unused letters. This is what a plain
#: ``[A-Z]{4}[0-9]{3}`` regular expression accepts.
FOUR_LETTERS_THREE_DIGITS = PlateFormat("4L3D", "LLLLDDD")

_FORMAT_SETS: dict[str, tuple[PlateFormat, ...]] = {
    "ontario": (ONTARIO_PASSENGER,),
    "ontario-extended": (
        ONTARIO_PASSENGER,
        # Earlier passenger series still on the road.
        PlateFormat("ontario-passenger-1986", "DDDLLL", frozenset("GIQU")),
        PlateFormat("ontario-passenger-1973", "LLLDDD", frozenset("GIQ")),
        # Commercial series; the source lists no unused letters for these.
        PlateFormat("ontario-commercial", "LLDDDDD"),
        PlateFormat("ontario-commercial-1996", "DDDDLL"),
    ),
    "4L3D": (FOUR_LETTERS_THREE_DIGITS,),
    # Three letters, four digits: Brazil's series from 1990 to 2018, which is
    # what the "br" subset of the OpenALPR benchmark consists of.
    "3L4D": (PlateFormat("3L4D", "LLLDDDD"),),
    "any": (),
}

#: Shortest and longest text accepted when no format is given.
ANY_FORMAT_LENGTHS = (4, 8)


def format_set_names() -> list[str]:
    """Names accepted by :func:`get_formats`."""
    return sorted(_FORMAT_SETS)


def get_formats(name: str) -> tuple[PlateFormat, ...]:
    """Look up a named set of plate formats.

    Args:
        name: ``"ontario"``, ``"ontario-extended"``, ``"4L3D"``, ``"3L4D"`` or
            ``"any"``.
            ``"any"`` returns no formats, which :func:`fit_plate` treats as
            "accept any text of plausible length".

    Raises:
        ValueError: If the name is unknown.
    """
    try:
        return _FORMAT_SETS[name]
    except KeyError:
        raise ValueError(
            f"unknown plate format {name!r}; choose from {format_set_names()}"
        ) from None


def fit_plate(
    text: str,
    formats: Sequence[PlateFormat],
    *,
    max_substitutions: int = 2,
) -> FormatFit | None:
    """Extract a plate serial from OCR text, correcting look-alike characters.

    Args:
        text: Raw or normalized OCR output.
        formats: Formats to try. With several, the fit needing the fewest
            corrections wins, then the longest pattern, then the earliest in
            the sequence. With none, any text of plausible length is accepted
            unchanged.
        max_substitutions: Most characters that may be corrected. The default
            of 2 keeps at least one genuine digit in a three-digit group,
            which stops slogans such as "YOURS TO DISCOVER" from being turned
            into plates. ``0`` accepts exact matches only.

    Returns:
        The best fit, or ``None`` if the text contains no plausible serial.
    """
    text = normalize(text)
    if not formats:
        shortest, longest = ANY_FORMAT_LENGTHS
        return FormatFit(text, 0, "any") if shortest <= len(text) <= longest else None

    best: FormatFit | None = None
    best_key: tuple[int, int] | None = None
    for plate_format in formats:
        candidate = plate_format.fit(text, max_substitutions)
        if candidate is None:
            continue
        key = (candidate.substitutions, -len(plate_format.pattern))
        if best_key is None or key < best_key:
            best, best_key = candidate, key
    return best
