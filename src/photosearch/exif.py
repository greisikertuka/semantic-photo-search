"""Parsers that turn messy EXIF strings into clean, filterable numeric types.

These are pure functions so they can be unit-tested (``tests/test_exif.py``) and
reused by *both* the Unsplash dataset prep (Session 2) and the local-folder
ingestion pipeline (Session 9). The Unsplash TSV stores EXIF as strings with many
blanks; real camera files store them as rationals and vendor-specific junk. Both
funnel through here into the same schema.

Convention: every parser returns ``None`` when the input can't be parsed (missing,
blank, or garbage). Callers map ``None`` to ``NaN`` so unparseable values simply
drop out of numeric filtering later — a photo with no aperture can't match
"aperture <= 2.0", which is the correct behaviour.
"""

from __future__ import annotations

import math
import re

# Matches the first (possibly decimal, possibly signed) number in a string.
_NUMBER = re.compile(r"[-+]?\d*\.?\d+")


def _clean(value: object) -> str | None:
    """Return a stripped string, or ``None`` for missing/blank input.

    Handles the three "missing" shapes we see: Python ``None``, pandas ``NaN``
    (a float), and empty/whitespace-only strings.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _first_number(text: str) -> float | None:
    match = _NUMBER.search(text)
    if match is None:
        return None
    try:
        return float(match.group())
    except ValueError:  # pragma: no cover - regex guarantees a numeric token
        return None


def parse_aperture(value: object) -> float | None:
    """``"f/1.8"`` -> ``1.8``. Also accepts ``"1.8"``, ``"F2.8"``. Garbage -> None."""
    text = _clean(value)
    if text is None:
        return None
    number = _first_number(text)
    if number is None or number <= 0:
        return None
    return number


def parse_focal_length(value: object) -> float | None:
    """``"35"`` / ``"35.0mm"`` -> ``35.0``. Garbage -> None."""
    text = _clean(value)
    if text is None:
        return None
    number = _first_number(text)
    if number is None or number <= 0:
        return None
    return number


def parse_iso(value: object) -> int | None:
    """``"800"`` -> ``800`` (int). Garbage -> None.

    Kept as a Python ``int``/``None`` so the DataFrame column can use the nullable
    ``"Int64"`` dtype — ordinary ``int`` can't represent missing values.
    """
    text = _clean(value)
    if text is None:
        return None
    number = _first_number(text)
    if number is None or number <= 0:
        return None
    return round(number)


def parse_exposure(value: object) -> float | None:
    """Exposure time to float seconds. ``"1/250"`` -> ``0.004``; ``"0.5"`` -> ``0.5``.

    Handled specially (not via ``_first_number``) because a fraction like
    ``"1/250"`` would otherwise be read as just ``1``.
    """
    text = _clean(value)
    if text is None:
        return None
    text = text.rstrip("sS").strip()  # tolerate a trailing "s" / "sec"
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        try:
            num = float(numerator)
            den = float(denominator)
        except ValueError:
            return None
        if den == 0:
            return None
        result = num / den
    else:
        try:
            result = float(text)
        except ValueError:
            return None
    return result if result > 0 else None


def _normalize_text(value: object) -> str | None:
    text = _clean(value)
    if text is None:
        return None
    # Collapse runs of internal whitespace to single spaces.
    return re.sub(r"\s+", " ", text)


def normalize_make(value: object) -> str | None:
    """Camera manufacturer, tidied: ``"NIKON CORPORATION"`` -> ``"Nikon Corporation"``.

    Title-casing gives one canonical spelling to filter/group on across the many
    ways makers write their own name (``Canon``, ``CANON``, ``SONY`` ...).
    """
    text = _normalize_text(value)
    if text is None:
        return None
    return text.title()


def normalize_model(value: object) -> str | None:
    """Camera model, whitespace-normalized but case-preserved.

    Unlike make, models must keep their casing (``"iPhone 13 Pro"``,
    ``"EOS R5"``) — title-casing would corrupt them.
    """
    return _normalize_text(value)
