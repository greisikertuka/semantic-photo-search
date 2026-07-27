"""Unit tests for the EXIF string parsers.

Pure functions, no model or network — these run fast in CI. They're also the
regression net for Session 9, where real camera files feed messier input in.
"""

import math

import pytest

from photosearch.exif import (
    normalize_make,
    normalize_model,
    parse_aperture,
    parse_exposure,
    parse_focal_length,
    parse_iso,
)


class TestParseAperture:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("f/1.8", 1.8),
            ("f/2.8", 2.8),
            ("1.8", 1.8),
            ("F2.0", 2.0),
            ("f/22", 22.0),
        ],
    )
    def test_valid(self, raw, expected):
        assert parse_aperture(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["", "   ", "n/a", "abc", None, float("nan"), "f/0"])
    def test_invalid_returns_none(self, raw):
        assert parse_aperture(raw) is None


class TestParseExposure:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1/250", 0.004),
            ("1/1000", 0.001),
            ("0.5", 0.5),
            ("2", 2.0),
            ("1/250 s", 0.004),
        ],
    )
    def test_valid(self, raw, expected):
        assert parse_exposure(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["", "abc", None, float("nan"), "1/0"])
    def test_invalid_returns_none(self, raw):
        assert parse_exposure(raw) is None


class TestParseFocalLength:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("35", 35.0), ("35.0mm", 35.0), ("135 mm", 135.0), ("50.0", 50.0)],
    )
    def test_valid(self, raw, expected):
        assert parse_focal_length(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["", "mm", None, float("nan")])
    def test_invalid_returns_none(self, raw):
        assert parse_focal_length(raw) is None


class TestParseIso:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("800", 800), ("100", 100), ("6400", 6400), ("200.0", 200)],
    )
    def test_valid(self, raw, expected):
        result = parse_iso(raw)
        assert result == expected
        assert isinstance(result, int)

    @pytest.mark.parametrize("raw", ["", "abc", None, float("nan"), "0"])
    def test_invalid_returns_none(self, raw):
        assert parse_iso(raw) is None


class TestNormalizeCamera:
    def test_make_titlecases(self):
        assert normalize_make("NIKON CORPORATION") == "Nikon Corporation"
        assert normalize_make("  Canon  ") == "Canon"
        assert normalize_make("SONY") == "Sony"

    def test_model_preserves_case(self):
        assert normalize_model("iPhone 13 Pro") == "iPhone 13 Pro"
        assert normalize_model("  EOS  R5 ") == "EOS R5"

    @pytest.mark.parametrize("raw", ["", "   ", None, float("nan")])
    def test_blank_returns_none(self, raw):
        assert normalize_make(raw) is None
        assert normalize_model(raw) is None


def test_nan_is_missing_not_a_value():
    # A defensive check: NaN floats must be treated as missing, never stringified.
    for parser in (parse_aperture, parse_exposure, parse_focal_length, parse_iso):
        assert parser(math.nan) is None
