"""The store-agnostic data types that flow through search.

``FilterSpec`` is the single most important design decision in the project: it is
the *one language* for EXIF filters, spoken by both back-ends. ``NumpyStore``
(Session 4) compiles it into a boolean mask; ``ChromaStore`` (Session 7) compiles
the *same* object into a Chroma ``where=`` clause. One interface, two stores — the
seam that later lets the deployed Space filter with NumPy alone.

Both are ``dataclass``es — Python's version of a Java ``record``: a typed bag of
fields with a generated constructor, ``repr``, and equality.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class FilterSpec:
    """EXIF filter criteria. Every field is optional; ``None`` means "don't filter".

    The numeric fields are inclusive bounds (``aperture_max=2.0`` keeps f/2.0 and
    wider). ``camera_make`` matches case-insensitively against the normalized make.
    A photo with no value for a filtered field is *excluded* when that filter is
    active — you can't honour "aperture <= 2.0" for a photo whose aperture is
    unknown, so it drops out. That is the correct, if slightly surprising,
    behaviour, and it's why the UI later surfaces "searching N photos with EXIF".
    """

    aperture_max: float | None = None
    iso_max: int | None = None
    focal_min: float | None = None
    focal_max: float | None = None
    camera_make: str | None = None

    def is_active(self) -> bool:
        """True if any filter is set — lets stores skip masking entirely when not."""
        return any(getattr(self, f.name) is not None for f in fields(self))


@dataclass
class Result:
    """One ranked search hit: the join of an embedding's score with its display row.

    Carries everything the UI needs to render a card and a detail view, so the API
    layer never has to reach back into the parquet.
    """

    photo_id: str
    score: float
    photo_image_url: str
    photo_url: str
    photographer: str
    width: int | None
    height: int | None
    blur_hash: str | None
    description: str | None
    ai_description: str | None
    aperture: float | None
    focal_length: float | None
    exposure_s: float | None
    iso: int | None
    camera_make: str | None
    camera_model: str | None
