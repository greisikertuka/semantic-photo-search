"""Session 9a — the local-folder ingestion pipeline, tested without a model.

Everything here runs on JPEGs this file writes itself, so the suite stays fast and
offline. The two things genuinely worth testing are the ones that silently corrupt
an index rather than crashing it: EXIF coercion (a 1/500s shutter read as ``1.0``)
and the incremental diff (a re-run that re-embeds everything, or deletes a folder
it wasn't asked about).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from photosearch.library import (
    LOCAL_MAX_PIXELS,
    ScannedFile,
    _rational,
    empty_manifest,
    library_row,
    load_manifest,
    load_photo,
    photo_id_for,
    plan_changes,
    read_exif,
    save_manifest,
    scan_folder,
    write_thumbnail,
)
from photosearch.models import FilterSpec
from photosearch.store import build_result, exif_metadata


def write_jpeg(
    path: Path,
    size: tuple[int, int] = (60, 40),
    *,
    make: str | None = "NIKON CORPORATION",
    model: str | None = "NIKON Z 7",
    fnumber: float | None = 1.8,
    iso: int | None = 400,
    focal: float | None = 35.0,
    exposure: tuple[int, int] | None = (1, 500),
) -> Path:
    """A tiny real JPEG carrying a real EXIF block — the fixture these tests run on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    if make is not None:
        exif[0x010F] = make
    if model is not None:
        exif[0x0110] = model
    sub: dict = {}
    if exposure is not None:
        sub[0x829A] = exposure
    if fnumber is not None:
        sub[0x829D] = fnumber
    if iso is not None:
        sub[0x8827] = iso
    if focal is not None:
        sub[0x920A] = focal
    sub[0x9003] = "2024:07:14 18:32:01"
    if sub:
        exif[0x8769] = sub
    Image.new("RGB", size, (90, 120, 160)).save(path, "JPEG", exif=exif)
    return path


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "MyPhotos"
    write_jpeg(root / "Iceland" / "a.jpg")
    write_jpeg(root / "Iceland" / "b.jpg", fnumber=8.0, iso=64)
    write_jpeg(root / "Tokyo" / "c.jpg", make="SONY", model="ILCE-7M3")
    return root


class TestPhotoId:
    def test_is_stable_for_the_same_file(self, tmp_path: Path) -> None:
        path = write_jpeg(tmp_path / "x.jpg")
        assert photo_id_for(path) == photo_id_for(path)

    def test_differs_between_files(self, tmp_path: Path) -> None:
        a = write_jpeg(tmp_path / "a.jpg")
        b = write_jpeg(tmp_path / "b.jpg")
        assert photo_id_for(a) != photo_id_for(b)

    @pytest.mark.skipif(os.name != "nt", reason="only Windows treats paths case-insensitively")
    def test_case_insensitive_on_windows(self, tmp_path: Path) -> None:
        path = write_jpeg(tmp_path / "Photo.jpg")
        assert photo_id_for(path) == photo_id_for(str(path).upper())


class TestRationalCoercion:
    """The bug this guards: a ``(1, 500)`` shutter read as its numerator, i.e. 1 second."""

    def test_two_tuple_is_a_fraction_not_a_first_element(self) -> None:
        assert _rational((1, 500)) == pytest.approx(0.002)

    def test_plain_float_passes_through(self) -> None:
        assert _rational(1.8) == pytest.approx(1.8)

    def test_zero_denominator_is_none(self) -> None:
        assert _rational((1, 0)) is None

    def test_missing_is_none(self) -> None:
        assert _rational(None) is None


class TestReadExif:
    def test_maps_real_tags_onto_the_session2_schema(self, tmp_path: Path) -> None:
        path = write_jpeg(tmp_path / "shot.jpg")
        with Image.open(path) as img:
            exif = read_exif(img)
        assert exif["aperture"] == pytest.approx(1.8)
        assert exif["focal_length"] == pytest.approx(35.0)
        assert exif["exposure_s"] == pytest.approx(0.002)  # 1/500s, not 1s
        assert exif["iso"] == 400
        # normalize_make title-cases so "NIKON CORPORATION" and "Nikon" group together
        assert exif["camera_make"] == "Nikon Corporation"
        assert exif["camera_model"] == "NIKON Z 7"
        assert exif["taken_at"] == "2024-07-14 18:32:01"

    def test_a_file_with_no_exif_yields_all_none(self, tmp_path: Path) -> None:
        path = tmp_path / "bare.png"
        Image.new("RGB", (20, 20)).save(path)
        with Image.open(path) as img:
            exif = read_exif(img)
        assert set(exif.values()) == {None}

    def test_partial_exif_keeps_what_is_there(self, tmp_path: Path) -> None:
        path = write_jpeg(tmp_path / "partial.jpg", fnumber=None, iso=None, exposure=None)
        with Image.open(path) as img:
            exif = read_exif(img)
        assert exif["aperture"] is None
        assert exif["iso"] is None
        assert exif["focal_length"] == pytest.approx(35.0)
        assert exif["camera_make"] == "Nikon Corporation"


class TestScan:
    def test_finds_images_recursively(self, folder: Path) -> None:
        found = scan_folder(folder)
        assert [Path(f.path).name for f in found] == ["a.jpg", "b.jpg", "c.jpg"]
        assert all(f.size > 0 for f in found)

    def test_ignores_non_images_and_dot_directories(self, folder: Path) -> None:
        (folder / "notes.txt").write_text("not a photo")
        write_jpeg(folder / ".trash" / "deleted.jpg")
        assert len(scan_folder(folder)) == 3


class TestPlanChanges:
    def test_first_run_is_all_new(self, folder: Path) -> None:
        plan = plan_changes(empty_manifest(), scan_folder(folder), folder)
        assert len(plan.new) == 3
        assert not plan.changed and not plan.deleted_ids
        assert not plan.is_noop()

    def test_second_run_is_a_noop(self, folder: Path) -> None:
        scanned = scan_folder(folder)
        manifest = pd.DataFrame(
            [library_row(f, dict.fromkeys(
                ["aperture", "focal_length", "exposure_s", "iso",
                 "camera_make", "camera_model", "taken_at"]), 10, 10) for f in scanned]
        )
        plan = plan_changes(manifest, scanned, folder)
        assert plan.is_noop()
        assert len(plan.unchanged) == 3

    def test_touched_file_is_modified_and_missing_file_is_deleted(self, folder: Path) -> None:
        scanned = scan_folder(folder)
        manifest = pd.DataFrame(
            [library_row(f, dict.fromkeys(
                ["aperture", "focal_length", "exposure_s", "iso",
                 "camera_make", "camera_model", "taken_at"]), 10, 10) for f in scanned]
        )
        # one file grew (a re-export), one vanished
        edited = ScannedFile(scanned[0].path, scanned[0].mtime + 5.0, scanned[0].size + 12,
                             scanned[0].photo_id)
        plan = plan_changes(manifest, [edited, scanned[1]], folder)
        assert [f.photo_id for f in plan.changed] == [scanned[0].photo_id]
        assert plan.deleted_ids == [scanned[2].photo_id]

    def test_deletion_is_scoped_to_the_scanned_root(self, tmp_path: Path, folder: Path) -> None:
        """A second indexed folder must survive a re-run over the first one."""
        other = tmp_path / "Archive"
        outsider = scan_folder(other.parent / "MyPhotos")[0]
        elsewhere = write_jpeg(other / "old.jpg")
        stray = ScannedFile(str(elsewhere), 1.0, 10, photo_id_for(elsewhere))
        rows = [library_row(f, dict.fromkeys(
            ["aperture", "focal_length", "exposure_s", "iso",
             "camera_make", "camera_model", "taken_at"]), 10, 10) for f in (outsider, stray)]
        plan = plan_changes(pd.DataFrame(rows), scan_folder(folder), folder)
        assert stray.photo_id not in plan.deleted_ids


class TestRowsAndManifest:
    def test_row_renders_through_build_result_like_any_photo(self, folder: Path) -> None:
        scanned = scan_folder(folder)[0]
        _, exif, (width, height) = load_photo(scanned.path)
        row = library_row(scanned, exif, width, height)
        result = build_result(pd.Series(row), 0.42)

        assert result.photo_id == scanned.photo_id
        assert result.score == pytest.approx(0.42)
        # the containing folder stands in for the photographer on a personal archive
        assert result.photographer == "Iceland"
        assert result.description == "a.jpg"
        assert result.photo_image_url == f"/api/photo/{scanned.photo_id}/thumb"
        assert result.aperture == pytest.approx(1.8)
        assert result.camera_make == "Nikon Corporation"

    def test_row_produces_filterable_chroma_metadata(self, folder: Path) -> None:
        scanned = scan_folder(folder)[0]
        _, exif, (w, h) = load_photo(scanned.path)
        meta = exif_metadata(pd.Series(library_row(scanned, exif, w, h)))
        assert meta["has_exif"] is True
        assert meta["aperture"] == pytest.approx(1.8)
        assert meta["camera_make"] == "nikon corporation"  # lowercased for $eq
        assert FilterSpec(aperture_max=2.0).is_active()

    def test_manifest_round_trips_with_pinned_dtypes(self, tmp_path: Path, folder: Path) -> None:
        """An all-missing aperture column must still be numeric, or filters go silent."""
        rows = []
        for scanned in scan_folder(folder):
            blank = dict.fromkeys(
                ["aperture", "focal_length", "exposure_s", "iso",
                 "camera_make", "camera_model", "taken_at"]
            )
            rows.append(library_row(scanned, blank, 10, 10))
        save_manifest(pd.DataFrame(rows), tmp_path)
        loaded = load_manifest(tmp_path)
        assert len(loaded) == 3
        assert loaded["aperture"].dtype == "float64"
        assert str(loaded["iso"].dtype) == "Int64"

    def test_load_manifest_is_empty_when_nothing_is_indexed(self, tmp_path: Path) -> None:
        assert load_manifest(tmp_path).empty


class TestDecompressionBombCeiling:
    """The local indexer lifts Pillow's pixel guard; the upload endpoint must not.

    Two 200 MP phone panoramas were skipped on the first real archive run — legitimate
    data, refused by a guard meant for untrusted bytes. Lifting it is correct *here*
    and dangerous everywhere else, because ``MAX_IMAGE_PIXELS`` is global to the
    process and ``api.py`` decodes stranger-supplied uploads with the same Pillow.
    """

    def test_ceiling_is_above_pillows_default(self) -> None:
        assert LOCAL_MAX_PIXELS > 200_000_000  # the panoramas that were skipped

    def test_limit_is_restored_after_a_successful_load(self, folder: Path) -> None:
        before = Image.MAX_IMAGE_PIXELS
        load_photo(scan_folder(folder)[0].path)
        assert Image.MAX_IMAGE_PIXELS == before

    def test_limit_is_restored_even_when_the_file_is_unreadable(self, tmp_path: Path) -> None:
        before = Image.MAX_IMAGE_PIXELS
        broken = tmp_path / "broken.jpg"
        broken.write_bytes(b"not a jpeg at all")
        with pytest.raises(Exception):  # noqa: B017 - any decode failure will do
            load_photo(broken)
        assert Image.MAX_IMAGE_PIXELS == before


class TestThumbnails:
    def test_bounds_the_long_edge_and_writes_jpeg(self, tmp_path: Path) -> None:
        source = write_jpeg(tmp_path / "big.jpg", size=(2000, 1000))
        image, _, _ = load_photo(source)
        dest = tmp_path / "thumbs" / "t.jpg"
        write_thumbnail(image, dest)
        with Image.open(dest) as thumb:
            assert max(thumb.size) <= 640
            assert thumb.format == "JPEG"
        assert dest.stat().st_size < source.stat().st_size
