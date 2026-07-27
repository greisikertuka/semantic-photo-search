"""Session 4 — a REPL to search the index from the terminal.

The first end-to-end proof the search core works on the real 25k corpus. Type a
query, get the top matches with scores and URLs. This is where the DoD queries
from the plan get their gut-check ("golden hour by the sea" had better return
golden-hour seascapes).

    uv run python scripts/03_search_cli.py
    uv run python scripts/03_search_cli.py --time          # print encode-ms / search-ms
    uv run python scripts/03_search_cli.py --k 5 "foggy empty street"   # one-shot

Filters (optional, combine freely):
    uv run python scripts/03_search_cli.py --aperture-max 2.0 --iso-max 800 "night street"
"""

from __future__ import annotations

import argparse

from photosearch.models import FilterSpec
from photosearch.search import SearchService
from photosearch.store import NumpyStore


def build_service() -> SearchService:
    from photosearch.encoder import Encoder

    print("[load] embeddings + parquet...")
    store = NumpyStore.load()
    print(f"[load] {len(store.photo_ids):,} photos indexed")
    print("[load] CLIP model (first run downloads ~600 MB)...")
    encoder = Encoder()
    return SearchService(encoder, store)


def filters_from_args(args: argparse.Namespace) -> FilterSpec:
    return FilterSpec(
        aperture_max=args.aperture_max,
        iso_max=args.iso_max,
        focal_min=args.focal_min,
        focal_max=args.focal_max,
        camera_make=args.camera_make,
    )


def run_query(service: SearchService, query: str, k: int, filters: FilterSpec, show_time: bool) -> None:
    results, timing = service.search_timed(query, k=k, filters=filters)
    if not results:
        print("  (no photos matched the active filters)")
        return
    for rank, r in enumerate(results, 1):
        exif = []
        if r.aperture is not None:
            exif.append(f"f/{r.aperture:g}")
        if r.focal_length is not None:
            exif.append(f"{r.focal_length:g}mm")
        if r.iso is not None:
            exif.append(f"ISO{r.iso}")
        tag = f"  [{', '.join(exif)}]" if exif else ""
        print(f"  {rank:>2}. {r.score:.3f}  {r.photographer:<22.22}  {r.photo_image_url}{tag}")
    if show_time:
        print(f"  -- encode {timing.encode_ms:.1f} ms | search {timing.search_ms:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the CLIP photo index.")
    parser.add_argument("query", nargs="?", help="run once and exit; omit for a REPL")
    parser.add_argument("--k", type=int, default=10, help="number of results (default 10)")
    parser.add_argument("--time", action="store_true", help="print encode-ms and search-ms")
    parser.add_argument("--aperture-max", type=float, default=None)
    parser.add_argument("--iso-max", type=int, default=None)
    parser.add_argument("--focal-min", type=float, default=None)
    parser.add_argument("--focal-max", type=float, default=None)
    parser.add_argument("--camera-make", type=str, default=None)
    args = parser.parse_args()

    service = build_service()
    filters = filters_from_args(args)

    if args.query:
        run_query(service, args.query, args.k, filters, args.time)
        return

    print('\nType a query (blank line or Ctrl-C to quit). e.g. "golden hour by the sea"\n')
    try:
        while True:
            query = input("search> ").strip()
            if not query:
                break
            run_query(service, query, args.k, filters, args.time)
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
