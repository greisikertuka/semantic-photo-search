"""Session 10 — build the judgment pool and label it.

Relevance judgments are the expensive part of evaluation, and that is the lesson:
metrics are cheap, *ground truth* is not. This is the smallest tool that makes
labeling a few hundred photos bearable.

    uv run python eval/label.py pool      # candidates per query (needs CLIP, ~1 min)
    uv run python eval/label.py sheets    # numbered contact sheets to eval/sheets/
    uv run python eval/label.py record --query e1 --relevant 1,4,7-9
    uv run python eval/label.py status    # coverage so far

**Pooling.** For each query the candidate set is the union of top-N from three
arms: the CLIP query itself, two hand-written rephrasings of it, and the BM25
baseline. Including *both systems'* top hits is the standard (TREC) way to keep a
comparison fair — a system whose results were never judged would otherwise be
scored as if every one of them were wrong. The honest caveat, which belongs in the
README: pooling from your own systems makes Recall@K *optimistic*, because a
relevant photo none of the arms surfaced is invisible to the metric.

**Labeling.** ``sheets`` writes one numbered contact sheet per query, so a human
judges thirty photos in one glance instead of thirty clicks; ``record`` takes the
numbers that were relevant. The policy for "relevant" lives in eval/POLICY.md —
writing it down first is what stops the labels from drifting halfway through.
"""

from __future__ import annotations

import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
QUERIES_PATH = EVAL_DIR / "queries.jsonl"
POOL_PATH = EVAL_DIR / "pool.json"
JUDGMENTS_PATH = EVAL_DIR / "judgments.json"
SHEET_DIR = EVAL_DIR / "sheets"
CACHE_DIR = EVAL_DIR / "cache"

# Pool depths per arm. Deep enough that both systems' top-10 are always judged
# (that's what makes P@10 meaningful), shallow enough that a human finishes.
CLIP_DEPTH = 12
REPHRASE_DEPTH = 6
BM25_DEPTH = 12

# Contact-sheet geometry.
COLUMNS = 6
CELL = (208, 156)
PAD = 6
HEADER = 34
THUMB_PARAMS = "?w=240&q=70"


def load_queries() -> list[dict]:
    with QUERIES_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


# --- pooling ---------------------------------------------------------------------


def build_pool() -> None:
    from photosearch.baseline import Bm25Baseline
    from photosearch.encoder import Encoder
    from photosearch.search import SearchService
    from photosearch.store import NumpyStore

    queries = load_queries()
    print(f"[pool] {len(queries)} queries, arms: clip({CLIP_DEPTH}) + "
          f"2 rephrasings({REPHRASE_DEPTH}) + bm25({BM25_DEPTH})")
    store = NumpyStore.load(DATA_DIR)
    clip = SearchService(Encoder(), store)
    bm25 = Bm25Baseline(store.photos)
    print(f"[pool] corpus {store.count():,} photos ({bm25.empty_docs:,} have no caption text)")

    pool: dict[str, dict] = {}
    for spec in queries:
        # candidate -> which arms found it, and its best rank in any arm (for ordering)
        found: dict[str, dict] = {}

        def add(results, arm: str, sink=found) -> None:
            for rank, r in enumerate(results, 1):
                entry = sink.setdefault(
                    r.photo_id,
                    {"photo_id": r.photo_id, "url": r.photo_image_url, "arms": [],
                     "best_rank": rank, "caption": r.ai_description or r.description or ""},
                )
                entry["arms"].append(arm)
                entry["best_rank"] = min(entry["best_rank"], rank)

        add(clip.search(spec["query"], k=CLIP_DEPTH), "clip")
        for i, rephrase in enumerate(spec.get("rephrasings", [])[:2]):
            add(clip.search(rephrase, k=REPHRASE_DEPTH), f"clip_rephrase{i + 1}")
        add(bm25.search(spec["query"], k=BM25_DEPTH), "bm25")

        # Deterministic order: photos several arms agreed on first, then by best rank.
        candidates = sorted(
            found.values(), key=lambda c: (-len(c["arms"]), c["best_rank"], c["photo_id"])
        )
        pool[spec["id"]] = {
            "query": spec["query"],
            "bucket": spec["bucket"],
            "candidates": candidates,
        }
        print(f"[pool] {spec['id']:>3}  {len(candidates):>3} candidates  {spec['query']}")

    save_json(POOL_PATH, {"clip_depth": CLIP_DEPTH, "bm25_depth": BM25_DEPTH, "queries": pool})
    total = sum(len(q["candidates"]) for q in pool.values())
    print(f"[pool] wrote {POOL_PATH.relative_to(PROJECT_ROOT)} — {total:,} judgments to make")


# --- contact sheets --------------------------------------------------------------


def cached_thumb(client: httpx.Client, photo_id: str, url: str) -> Path | None:
    """Fetch a small thumbnail once and keep it — sheets get rebuilt a lot."""
    dest = CACHE_DIR / f"{photo_id}.jpg"
    if dest.exists():
        return dest
    try:
        resp = client.get(url + THUMB_PARAMS, timeout=20, follow_redirects=True)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        image.thumbnail((CELL[0], CELL[1]))
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest, "JPEG", quality=78)
        return dest
    except Exception as exc:  # noqa: BLE001 - a deleted photo shouldn't kill the sheet
        print(f"[sheet]   !! {photo_id}: {exc}")
        return None


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


def build_sheet(query_id: str, entry: dict, client: httpx.Client) -> Path:
    candidates = entry["candidates"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(
            pool.map(lambda c: cached_thumb(client, c["photo_id"], c["url"]), candidates)
        )

    rows = (len(candidates) + COLUMNS - 1) // COLUMNS
    width = COLUMNS * (CELL[0] + PAD) + PAD
    height = HEADER + rows * (CELL[1] + PAD) + PAD
    sheet = Image.new("RGB", (width, height), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD, 9), f"{query_id}  |  {entry['query']}", fill=(240, 240, 240), font=_font(18))

    for i, (candidate, path) in enumerate(zip(candidates, paths, strict=True)):
        col, row = i % COLUMNS, i // COLUMNS
        x = PAD + col * (CELL[0] + PAD)
        y = HEADER + row * (CELL[1] + PAD)
        draw.rectangle([x, y, x + CELL[0], y + CELL[1]], fill=(34, 34, 38))
        if path is not None:
            with Image.open(path) as thumb:
                sheet.paste(thumb, (x + (CELL[0] - thumb.width) // 2,
                                    y + (CELL[1] - thumb.height) // 2))
        # the number a labeller types into `record`
        draw.rectangle([x, y, x + 30, y + 19], fill=(232, 134, 63))
        draw.text((x + 5, y + 3), str(i + 1), fill=(20, 16, 10), font=_font(15))
        arms = "".join(sorted({a[0] for a in candidate["arms"]}))  # c=clip, b=bm25
        draw.text((x + 34, y + 3), arms, fill=(150, 150, 150), font=_font(13))

    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    dest = SHEET_DIR / f"{query_id}.jpg"
    sheet.save(dest, "JPEG", quality=84)
    return dest


def build_sheets(only: str | None) -> None:
    pool = load_json(POOL_PATH, {})
    if not pool:
        raise SystemExit("no pool yet — run: python eval/label.py pool")
    with httpx.Client() as client:
        for query_id, entry in pool["queries"].items():
            if only and query_id != only:
                continue
            dest = build_sheet(query_id, entry, client)
            print(f"[sheet] {query_id}: {len(entry['candidates']):>3} cells -> {dest}")


# --- recording judgments ---------------------------------------------------------


def parse_numbers(spec: str) -> list[int]:
    """``"1,4,7-9"`` -> ``[1, 4, 7, 8, 9]``. Empty string -> no relevant photos."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def record(query_id: str, numbers_spec: str) -> None:
    pool = load_json(POOL_PATH, {})
    entry = pool.get("queries", {}).get(query_id)
    if entry is None:
        raise SystemExit(f"unknown query id {query_id!r}")
    candidates = entry["candidates"]

    numbers = parse_numbers(numbers_spec)
    bad = [n for n in numbers if not 1 <= n <= len(candidates)]
    if bad:
        raise SystemExit(f"out of range for {query_id} (1..{len(candidates)}): {bad}")

    judgments = load_json(JUDGMENTS_PATH, {})
    judgments[query_id] = {
        "query": entry["query"],
        "bucket": entry["bucket"],
        # every pooled photo was looked at; the rest of the corpus is simply unjudged
        "judged": [c["photo_id"] for c in candidates],
        "relevant": [candidates[n - 1]["photo_id"] for n in numbers],
    }
    save_json(JUDGMENTS_PATH, judgments)
    print(f"[record] {query_id}: {len(numbers)}/{len(candidates)} relevant — {entry['query']}")


def status() -> None:
    pool = load_json(POOL_PATH, {}).get("queries", {})
    judgments = load_json(JUDGMENTS_PATH, {})
    if not pool:
        raise SystemExit("no pool yet — run: python eval/label.py pool")

    rows = []
    for query_id, entry in pool.items():
        judged = judgments.get(query_id)
        rows.append(
            {
                "id": query_id,
                "bucket": entry["bucket"],
                "pooled": len(entry["candidates"]),
                "relevant": len(judged["relevant"]) if judged else None,
                "query": entry["query"],
            }
        )
    frame = pd.DataFrame(rows)
    done = frame["relevant"].notna().sum()
    print(frame.to_string(index=False))
    print(f"\nlabeled {done}/{len(frame)} queries, "
          f"{int(frame['pooled'].sum()):,} pooled judgments, "
          f"{int(frame['relevant'].fillna(0).sum()):,} marked relevant")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and label the evaluation pool.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pool", help="build the candidate pool (runs CLIP + BM25)")
    sheets = sub.add_parser("sheets", help="write numbered contact sheets")
    sheets.add_argument("--query", type=str, default=None, help="only this query id")
    rec = sub.add_parser("record", help="save which numbered cells were relevant")
    rec.add_argument("--query", type=str, required=True)
    rec.add_argument("--relevant", type=str, required=True, help='e.g. "1,4,7-9" (or "" for none)')
    sub.add_parser("status", help="labeling coverage so far")
    args = parser.parse_args()

    if args.command == "pool":
        build_pool()
    elif args.command == "sheets":
        build_sheets(args.query)
    elif args.command == "record":
        record(args.query, args.relevant)
    else:
        status()


if __name__ == "__main__":
    main()
