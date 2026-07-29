"""Session 10 — the evaluation harness: how good is this thing, and versus what?

    uv run python eval/run_eval.py                    # everything: metrics + latency
    uv run python eval/run_eval.py --system clip      # one system
    uv run python eval/run_eval.py --no-latency       # skip the timing pass
    uv run python eval/run_eval.py --failures 5       # worst queries, with results
    uv run python eval/run_eval.py --markdown         # README-ready tables

Three metrics, because each answers a different question:

* **Precision@10** — of the ten frames on screen, how many are right? This is what
  the user experiences.
* **Recall@10** — of all the relevant photos we know about, how many made the first
  page? Coverage. Needs "all relevant", which at 25k photos we approximate by
  pooling — so it is *optimistic* and only meaningful as a comparison between
  systems that were pooled together (see eval/POLICY.md).
* **MRR** — 1/(rank of the first relevant hit), averaged. How far down the page
  before something useful appears; a query with a great #1 and nothing else still
  scores well, which matches how people actually search.

Unjudged photos count as not relevant — the standard convention, and the reason the
pool includes both systems' top hits (a system whose results were never looked at
would otherwise score zero by construction).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from photosearch.evaluation import (
    QueryScore,
    by_bucket,
    drop_unanswerable,
    score_query,
    summarize,
)

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
JUDGMENTS_PATH = EVAL_DIR / "judgments.json"

K = 10
BUCKET_ORDER = ["easy", "compositional", "abstract", "jargon"]


def load_judgments() -> dict:
    if not JUDGMENTS_PATH.exists():
        raise SystemExit(
            "no judgments yet — run:\n"
            "  python eval/label.py pool\n"
            "  python eval/label.py sheets\n"
            "  python eval/label.py record --query e1 --relevant 1,4,7"
        )
    with JUDGMENTS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def build_system(name: str, store_kind: str):
    """Return an object with ``search(query, k) -> list[Result]``.

    Both a CLIP ``SearchService`` and the BM25 baseline already have that shape, which
    is the whole reason the harness can treat them interchangeably.
    """
    if name == "bm25":
        from photosearch.baseline import Bm25Baseline

        return Bm25Baseline.load(DATA_DIR)

    from photosearch.encoder import load_encoder
    from photosearch.search import SearchService
    from photosearch.store import load_store

    # "clip-onnx" is the torch-free encoder Session 11b deploys. Running the *same*
    # gold set through it turns "cosine 0.9999 vs the reference" — a number nobody can
    # interpret — into "P@10 moved by this much", which is the only question that
    # matters when deciding whether a deploy-shaped model is good enough to ship.
    kind = "onnx" if name == "clip-onnx" else "clip"
    return SearchService(load_encoder(kind), load_store(store_kind, DATA_DIR))


def evaluate(system, judgments: dict) -> list[QueryScore]:
    scores = []
    for query_id, judged in judgments.items():
        results = system.search(judged["query"], k=K)
        scores.append(score_query(query_id, judged, [r.photo_id for r in results], k=K))
    return scores


# --- reporting -------------------------------------------------------------------


def table(rows: list[list[str]], headers: list[str], markdown: bool) -> str:
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    def line(cells, pad=" "):
        body = " | ".join(str(c).ljust(w, pad) for c, w in zip(cells, widths, strict=True))
        return f"| {body} |" if markdown else f"  {body}"
    out = [line(headers)]
    out.append(
        "|" + "|".join("-" * (w + 2) for w in widths) + "|"
        if markdown
        else "  " + "-+-".join("-" * w for w in widths)
    )
    out.extend(line(r) for r in rows)
    return "\n".join(out)


def report_metrics(results: dict[str, list[QueryScore]], markdown: bool) -> None:
    print("\n=== retrieval quality (K=10) ===\n")
    rows = [
        [name, f"{s['P@10']:.3f}", f"{s['R@10']:.3f}", f"{s['MRR']:.3f}", str(s["n"])]
        for name, s in ((n, summarize(v)) for n, v in results.items())
    ]
    print(table(rows, ["system", "P@10", "R@10", "MRR", "queries"], markdown))

    if len(results) == 2 and "clip" in results and "bm25" in results:
        clip, bm25 = summarize(results["clip"]), summarize(results["bm25"])
        gap = clip["P@10"] - bm25["P@10"]
        print(
            f"\n  CLIP beats BM25-over-captions by {gap:+.3f} P@10 "
            f"({clip['P@10']:.1%} vs {bm25['P@10']:.1%} of the first page relevant)."
        )

    print("\n=== P@10 by query bucket ===\n")
    buckets = {name: by_bucket(scores, BUCKET_ORDER, K) for name, scores in results.items()}
    rows = []
    for bucket in BUCKET_ORDER:
        row = [bucket]
        for name in results:
            stats = buckets[name].get(bucket)
            row.append(f"{stats['P@10']:.3f}" if stats else "-")
        rows.append(row)
    print(table(rows, ["bucket", *results.keys()], markdown))


def report_latency(stores: list[str], queries: list[str], markdown: bool) -> None:
    from photosearch.encoder import Encoder
    from photosearch.search import SearchService
    from photosearch.store import load_store

    print("\n=== latency over the eval queries (median) ===\n")
    encoder = Encoder()
    rows = []
    for kind in stores:
        try:
            service = SearchService(encoder, load_store(kind, DATA_DIR))
        except Exception as exc:  # noqa: BLE001 - chroma may simply not be ingested
            print(f"  ({kind} unavailable: {exc})")
            continue
        service.search("warmup", k=K)  # never time the first call
        encode_ms, search_ms = [], []
        for query in queries:
            _, timing = service.search_timed(query, k=K)
            encode_ms.append(timing.encode_ms)
            search_ms.append(timing.search_ms)
        rows.append(
            [
                kind,
                f"{service.store.count():,}",
                f"{statistics.median(encode_ms):.1f}",
                f"{statistics.median(search_ms):.2f}",
                f"{statistics.median(e + s for e, s in zip(encode_ms, search_ms, strict=True)):.1f}",
            ]
        )
    print(table(rows, ["store", "vectors", "encode ms", "search ms", "total ms"], markdown))
    print("\n  Encoding dominates: the text encoder is a neural forward pass, the search")
    print("  is a dot product. That is why brute force was never the bottleneck.")


def report_failures(scores: list[QueryScore], n: int, system) -> None:
    print(f"\n=== worst {n} queries (CLIP) ===\n")
    for score in sorted(scores, key=lambda s: (s.precision, s.reciprocal_rank))[:n]:
        print(f"  [{score.bucket}] {score.query!r}")
        print(
            f"    P@10={score.precision:.2f}  R@10={score.recall:.2f}  "
            f"RR={score.reciprocal_rank:.2f}  ({score.n_relevant} relevant in the pool)"
        )
        for rank, (result, hit) in enumerate(
            zip(system.search(score.query, k=5), score.hits, strict=False), 1
        ):
            mark = "OK  " if hit else "MISS"
            caption = (result.ai_description or result.description or "")[:64]
            print(f"    {mark} {rank}. {result.score:.3f}  {caption}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument(
        "--system",
        choices=["clip", "clip-onnx", "bm25", "both", "deploy"],
        default="both",
        help='"deploy" runs clip vs clip-onnx — the deployed encoder against the real one',
    )
    parser.add_argument("--store", choices=["numpy", "chroma"], default="numpy",
                        help="which back-end serves the CLIP system")
    parser.add_argument("--no-latency", action="store_true", help="skip the timing pass")
    parser.add_argument("--latency-stores", default="numpy,chroma")
    parser.add_argument("--failures", type=int, default=3, help="show the N worst queries")
    parser.add_argument("--markdown", action="store_true", help="emit README-ready tables")
    args = parser.parse_args()

    judgments = load_judgments()
    print(f"[eval] {len(judgments)} labeled queries, "
          f"{sum(len(j['relevant']) for j in judgments.values())} relevant photos, "
          f"{sum(len(j['judged']) for j in judgments.values())} pooled judgments")

    # A query with nothing relevant in the pool scores 0 for every system and tells you
    # nothing about ranking — the standard practice (TREC) is to drop it from the
    # averages and report it separately. It's a finding about the *corpus*, not a bug.
    kept, unanswerable = drop_unanswerable(judgments)
    for qid in unanswerable:
        print(f"[eval] excluded (no relevant photo in the pool): "
              f"{qid} — {judgments[qid]['query']!r}")
    judgments = kept

    presets = {"both": ["clip", "bm25"], "deploy": ["clip", "clip-onnx"]}
    names = presets.get(args.system, [args.system])
    results: dict[str, list[QueryScore]] = {}
    clip_system = None
    for name in names:
        t0 = time.perf_counter()
        system = build_system(name, args.store)
        if name == "clip":
            clip_system = system
        results[name] = evaluate(system, judgments)
        print(f"[eval] {name}: {len(results[name])} queries in {time.perf_counter() - t0:.1f}s")

    report_metrics(results, args.markdown)

    if args.failures and clip_system is not None:
        report_failures(results["clip"], args.failures, clip_system)

    if not args.no_latency:
        report_latency(
            [s.strip() for s in args.latency_stores.split(",") if s.strip()],
            [j["query"] for j in judgments.values()],
            args.markdown,
        )


if __name__ == "__main__":
    main()
