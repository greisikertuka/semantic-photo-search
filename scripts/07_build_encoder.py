"""Session 11b — build the torch-free CLIP text encoder the Render deploy runs on.

    uv run --group render python scripts/07_build_encoder.py
    uv run --group render python scripts/07_build_encoder.py --sweep      # the evidence
    uv run --group render python scripts/07_build_encoder.py --quantize   # small fallback

Writes ``data/encoder/``: the ONNX text tower plus its tokenizer, ~256 MB, no PyTorch
required to run it.

Why this script exists, and why it does *not* quantize by default
-----------------------------------------------------------------
The plan for this session said: Render's free tier is 512 MB, the fp32 text encoder
is 254 MB, so quantize it to the ready-made 64 MB int8 export. Two measurements
killed that plan, in a useful order.

**First: the 64 MB int8 export is broken for this model.** Measured against fp32 on
the same queries, it agrees at cosine 0.88 and keeps 8% of top-1 results (``--sweep``
reproduces the table). The cause is *per-tensor* quantization — one scale factor for
an entire weight matrix — meeting a transformer's activation outliers. The 4-bit
exports are far better (cosine 0.988) but still cost 0.09 P@10 on the Session 10 gold
set, which is a real, user-visible chunk of quality.

**Then: the memory problem wasn't the model's size at all.** ONNX stores weights
*inside* the graph protobuf by default, so loading a 254 MB model means parsing 254 MB
and then materializing every initializer again — the whole app peaked at 598 MB. Move
those same weights to an external data file and ONNX Runtime memory-maps them instead:

    peak RSS 598 MB -> 400 MB,  encode 26 ms,  accuracy loss exactly zero

That fits, with headroom, and mmapped pages are file-backed — under memory pressure
the kernel evicts them rather than the OOM killer taking the process. So the fix was
never fewer bits; it was not copying the weights. Quantization was the answer to a
question we hadn't measured yet.

``--quantize`` still builds a small model (block-wise 8-bit, cosine 0.9999 — the
*right* way to quantize this model, unlike the ready-made export) as a fallback if the
free tier ever gets tighter. It is a third the size and, on this graph, ~12x slower
per query, which is its own lesson about what "smaller" costs.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "encoder"

SOURCE_REPO = "Xenova/clip-vit-base-patch32"  # the same weights as clip-ViT-B-32, exported
SOURCE_FILE = "onnx/text_model.onnx"
TOKENIZER_FILE = "tokenizer.json"

FP32_NAME = "text_model.onnx"
QUANTIZED_NAME = "text_model_int8_block128.onnx"
BITS, BLOCK_SIZE = 8, 128

PROBES = [
    "golden hour by the sea",
    "a dog running on the beach",
    "loneliness",
    "long exposure light trails",
    "a man in a red jacket on a mountain",
    "shallow depth of field portrait with creamy bokeh",
    "night street, neon reflections in the rain",
    "leading lines",
    "nostalgia",
    "two people sitting on a bench by the water",
    "a street without any people",
    "misty forest at dawn",
]

# Every ready-made export in the source repo — so --sweep reproduces DECISIONS.md
# rather than asking anyone to take the table on faith.
SWEEP_VARIANTS = [
    "text_model.onnx",
    "text_model_fp16.onnx",
    "text_model_q4.onnx",
    "text_model_q4f16.onnx",
    "text_model_quantized.onnx",  # the plan's pick — the one that fails
    "text_model_uint8.onnx",
]


def fetch(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(SOURCE_REPO, filename))


def encode_all(model_path: Path, tokenizer_path: Path) -> np.ndarray:
    """Unit vectors for every probe, straight through ONNX Runtime."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=77)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    vectors = []
    for text in PROBES:
        ids = np.asarray([tokenizer.encode(text).ids], dtype=np.int64)
        vec = session.run(["text_embeds"], {"input_ids": ids})[0][0].astype(np.float32)
        vectors.append(vec / np.linalg.norm(vec))
    return np.stack(vectors)


def ranking_agreement(vectors: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Do the top-10 lists over the *real* index survive?

    Cosine between two query vectors is an abstraction; what a user sees is the ranked
    page. Skipped rather than faked when the index hasn't been built.
    """
    embeddings_path = PROJECT_ROOT / "data" / "embeddings.npy"
    if not embeddings_path.is_file():
        return float("nan"), float("nan")
    embeddings = np.load(embeddings_path)

    def top10(vec: np.ndarray) -> np.ndarray:
        return np.argsort(-(embeddings @ vec))[:10]

    pairs = [(top10(a), top10(b)) for a, b in zip(vectors, reference, strict=True)]
    overlap = float(np.mean([len(set(a) & set(b)) for a, b in pairs]))
    top1 = float(np.mean([a[0] == b[0] for a, b in pairs]))
    return overlap, top1


def report(label: str, model: Path, tokenizer: Path, reference: np.ndarray) -> np.ndarray:
    vectors = encode_all(model, tokenizer)
    cosines = (vectors * reference).sum(axis=1)
    overlap, top1 = ranking_agreement(vectors, reference)
    print(
        f"  {label:30s} {payload_mb(model):7.1f} MB  cos min {cosines.min():.4f} "
        f"mean {cosines.mean():.4f}   top-10 {overlap:4.1f}/10   top-1 {top1:4.0%}"
    )
    return vectors


def payload_mb(model: Path) -> float:
    """Size of the graph *plus* its external weights — what actually gets downloaded."""
    total = model.stat().st_size
    external = model.with_suffix(model.suffix + ".data")
    if external.is_file():
        total += external.stat().st_size
    return total / 1e6


def write_external(source: Path, destination: Path) -> None:
    """Re-save the graph with weights in a sidecar file, so ORT can memory-map them.

    This one call is the whole memory fix: identical weights, identical outputs, but
    the runtime maps them from disk instead of parsing and copying them into RAM.
    """
    import onnx

    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(
        onnx.load(str(source)),
        str(destination),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=destination.name + ".data",
        size_threshold=1024,  # tiny constants stay inline; only real weights move out
    )


def write_quantized(source: Path, destination: Path) -> None:
    """Block-wise 8-bit weights — the *correct* way to quantize this model.

    ``block_size`` is the knob the broken export got wrong: 128 weights share one
    scale factor instead of a whole matrix sharing one. Same 8 bits, ~200x more scale
    factors, and the error collapses from cosine 0.88 to 0.9999.
    """
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    quantizer = MatMulNBitsQuantizer(onnx.load(str(source)), bits=BITS, block_size=BLOCK_SIZE)
    quantizer.process()
    destination.parent.mkdir(parents=True, exist_ok=True)
    quantizer.model.save_model_to_file(str(destination), use_external_data_format=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", action="store_true",
                        help="measure every ready-made export against fp32 (the evidence)")
    parser.add_argument("--quantize", action="store_true",
                        help="build the block-wise 8-bit fallback instead of fp32")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    tokenizer = fetch(TOKENIZER_FILE)
    source = fetch(SOURCE_FILE)
    print(f"[source] {SOURCE_REPO}/{SOURCE_FILE}  {source.stat().st_size / 1e6:.1f} MB\n")

    reference = encode_all(source, tokenizer)
    print(f"[ref] {len(PROBES)} probe vectors from the fp32 model")

    if args.sweep:
        print("\n[sweep] every ready-made export, measured against fp32:")
        for name in SWEEP_VARIANTS:
            report(name, fetch(f"onnx/{name}"), tokenizer, reference)

    name = QUANTIZED_NAME if args.quantize else FP32_NAME
    destination = args.out / name
    if args.quantize:
        print(f"\n[build] block-wise {BITS}-bit, block_size={BLOCK_SIZE}, external weights ...")
        write_quantized(source, destination)
    else:
        print("\n[build] fp32, weights moved to an external file so ORT can mmap them ...")
        write_external(source, destination)

    shutil.copyfile(tokenizer, args.out / TOKENIZER_FILE)

    print("[verify] the built encoder, against the same fp32 reference:")
    report(name, destination, args.out / TOKENIZER_FILE, reference)

    total = sum(p.stat().st_size for p in args.out.iterdir()) / 1e6
    print(f"\n[done] {args.out}  —  {total:.1f} MB")
    for path in sorted(args.out.iterdir()):
        print(f"       {path.stat().st_size / 1e6:8.1f} MB  {path.name}")


if __name__ == "__main__":
    main()
