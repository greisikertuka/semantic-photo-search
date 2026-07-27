# Decisions

An engineering log — every non-obvious choice, and *why* it beat the alternatives. Written as decisions are made, not reconstructed after the fact. (This file is deliberately part of the portfolio: it's the "why" behind the code.)

---

## Session 0 — Toolchain & project skeleton

### uv as the package manager
**Decision:** Use [uv](https://docs.astral.sh/uv/) (v0.11.x) for dependency management, virtual environments, Python installation, and command running.

**Why:** It is the mainstream Python tool in 2026 and maps cleanly onto tooling I already know:
`pyproject.toml` ≈ `package.json`, `uv.lock` ≈ `package-lock.json`, `.venv` ≈ `node_modules`, `uv run` ≈ `npx`. It also manages the Python interpreter itself, so there is no separate Python install step. Both `pyproject.toml` and `uv.lock` are committed; the lock is never hand-edited.

**Alternatives:** pip + venv + pyenv + pip-tools is the traditional stack — four tools where uv is one, and slower. Poetry/PDM are closer but less fast and less dominant now.

### Python 3.12 (pinned)
**Decision:** Pin Python to 3.12 via `.python-version` and `requires-python = ">=3.12"`.

**Why:** 3.12 is the newest version supported by the *entire* stack **and** it matches the Hugging Face Space runtime (ZeroGPU pins Python 3.12.12), giving local/deploy parity. 3.13 works locally but buys nothing here and loses that parity.

### PyTorch CPU index pin — the one non-obvious config
**Decision:** Pin torch to the CPU-only wheel index in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
```

**Why:** On Windows, PyPI serves CPU-only torch wheels anyway — but on **Linux** (any deploy target: HF Space, Render) the default resolves to multi-GB CUDA builds. This app never uses a GPU, so those builds are pure bloat that would blow the free-tier RAM/disk budgets. Pinning the CPU index on day one keeps every future Linux build small. A classic "works on my machine" trap, inverted — the machine that would break is the *server*, not the laptop.
Verified: the resolved install reports `torch==2.13.0+cpu`.
