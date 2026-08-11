"""BioBERT embedding backbone shared by the classifiers, eligibility ranker, and
scoring. Embeddings are L2-normalised and cached on disk per text-hash so the
expensive CPU encoding is computed once and reused across train/eval/predict.

Model: pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb (the same family
the official metric uses for semantic similarity). Runs on CPU; offline once the
weights are cached locally by huggingface.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "models" / "emb_cache"
# Prefer a bundled local copy (fully offline); else the HuggingFace id.
_LOCAL_BIOBERT = ROOT / "models" / "biobert"
BIOBERT = (str(_LOCAL_BIOBERT) if _LOCAL_BIOBERT.exists()
           else "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

_model = None


def _device() -> str:
    """COHORTX_DEVICE env override, else CUDA if available, else CPU.

    The final eval environment is CPU-only; on the DGX we use CUDA for speed.
    Embeddings are identical across devices, and are cached on disk regardless.
    """
    dev = os.environ.get("COHORTX_DEVICE")
    if dev:
        return dev
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(BIOBERT, device=_device())
    return _model


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def embed(texts: list[str], batch_size: int = 64, use_cache: bool = True) -> np.ndarray:
    """Return an (n, dim) float32 array of normalised embeddings for `texts`.

    Per-text disk cache keyed by content hash; only uncached texts are encoded.
    """
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    out: list[np.ndarray | None] = [None] * len(texts)
    todo_idx, todo_txt = [], []
    for i, t in enumerate(texts):
        t = t if isinstance(t, str) else ""
        if use_cache:
            fp = CACHE_DIR / f"{_key(t)}.npy"
            if fp.exists():
                out[i] = np.load(fp)
                continue
        todo_idx.append(i)
        todo_txt.append(t)

    if todo_txt:
        model = get_embedder()
        vecs = model.encode(todo_txt, batch_size=batch_size, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        vecs = vecs.astype(np.float32)
        for j, i in enumerate(todo_idx):
            out[i] = vecs[j]
            if use_cache:
                np.save(CACHE_DIR / f"{_key(texts[i] or '')}.npy", vecs[j])

    return np.vstack(out)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarities between rows of a and rows of b (both assumed L2-normed)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return a @ b.T
