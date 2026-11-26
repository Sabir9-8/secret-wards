"""Pure-Python runtime scorer for the Secret Wards span disambiguator.

Uses ONLY Python 3.10+ standard library and lazily loads `rakshak/models/disambiguator.json`.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

from .ml_features import extract_features, vectorize_sparse

_MODEL_PATH = Path(__file__).resolve().parent / "models" / "disambiguator.json"
_MODEL_CACHE: dict | None = None
_MODEL_LOAD_FAILED = False


def _sigmoid(z: float) -> float:
    if z >= 30.0:
        return 1.0 - 1e-13
    if z <= -30.0:
        return 1e-13
    return 1.0 / (1.0 + math.exp(-z))


def _load_model() -> dict | None:
    global _MODEL_CACHE, _MODEL_LOAD_FAILED
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if _MODEL_LOAD_FAILED:
        return None
    try:
        raw = _MODEL_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict) or "vocab" not in data or "weights" not in data or "idf" not in data:
            _MODEL_LOAD_FAILED = True
            return None
        _MODEL_CACHE = data
        return _MODEL_CACHE
    except (OSError, ValueError):
        _MODEL_LOAD_FAILED = True
        if os.environ.get("RAKSHA_ML_DEBUG") == "1":
            print("[ml_scorer] disambiguator model missing or unreadable", file=sys.stderr)
        return None


def get_threshold(default: float = 0.72) -> float:
    model = _load_model()
    if model is None:
        return default
    return float(model.get("threshold", default))


def score_span(span: str, left_ctx: str = "", right_ctx: str = "") -> float | None:
    """Return P(PII_ENTITY | span, context) in [0, 1], or None if model is unavailable."""
    model = _load_model()
    if model is None:
        return None

    ngrams, struct = extract_features(span, left_ctx=left_ctx, right_ctx=right_ctx)
    vec = vectorize_sparse(ngrams, struct, model["vocab"], model["idf"])
    weights = model["weights"]
    z = float(model.get("bias", 0.0))
    for idx, val in vec.items():
        z += weights[idx] * val
    prob = _sigmoid(z)

    if os.environ.get("RAKSHA_ML_DEBUG") == "1":
        # Emit structural diagnostics only (never log raw PII spans or RAKSHA_SECRET)
        print(f"[ml_scorer] p={prob:.4f} active_feats={len(vec)}", file=sys.stderr)
    return prob
