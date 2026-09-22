from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SEED = 42


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def normalize_doi(value: object) -> str:
    if pd.isna(value):
        return "missing_doi"
    s = str(value).strip().lower()
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s.rstrip("/.").strip()


def stable_lipid_key(smiles: object, lnp_id: object) -> str:
    if isinstance(smiles, str) and smiles.strip():
        return smiles.strip()
    return f"structure_missing_lnp_{lnp_id}"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(obj: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Unsupervised training-fold filter for highly correlated numeric columns."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        if arr.shape[1] < 2:
            self.keep_indices_ = np.arange(arr.shape[1])
            return self
        corr = np.corrcoef(arr, rowvar=False)
        corr = np.nan_to_num(np.abs(corr), nan=0.0)
        upper = np.triu(corr, k=1)
        drop = set(np.where(upper > self.threshold)[1].tolist())
        self.keep_indices_ = np.array([i for i in range(arr.shape[1]) if i not in drop])
        return self

    def transform(self, X):
        return X[:, self.keep_indices_]

