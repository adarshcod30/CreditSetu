"""
Filesystem-based model registry: versioned artifacts + a metadata sidecar,
instead of overwriting one fixed pickle path with no history.

For each logical model path (e.g. "data/models/capacity_model.pkl"), save()
writes a new versioned file (".v{N}.pkl"), a ".meta.json" sidecar recording
when it was trained and with what metrics/profile, and updates a
".latest.json" pointer. load() follows the pointer to the current version and
returns both the model and its metadata.

This is deliberately filesystem-only so the library has zero new
infrastructure dependencies — swap this module for an MLflow/S3-backed
registry in a larger deployment without touching the engines that call it.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _pointer_path(path: Path) -> Path:
    return path.with_suffix(".latest.json")


def _versioned_path(path: Path, version: int) -> Path:
    return path.with_suffix(f".v{version}{path.suffix}")


def _metadata_path(versioned_path: Path) -> Path:
    return versioned_path.with_suffix(versioned_path.suffix + ".meta.json")


def save(model: Any, path: str | Path, metadata: Optional[dict] = None) -> Path:
    """Save `model` as the next version under the logical `path`, with metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pointer = _pointer_path(path)
    next_version = 1
    if pointer.exists():
        try:
            next_version = int(json.loads(pointer.read_text())["version"]) + 1
        except (json.JSONDecodeError, KeyError, ValueError):
            next_version = 1

    versioned = _versioned_path(path, next_version)
    with open(versioned, "wb") as f:
        pickle.dump(model, f)

    meta = dict(metadata or {})
    meta.setdefault("trained_at", datetime.now(timezone.utc).isoformat())
    meta["version"] = next_version
    with open(_metadata_path(versioned), "w") as f:
        json.dump(meta, f, indent=2)

    pointer.write_text(json.dumps({"version": next_version, "file": versioned.name}, indent=2))
    return versioned


def load(path: str | Path) -> tuple[Any, dict]:
    """Load the latest version registered under the logical `path`. Returns (model, metadata)."""
    path = Path(path)
    pointer = _pointer_path(path)

    if pointer.exists():
        data = json.loads(pointer.read_text())
        versioned = path.parent / data["file"]
    elif path.exists():
        # Backward-compat: a plain pickle file not produced by this registry.
        versioned = path
    else:
        raise FileNotFoundError(f"No model found at '{path}' or its registry pointer '{pointer}'")

    with open(versioned, "rb") as f:
        model = pickle.load(f)

    metadata: dict = {}
    meta_path = _metadata_path(versioned)
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    return model, metadata
