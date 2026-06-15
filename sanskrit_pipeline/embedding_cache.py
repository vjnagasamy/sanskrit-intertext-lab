"""Persistent on-disk cache for segmented and embedded Sanskrit documents."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class CacheEntry:
    """One cached segmentation + embedding result."""

    segments: list[str]
    embeddings: np.ndarray


class EmbeddingCache:
    """File-content-keyed cache for document segments and embeddings."""

    def __init__(self, cache_root: str | Path = ".embedding_cache") -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def get(
        self,
        file_path: str | Path,
        model_id: str,
        engine: str,
        split_on_single_danda: bool,
        *,
        is_query: bool = False,
        source_format: str = "devanagari",
    ) -> CacheEntry | None:
        """Return a cached entry or None if missing or invalid."""
        key = self._cache_key(
            file_path,
            model_id,
            engine,
            split_on_single_danda,
            is_query=is_query,
            source_format=source_format,
        )
        entry_dir = self.cache_root / key
        segments_path = entry_dir / "segments.json"
        embeddings_path = entry_dir / "embeddings.npy"
        meta_path = entry_dir / "meta.json"

        if not segments_path.exists() or not embeddings_path.exists() or not meta_path.exists():
            return None

        try:
            segments = json.loads(segments_path.read_text(encoding="utf-8"))
            if not isinstance(segments, list):
                return None
            embeddings = np.load(embeddings_path, mmap_mode="r")
            if embeddings.ndim != 2:
                return None
            if embeddings.shape[0] != len(segments):
                return None
        except (OSError, ValueError, json.JSONDecodeError):
            return None

        return CacheEntry(segments=segments, embeddings=embeddings)

    def put(
        self,
        file_path: str | Path,
        model_id: str,
        engine: str,
        split_on_single_danda: bool,
        entry: CacheEntry,
        *,
        is_query: bool = False,
        source_format: str = "devanagari",
    ) -> None:
        """Persist one cache entry using atomic embedding writes."""
        key = self._cache_key(
            file_path,
            model_id,
            engine,
            split_on_single_danda,
            is_query=is_query,
            source_format=source_format,
        )
        entry_dir = self.cache_root / key
        entry_dir.mkdir(parents=True, exist_ok=True)

        embeddings_path = entry_dir / "embeddings.npy"
        tmp_path = entry_dir / f"embeddings.tmp.{os.getpid()}.npy"
        np.save(tmp_path, entry.embeddings.astype(np.float32, copy=False))
        os.replace(tmp_path, embeddings_path)

        segments_path = entry_dir / "segments.json"
        segments_path.write_text(
            json.dumps(entry.segments, ensure_ascii=False),
            encoding="utf-8",
        )

        meta = {
            "file_path": str(Path(file_path).resolve()),
            "model_id": model_id,
            "engine": engine,
            "split_on_single_danda": split_on_single_danda,
            "is_query": is_query,
            "source_format": source_format,
            "created_at": time.time(),
            "n_segments": len(entry.segments),
        }
        meta_path = entry_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def stats(self) -> dict[str, float | int]:
        """Return entry count and total on-disk size in megabytes."""
        entry_count = 0
        total_bytes = 0
        for entry_dir in self.cache_root.iterdir():
            if not entry_dir.is_dir():
                continue
            if not (entry_dir / "meta.json").exists():
                continue
            entry_count += 1
            for path in entry_dir.rglob("*"):
                if path.is_file():
                    total_bytes += path.stat().st_size
        return {
            "entry_count": entry_count,
            "total_size_mb": round(total_bytes / (1024 * 1024), 3),
        }

    def evict_old(self, days: int) -> int:
        """Delete entries older than ``days``; return the number removed."""
        if days < 0:
            raise ValueError("days must be non-negative.")
        cutoff = time.time() - days * 86400
        removed = 0
        for entry_dir in self.cache_root.iterdir():
            if not entry_dir.is_dir():
                continue
            meta_path = entry_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                created_at = float(meta.get("created_at", 0))
            except (OSError, ValueError, json.JSONDecodeError, TypeError):
                created_at = 0.0
            if created_at < cutoff:
                shutil.rmtree(entry_dir)
                removed += 1
        return removed

    def _cache_key(
        self,
        file_path: str | Path,
        model_id: str,
        engine: str,
        split_on_single_danda: bool,
        *,
        is_query: bool,
        source_format: str,
    ) -> str:
        path = Path(file_path)
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        config = f"{model_id}|{engine}|{split_on_single_danda}|{is_query}|{source_format}"
        config_hash = hashlib.sha256(config.encode("utf-8")).hexdigest()[:8]
        return f"{content_hash}_{config_hash}"
