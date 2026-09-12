"""Unit tests for the persistent embedding cache."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sanskrit_pipeline.corpus_pairwise import run_corpus_pairwise_similarity
from sanskrit_pipeline.embedding_cache import CacheEntry, EmbeddingCache
from sanskrit_pipeline.sdk import SanskritResearchSDK
from tests.test_corpus_pairwise import FakeSDK


class EmbeddingCacheTests(unittest.TestCase):
    def test_cache_miss_writes_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "cache"
            cache = EmbeddingCache(cache_root)
            file_path = Path(temp_dir) / "sample.txt"
            file_path.write_text("alpha|shared", encoding="utf-8")
            entry = CacheEntry(
                segments=["alpha", "shared"],
                embeddings=np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
            )

            cache.put(file_path, "fake/model", "dandas", False, entry, is_query=True)

            key_dirs = [path for path in cache_root.iterdir() if path.is_dir()]
            self.assertEqual(len(key_dirs), 1)
            entry_dir = key_dirs[0]
            self.assertTrue((entry_dir / "segments.json").exists())
            self.assertTrue((entry_dir / "embeddings.npy").exists())
            self.assertTrue((entry_dir / "meta.json").exists())
            meta = json.loads((entry_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["model_id"], "fake/model")
            self.assertEqual(meta["n_segments"], 2)

    def test_cache_hit_returns_same_data_without_embedder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(Path(temp_dir) / "cache")
            file_path = Path(temp_dir) / "sample.txt"
            file_path.write_text("alpha|shared", encoding="utf-8")
            expected = CacheEntry(
                segments=["alpha", "shared"],
                embeddings=np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
            )
            cache.put(file_path, "fake/model", "dandas", False, expected, is_query=False)

            sdk = SanskritResearchSDK(model_id="fake/model", cache=cache, device="cpu")
            with patch.object(sdk, "embed_sentences") as mock_embed:
                hit = sdk.cached_embed_file(file_path, is_query=False)
            mock_embed.assert_not_called()
            self.assertEqual(hit.segments, expected.segments)
            np.testing.assert_allclose(np.asarray(hit.embeddings), expected.embeddings)

    def test_evict_old_removes_only_old_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(Path(temp_dir) / "cache")
            file_path = Path(temp_dir) / "sample.txt"
            file_path.write_text("alpha", encoding="utf-8")
            entry = CacheEntry(segments=["alpha"], embeddings=np.array([[1.0, 0.0]], dtype=np.float32))
            cache.put(file_path, "fake/model", "dandas", False, entry)

            entry_dir = next(path for path in cache.cache_root.iterdir() if path.is_dir())
            meta_path = entry_dir / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["created_at"] = time.time() - 10 * 86400
            meta_path.write_text(json.dumps(meta), encoding="utf-8")

            removed = cache.evict_old(days=7)
            self.assertEqual(removed, 1)
            self.assertEqual(cache.stats()["entry_count"], 0)

    def test_corrupted_missing_npy_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(Path(temp_dir) / "cache")
            file_path = Path(temp_dir) / "sample.txt"
            file_path.write_text("alpha", encoding="utf-8")
            entry = CacheEntry(segments=["alpha"], embeddings=np.array([[1.0, 0.0]], dtype=np.float32))
            cache.put(file_path, "fake/model", "dandas", False, entry)

            entry_dir = next(path for path in cache.cache_root.iterdir() if path.is_dir())
            (entry_dir / "embeddings.npy").unlink()

            self.assertIsNone(cache.get(file_path, "fake/model", "dandas", False))

    def test_corpus_workflow_uses_cache_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dir_a = root / "a"
            dir_b = root / "b"
            out = root / "out"
            dir_a.mkdir()
            dir_b.mkdir()
            (dir_a / "a1.txt").write_text("alpha|shared", encoding="utf-8")
            (dir_b / "b1.txt").write_text("shared", encoding="utf-8")

            cache = EmbeddingCache(root / "cache")
            FakeSDK.instances.clear()
            with patch("sanskrit_pipeline.corpus_pairwise.SanskritResearchSDK", FakeSDK):
                run_corpus_pairwise_similarity(
                    dir_a=dir_a,
                    dir_b=dir_b,
                    output_dir=out,
                    model_id="fake/model",
                    device="cpu",
                    top_k=1,
                    embedding_cache=cache,
                )
                sdk = FakeSDK.instances[0]
                self.assertIsNotNone(sdk.cache)
                self.assertEqual(len(sdk.embed_calls), 2)
                self.assertEqual(cache.stats()["entry_count"], 2)

            FakeSDK.instances.clear()
            with patch("sanskrit_pipeline.corpus_pairwise.SanskritResearchSDK", FakeSDK):
                run_corpus_pairwise_similarity(
                    dir_a=dir_a,
                    dir_b=dir_b,
                    output_dir=out / "second",
                    model_id="fake/model",
                    device="cpu",
                    top_k=1,
                    embedding_cache=cache,
                )
                sdk = FakeSDK.instances[0]
                self.assertEqual(len(sdk.embed_calls), 0)


if __name__ == "__main__":
    unittest.main()
