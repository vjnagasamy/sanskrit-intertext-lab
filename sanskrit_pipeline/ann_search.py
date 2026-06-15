"""Approximate nearest-neighbour search over embedding vectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_FLAT_THRESHOLD = 10_000


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "Install faiss-cpu or faiss-gpu to use ANN search."
        ) from exc
    return faiss


def _row_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return (embeddings / norms).astype(np.float32, copy=False)


class ANNIndex:
    """FAISS index for cosine similarity via inner product on unit vectors."""

    def __init__(self, embeddings: np.ndarray, use_gpu: bool = False) -> None:
        faiss = _import_faiss()

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be rank-2.")
        if embeddings.shape[0] == 0:
            raise ValueError("Cannot build ANN index from empty embeddings.")

        self._dim = int(embeddings.shape[1])
        self._n_vectors = int(embeddings.shape[0])
        normalized = _row_normalize(np.asarray(embeddings, dtype=np.float32))

        if self._n_vectors < _FLAT_THRESHOLD:
            self._index = faiss.IndexFlatIP(self._dim)
            self._index_type = "flat"
            self._nlist = 0
        else:
            nlist = min(4096, max(1, self._n_vectors // 40))
            quantizer = faiss.IndexFlatIP(self._dim)
            self._index = faiss.IndexIVFFlat(quantizer, self._dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self._index.train(normalized)
            self._index_type = "ivf"
            self._nlist = nlist

        self._index.add(normalized)

        if use_gpu and faiss.get_num_gpus() > 0:
            resources = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(resources, 0, self._index)

    def search(self, query_embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, indices)`` arrays with shape ``(Q, top_k)``."""
        if query_embeddings.ndim != 2:
            raise ValueError("Query embeddings must be rank-2.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        faiss = _import_faiss()
        queries = _row_normalize(np.asarray(query_embeddings, dtype=np.float32))
        k = min(top_k, self._n_vectors)

        if self._index_type == "ivf" and hasattr(self._index, "nprobe"):
            self._index.nprobe = min(64, self._nlist)

        scores, indices = self._index.search(queries, k)
        return scores.astype(np.float32, copy=False), indices.astype(np.int64, copy=False)

    def save(self, path: Path) -> None:
        faiss = _import_faiss()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path, *, use_gpu: bool = False) -> ANNIndex:
        faiss = _import_faiss()
        instance = cls.__new__(cls)
        instance._index = faiss.read_index(str(path))
        instance._dim = instance._index.d
        instance._n_vectors = instance._index.ntotal
        instance._index_type = "ivf" if hasattr(instance._index, "nlist") and instance._index.nlist > 0 else "flat"
        instance._nlist = getattr(instance._index, "nlist", 0)
        if use_gpu and faiss.get_num_gpus() > 0:
            resources = faiss.StandardGpuResources()
            instance._index = faiss.index_cpu_to_gpu(resources, 0, instance._index)
        return instance
