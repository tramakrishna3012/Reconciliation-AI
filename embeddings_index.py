from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger


class EmbeddingsIndex:
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        persist_dir: Optional[str] = None,
        metric: str = "cosine",
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.metric = metric
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.vectors: Optional[np.ndarray] = None
        self.ids: List[int] = []

    def load_model(self) -> None:
        if self.model is None:
            logger.info(f"Loading embeddings model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)

    def _create_index(self, dim: int) -> faiss.Index:
        if self.metric == "cosine":
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)
        return index

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        self.load_model()
        assert self.model is not None
        vecs = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype("float32")

    def build(self, texts: List[str], ids: Optional[List[int]] = None) -> None:
        vecs = self.encode(texts)
        dim = vecs.shape[1]
        self.index = self._create_index(dim)
        self.index.add(vecs)
        self.vectors = vecs
        self.ids = ids if ids is not None else list(range(1, len(texts) + 1))
        logger.info(f"Built FAISS index with {len(texts)} vectors, dim={dim}")

    def query(self, query_texts: List[str], top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index not built")
        q = self.encode(query_texts)
        sims, idxs = self.index.search(q, top_k)
        return sims, idxs

    def save(self) -> None:
        if self.persist_dir is None:
            logger.warning("persist_dir not set; skipping save")
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        if self.index is None or self.vectors is None:
            logger.warning("Nothing to save: index not built")
            return
        faiss.write_index(self.index, str(self.persist_dir / "index.faiss"))
        np.save(self.persist_dir / "vectors.npy", self.vectors)
        np.save(self.persist_dir / "ids.npy", np.array(self.ids, dtype=np.int64))
        logger.info(f"Saved index to {self.persist_dir}")

    def load(self) -> bool:
        if self.persist_dir is None:
            return False
        idx_path = self.persist_dir / "index.faiss"
        vec_path = self.persist_dir / "vectors.npy"
        ids_path = self.persist_dir / "ids.npy"
        if not (idx_path.exists() and vec_path.exists() and ids_path.exists()):
            return False
        self.index = faiss.read_index(str(idx_path))
        self.vectors = np.load(vec_path)
        self.ids = np.load(ids_path).tolist()
        logger.info(f"Loaded index from {self.persist_dir}")
        return True
