import os
import glob
import asyncio

import numpy as np

from ..config import settings
from .embedder import Embedder


class RagIndex:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._chunks: list[dict] = []
        self._embeddings: np.ndarray | None = None

    def load(self) -> None:
        self._index_corpus()

    def _chunk(self, text: str, size: int = 700) -> list[str]:
        paragrafos = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks, buf = [], ""
        for p in paragrafos:
            if len(buf) + len(p) < size:
                buf = f"{buf}\n\n{p}" if buf else p
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return chunks

    def _index_corpus(self) -> None:
        for path in sorted(glob.glob(os.path.join(settings.corpus_dir, "*"))):
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as f:
                text = f.read()
            titulo = os.path.basename(path)
            for ch in self._chunk(text):
                self._chunks.append({"title": titulo, "text": ch})

        if self._chunks:
            textos = [c["text"] for c in self._chunks]
            self._embeddings = self._embedder.encode(textos)
        print(f"[rag] {len(self._chunks)} trecho(s) indexado(s) do corpus curado.")

    def _search_sync(self, query: str, top_k: int) -> list[dict]:
        if self._embeddings is None or not self._chunks:
            return []
        q = self._embedder.encode_one(query)
        scores = self._embeddings @ q
        ordem = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "title": self._chunks[i]["title"],
                "snippet": self._chunks[i]["text"],
                "score": float(scores[i]),
            }
            for i in ordem
        ]

    async def search(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or settings.rag_top_k
        return await asyncio.to_thread(self._search_sync, query, top_k)
