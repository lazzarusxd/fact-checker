import asyncio
from datetime import datetime

import numpy as np

from ..config import settings
from .embedder import Embedder


def _age_days(created_at) -> int:
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at)
        except ValueError:
            return 0
    if isinstance(created_at, datetime):
        return max(0, (datetime.now() - created_at).days)
    return 0


class LearnedStore:

    def __init__(self, embedder: Embedder, db) -> None:
        self._db = db
        self._embedder = embedder
        self._seen: set[str] = set()
        self._items: list[dict] = []
        self._embeddings: np.ndarray | None = None

    async def load(self) -> None:
        rows = await self._db.load_learned_evidence(settings.learned_ttl_days)
        for r in rows:
            self._items.append(r)
            self._seen.add(r["snippet"])
        self._rebuild()
        print(f"[learned] {len(self._items)} evidência(s) aprendida(s) carregada(s).")

    def _rebuild(self) -> None:
        if self._items:
            self._embeddings = np.array(
                [it["embedding"] for it in self._items], dtype=np.float32
            )
        else:
            self._embeddings = None

    def _search_sync(self, query: str, top_k: int) -> list[dict]:
        if self._embeddings is None or not self._items:
            return []
        q = self._embedder.encode_one(query)
        scores = self._embeddings @ q
        ordem = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "title": self._items[i]["title"],
                "snippet": self._items[i]["snippet"],
                "url": self._items[i].get("url"),
                "age_days": _age_days(self._items[i].get("created_at")),
                "score": float(scores[i]),
            }
            for i in ordem
        ]

    async def search(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or settings.learned_top_k
        return await asyncio.to_thread(self._search_sync, query, top_k)

    async def add(self, title: str, snippet: str, url: str | None) -> None:
        if not snippet or snippet in self._seen:
            return
        emb = await asyncio.to_thread(self._embedder.encode_one, snippet)
        emb_list = emb.astype(float).tolist()
        await self._db.insert_learned_evidence(title, snippet, url, emb_list)
        self._seen.add(snippet)
        self._items.append(
            {
                "title": title,
                "snippet": snippet,
                "url": url,
                "created_at": datetime.now(),
                "embedding": emb_list,
            }
        )
        self._rebuild()
