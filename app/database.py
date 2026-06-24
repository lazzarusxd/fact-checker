import json

import aiosqlite

from .config import settings
from .schemas import Critique, Evidence, Verification

_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS claims (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        overall_label TEXT,
        overall_confidence REAL,
        summary TEXT,
        created_at TEXT DEFAULT (datetime('now', 'localtime'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sub_claims (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id INTEGER,
        text TEXT,
        label TEXT,
        confidence REAL,
        rationale TEXT,
        created_at TEXT DEFAULT (datetime('now', 'localtime')),
        FOREIGN KEY (claim_id) REFERENCES claims(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS debate_rounds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id INTEGER,
        sub_claim TEXT,
        round_idx INTEGER,
        query TEXT,
        evidence TEXT,
        verification TEXT,
        critique TEXT,
        created_at TEXT DEFAULT (datetime('now', 'localtime')),
        FOREIGN KEY (claim_id) REFERENCES claims(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learned_evidence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        snippet TEXT,
        url TEXT,
        embedding TEXT,
        created_at TEXT DEFAULT (datetime('now', 'localtime'))
    )
    """,
]


class Database:

    def __init__(self) -> None:
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(settings.sqlite_path)
        self._db.row_factory = aiosqlite.Row
        for ddl in _TABLES:
            await self._db.execute(ddl)
        await self._db.commit()
        print(f"[db] SQLite pronto em '{settings.sqlite_path}'.")

    async def insert_claim(self, text: str) -> int:
        cur = await self._db.execute("INSERT INTO claims (text) VALUES (?)", (text,))
        await self._db.commit()
        return cur.lastrowid

    async def insert_sub_claim(self, claim_id: int, text: str, label: str, confidence: float, rationale: str) -> int:
        cur = await self._db.execute(
            "INSERT INTO sub_claims "
            "(claim_id, text, label, confidence, rationale) "
            "VALUES (?, ?, ?, ?, ?)",
            (claim_id, text, label, confidence, rationale),
        )
        await self._db.commit()
        return cur.lastrowid

    async def insert_round(
        self,
        query: str,
        claim_id: int,
        sub_claim: str,
        round_idx: int,
        critique: Critique,
        evidence: list[Evidence],
        verification: Verification
    ) -> None:
        ev = json.dumps([e.model_dump() for e in evidence], ensure_ascii=False)
        ver = json.dumps(verification.model_dump(mode="json"), ensure_ascii=False)
        cri = json.dumps(critique.model_dump(mode="json"), ensure_ascii=False)
        await self._db.execute(
            "INSERT INTO debate_rounds "
            "(claim_id, sub_claim, round_idx, query, evidence, "
            "verification, critique) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (claim_id, sub_claim, round_idx, query, ev, ver, cri),
        )
        await self._db.commit()

    async def update_claim_verdict(self, claim_id: int, label: str, confidence: float, summary: str) -> None:
        await self._db.execute(
            "UPDATE claims SET overall_label=?, overall_confidence=?, "
            "summary=? WHERE id=?",
            (label, confidence, summary, claim_id),
        )
        await self._db.commit()

    async def insert_learned_evidence(self, title: str, snippet: str, url: str | None, embedding: list[float]) -> None:
        emb = json.dumps(embedding)
        await self._db.execute(
            "INSERT INTO learned_evidence (title, snippet, url, embedding) "
            "VALUES (?, ?, ?, ?)",
            (title, snippet, url, emb),
        )
        await self._db.commit()

    async def load_learned_evidence(self, ttl_days: int) -> list[dict]:
        cur = await self._db.execute(
            "SELECT title, snippet, url, embedding, created_at "
            "FROM learned_evidence "
            "WHERE created_at >= datetime('now', 'localtime', ?)",
            (f"-{ttl_days} days",),
        )
        rows = await cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            emb = d["embedding"]
            d["embedding"] = json.loads(emb) if isinstance(emb, str) else emb
            result.append(d)
        return result

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
