import json

import aiosqlite

from backend.db.models import SCHEMA


class Repository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()

    async def save_analysis(
        self,
        question: str,
        document_text: str,
        answer: str,
        overall_score: float,
        claims: list[dict],
    ) -> int:
        cursor = await self._db.execute(
            "INSERT INTO analyses (question, document_text, answer, overall_score, claims_json) VALUES (?, ?, ?, ?, ?)",
            (question, document_text, answer, overall_score, json.dumps(claims)),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_analysis(self, analysis_id: int) -> dict | None:
        cursor = await self._db.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "question": row["question"],
            "document_text": row["document_text"],
            "answer": row["answer"],
            "overall_score": row["overall_score"],
            "claims": json.loads(row["claims_json"]),
            "created_at": row["created_at"],
        }

    async def list_analyses(self, limit: int = 50) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT id, question, answer, overall_score, created_at FROM analyses ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "overall_score": row["overall_score"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    async def save_feedback(
        self,
        analysis_id: int,
        claim_index: int,
        is_correct: bool,
        note: str | None = None,
    ) -> int:
        cursor = await self._db.execute(
            "INSERT INTO feedback (analysis_id, claim_index, is_correct, note) VALUES (?, ?, ?, ?)",
            (analysis_id, claim_index, is_correct, note),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def list_feedback(self, analysis_id: int) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM feedback WHERE analysis_id = ? ORDER BY created_at",
            (analysis_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "analysis_id": row["analysis_id"],
                "claim_index": row["claim_index"],
                "is_correct": bool(row["is_correct"]),
                "note": row["note"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
