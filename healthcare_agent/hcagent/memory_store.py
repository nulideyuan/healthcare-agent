"""Persistent store for memory trees (data/memory.sqlite).

* Write-through: the session's tree is saved after every turn.
* Sessions can be resumed after a page refresh or a server restart, only by the user who owns them.
* On resume, each patient branch is re-authorised against the current policy; branches the user may no
  longer access are dropped.
* Earlier sessions of the same user are readable for cross-session recall, never another user's.
* Retention: sessions untouched for HC_MEMORY_RETENTION_DAYS are purged at startup. forget_user()
  deletes everything for a user.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .memory import MemoryTree, default_title

SCHEMA = """
CREATE TABLE IF NOT EXISTS mem_sessions (
    session_id TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    role       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    turn_count INTEGER NOT NULL,
    patients   TEXT NOT NULL,          -- comma-separated patient IDs, for listing
    tree_json  TEXT NOT NULL,
    title      TEXT                    -- chat-list title
);
CREATE INDEX IF NOT EXISTS idx_mem_user ON mem_sessions(user_id, updated_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, path: Path, retention_days: int = 30):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            if "title" not in {r["name"] for r in self._conn.execute("PRAGMA table_info(mem_sessions)")}:
                self._conn.execute("ALTER TABLE mem_sessions ADD COLUMN title TEXT")   # stores from before titles
            cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat(timespec="seconds")
            self._conn.execute("DELETE FROM mem_sessions WHERE updated_at < ?", (cutoff,))
            self._conn.commit()

    def save(self, tree: MemoryTree) -> None:
        now = _now()
        with self._lock:
            self._conn.execute(
                """INSERT INTO mem_sessions (session_id, user_id, role, created_at, updated_at, turn_count, patients,
                   tree_json, title) VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at, turn_count=excluded.turn_count,
                   patients=excluded.patients, tree_json=excluded.tree_json, title=excluded.title""",
                (tree.session_id, tree.user_id, tree.role, tree.created_at, now, tree.turn_count,
                 ",".join(tree.patients), tree.model_dump_json(), tree.title),
            )
            self._conn.commit()

    def load(self, session_id: str, user_id: str, may_access: Callable[[str], bool]) -> MemoryTree | None:
        """Owner-only resume. Branches the user is no longer authorised for are removed."""
        row = self._conn.execute("SELECT * FROM mem_sessions WHERE session_id=? AND user_id=?",
                                 (session_id, user_id)).fetchone()
        if row is None:
            return None
        tree = MemoryTree.model_validate_json(row["tree_json"])
        for pid in [p for p in tree.patients if not may_access(p)]:
            tree.patients.pop(pid)
        if tree.focus_patient_id not in tree.patients:
            tree.focus_patient_id = None
        return tree

    def list_sessions(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT session_id, title, created_at, updated_at, turn_count, patients FROM mem_sessions "
            "WHERE user_id=? ORDER BY updated_at DESC LIMIT ?", (user_id, limit)).fetchall()
        out = [dict(r) for r in rows]
        for r in out:
            if not r["title"]:   # stored before titles existed: use the first question still in the tree
                tree = self._conn.execute("SELECT tree_json FROM mem_sessions WHERE session_id=?",
                                          (r["session_id"],)).fetchone()["tree_json"]
                t = MemoryTree.model_validate_json(tree)
                turns = [x for b in [*t.patients.values(), t.operational, t.general] for x in b.turns]
                r["title"] = default_title(min(turns, key=lambda x: x.turn_no).question) if turns else "Untitled chat"
        return out

    def earlier_turns(self, user_id: str, patient_id: str, exclude_session: str | None, limit: int = 20) -> list[dict[str, Any]]:
        """This user's questions about a patient in *other* sessions, newest first."""
        out = []
        rows = self._conn.execute(
            "SELECT session_id, tree_json FROM mem_sessions WHERE user_id=? AND session_id IS NOT ? "
            "AND (',' || patients || ',') LIKE ? ORDER BY updated_at DESC",
            (user_id, exclude_session, f"%,{patient_id},%")).fetchall()
        for r in rows:
            branch = MemoryTree.model_validate_json(r["tree_json"]).patients.get(patient_id)
            for t in reversed(branch.turns if branch else []):
                out.append({"session_id": r["session_id"], "asked_at": t.asked_at, "question": t.question,
                            "route": t.route, "answer_excerpt": t.answer[:200]})
        return out[:limit]

    def rename(self, session_id: str, user_id: str, title: str) -> bool:
        """Owner-only. Keeps updated_at, so renaming does not reorder the chat list."""
        with self._lock:
            row = self._conn.execute("SELECT tree_json FROM mem_sessions WHERE session_id=? AND user_id=?",
                                     (session_id, user_id)).fetchone()
            if row is None:
                return False
            tree = MemoryTree.model_validate_json(row["tree_json"])
            tree.title = title
            self._conn.execute("UPDATE mem_sessions SET title=?, tree_json=? WHERE session_id=?",
                               (title, tree.model_dump_json(), session_id))
            self._conn.commit()
        return True

    def delete_session(self, session_id: str, user_id: str) -> int:
        with self._lock:
            n = self._conn.execute("DELETE FROM mem_sessions WHERE session_id=? AND user_id=?", (session_id, user_id)).rowcount
            self._conn.commit()
        return n

    def forget_user(self, user_id: str) -> int:
        with self._lock:
            n = self._conn.execute("DELETE FROM mem_sessions WHERE user_id=?", (user_id,)).rowcount
            self._conn.commit()
        return n
