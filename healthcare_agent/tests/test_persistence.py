"""Persistent memory: resume after restart, owner-only access, re-authorisation, cross-session recall, forget.

tests/__init__.py points HC_MEMORY_DB_PATH at a temporary file, so the real store is never touched.
"""

from __future__ import annotations

import asyncio
import unittest

from hcagent.schemas import Role, UserContext
from hcagent.system import HealthcareAgentSystem, resolve_user


def ask(system, session, *questions):
    async def run():
        return [await system.ask(session, q) for q in questions]
    return asyncio.run(run())


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.system = HealthcareAgentSystem(use_llm=False)
        self.assertIsNotNone(self.system.store)

    def test_resume_after_restart(self):
        user = resolve_user("U101")
        s = self.system.new_session(user)
        ask(self.system, s, "What is the age and sex of patient P0042?")
        restarted = HealthcareAgentSystem(use_llm=False)           # new process, same database
        resumed = restarted.resume_session(s.session_id, user)
        self.assertIsNotNone(resumed)
        self.assertEqual(resumed.tree.focus_patient_id, "P0042")
        (st,) = ask(restarted, resumed, "Has this patient attended the ED in the last six months?")
        self.assertEqual(st.target_patient_id, "P0042")              # "this patient" survives the restart
        self.assertEqual(st.tool_calls[0].status, "cached")          # profile reused within its TTL

    def test_only_owner_can_resume(self):
        s = self.system.new_session(resolve_user("U101"))
        ask(self.system, s, "What chronic conditions does P0306 have?")
        self.assertIsNone(self.system.resume_session(s.session_id, resolve_user("U201")))

    def test_resume_reauthorises_branches(self):
        before = UserContext(user_id="U900", role=Role.CLINICIAN, authorized_patients=["P0109", "P0042"])
        s = self.system.new_session(before)
        ask(self.system, s, "Show me the latest vital signs for P0109.", "What is the age and sex of patient P0042?")
        after = before.model_copy(update={"authorized_patients": ["P0109"]})   # P0042 removed from care team
        resumed = self.system.resume_session(s.session_id, after)
        self.assertEqual(set(resumed.tree.patients), {"P0109"})
        self.assertIsNone(resumed.tree.focus_patient_id)

    def test_cross_session_recall_same_user_only(self):
        u201 = resolve_user("U201")
        ask(self.system, self.system.new_session(u201), "What is DM2011's HbA1c?")
        (st,) = ask(self.system, self.system.new_session(u201),
                    "What information has already been retrieved about DM2011?")
        earlier = st.last_ok("recall_session_memory").data["earlier_sessions"]
        self.assertTrue(any("HbA1c" in t["question"] for t in earlier))
        (other,) = ask(self.system, self.system.new_session(resolve_user("U101")),
                       "What information has already been retrieved about DM2011?")
        self.assertIsNone(other.last_ok("recall_session_memory"))    # U101 has no history: NO_DATA

    def test_forget_user(self):
        user = resolve_user("U101")
        s = self.system.new_session(user)
        ask(self.system, s, "What chronic conditions does P0306 have?")
        self.assertGreater(self.system.forget_user(user), 0)
        self.assertEqual(self.system.list_sessions(user), [])
        self.assertIsNone(self.system.resume_session(s.session_id, user))

    def test_chat_list_titles_rename_delete(self):
        user = resolve_user("U102")
        s = self.system.new_session(user)
        ask(self.system, s, "Show me the latest vital signs for P0109.", "What is the age and sex of patient P0042?")
        row = next(r for r in self.system.list_sessions(user) if r["session_id"] == s.session_id)
        self.assertEqual(row["title"], "Latest vital signs for P0109")        # from the first question only
        self.assertTrue(self.system.rename_session(s.session_id, user, "  Ward round  "))
        ask(self.system, s, "Show me the latest vital signs for P0109.")      # write-through keeps the new title
        row = next(r for r in self.system.list_sessions(user) if r["session_id"] == s.session_id)
        self.assertEqual(row["title"], "Ward round")
        self.assertFalse(self.system.rename_session(s.session_id, resolve_user("U101"), "x"))   # owner only
        self.assertFalse(self.system.delete_session(s.session_id, resolve_user("U101")))
        self.assertTrue(self.system.delete_session(s.session_id, user))
        self.assertIsNone(self.system.resume_session(s.session_id, user))

    def test_store_without_title_column_is_migrated(self):
        import sqlite3, tempfile
        from pathlib import Path
        from hcagent.memory_store import MemoryStore
        path = Path(tempfile.mkdtemp()) / "old.sqlite"
        s = self.system.new_session(resolve_user("U101"))
        ask(self.system, s, "What chronic conditions does P0306 have?")
        tree = s.tree.model_copy(update={"title": None})
        conn = sqlite3.connect(path)   # schema as it was before chat titles
        conn.execute("CREATE TABLE mem_sessions (session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, role TEXT NOT NULL, "
                     "created_at TEXT NOT NULL, updated_at TEXT NOT NULL, turn_count INTEGER NOT NULL, "
                     "patients TEXT NOT NULL, tree_json TEXT NOT NULL)")
        conn.execute("INSERT INTO mem_sessions VALUES (?,?,?,?,?,?,?,?)", (tree.session_id, "U101", "clinician",
                     tree.created_at, "2999-01-01T00:00:00+00:00", 1, "P0306", tree.model_dump_json()))
        conn.commit(); conn.close()
        (row,) = MemoryStore(path).list_sessions("U101")
        self.assertEqual(row["title"], "What chronic conditions does P0306 have")

    def test_memory_can_be_disabled(self):
        self.assertIsNone(HealthcareAgentSystem(use_llm=False, persist_memory=False).store)


if __name__ == "__main__":
    unittest.main()
