"""Memory-tree behaviour and its safety properties (offline)."""

from __future__ import annotations

import asyncio
import unittest

from hcagent.memory import MemoryTree
from hcagent.schemas import PlannedCall, SharedState
from hcagent.system import HealthcareAgentSystem, resolve_user

SYSTEM = HealthcareAgentSystem(use_llm=False)


def converse(user_id: str, questions: list[str]):
    session = SYSTEM.new_session(resolve_user(user_id))

    async def run():
        return [await SYSTEM.ask(session, q) for q in questions]

    return session, asyncio.run(run())


class MemoryTreeTests(unittest.TestCase):
    def test_previous_patient_and_focus(self):
        session, states = converse("U101", [
            "What is the age and sex of patient P0042?",
            "Show me the latest vital signs for P0109.",
            "Go back to the previous patient: has he attended the ED in the last six months?",
        ])
        self.assertEqual(states[-1].target_patient_id, "P0042")
        self.assertEqual(session.tree.focus_patient_id, "P0042")
        self.assertEqual(session.tree.previous_patient(), "P0109")

    def test_llm_context_is_isolated_to_target_patient(self):
        _, states = converse("U101", [
            "Why is P0002 still waiting in the ED?",
            "What chronic conditions does P0306 have?",
            "What were this patient's latest observations?",
        ])
        last = states[-1]
        self.assertEqual(last.target_patient_id, "P0306")
        self.assertEqual({t["patient_id"] for t in last.conversation}, {"P0306"})

    def test_reuse_is_marked_and_keeps_provenance(self):
        _, states = converse("U101", ["What chronic conditions does P0306 have?",
                                      "What chronic conditions does P0306 have?"])
        first, second = states
        self.assertEqual(second.tool_calls[0].status, "cached")
        self.assertTrue(second.tool_results[0].from_memory)
        self.assertEqual(second.tool_results[0].memory_request_id, first.request_id)

    def test_denied_and_not_found_create_no_patient_branch(self):
        session, _ = converse("U102", ["Show me the latest vital signs for P0200."])
        self.assertNotIn("P0200", session.tree.patients)
        session, _ = converse("U101", ["Show vitals for P9999"])
        self.assertNotIn("P9999", session.tree.patients)

    def test_cache_is_reauthorised_on_reuse(self):
        # Even if a tree somehow held another patient's fact, the policy runs before the lookup.
        session, _ = converse("U101", ["What chronic conditions does P0306 have?"])
        st = SharedState(clock=SYSTEM.clock, user=resolve_user("PAT-P0109"), message="", session_id=session.session_id)
        r = SYSTEM.services.executor.execute(st, PlannedCall(tool="get_patient_profile", args={"patient_id": "P0306"}), "llm")
        self.assertEqual(r.error_code, "ACCESS_DENIED")
        self.assertEqual(st.tool_calls[-1].status, "denied")

    def test_clock_change_invalidates_facts(self):
        session, _ = converse("OPS1", ["How busy is the ED right now?"])
        tree: MemoryTree = session.tree
        args = {"timestamp": None, "trend_hours": 6}
        self.assertIsNotNone(tree.lookup("get_ed_status", args, SYSTEM.clock))
        self.assertIsNone(tree.lookup("get_ed_status", args, "2026-09-28T22:00"))

    def test_sessions_do_not_share_memory(self):
        s1, _ = converse("U101", ["What chronic conditions does P0306 have?"])
        _, states = converse("U101", ["What chronic conditions does P0306 have?"])
        self.assertEqual(states[0].tool_calls[0].status, "ok")


if __name__ == "__main__":
    unittest.main()
