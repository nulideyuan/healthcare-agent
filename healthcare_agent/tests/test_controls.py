"""Offline tests for the deterministic components (no LLM needed).

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import json
import unittest

from agent_framework import FunctionInvocationContext

from hcagent.agent_layer import PolicyGuardMiddleware, build_llm_tools
from hcagent.grounding import check_answer
from hcagent.interpreter import RuleInterpreter
from hcagent.rules import news2_partial
from hcagent.schemas import Intent, ParsedRequest, SharedState
from hcagent.system import HealthcareAgentSystem, patient_user, resolve_user

SYSTEM = HealthcareAgentSystem(use_llm=False)
CLOCK = SYSTEM.clock


def ask(user_id: str, question: str, history: list[str] = ()) -> SharedState:
    session = SYSTEM.new_session(resolve_user(user_id))

    async def run():
        for q in history:
            await SYSTEM.ask(session, q)
        return await SYSTEM.ask(session, question)

    return asyncio.run(run())


class InterpreterTests(unittest.TestCase):
    rules = RuleInterpreter()

    def test_spec_examples(self):
        cases = {
            "What was my latest potassium result?": [Intent.LAB_RESULTS],
            "Why have I been waiting in the ED for four hours?": [Intent.PATIENT_ED_STATUS],
            "Is the Emergency Department busy today?": [Intent.ED_STATUS],
            "How many inpatient beds are currently available?": [Intent.BED_AVAILABILITY],
            "Which wards are close to capacity?": [Intent.WARD_OCCUPANCY],
            "Has this patient attended the ED in the last six months?": [Intent.PATIENT_HISTORY],
        }
        for text, intents in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.rules.parse(text).intents, intents)

    def test_patient_id_normalisation_and_window(self):
        p = self.rules.parse("Has P042 attended the ED in the last six months?")
        self.assertEqual(p.patient_id, "P0042")
        self.assertEqual(p.days, 180)


class TimeConsistencyTests(unittest.TestCase):
    def test_future_fields_are_masked_for_patient_in_ed(self):
        enc = SYSTEM.services.repo.get_encounters("P0109", CLOCK, limit=1)[0]
        self.assertEqual(enc["status_at_now"], "IN_ED")
        self.assertIsNone(enc["disposition"])
        self.assertIsNone(enc["admission_time"])

    def test_tools_reject_future_timestamps(self):
        st = SharedState(clock=CLOCK, user=resolve_user("OPS1"), message="")
        from hcagent.schemas import PlannedCall
        r = SYSTEM.services.executor.execute(st, PlannedCall(tool="get_ed_status", args={"timestamp": "2027-01-01T00:00"}), "router")
        self.assertEqual(r.error_code, "INVALID_INPUT")


class PolicyTests(unittest.TestCase):
    def test_patient_cannot_read_other_patient(self):
        st = ask("PAT-P0109", "Show me patient P0005's medications.")
        self.assertEqual(st.route, "denied")
        self.assertEqual(st.tool_calls, [])

    def test_operations_cannot_read_phi(self):
        st = ask("OPS1", "Show me the vital signs of patient P0109.")
        self.assertEqual(st.route, "denied")
        self.assertEqual(st.permissions[-1].rule, "R1 ops_no_phi")

    def test_clinician_scope(self):
        self.assertEqual(ask("U102", "Show me the latest vital signs for P0200.").route, "denied")
        self.assertEqual(ask("U102", "Show me the latest vital signs for P0109.").route, "deterministic")

    def test_existence_not_revealed_before_authorisation(self):
        self.assertEqual(ask("PAT-P0109", "Show vitals for P9999").route, "denied")
        self.assertEqual(ask("U101", "Show vitals for P9999").route, "not_found")

    def test_executor_denies_even_if_plan_is_wrong(self):
        st = SharedState(clock=CLOCK, user=patient_user("P0109"), message="")
        from hcagent.schemas import PlannedCall
        r = SYSTEM.services.executor.execute(st, PlannedCall(tool="get_patient_profile", args={"patient_id": "P0005"}), "router")
        self.assertEqual(r.error_code, "ACCESS_DENIED")
        self.assertEqual(st.tool_results, [])


class RoutingTests(unittest.TestCase):
    def test_patient_wait_scenario(self):
        st = ask("PAT-P0109", "Why have I been waiting in the ED so long?")
        self.assertEqual([c.tool for c in st.tool_calls], [
            "get_patient_profile", "get_patient_encounters", "get_recent_vitals", "get_ed_status", "get_bed_capacity"])
        codes = {f.code for f in st.findings}
        self.assertIn("WAIT_DRIVERS_OPERATIONAL", codes)
        self.assertIn("ED_CROWDING", codes)

    def test_conversation_memory_resolves_this_patient(self):
        st = ask("U101", "Has this patient attended the ED in the last six months?",
                 history=["What is the age and sex of patient P0042?"])
        self.assertEqual(st.target_patient_id, "P0042")
        self.assertEqual(st.tool_calls[-1].args, {"patient_id": "P0042", "limit": 10, "days": 180})

    def test_unsupported_data_is_not_fabricated(self):
        st = ask("PAT-P0109", "What was my latest potassium result?")
        self.assertEqual(st.route, "unsupported")
        self.assertIn("not available", st.answer)

    def test_clarify_without_patient(self):
        self.assertEqual(ask("U101", "Show me the recent observations.").route, "clarify")


class RulesTests(unittest.TestCase):
    def test_copd_uses_spo2_scale_2(self):
        obs = {"respiratory_rate_bpm": 18, "spo2_pct": 89, "systolic_bp_mmHg": 130,
               "heart_rate_bpm": 80, "temperature_c": 37.0, "timestamp": "t"}
        self.assertEqual(news2_partial(obs, ["COPD"])["components"]["spo2_pct"], 0)
        self.assertEqual(news2_partial(obs, [])["components"]["spo2_pct"], 3)


class GuardTests(unittest.TestCase):
    def _state(self):
        st = SharedState(clock=CLOCK, user=patient_user("P0109"), message="Why the wait?")
        st.parsed = ParsedRequest(intents=[Intent.PATIENT_ED_STATUS])
        st.target_patient_id = "P0109"
        st.llm_allowed_tools = ["get_recent_vitals"]
        return st

    def _invoke(self, st, tool, args):
        mw = PolicyGuardMiddleware(SYSTEM.services.policy, SYSTEM.services.executor)
        ctx = FunctionInvocationContext(function=build_llm_tools(SYSTEM.services.executor)[tool],
                                        arguments=args, kwargs={"state": st})
        called = []

        async def call_next():
            called.append(True)

        asyncio.run(mw.process(ctx, call_next))
        return ctx, called

    def test_middleware_blocks_tool_outside_allowlist(self):
        st = self._state()
        ctx, called = self._invoke(st, "get_bed_capacity", {})
        self.assertFalse(called)
        self.assertEqual(json.loads(ctx.result)["error_code"], "BLOCKED")

    def test_middleware_denies_other_patient(self):
        st = self._state()
        ctx, called = self._invoke(st, "get_recent_vitals", {"patient_id": "P0005"})
        self.assertFalse(called)
        self.assertEqual(json.loads(ctx.result)["error_code"], "ACCESS_DENIED")
        self.assertEqual(st.tool_calls[-1].status, "denied")

    def test_middleware_allows_own_record(self):
        _, called = self._invoke(self._state(), "get_recent_vitals", {"patient_id": "P0109"})
        self.assertTrue(called)

    def test_grounding_flags_invented_numbers_and_blocks_foreign_ids(self):
        st = ask("PAT-P0109", "Why have I been waiting in the ED so long?")
        ok = check_answer(st, f"The ED occupancy is {st.hospital_context['ed_occupancy']:.0%}.")
        self.assertEqual(ok.unsupported_numbers, [])
        bad = check_answer(st, "Your potassium was 5.9 mmol/L, like patient P0005.")
        self.assertIn("5.9", bad.unsupported_numbers)
        self.assertTrue(bad.blocked)


if __name__ == "__main__":
    unittest.main()
