"""Diabetes-cohort agents: routing, model integrity, rules and safety (offline, no LLM)."""

from __future__ import annotations

import asyncio
import unittest

import numpy as np

from hcagent.dm_features import build_features, targets
from hcagent.dm_rules import glucose_band, risk_level, trend_arrow
from hcagent.schemas import PlannedCall, SharedState
from hcagent.system import HealthcareAgentSystem, resolve_user

SYSTEM = HealthcareAgentSystem(use_llm=False)
DM = SYSTEM.diabetes


def converse(user_id: str, questions: list[str]):
    session = SYSTEM.new_session(resolve_user(user_id))

    async def run():
        return [await SYSTEM.ask(session, q) for q in questions]

    return session, asyncio.run(run())


@unittest.skipIf(DM is None, "diabetes database/model not built")
class RoutingTests(unittest.TestCase):
    def test_agent_assignment(self):
        _, (trend, meals, both) = converse("U201", [
            "What is DM2011's current glucose trend?",
            "What happened after the patient's previous meals?",
            "Is there an increased risk of hypoglycaemia, and what happened after this patient's previous meals?",
        ])
        self.assertEqual(trend.agents, ["glucose_prediction"])
        self.assertEqual(meals.agents, ["patient_context"])
        self.assertEqual(both.agents, ["glucose_prediction", "patient_context"])
        self.assertEqual(both.agent_tasks, {"glucose_prediction": ["GLYCEMIC_RISK"], "patient_context": ["MEAL_RESPONSE"]})

    def test_forecast_reuses_trend_from_memory(self):
        _, (_, fc) = converse("U201", ["What is DM2011's current glucose trend?",
                                       "What will the glucose level be in 30 or 60 minutes?"])
        self.assertEqual([(c.tool, c.status) for c in fc.tool_calls],
                         [("dm_get_glucose_trend", "cached"), ("dm_forecast_glucose", "ok")])

    def test_as_of_replay(self):
        _, (st,) = converse("U201", ["Forecast glucose for DM2001 at 2020-11-10 07:30"])
        f = st.last_ok("dm_forecast_glucose").data
        self.assertLessEqual(f["base_time"], "2020-11-10T07:30")

    def test_hospital_patient_has_no_cgm(self):
        _, (st,) = converse("U101", ["What is P0109's current glucose trend?"])
        self.assertEqual(st.route, "unsupported")

    def test_dm_patient_has_no_vitals(self):
        _, (st,) = converse("U201", ["Show me the vital signs for DM2011."])
        self.assertEqual(st.route, "unsupported")


@unittest.skipIf(DM is None, "diabetes database/model not built")
class SafetyTests(unittest.TestCase):
    def test_patient_cannot_read_other_dm_patient(self):
        _, (st,) = converse("PAT-DM2011", ["Show me DM2041's glucose."])
        self.assertEqual(st.route, "denied")
        self.assertEqual(st.tool_calls, [])

    def test_hospital_care_team_cannot_read_dm_cohort(self):
        _, (st,) = converse("U102", ["What is DM2011's current glucose trend?"])
        self.assertEqual(st.route, "denied")

    def test_diabetes_clinician_cannot_read_hospital_patients(self):
        _, (st,) = converse("U201", ["Show me P0109's vital signs."])
        self.assertEqual(st.route, "denied")

    def test_operations_cannot_read_glucose(self):
        _, (st,) = converse("OPS1", ["What is DM2011's current glucose trend?"])
        self.assertEqual(st.route, "denied")

    def test_memory_recall_is_never_cached(self):
        _, states = converse("U201", ["What is DM2011's HbA1c?",
                                      "What information has already been retrieved in this conversation?",
                                      "What information has already been retrieved in this conversation?"])
        self.assertEqual(states[-1].tool_calls[-1].status, "ok")
        self.assertEqual(len(states[-1].last_ok("recall_session_memory").data["this_session"]["earlier_questions"]), 2)


@unittest.skipIf(DM is None, "diabetes database/model not built")
class ModelTests(unittest.TestCase):
    def test_no_lookahead_and_matches_training(self):
        """Truncated (runtime) predictions equal full-history ones; replay MAE equals the stored OOF MAE."""
        pid = "DM2000"
        rec = DM.repo.recordings(pid)[-1]
        full = DM.repo.recording_data(rec, rec["cgm_end"])
        X, valid, t = build_features(full)
        Y, yv = targets(full, t)
        m = DM.forecaster.a["fold_models"][DM.forecaster.fold_for(pid)][0]
        pred = ((X - np.array(m["mean"])) / np.array(m["std"])) @ np.array(m["coef"]) + m["intercept"]
        ok = valid & yv
        self.assertAlmostEqual(float(np.abs(pred[ok] - Y[ok, 0]).mean()),
                               DM.forecaster.a["metrics_per_patient"][pid]["30min"]["mae"], places=1)
        i = int(np.flatnonzero(ok)[100])
        r = DM.forecaster.predict(pid, DM.repo.recording_data(rec, full.cgm_ts[i]))
        self.assertAlmostEqual(r["forecasts"][0]["predicted_mgdl"], full.cgm[i] + pred[i], delta=0.06)

    def test_forecast_is_from_model_not_llm(self):
        st = SharedState(clock=SYSTEM.clock, user=resolve_user("U201"), message="")
        r = SYSTEM.services.executor.execute(st, PlannedCall(tool="dm_forecast_glucose", args={"patient_id": "DM2011"}), "router")
        self.assertEqual(r.source, "dm_forecast_model")
        self.assertEqual(r.data["model"], DM.forecaster.name)

    def test_flagged_recording_carries_warning(self):
        _, (st,) = converse("U201", ["Forecast glucose for DM2077"])
        self.assertIn("SENSOR_BIAS_SUSPECTED", {f.code for f in st.findings})


class RuleTests(unittest.TestCase):
    def test_bands_arrows_and_risk(self):
        self.assertTrue(glucose_band(50)[0].startswith("level 2 hypo"))
        self.assertTrue(glucose_band(120)[0].startswith("in target"))
        self.assertTrue(glucose_band(260)[0].startswith("level 2 hyper"))
        self.assertEqual(trend_arrow(-2.5), "falling (↓)")
        self.assertEqual(trend_arrow(0.4), "stable (→)")
        self.assertEqual(risk_level(65, 50, True, 70), "high")
        self.assertEqual(risk_level(80, 60, True, 70), "elevated")
        self.assertEqual(risk_level(120, 90, True, 70), "not elevated")


if __name__ == "__main__":
    unittest.main()
