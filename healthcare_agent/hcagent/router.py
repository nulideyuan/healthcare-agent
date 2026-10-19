"""Deterministic task router: intent -> tool plan. The LLM router is only the fallback."""

from __future__ import annotations

from dataclasses import dataclass, field

from .policy import PolicyEngine
from .schemas import Intent, ParsedRequest, PlannedCall, Route, SharedState, UserContext
from .tools import TOOLS

# Intents recognised by the interpreter but without a data source in the v1 dataset.
UNSUPPORTED_V1: dict[Intent, str] = {
    Intent.LAB_RESULTS: "laboratory results",
    Intent.MEDICATION_QUERY: "medication records",
    Intent.PREDICTION: "prediction models (e.g. ED demand forecasting)",
    Intent.RISK_ASSESSMENT: "patient risk-prediction models",
}

DEFAULT_ED_TREND_HOURS = 6

DM_INTENTS = {Intent.GLUCOSE_TREND, Intent.GLUCOSE_FORECAST, Intent.GLYCEMIC_RISK, Intent.FORECAST_ACCURACY,
              Intent.MEAL_RESPONSE, Intent.GLUCOSE_PATTERN_HISTORY}
# Hospital-only data a diabetes-cohort patient doesn't have, and vice versa.
NOT_IN_DM_COHORT = {Intent.VITAL_SIGNS: "vital signs", Intent.PATIENT_ED_STATUS: "ED encounters"}


def is_dm_patient(patient_id: str | None) -> bool:
    return bool(patient_id and patient_id.startswith("DM"))


def _dm_plan(intent: Intent, p: ParsedRequest, pid: str) -> list[PlannedCall]:
    """Diabetes-cohort plans. Tool ownership (ToolSpec.agent) decides which specialist explains."""
    t = {"patient_id": pid, **({"as_of": p.as_of} if p.as_of else {})}
    trend = PlannedCall(tool="dm_get_glucose_trend", args=dict(t), reason="current CGM and trend")
    forecast = PlannedCall(tool="dm_forecast_glucose", args=dict(t), reason="model forecast")
    profile = PlannedCall(tool="dm_get_patient_profile", args={"patient_id": pid}, reason="profile, labs, agents")
    if intent == Intent.GLUCOSE_TREND:
        return [trend]
    if intent in (Intent.GLUCOSE_FORECAST, Intent.GLYCEMIC_RISK, Intent.PREDICTION, Intent.RISK_ASSESSMENT):
        return [trend, forecast]
    if intent == Intent.FORECAST_ACCURACY:
        return [PlannedCall(tool="dm_get_forecast_accuracy", args={"patient_id": pid}, reason="validation accuracy")]
    if intent in (Intent.PATIENT_INFORMATION, Intent.LAB_RESULTS, Intent.PATIENT_HISTORY):
        return [profile]
    if intent == Intent.MEDICATION_QUERY:
        hours = p.hours or (p.days * 24 if p.days else 72)
        return [profile, PlannedCall(tool="dm_get_medication_events", args={**t, "hours": min(hours, 336)},
                                     reason="recorded insulin / oral agents")]
    if intent == Intent.MEAL_RESPONSE:
        return [PlannedCall(tool="dm_get_meal_responses", args=dict(t), reason="post-meal CGM responses")]
    if intent == Intent.GLUCOSE_PATTERN_HISTORY:
        return [PlannedCall(tool="dm_find_similar_patterns", args=dict(t), reason="similar past CGM windows")]
    if intent == Intent.MEMORY_RECALL:
        return [PlannedCall(tool="recall_session_memory", args={"patient_id": pid}, reason="session memory")]
    return []


@dataclass
class RouteDecision:
    route: Route
    reason: str
    plan: list[PlannedCall] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    llm_allowed_tools: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    agent_tasks: dict[str, list[str]] = field(default_factory=dict)


def _patient_plan(intent: Intent, p: ParsedRequest, pid: str) -> list[PlannedCall]:
    profile = PlannedCall(tool="get_patient_profile", args={"patient_id": pid}, reason="patient context")
    if intent == Intent.PATIENT_INFORMATION:
        return [profile]
    if intent == Intent.PATIENT_HISTORY:
        args = {"patient_id": pid, "limit": 10 if p.days else 5}
        if p.days:
            args["days"] = p.days
        return [profile, PlannedCall(tool="get_patient_encounters", args=args, reason="encounter history")]
    if intent == Intent.VITAL_SIGNS:
        args = {"patient_id": pid, "limit": 10}
        if p.hours:
            args["hours"] = p.hours
        return [profile, PlannedCall(tool="get_recent_vitals", args=args, reason="observations")]
    if intent == Intent.PATIENT_ED_STATUS:
        # Vitals, ED and bed context depend on the encounter; see followups().
        return [profile, PlannedCall(tool="get_patient_encounters", args={"patient_id": pid, "limit": 1},
                                     reason="current/latest ED encounter")]
    return []


def _ops_plan(intent: Intent, p: ParsedRequest) -> list[PlannedCall]:
    if intent == Intent.ED_STATUS:
        trend = min(p.hours, 72) if p.hours else DEFAULT_ED_TREND_HOURS
        return [PlannedCall(tool="get_ed_status", args={"trend_hours": trend}, reason="ED traffic")]
    if intent == Intent.BED_AVAILABILITY:
        args = {"ward": p.ward} if p.ward else {}
        return [PlannedCall(tool="get_bed_capacity", args=args, reason="bed availability")]
    if intent == Intent.WARD_OCCUPANCY:
        return [PlannedCall(tool="get_bed_capacity", args={}, reason="ward occupancy")]
    return []


class DeterministicRouter:
    def __init__(self, policy: PolicyEngine):
        self.policy = policy

    def route(self, parsed: ParsedRequest, target_patient_id: str | None, user: UserContext) -> RouteDecision:
        intents = parsed.intents
        if intents == [Intent.UNKNOWN]:
            return RouteDecision(
                route="llm_fallback",
                reason="No deterministic rule matched; LLM may choose among role-permitted tools.",
                llm_allowed_tools=self.policy.allowed_tools(user),
            )
        if intents == [Intent.GENERAL_MEDICAL_INFORMATION]:
            return RouteDecision(route="general_llm", reason="General medical information; no record access.")

        dm = is_dm_patient(target_patient_id)
        if dm:
            unsupported = [f"{NOT_IN_DM_COHORT[i]} (not recorded for the diabetes cohort)" for i in intents if i in NOT_IN_DM_COHORT]
        else:
            unsupported = [UNSUPPORTED_V1[i] for i in intents if i in UNSUPPORTED_V1]
            if any(i in DM_INTENTS for i in intents):
                unsupported.append("CGM glucose data (only available for diabetes-cohort patients, IDs DMxxxx)")
        plan: list[PlannedCall] = []
        agent_tasks: dict[str, list[str]] = {}
        for intent in intents:
            if dm:
                calls = _dm_plan(intent, parsed, target_patient_id)
            elif intent == Intent.MEMORY_RECALL and target_patient_id:
                calls = _dm_plan(intent, parsed, target_patient_id)
            else:
                calls = _patient_plan(intent, parsed, target_patient_id) if target_patient_id else []
            calls += _ops_plan(intent, parsed)
            for agent in {TOOLS[c.tool].agent for c in calls}:
                agent_tasks.setdefault(agent, []).append(intent.value)
            for c in calls:
                if not any(c.tool == x.tool and c.args == x.args for x in plan):
                    plan.append(c)

        if not plan:
            return RouteDecision(
                route="unsupported", unsupported=unsupported,
                reason="Requested data is not available in the v1 dataset.",
            )
        order = ["glucose_prediction", "patient_context", "hospital_explainer"]
        agents = sorted(dict.fromkeys(TOOLS[c.tool].agent for c in plan), key=order.index)
        return RouteDecision(
            route="deterministic",
            reason="Rule-based plan for intents: " + ", ".join(i.value for i in intents)
                   + f" -> agent(s): {', '.join(agents)}",
            plan=plan,
            unsupported=unsupported,
            llm_allowed_tools=sorted({c.tool for c in plan}),
            agents=agents,
            agent_tasks=agent_tasks,
        )

    @staticmethod
    def followups(state: SharedState) -> list[PlannedCall]:
        """Second-stage calls whose arguments depend on first-stage results."""
        if state.parsed is None or Intent.PATIENT_ED_STATUS not in state.parsed.intents:
            return []
        enc_result = state.last_ok("get_patient_encounters")
        if enc_result is None:
            return []
        enc = enc_result.data["encounters"][0]
        in_ed = enc["status_at_now"] == "IN_ED"
        ref_time = state.clock if in_ed else enc["ed_departure_time"]
        stay_hours = max(1, min(72, enc["ed_minutes"] // 60))
        why = "current ED stay" if in_ed else "conditions during the latest ED stay"
        return [
            PlannedCall(tool="get_recent_vitals",
                        args={"patient_id": enc["patient_id"], "encounter_id": enc["encounter_id"], "limit": 10},
                        reason=f"observations for {enc['encounter_id']}"),
            PlannedCall(tool="get_ed_status",
                        args={"timestamp": ref_time[:16], "trend_hours": stay_hours}, reason=why),
            PlannedCall(tool="get_bed_capacity", args={"timestamp": ref_time[:16]}, reason=why),
        ]
