"""MAF agent layer: the LLM interprets evidence and may call only the tools it is given.

Guarantees enforced in code, not in the prompt:
  * per-request tool allowlist (the LLM is only shown routed tools),
  * PolicyGuardMiddleware re-checks every LLM tool call against the PolicyEngine,
  * identity and shared state reach tools through runtime kwargs the LLM cannot see or set.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agent_framework import Agent, FunctionInvocationContext, FunctionMiddleware, FunctionTool

from .executor import ToolExecutor
from .policy import PolicyEngine
from .schemas import PlannedCall, Role, SharedState
from .tools import TOOLS, ToolSpec

EVIDENCE_POLICY = """You are the explanation layer of a controlled hospital information system that runs on synthetic research data.
Software has already interpreted the request, checked permissions, retrieved data and applied the clinical and operational rules. Your job is to explain that evidence.

Rules:
1. Only make patient-specific or hospital-specific claims that are supported by the EVIDENCE block or by tool results you retrieve.
2. If information is unavailable, say explicitly that it is unavailable. Never estimate or invent values, diagnoses, laboratory results, medications or hospital status.
3. Use the deterministic findings for thresholds, scores, risk bands, crowding levels and clinical stability. Do not compute your own scores or contradict a finding.
4. Keep clinical risk and operational congestion as separate points.
5. Give the time of important facts, e.g. "(observations at 2026-09-28T22:10)".
6. Never mention any patient other than the target patient.
7. Do not recommend medication changes, diagnoses or treatment decisions; suggest speaking to clinical staff where appropriate.
8. Call a tool only if the evidence is insufficient. Access is enforced by software: if a tool returns ACCESS_DENIED, say access was denied and do not try another patient.
9. All times are simulated; "now" is CURRENT_TIME in the evidence.
Answer in the language of the question. Be brief: 3-8 sentences or short bullets."""

ROLE_STYLE = {
    Role.PATIENT: ("You are talking to the patient. Use plain, calm language without jargon. If any clinical finding has "
                   "severity warning or critical, advise them to tell a member of staff now."),
    Role.CLINICIAN: "You are talking to a clinician. Lead with the answer; be concise and clinical.",
    Role.OPERATIONS: ("You are talking to a hospital operations manager. Focus on capacity and flow. "
                      "Never include patient-identifiable information."),
}

# Specialist agents for the diabetes cohort. Each gets EVIDENCE_POLICY plus its own remit.
SPECIALISTS = {
    "glucose_prediction": ("Glucose Prediction Agent", """You are the Glucose Prediction Agent for a diabetes (CGM) cohort.
Your remit: the current glucose and its trend, the model forecast for +30/+60 min, the risk of hypo- or
hyperglycaemia, and how accurate the forecast is.
- You never predict glucose yourself. Every future value must come from the dm_forecast_glucose result, and
  you always give its 90% interval and say it is a model estimate.
- Use the deterministic findings (GLUCOSE_CURRENT, GLUCOSE_TREND, HYPO_RISK, HYPER_RISK) for bands, arrows and
  risk levels. When a risk is elevated or high, quote the validated sensitivity/precision given in the finding.
- For accuracy questions, use FORECAST_ACCURACY findings and compare with the no-change baseline.
- Factor effects describe what moves this forecast; they are not a description of past meals or events.
- The question may also contain parts for the Patient Context Agent (history, labs, medications, meals,
  past patterns). Do not answer or mention those parts; that agent answers them separately."""),
    "patient_context": ("Patient Context & Analysis Agent", """You are the Patient Context & Analysis Agent for a diabetes (CGM) cohort.
Your remit: the patient's profile and labs (e.g. HbA1c), recorded insulin and oral-agent events, glucose responses
after previous meals, similar past glucose patterns, and what this conversation has already retrieved.
- Describe history and patterns; do not forecast future glucose (that is the Glucose Prediction Agent's job).
- Similar-pattern results describe what happened before; say they are not a prediction.
- The 'hypoglycaemia history' summary field is clinical history, not a CGM finding; say so if you use it.
- When asked what has been retrieved, list the facts from recall_session_memory with their sources; mention
  questions from this user's earlier sessions separately, with their dates.
- The question may also contain parts for the Glucose Prediction Agent (current trend, forecast, hypo/hyper
  risk, forecast accuracy). Do not answer or mention those parts; that agent answers them separately."""),
}

# Plain-language description of each intent, used to tell a specialist which part of a question is theirs.
TASK_TEXT = {
    "GLUCOSE_TREND": "the current glucose and its trend", "GLUCOSE_FORECAST": "the glucose forecast for +30/+60 min",
    "GLYCEMIC_RISK": "the risk of hypo- or hyperglycaemia in the next hour", "PREDICTION": "the glucose forecast",
    "RISK_ASSESSMENT": "the risk of hypo- or hyperglycaemia", "FORECAST_ACCURACY": "how accurate the forecast is",
    "PATIENT_INFORMATION": "the patient's profile", "LAB_RESULTS": "the patient's lab results",
    "PATIENT_HISTORY": "the patient's recordings and history", "MEDICATION_QUERY": "recorded medications and insulin",
    "MEAL_RESPONSE": "what happened after previous meals", "GLUCOSE_PATTERN_HISTORY": "similar past glucose patterns",
    "MEMORY_RECALL": "what this conversation has already retrieved",
}

GENERAL_INFO_INSTRUCTIONS = """You give general, educational health information for a hospital assistant.
You have no access to any patient's record in this mode: do not make claims about the user's own results, diagnoses or care.
Do not give personalised medical advice or medication changes; suggest speaking to a clinician for personal concerns.
Answer in the language of the question, in at most 6 sentences."""


# --------------------------------------------------------------------------- guard middleware


def _args_dict(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if hasattr(arguments, "model_dump"):
        return arguments.model_dump()
    return dict(arguments)


def _denial(tool: str, code: str, reason: str) -> str:
    return json.dumps({"tool": tool, "ok": False, "error_code": code, "error": reason})


class PolicyGuardMiddleware(FunctionMiddleware):
    """Fail-closed check in front of every LLM-initiated tool call."""

    def __init__(self, policy: PolicyEngine, executor: ToolExecutor):
        self.policy = policy
        self.executor = executor

    async def process(self, context: FunctionInvocationContext, call_next) -> None:
        name = context.function.name
        args = _args_dict(context.arguments)
        state: SharedState | None = context.kwargs.get("state")
        if state is None:
            context.result = _denial(name, "BLOCKED", "No request context.")
            return
        if name not in state.llm_allowed_tools:
            self.executor.record_rejection(state, name, args, "blocked", "NOT_IN_ALLOWLIST")
            context.result = _denial(name, "BLOCKED", "This tool is not permitted for this request.")
            return
        decision = self.policy.check_tool(state.user, TOOLS[name], args)
        if not decision.allowed:
            state.permissions.append(decision)
            self.executor.record_rejection(state, name, args, "denied", "ACCESS_DENIED")
            context.result = _denial(name, "ACCESS_DENIED", decision.reason)
            return
        await call_next()


# --------------------------------------------------------------------------- tools exposed to the LLM


def _make_invoker(spec: ToolSpec, executor: ToolExecutor):
    async def invoke(ctx: FunctionInvocationContext, **kwargs: Any) -> str:
        state: SharedState = ctx.kwargs["state"]
        args = {k: v for k, v in kwargs.items() if v is not None}
        result = executor.execute(state, PlannedCall(tool=spec.name, args=args, reason="LLM call"), caller="llm")
        return result.model_dump_json(exclude={"retrieved_at"})

    return invoke


def build_llm_tools(executor: ToolExecutor) -> dict[str, FunctionTool]:
    return {
        spec.name: FunctionTool(
            name=spec.name,
            description=spec.description,
            func=_make_invoker(spec, executor),
            input_model=spec.input_model,
        )
        for spec in TOOLS.values()
    }


# --------------------------------------------------------------------------- prompt


def _compact_result(r) -> dict[str, Any]:
    data = r.data
    if r.ok and r.tool == "get_ed_status" and data["trend"]:
        keys = ("timestamp", "ed_occupancy_ratio", "patients_waiting", "patients_on_trolleys", "average_wait_minutes")
        data = {**data, "trend": [{k: row[k] for k in keys} for row in data["trend"]]}
    out = {"tool": r.tool, "source": r.source, "args": r.args, "ok": r.ok}
    if r.ok:
        out.update({"data_timestamp": r.data_timestamp, "data": data})
        if r.from_memory:
            out["from_session_memory"] = r.memory_request_id
    else:
        out.update({"error_code": r.error_code, "error": r.error})
    return out


def evidence_packet(state: SharedState, agent: str | None = None) -> dict[str, Any]:
    """The evidence an agent may use. A specialist sees only its own tools' results and findings."""
    results = [r for r in state.tool_results if agent is None or (r.tool in TOOLS and TOOLS[r.tool].agent == agent)]
    findings = [f for f in state.findings if agent is None or f.owner in (None, agent)]
    return {
        "CURRENT_TIME": state.clock,
        "request": {
            "intents": [i.value for i in state.parsed.intents] if state.parsed else [],
            "target_patient_id": state.target_patient_id,
            "not_available_in_this_system": state.unsupported,
        },
        "findings": [f.model_dump(include={"category", "code", "severity", "message"}) for f in findings],
        "patient": state.patient,
        # The glucose/forecast context belongs to the prediction agent only.
        "clinical_context": {k: v for k, v in state.clinical_context.items()
                             if agent in (None, "glucose_prediction") or k != "glucose"},
        "hospital_context": state.hospital_context,
        "retrieved": [_compact_result(r) for r in results],
    }


def build_prompt(state: SharedState, with_evidence: bool, agent: str | None = None) -> str:
    parts = [f"ROLE GUIDANCE: {ROLE_STYLE[state.user.role]}"]
    if state.user.role == Role.PATIENT:
        parts.append(f"The user is patient {state.user.patient_id}; they may only see their own record.")
    if state.conversation:
        turns = "\n".join(f"- Q (turn {t['turn_no']}): {t['question']}\n  A: {t['answer'][:300]}"
                          for t in state.conversation)
        scope = f"about {state.target_patient_id} only" if state.target_patient_id else "without patient data"
        parts.append(f"EARLIER TURNS ({scope}; for context, not evidence):\n{turns}")
    if with_evidence:
        parts.append("EVIDENCE (shared state):\n" + json.dumps(evidence_packet(state, agent), default=str, indent=1))
    else:
        parts.append(f"CURRENT_TIME: {state.clock}\nNo evidence has been retrieved yet. Use the available tools "
                     "only if they are needed; otherwise say the request is outside what this system can answer.")
        if state.target_patient_id:
            parts.append(f"Patient in context: {state.target_patient_id}")
    parts.append(f"QUESTION: {state.message}")
    if agent and state.agent_tasks.get(agent):
        task = "; ".join(TASK_TEXT.get(i, i) for i in state.agent_tasks[agent])
        parts.append(f"YOUR PART: answer only about {task}. Other parts of the question are answered by another "
                     "agent; do not answer or comment on them, and do not refer the user to another agent.")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- chat titles

TITLE_INSTRUCTIONS = """Write a short title (3 to 6 words) naming the topic of a conversation that starts with the user's question.
Keep any patient ID exactly as written. No quotes, no trailing punctuation. Reply with the title only."""


class TitleWriter:
    """Topic title for the chat list. Sees only the first question, never evidence or answers."""

    def __init__(self, client):
        self._agent = Agent(client=client, name="chat_titler", instructions=TITLE_INSTRUCTIONS)

    async def title(self, question: str) -> str | None:
        response = await self._agent.run(question, options={"temperature": 0, "max_tokens": 20})
        t = " ".join((response.text or "").split()).strip("\"'“”.").strip()
        return t[:60] or None


# --------------------------------------------------------------------------- responder


class LLMResponder:
    def __init__(self, client, policy: PolicyEngine, executor: ToolExecutor):
        self.tools = build_llm_tools(executor)
        self.explainer = Agent(
            client=client, name="healthcare_explainer", instructions=EVIDENCE_POLICY,
            middleware=[PolicyGuardMiddleware(policy, executor)],
        )
        self.general = Agent(client=client, name="general_health_info", instructions=GENERAL_INFO_INSTRUCTIONS)
        self.specialists = {
            key: Agent(client=client, name=key, instructions=EVIDENCE_POLICY + "\n\n" + remit,
                       middleware=[PolicyGuardMiddleware(policy, executor)])
            for key, (_, remit) in SPECIALISTS.items()
        }

    async def explain(self, state: SharedState, with_evidence: bool = True) -> str:
        specialists = [a for a in state.agents if a in self.specialists]
        if specialists and with_evidence:
            return await self._run_specialists(state, specialists)
        return await self._run_explainer(state, with_evidence)

    async def _run_specialists(self, state: SharedState, agents: list[str]) -> str:
        """Each specialist explains its own part of the evidence; the parts run concurrently."""

        async def run(agent: str) -> str:
            tools = [self.tools[n] for n in state.llm_allowed_tools if n in self.tools and TOOLS[n].agent == agent]
            response = await self.specialists[agent].run(
                build_prompt(state, True, agent), tools=tools or None, options={"temperature": 0},
                function_invocation_kwargs={"state": state},
            )
            state.llm_calls += 1
            return response.text

        texts = await asyncio.gather(*(run(a) for a in agents))
        if len(agents) == 1:
            return texts[0]
        return "\n\n".join(f"**{SPECIALISTS[a][0]}**\n{t}" for a, t in zip(agents, texts))

    async def _run_explainer(self, state: SharedState, with_evidence: bool) -> str:
        tools = [self.tools[n] for n in state.llm_allowed_tools if n in self.tools]
        response = await self.explainer.run(
            build_prompt(state, with_evidence),
            tools=tools or None,
            options={"temperature": 0},
            function_invocation_kwargs={"state": state},
        )
        state.llm_calls += 1
        return response.text

    async def general_info(self, state: SharedState) -> str:
        response = await self.general.run(state.message, options={"temperature": 0})
        state.llm_calls += 1
        return response.text
