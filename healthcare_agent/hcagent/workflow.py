"""The controlled pipeline as a MAF Workflow.

    interpret -> policy --(denied/clarify)--------------------------> respond
                       \\-> route --(unsupported)--------------------> respond
                                 |--(deterministic)-> tools -> validate -> llm -> guard
                                 \\--(general_llm / llm_fallback)-------> llm -> guard

Every stage is plain, self-written Python; MAF provides the graph, the edges and the agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from agent_framework import Case, Default, Executor, WorkflowBuilder, WorkflowContext, handler
from typing_extensions import Never

from . import renderer
from .agent_layer import LLMResponder
from .data import ClinicalRepository
from .executor import ToolExecutor
from .grounding import check_answer
from .interpreter import LLMInterpreter, RuleInterpreter
from .memory import MemoryTree
from .policy import PolicyEngine
from .dm_tools import DiabetesServices
from .router import DeterministicRouter, is_dm_patient
from .schemas import PATIENT_SCOPED_INTENTS, Intent, ParsedRequest, Role, SharedState
from .validator import ResultValidator


@dataclass
class Services:
    repo: ClinicalRepository
    policy: PolicyEngine
    router: DeterministicRouter
    executor: ToolExecutor
    validator: ResultValidator
    rules: RuleInterpreter
    llm_interpreter: LLMInterpreter | None
    responder: LLMResponder | None
    memory: dict[str, MemoryTree]
    diabetes: DiabetesServices | None = None

    def patient_exists(self, patient_id: str) -> bool:
        if is_dm_patient(patient_id):
            return self.diabetes is not None and self.diabetes.repo.patient_exists(patient_id)
        return self.repo.patient_exists(patient_id)


class _Stage(Executor):
    def __init__(self, stage_id: str, svc: Services):
        super().__init__(id=stage_id)
        self.svc = svc

    def _timed(self, state: SharedState, t0: float) -> None:
        state.timings_ms[self.id] = round((time.perf_counter() - t0) * 1000, 1)


def resolve_target(state: SharedState, parsed: ParsedRequest, tree: MemoryTree | None) -> str | None:
    """Deterministic patient identification. The LLM never sets the authenticated identity.

    The memory tree only holds patients the user was previously granted, and PolicyStage
    re-checks access anyway.
    """
    if parsed.patient_id:
        return parsed.patient_id
    if state.user.role == Role.PATIENT:
        return state.user.patient_id
    if state.user.role == Role.CLINICIAN and tree is not None:
        if parsed.refers_to_previous_patient:
            return tree.previous_patient()
        if parsed.refers_to_context_patient or any(i in PATIENT_SCOPED_INTENTS for i in parsed.intents):
            return tree.focus_patient_id
    return None


class InterpretStage(_Stage):
    @handler
    async def run(self, state: SharedState, ctx: WorkflowContext[SharedState]) -> None:
        t0 = time.perf_counter()
        parsed = self.svc.rules.parse(state.message)
        if parsed.intents == [Intent.UNKNOWN] and self.svc.llm_interpreter is not None:
            try:
                parsed = await self.svc.llm_interpreter.parse(state.message)
                state.llm_calls += 1
            except Exception as e:  # keep the rule result; the router falls back safely
                state.errors.append(f"llm_interpreter: {type(e).__name__}: {e}")
        state.parsed = parsed
        tree = self.svc.memory.get(state.session_id) if state.session_id else None
        state.target_patient_id = resolve_target(state, parsed, tree)
        state.conversation = tree.context_for(state.target_patient_id) if tree else []
        self._timed(state, t0)
        await ctx.send_message(state)


class PolicyStage(_Stage):
    @handler
    async def run(self, state: SharedState, ctx: WorkflowContext[SharedState]) -> None:
        t0 = time.perf_counter()
        intents = state.parsed.intents
        needs_patient = any(i in PATIENT_SCOPED_INTENTS for i in intents)
        if needs_patient and state.target_patient_id is None and state.user.role != Role.OPERATIONS:
            state.route, state.route_reason = "clarify", "Patient-level request without an identifiable patient."
        else:
            decision = self.svc.policy.check_request(state.user, intents, state.target_patient_id)
            state.permissions.append(decision)
            if not decision.allowed:
                state.route, state.route_reason = "denied", decision.reason
            # Existence is checked only after authorisation, so it cannot be probed by unauthorised users.
            elif needs_patient and not self.svc.patient_exists(state.target_patient_id):
                state.route, state.route_reason = "not_found", "Target patient does not exist."
        self._timed(state, t0)
        await ctx.send_message(state)


class RouteStage(_Stage):
    @handler
    async def run(self, state: SharedState, ctx: WorkflowContext[SharedState]) -> None:
        t0 = time.perf_counter()
        d = self.svc.router.route(state.parsed, state.target_patient_id, state.user)
        state.route, state.route_reason = d.route, d.reason
        state.plan, state.unsupported, state.llm_allowed_tools = d.plan, d.unsupported, d.llm_allowed_tools
        state.agents, state.agent_tasks = d.agents, d.agent_tasks
        self._timed(state, t0)
        await ctx.send_message(state)


class ToolStage(_Stage):
    @handler
    async def run(self, state: SharedState, ctx: WorkflowContext[SharedState]) -> None:
        t0 = time.perf_counter()
        self.svc.executor.run_plan(state, state.plan)
        followups = self.svc.router.followups(state)
        if followups:
            state.plan.extend(followups)
            state.llm_allowed_tools = sorted(set(state.llm_allowed_tools) | {c.tool for c in followups})
            self.svc.executor.run_plan(state, followups)
        self._timed(state, t0)
        await ctx.send_message(state)


class ValidateStage(_Stage):
    @handler
    async def run(self, state: SharedState, ctx: WorkflowContext[SharedState]) -> None:
        t0 = time.perf_counter()
        self.svc.validator.validate(state)
        self._timed(state, t0)
        await ctx.send_message(state)


class LLMStage(_Stage):
    @handler
    async def run(self, state: SharedState, ctx: WorkflowContext[SharedState]) -> None:
        t0 = time.perf_counter()
        responder = self.svc.responder
        n_calls_before = len(state.tool_calls)
        try:
            if responder is None:
                raise RuntimeError("LLM not configured")
            if state.route == "general_llm":
                state.answer = await responder.general_info(state)
            else:
                state.answer = await responder.explain(state, with_evidence=state.route == "deterministic")
            state.answer_source = "llm"
        except Exception as e:
            if responder is not None:
                state.errors.append(f"llm: {type(e).__name__}: {e}")
            state.answer_source = "deterministic"
            if state.route == "deterministic":
                state.answer = renderer.evidence_summary(state)
            else:
                state.answer = ("I could not match this request to a supported query, and the language model "
                                "is unavailable, so no answer was generated.")
        if len(state.tool_calls) > n_calls_before:
            self.svc.validator.validate(state)  # LLM retrieved more evidence: recompute findings
        self._timed(state, t0)
        await ctx.send_message(state)


class GuardStage(_Stage):
    @handler
    async def run(self, state: SharedState, ctx: WorkflowContext[Never, SharedState]) -> None:
        t0 = time.perf_counter()
        state.grounding = check_answer(state, state.answer or "")
        if state.grounding.blocked:
            state.errors.append(f"guard: blocked answer mentioning {state.grounding.foreign_patient_ids}")
            state.answer = "The answer was withheld because it referenced records you are not authorised to view."
            state.answer_source = "deterministic"
        self._timed(state, t0)
        await ctx.yield_output(state)


class RespondStage(_Stage):
    @handler
    async def run(self, state: SharedState, ctx: WorkflowContext[Never, SharedState]) -> None:
        render = {"denied": renderer.denied, "clarify": renderer.clarify, "unsupported": renderer.unsupported,
                  "not_found": renderer.not_found}
        state.answer = render[state.route](state)
        state.answer_source = "deterministic"
        await ctx.yield_output(state)


def build_workflow(svc: Services):
    interpret = InterpretStage("interpret", svc)
    policy = PolicyStage("policy", svc)
    route = RouteStage("route", svc)
    tools = ToolStage("tools", svc)
    validate = ValidateStage("validate", svc)
    llm = LLMStage("llm", svc)
    guard = GuardStage("guard", svc)
    respond = RespondStage("respond", svc)

    return (
        WorkflowBuilder(start_executor=interpret, name="controlled_healthcare_agent")
        .add_edge(interpret, policy)
        .add_switch_case_edge_group(policy, [
            Case(condition=lambda s: s.route in ("denied", "clarify", "not_found"), target=respond),
            Default(target=route),
        ])
        .add_switch_case_edge_group(route, [
            Case(condition=lambda s: s.route == "unsupported", target=respond),
            Case(condition=lambda s: s.route == "deterministic", target=tools),
            Default(target=llm),
        ])
        .add_edge(tools, validate)
        .add_edge(validate, llm)
        .add_edge(llm, guard)
        .build()
    )
