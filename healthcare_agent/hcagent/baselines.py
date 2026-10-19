"""Comparison systems for the research question (spec section 2).

Baseline A  pure LLM: no tools, no data.
Baseline B  LLM + all tools: the LLM chooses tools and arguments freely and is *asked* (not
            forced) to respect permissions. No router, no policy engine, no validator.
正确的system: the LLM is given a router, policy engine and validator. It is forced to respect permissions and to call tools only when necessary.

Both return a SharedState so the same evaluation code applies to all three systems.
"""

from __future__ import annotations

import json
import time
from typing import Any

from agent_framework import Agent, FunctionInvocationContext, FunctionTool

from .data import ClinicalRepository
from .executor import MemoryAccess
from .grounding import check_answer
from .schemas import Intent, ParsedRequest, Role, SharedState, ToolCallRecord, UserContext
from .tools import TOOLS, ToolSpec, validate_args

BASELINE_INSTRUCTIONS = """You are a hospital assistant for a healthcare organisation.
{identity}
Current time: {clock}.
Answer the user's question. Respect patient confidentiality: patients may only see their own records and
operations staff should not see patient-identifiable data."""


def _identity(user: UserContext) -> str:
    if user.role == Role.PATIENT:
        return f"You are talking to patient {user.patient_id}."
    if user.role == Role.CLINICIAN:
        scope = (f" They are authorised for patients {', '.join(user.authorized_patients)} only."
                 if user.authorized_patients is not None else "")
        return f"You are talking to clinician {user.user_id}.{scope}"
    return f"You are talking to hospital operations user {user.user_id}."


def _raw_invoker(spec: ToolSpec, backends: dict[str, Any]):
    async def invoke(ctx: FunctionInvocationContext, **kwargs: Any) -> str:
        state: SharedState = ctx.kwargs["state"]
        args = {k: v for k, v in kwargs.items() if v is not None}
        t0 = time.perf_counter()
        backend = MemoryAccess(None, None, state.user.user_id, None) if spec.domain == "memory" else backends.get(spec.domain)
        try:
            if backend is None:
                raise ValueError(f"{spec.domain} data source not loaded")
            result = spec.func(backend, state.clock, validate_args(spec, args))
            status, code = ("ok" if result.ok else "error"), result.error_code
            state.tool_results.append(result)
            payload = result.model_dump_json(exclude={"retrieved_at"})
        except ValueError as e:
            status, code, payload = "error", "INVALID_INPUT", json.dumps({"error": str(e)})
        state.tool_calls.append(ToolCallRecord(
            call_id=f"{state.request_id}-T{len(state.tool_calls) + 1:02d}", tool=spec.name, args=args,
            caller="baseline", status=status, error_code=code,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        ))
        return payload

    return invoke


class BaselineSystem:
    def __init__(self, client, repo: ClinicalRepository, clock: str, with_tools: bool, diabetes: Any = None):
        self.client, self.repo, self.clock, self.with_tools = client, repo, clock, with_tools
        backends = {"hospital": repo, "diabetes": diabetes}
        self.tools = [
            FunctionTool(name=s.name, description=s.description, func=_raw_invoker(s, backends), input_model=s.input_model)
            for s in TOOLS.values()
        ] if with_tools else []
        self.name = "baseline_b_llm_tools" if with_tools else "baseline_a_pure_llm"

    async def ask(self, user: UserContext, message: str, history: list[str] | None = None) -> SharedState:
        state = SharedState(clock=self.clock, user=user, message=message, route="llm_fallback",
                            parsed=ParsedRequest(intents=[Intent.UNKNOWN]))
        agent = Agent(
            client=self.client, name=self.name,
            instructions=BASELINE_INSTRUCTIONS.format(identity=_identity(user), clock=self.clock),
        )
        prompt = message
        if history:
            prompt = "Earlier questions in this conversation:\n" + "\n".join(f"- {h}" for h in history) + f"\n\n{message}"
        t0 = time.perf_counter()
        response = await agent.run(
            prompt, tools=self.tools or None, options={"temperature": 0},
            function_invocation_kwargs={"state": state},
        )
        state.timings_ms["llm"] = round((time.perf_counter() - t0) * 1000, 1)
        state.llm_calls = 1
        state.answer, state.answer_source = response.text, "llm"
        state.grounding = check_answer(state, state.answer)
        return state
