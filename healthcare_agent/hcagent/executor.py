"""Tool executor: the only path to data. Validates input, checks policy, records provenance."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from .data import ClinicalRepository
from .memory import MemoryTree
from .policy import PolicyEngine
from .schemas import PlannedCall, SharedState, SourceRef, ToolCallRecord, ToolResult
from .tools import TOOLS, validate_args

_RECORD_KEYS = {
    "patients": "patient_id",
    "encounters": "encounter_id",
    "vitals": "vital_id",
    "ed_status": "status_id",
    "beds": "bed_status_id",
}


def _record_ids(result: ToolResult) -> list[str]:
    key = _RECORD_KEYS.get(result.source)
    d = result.data
    if not key or d is None:
        return []
    rows: list[dict[str, Any]]
    if result.source == "patients":
        rows = [d]
    elif result.source == "encounters":
        rows = d["encounters"]
    elif result.source == "vitals":
        rows = d["observations"]
    elif result.source == "ed_status":
        rows = [d["current"], *d["trend"]]
    else:
        rows = d["wards"]
    return [r[key] for r in rows]


@dataclass
class MemoryAccess:
    """What a memory tool may read: this session's tree and the same user's earlier sessions."""

    tree: MemoryTree | None
    store: Any
    user_id: str
    session_id: str | None

    def earlier_turns(self, patient_id: str) -> list[dict[str, Any]]:
        return self.store.earlier_turns(self.user_id, patient_id, self.session_id) if self.store else []


class ToolExecutor:
    def __init__(self, repo: ClinicalRepository, policy: PolicyEngine, memory: dict[str, MemoryTree] | None = None,
                 diabetes: Any = None, store: Any = None):
        self.repo = repo
        self.policy = policy
        self.memory = memory if memory is not None else {}
        self.backends = {"hospital": repo, "diabetes": diabetes}
        self.store = store  # persistent memory store, for cross-session recall

    def execute(
        self, state: SharedState, call: PlannedCall, caller: Literal["router", "llm", "baseline"]
    ) -> ToolResult:
        call_id = f"{state.request_id}-T{len(state.tool_calls) + 1:02d}"
        t0 = time.perf_counter()

        def record(status, code=None):
            state.tool_calls.append(ToolCallRecord(
                call_id=call_id, tool=call.tool, args=call.args, caller=caller, status=status,
                error_code=code, duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ))

        spec = TOOLS.get(call.tool)
        if spec is None:
            record("blocked", "UNKNOWN_TOOL")
            return ToolResult(tool=call.tool, args=call.args, ok=False, source="none",
                              error_code="UNKNOWN_TOOL", error=f"Unknown tool {call.tool}")
        try:
            args = validate_args(spec, call.args)
        except ValueError as e:
            record("error", "INVALID_INPUT")
            result = ToolResult(tool=call.tool, args=call.args, ok=False, source=spec.data_source,
                                error_code="INVALID_INPUT", error=str(e))
            state.tool_results.append(result)
            return result

        decision = self.policy.check_tool(state.user, spec, args.model_dump())
        state.permissions.append(decision)
        if not decision.allowed:
            record("denied", "ACCESS_DENIED")
            return ToolResult(tool=call.tool, args=args.model_dump(), ok=False, source=spec.data_source,
                              error_code="ACCESS_DENIED", error=decision.reason)

        # Memory is consulted only after the policy has approved this exact call.
        tree = self.memory.get(state.session_id) if state.session_id else None
        fact = tree.lookup(call.tool, args.model_dump(), state.clock) if tree and spec.domain != "memory" else None
        backend = (MemoryAccess(tree, self.store, state.user.user_id, state.session_id) if spec.domain == "memory"
                   else self.backends.get(spec.domain))
        if fact is not None:
            result = fact.result.model_copy(deep=True, update={"from_memory": True, "memory_request_id": fact.request_id})
            record("cached")
        elif backend is None and spec.domain != "memory":
            record("error", "UNAVAILABLE")
            result = ToolResult(tool=call.tool, args=args.model_dump(), ok=False, source=spec.data_source,
                                error_code="UNAVAILABLE", error=f"The {spec.domain} data source is not loaded.")
        else:
            result = spec.func(backend, state.clock, args)
            record("ok" if result.ok else "error", result.error_code)
        state.tool_results.append(result)
        if result.ok:
            state.sources.append(SourceRef(
                tool=call.tool, source=result.source, record_ids=_record_ids(result),
                data_timestamp=result.data_timestamp, patient_id=args.model_dump().get("patient_id"),
                from_memory=result.from_memory,
            ))
        return result

    @staticmethod
    def record_rejection(state: SharedState, tool: str, args: dict[str, Any], status: str, code: str) -> None:
        """Log a call that a guard stopped before it reached execute()."""
        state.tool_calls.append(ToolCallRecord(
            call_id=f"{state.request_id}-T{len(state.tool_calls) + 1:02d}", tool=tool, args=args,
            caller="llm", status=status, error_code=code,
        ))

    def run_plan(self, state: SharedState, plan: list[PlannedCall]) -> None:
        for call in plan:
            self.execute(state, call, caller="router")
