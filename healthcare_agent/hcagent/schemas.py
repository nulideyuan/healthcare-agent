"""Typed contracts shared by every component. SharedState is the single source of truth."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- users


class Role(str, Enum):
    PATIENT = "patient"
    CLINICIAN = "clinician"
    OPERATIONS = "operations"


class UserContext(BaseModel):
    """Authenticated identity. Never derived from the chat message."""

    user_id: str
    role: Role
    display_name: str = ""
    # For role=patient: the patient record this login is bound to.
    patient_id: str | None = None
    # For role=clinician: care-team scope. None means all patients.
    authorized_patients: list[str] | None = None


# --------------------------------------------------------------------------- intents


class Intent(str, Enum):
    PATIENT_INFORMATION = "PATIENT_INFORMATION"
    PATIENT_HISTORY = "PATIENT_HISTORY"
    VITAL_SIGNS = "VITAL_SIGNS"
    PATIENT_ED_STATUS = "PATIENT_ED_STATUS"
    LAB_RESULTS = "LAB_RESULTS"
    MEDICATION_QUERY = "MEDICATION_QUERY"
    ED_STATUS = "ED_STATUS"
    BED_AVAILABILITY = "BED_AVAILABILITY"
    WARD_OCCUPANCY = "WARD_OCCUPANCY"
    PREDICTION = "PREDICTION"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    GENERAL_MEDICAL_INFORMATION = "GENERAL_MEDICAL_INFORMATION"
    # Diabetes cohort (CGM)
    GLUCOSE_TREND = "GLUCOSE_TREND"
    GLUCOSE_FORECAST = "GLUCOSE_FORECAST"
    GLYCEMIC_RISK = "GLYCEMIC_RISK"
    FORECAST_ACCURACY = "FORECAST_ACCURACY"
    MEAL_RESPONSE = "MEAL_RESPONSE"
    GLUCOSE_PATTERN_HISTORY = "GLUCOSE_PATTERN_HISTORY"
    MEMORY_RECALL = "MEMORY_RECALL"
    UNKNOWN = "UNKNOWN"


PATIENT_SCOPED_INTENTS = {
    Intent.PATIENT_INFORMATION,
    Intent.PATIENT_HISTORY,
    Intent.VITAL_SIGNS,
    Intent.PATIENT_ED_STATUS,
    Intent.LAB_RESULTS,
    Intent.MEDICATION_QUERY,
    Intent.RISK_ASSESSMENT,
    Intent.GLUCOSE_TREND,
    Intent.GLUCOSE_FORECAST,
    Intent.GLYCEMIC_RISK,
    Intent.FORECAST_ACCURACY,
    Intent.MEAL_RESPONSE,
    Intent.GLUCOSE_PATTERN_HISTORY,
    Intent.MEMORY_RECALL,
}


class ParsedRequest(BaseModel):
    """Structured output of the request interpreter."""

    intents: list[Intent] = Field(description="Detected intents, primary first.")
    patient_id: str | None = Field(
        default=None, description="Explicit target patient ID such as P0042, if any."
    )
    refers_to_self: bool = Field(
        default=False, description="True when the user asks about themselves (I/my/me)."
    )
    refers_to_context_patient: bool = Field(
        default=False,
        description="True when the user says 'this patient' / 'the patient' without an ID.",
    )
    refers_to_previous_patient: bool = Field(
        default=False, description="True when the user says 'the previous / other patient'."
    )
    ward: str | None = Field(default=None, description="Ward name if one is mentioned.")
    hours: int | None = Field(default=None, description="Time window in hours, if stated.")
    days: int | None = Field(default=None, description="Look-back window in days, if stated.")
    as_of: str | None = Field(default=None, description="Replay time YYYY-MM-DDTHH:MM, if stated.")
    parser: Literal["rules", "llm"] = "rules"

    @property
    def primary(self) -> Intent:
        return self.intents[0] if self.intents else Intent.UNKNOWN


class LLMParsedRequest(BaseModel):
    """Schema the LLM must fill when rules cannot classify the request."""

    intents: list[Intent]
    patient_id: str | None = None
    refers_to_self: bool = False
    refers_to_context_patient: bool = False
    refers_to_previous_patient: bool = False
    ward: str | None = None
    hours: int | None = None
    days: int | None = None
    as_of: str | None = None


# --------------------------------------------------------------------------- tools / provenance


class PlannedCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class ToolResult(BaseModel):
    tool: str
    args: dict[str, Any]
    ok: bool
    error_code: str | None = None  # NOT_FOUND | NO_DATA | INVALID_INPUT | ACCESS_DENIED
    error: str | None = None
    source: str  # data source (table)
    data_timestamp: str | None = None  # timestamp of the newest record returned
    retrieved_at: str = Field(default_factory=utc_now)
    data: Any = None
    from_memory: bool = False  # served from the session memory tree
    memory_request_id: str | None = None  # request that originally retrieved it


class SourceRef(BaseModel):
    tool: str
    source: str
    record_ids: list[str] = Field(default_factory=list)
    data_timestamp: str | None = None
    patient_id: str | None = None
    from_memory: bool = False


class ToolCallRecord(BaseModel):
    call_id: str
    tool: str
    args: dict[str, Any]
    caller: Literal["router", "llm", "baseline"]
    status: Literal["ok", "error", "denied", "blocked", "cached"]
    error_code: str | None = None
    duration_ms: float = 0.0
    started_at: str = Field(default_factory=utc_now)


class PermissionDecision(BaseModel):
    allowed: bool
    rule: str
    reason: str
    tool: str | None = None
    patient_id: str | None = None
    checked_at: str = Field(default_factory=utc_now)


class Finding(BaseModel):
    """Output of a deterministic rule. Clinical and operational findings stay separate."""

    category: Literal["clinical", "operational", "data_quality", "context"]
    code: str
    message: str
    severity: Literal["info", "warning", "critical"] = "info"
    evidence: dict[str, Any] = Field(default_factory=dict)
    owner: str | None = None  # specialist agent this finding belongs to (None = shared)


class GroundingReport(BaseModel):
    numbers_checked: int = 0
    numbers_supported: int = 0
    unsupported_numbers: list[str] = Field(default_factory=list)
    foreign_patient_ids: list[str] = Field(default_factory=list)
    blocked: bool = False

    @property
    def score(self) -> float:
        return 1.0 if self.numbers_checked == 0 else self.numbers_supported / self.numbers_checked


# --------------------------------------------------------------------------- shared state

Route = Literal[
    "deterministic",
    "general_llm",
    "llm_fallback",
    "denied",
    "unsupported",
    "clarify",
    "not_found",
]


class SharedState(BaseModel):
    request_id: str = Field(default_factory=lambda: "REQ-" + uuid.uuid4().hex[:8].upper())
    created_at: str = Field(default_factory=utc_now)
    clock: str  # simulated "now" used for every data query
    session_id: str | None = None  # key of this conversation's memory tree
    user: UserContext
    message: str
    conversation: list[dict[str, Any]] = Field(default_factory=list)  # turns selected from the memory tree

    parsed: ParsedRequest | None = None
    target_patient_id: str | None = None
    route: Route | None = None
    route_reason: str = ""
    plan: list[PlannedCall] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)  # requested data with no v1 source
    llm_allowed_tools: list[str] = Field(default_factory=list)
    agents: list[str] = Field(default_factory=list)  # specialist agents the router assigned
    agent_tasks: dict[str, list[str]] = Field(default_factory=dict)  # agent -> intents it must answer

    patient: dict[str, Any] | None = None
    clinical_context: dict[str, Any] = Field(default_factory=dict)
    hospital_context: dict[str, Any] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)

    tool_results: list[ToolResult] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    permissions: list[PermissionDecision] = Field(default_factory=list)

    answer: str | None = None
    answer_source: Literal["llm", "deterministic"] | None = None
    grounding: GroundingReport | None = None
    llm_calls: int = 0
    errors: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)

    def results_for(self, tool: str) -> list[ToolResult]:
        return [r for r in self.tool_results if r.tool == tool]

    def last_ok(self, tool: str) -> ToolResult | None:
        for r in reversed(self.tool_results):
            if r.tool == tool and r.ok:
                return r
        return None
