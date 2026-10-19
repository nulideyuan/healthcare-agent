"""Tool registry: deterministic data-access functions with explicit contracts.

Each ToolSpec documents name, description, inputs, output, permission scope, errors and
data source. The functions here never check permissions themselves; the ToolExecutor
always asks the PolicyEngine first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from .data import ClinicalRepository, fmt_ts, parse_ts
from .schemas import Role, ToolResult

PATIENT_ID_PATTERN = r"^P\d{4}$"
TS_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$"


# --------------------------------------------------------------------------- input schemas


class PatientProfileArgs(BaseModel):
    patient_id: str = Field(pattern=PATIENT_ID_PATTERN, description="Patient ID, e.g. P0042")


class PatientEncountersArgs(BaseModel):
    patient_id: str = Field(pattern=PATIENT_ID_PATTERN, description="Patient ID, e.g. P0042")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum encounters to return")
    days: int | None = Field(default=None, ge=1, le=3650, description="Only encounters in the last N days")


class RecentVitalsArgs(BaseModel):
    patient_id: str = Field(pattern=PATIENT_ID_PATTERN, description="Patient ID, e.g. P0042")
    hours: int | None = Field(
        default=None, ge=1, le=720,
        description="Only observations from the last N hours. Omit for the most recent observations.",
    )
    encounter_id: str | None = Field(default=None, description="Restrict to one encounter")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum observation sets")


class EDStatusArgs(BaseModel):
    timestamp: str | None = Field(
        default=None, pattern=TS_PATTERN,
        description="Snapshot time YYYY-MM-DDTHH:MM. Omit for the current snapshot.",
    )
    trend_hours: int = Field(default=0, ge=0, le=72, description="Also return the previous N hourly snapshots")


class BedCapacityArgs(BaseModel):
    ward: str | None = Field(default=None, description="Ward name. Omit for all wards.")
    timestamp: str | None = Field(
        default=None, pattern=TS_PATTERN,
        description="Snapshot time YYYY-MM-DDTHH:MM. Omit for the current snapshot.",
    )


# --------------------------------------------------------------------------- spec


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output: str
    scope: Literal["patient", "operational"]
    allowed_roles: frozenset[Role]
    errors: tuple[str, ...]
    data_source: str
    func: Callable[[Any, str, Any], ToolResult] = field(repr=False)
    # Which backend the executor passes as the first argument: the hospital repository,
    # the diabetes services (repository + forecaster), or the session's memory tree.
    domain: Literal["hospital", "diabetes", "memory"] = "hospital"
    agent: str = "hospital_explainer"  # specialist agent that owns this tool

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "agent": self.agent,
            "description": self.description,
            "inputs": self.input_model.model_json_schema()["properties"],
            "output": self.output,
            "scope": self.scope,
            "allowed_roles": sorted(r.value for r in self.allowed_roles),
            "errors": list(self.errors),
            "data_source": self.data_source,
        }


def _result(tool: str, args: BaseModel, source: str, data: Any, data_ts: str | None) -> ToolResult:
    return ToolResult(
        tool=tool, args=args.model_dump(), ok=True, source=source, data=data, data_timestamp=data_ts
    )


def _error(tool: str, args: dict[str, Any], source: str, code: str, msg: str) -> ToolResult:
    return ToolResult(tool=tool, args=args, ok=False, source=source, error_code=code, error=msg)


def _resolve_time(ts: str | None, now: str) -> str:
    if ts is None:
        return now
    if ts > now:
        raise ValueError(f"timestamp {ts} is after the current time {now}")
    return ts


# --------------------------------------------------------------------------- implementations


def get_patient_profile(repo: ClinicalRepository, now: str, a: PatientProfileArgs) -> ToolResult:
    p = repo.get_patient(a.patient_id)
    if p is None:
        return _error("get_patient_profile", a.model_dump(), "patients", "NOT_FOUND",
                      f"No patient with ID {a.patient_id}")
    return _result("get_patient_profile", a, "patients", p, None)


def get_patient_encounters(repo: ClinicalRepository, now: str, a: PatientEncountersArgs) -> ToolResult:
    if not repo.patient_exists(a.patient_id):
        return _error("get_patient_encounters", a.model_dump(), "encounters", "NOT_FOUND",
                      f"No patient with ID {a.patient_id}")
    since = fmt_ts(parse_ts(now) - timedelta(days=a.days)) if a.days else None
    rows = repo.get_encounters(a.patient_id, now, limit=a.limit, since=since)
    window = f" in the last {a.days} days" if a.days else ""
    if not rows:
        return _error("get_patient_encounters", a.model_dump(), "encounters", "NO_DATA",
                      f"No encounters found for {a.patient_id}{window}")
    return _result("get_patient_encounters", a, "encounters",
                   {"encounters": rows, "count": len(rows), "window_days": a.days, "as_of": now},
                   rows[0]["arrival_time"])


def get_recent_vitals(repo: ClinicalRepository, now: str, a: RecentVitalsArgs) -> ToolResult:
    if not repo.patient_exists(a.patient_id):
        return _error("get_recent_vitals", a.model_dump(), "vitals", "NOT_FOUND",
                      f"No patient with ID {a.patient_id}")
    rows = repo.get_vitals(a.patient_id, now, hours=a.hours, encounter_id=a.encounter_id, limit=a.limit)
    if not rows:
        latest = repo.latest_vitals_time(a.patient_id, now)
        window = f" in the last {a.hours} hours" if a.hours else ""
        hint = f" (most recent observations on record: {latest})" if latest else " (no observations on record)"
        return _error("get_recent_vitals", a.model_dump(), "vitals", "NO_DATA",
                      f"No observations found for {a.patient_id}{window}{hint}")
    return _result("get_recent_vitals", a, "vitals",
                   {"observations": rows, "count": len(rows), "as_of": now}, rows[0]["timestamp"])


def get_ed_status(repo: ClinicalRepository, now: str, a: EDStatusArgs) -> ToolResult:
    try:
        at = _resolve_time(a.timestamp, now)
    except ValueError as e:
        return _error("get_ed_status", a.model_dump(), "ed_status", "INVALID_INPUT", str(e))
    rows = repo.get_ed_status(at, a.trend_hours)
    if not rows:
        return _error("get_ed_status", a.model_dump(), "ed_status", "NO_DATA", f"No ED status at or before {at}")
    return _result("get_ed_status", a, "ed_status",
                   {"current": rows[0], "trend": rows[1:], "requested_time": at}, rows[0]["timestamp"])


def get_bed_capacity(repo: ClinicalRepository, now: str, a: BedCapacityArgs) -> ToolResult:
    try:
        at = _resolve_time(a.timestamp, now)
    except ValueError as e:
        return _error("get_bed_capacity", a.model_dump(), "beds", "INVALID_INPUT", str(e))
    if a.ward and a.ward.lower() not in {w.lower() for w in repo.wards()}:
        return _error("get_bed_capacity", a.model_dump(), "beds", "NOT_FOUND",
                      f"Unknown ward '{a.ward}'. Known wards: {', '.join(repo.wards())}")
    rows = repo.get_beds(at, a.ward)
    if not rows:
        return _error("get_bed_capacity", a.model_dump(), "beds", "NO_DATA", f"No bed data at or before {at}")
    totals = {k: sum(r[k] for r in rows) for k in ("total_beds", "occupied_beds", "available_beds", "blocked_beds")}
    totals["occupancy_ratio"] = round(totals["occupied_beds"] / totals["total_beds"], 3)
    return _result("get_bed_capacity", a, "beds",
                   {"wards": rows, "totals": totals, "requested_time": at}, rows[0]["timestamp"])


# --------------------------------------------------------------------------- registry

ALL_ROLES = frozenset(Role)
CLINICAL_ROLES = frozenset({Role.PATIENT, Role.CLINICIAN})
PATIENT_ERRORS = ("INVALID_INPUT", "NOT_FOUND", "NO_DATA", "ACCESS_DENIED")
OPS_ERRORS = ("INVALID_INPUT", "NOT_FOUND", "NO_DATA")

TOOLS: dict[str, ToolSpec] = {
    s.name: s
    for s in [
        ToolSpec(
            "get_patient_profile",
            "Retrieve a patient's demographics (age, sex, height, weight, BMI, blood group) and chronic conditions.",
            PatientProfileArgs,
            "patient record: patient_id, age, sex, height_m, weight_kg, bmi, blood_group, chronic_conditions[]",
            "patient", CLINICAL_ROLES, PATIENT_ERRORS, "patients", get_patient_profile,
        ),
        ToolSpec(
            "get_patient_encounters",
            "Retrieve a patient's previous and current hospital encounters (ED attendances), most recent first.",
            PatientEncountersArgs,
            "encounters[]: encounter_id, arrival_time, chief_complaint, triage_level, disposition, "
            "admission_time, discharge_time, status_at_now, ed_minutes",
            "patient", CLINICAL_ROLES, PATIENT_ERRORS, "encounters", get_patient_encounters,
        ),
        ToolSpec(
            "get_recent_vitals",
            "Retrieve a patient's recent vital-sign observations (HR, RR, SpO2, temperature, BP), most recent first.",
            RecentVitalsArgs,
            "observations[]: timestamp, heart_rate_bpm, respiratory_rate_bpm, spo2_pct, temperature_c, "
            "systolic_bp_mmHg, diastolic_bp_mmHg",
            "patient", CLINICAL_ROLES, PATIENT_ERRORS, "vitals", get_recent_vitals,
        ),
        ToolSpec(
            "get_ed_status",
            "Retrieve Emergency Department traffic: patients in ED, waiting, on trolleys, wait times, "
            "ambulance arrivals and occupancy. Contains no patient-identifiable data.",
            EDStatusArgs,
            "current snapshot + optional hourly trend",
            "operational", ALL_ROLES, OPS_ERRORS, "ed_status", get_ed_status,
        ),
        ToolSpec(
            "get_bed_capacity",
            "Retrieve inpatient bed capacity and occupancy by ward (Medical, Surgical, ICU, Cardiology, "
            "Respiratory, Orthopaedics). Contains no patient-identifiable data.",
            BedCapacityArgs,
            "wards[]: ward, total_beds, occupied_beds, available_beds, blocked_beds, occupancy_ratio; totals",
            "operational", ALL_ROLES, OPS_ERRORS, "beds", get_bed_capacity,
        ),
    ]
}


def validate_args(spec: ToolSpec, args: dict[str, Any]) -> BaseModel:
    """Raises ValueError with a compact message on invalid input."""
    try:
        return spec.input_model.model_validate(args)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        raise ValueError(problems) from None


# Diabetes-cohort tools (ShanghaiT2DM) live in dm_tools.py and join the same registry, so they get the
# same policy checks, provenance and memory caching as the hospital tools.
from .dm_tools import DM_TOOLS  # noqa: E402

TOOLS.update(DM_TOOLS)
