"""Response guard: checks that numbers and patient IDs in an answer are backed by evidence.

Numbers are compared against every numeric value in the shared state (tool data, findings,
contexts, clock and the user's own question), including common unit conversions
(ratio -> %, minutes -> hours). This is a conservative lexical check, used both as a
runtime warning and as the data-grounding metric in evaluation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from .schemas import GroundingReport, Role, SharedState

ID_RE = re.compile(r"\b(P\d{4}|E\d{5}|V\d{6}|ED\d{5}|B\d{6}|REQ-[0-9A-F]{8}|U\w+)\b")
PATIENT_ID_RE = re.compile(r"\bP\d{4}\b")
DATE_RE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
TIME_RE = re.compile(r"(?<![\d:])\d{2}:\d{2}(?![\d:])")
NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])")
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty-four": 24}
# Small integers and common clinical constants that do not assert a patient/hospital fact.
TRIVIAL = {0, 1, 2, 24, 100}


def _walk(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def _expand(x: float, key_hint: str = "") -> set[float]:
    out = {x, round(x), round(x, 1)}
    if 0 < x <= 2.5:  # ratios -> percentages
        out |= {round(x * 100), round(x * 100, 1)}
    if x >= 30:  # minutes -> hours
        out |= {round(x / 60), round(x / 60, 1), x // 60, x % 60}
    if key_hint == "days":
        out.add(round(x / 30))
    return out


def evidence_numbers(state: SharedState) -> set[float]:
    nums: set[float] = set()
    payload = {
        "results": [r.data for r in state.tool_results if r.ok],
        "args": [r.args for r in state.tool_results],
        "patient": state.patient,
        "clinical": state.clinical_context,
        "hospital": state.hospital_context,
        "findings": [f.model_dump() for f in state.findings],
        "clock": state.clock,
        "question": state.message,
    }
    text_blob = json.dumps(payload, default=str)
    for v in _walk(payload):
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            nums |= _expand(float(v))
    for m in NUM_RE.finditer(ID_RE.sub(" ", text_blob)):
        nums |= _expand(float(m.group(1)))
    for w, n in WORD_NUMBERS.items():
        if re.search(rf"\b{w}\b", state.message, re.IGNORECASE):
            nums.add(n)
    return nums


def _supported(x: float, pool: set[float]) -> bool:
    return any(abs(x - y) <= max(0.051, 0.01 * abs(y)) for y in pool)


def _evidence_blob(state: SharedState) -> str:
    return json.dumps([state.model_dump(include={"tool_results", "findings", "clock", "message"})], default=str)


def check_answer(state: SharedState, answer: str) -> GroundingReport:
    report = GroundingReport()
    pool = evidence_numbers(state)
    blob = _evidence_blob(state)
    stripped = ID_RE.sub(" ", answer)

    # Dates and clock times are checked as whole tokens, then removed from the number scan.
    for rx in (DATE_RE, TIME_RE):
        for m in rx.finditer(stripped):
            report.numbers_checked += 1
            if m.group(0) in blob:
                report.numbers_supported += 1
            else:
                report.unsupported_numbers.append(m.group(0))
        stripped = rx.sub(" ", stripped)

    for m in NUM_RE.finditer(stripped):
        x = float(m.group(1))
        if x in TRIVIAL:
            continue
        report.numbers_checked += 1
        if _supported(x, pool):
            report.numbers_supported += 1
        else:
            report.unsupported_numbers.append(m.group(1))

    allowed_ids = {state.target_patient_id, state.user.patient_id} | {
        s.patient_id for s in state.sources if s.patient_id
    }
    report.foreign_patient_ids = sorted({p for p in PATIENT_ID_RE.findall(answer) if p not in allowed_ids})
    # Patients and operations users must never see an identifier they were not authorised for.
    report.blocked = state.user.role in (Role.PATIENT, Role.OPERATIONS) and bool(report.foreign_patient_ids)
    return report
