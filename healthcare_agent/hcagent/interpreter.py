"""Request interpreter: deterministic rules first, LLM structured output only as a fallback."""

from __future__ import annotations

import re

from .schemas import PATIENT_SCOPED_INTENTS, Intent, LLMParsedRequest, ParsedRequest

_I = re.IGNORECASE

PATIENT_ID_RE = re.compile(r"\bP[-\s]?0*(\d{1,4})\b", _I)
DM_ID_RE = re.compile(r"\bDM[-\s]?(\d{4})\b", _I)                      # diabetes cohort, e.g. DM2045
AS_OF_RE = re.compile(r"\b(?:at|as of|on)\s+(\d{4}-\d{2}-\d{2})[ T](\d{1,2}):(\d{2})\b", _I)
PATIENT_NUM_RE = re.compile(r"\bpatient\s+(?:id\s+)?#?0*(\d{1,4})\b", _I)
SELF_RE = re.compile(r"\b(i|i'm|i've|im|me|my|mine|myself)\b", _I)
PREVIOUS_PATIENT_RE = re.compile(r"\b(previous|earlier|other|first|last|prior) patient\b", _I)
CONTEXT_PATIENT_RE =re.compile(r"\b(this|the|that|same)\s+patient('s)?\b|\b(he|she|him|her|his|hers)\b", _I)
GENERAL_Q_RE = re.compile(r"^\s*(what\s+(is|are|does|causes)|explain|define|tell me about|how does|why do)\b", _I)

WARDS = {
    "medical": "Medical", "surgical": "Surgical", "icu": "ICU", "intensive care": "ICU",
    "cardiology": "Cardiology", "cardiac": "Cardiology", "respiratory": "Respiratory",
    "orthopaedics": "Orthopaedics", "orthopedics": "Orthopaedics", "orthopaedic": "Orthopaedics",
    "orthopedic": "Orthopaedics", "ortho": "Orthopaedics",
}
WARD_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, WARDS), key=len, reverse=True)) + r")\b", _I)

NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty-four": 24, "forty-eight": 48,
}
WINDOW_RE = re.compile(
    r"\b(?:last|past|previous|over the last|in the last)\s+(\d+|" + "|".join(NUMBER_WORDS) + r")?\s*"
    r"(hours?|hrs?|days?|weeks?|months?|years?)\b",
    _I,
)

# Ordered (intent, pattern). Order only matters for the order of `intents`.
INTENT_RULES: list[tuple[Intent, re.Pattern]] = [
    (Intent.PREDICTION, re.compile(r"\b(predict\w*|forecast\w*|projected|expected demand|next\s+\d+\s+hours|next (day|week|24 hours)|tomorrow)\b", _I)),
    (Intent.RISK_ASSESSMENT, re.compile(r"\b(risk|deteriorat\w*|early warning|news2?)\b", _I)),
    (Intent.LAB_RESULTS, re.compile(r"\b(labs?|laborator\w*|blood tests?|bloods|blood results?|potassium|sodium|creatinine|ha?emoglobin|hba1c|a1c|c-?peptide|glycated albumin|cholesterol|lipids?|egfr|triglycerides?|troponin|crp|white cells?|wbc|electrolytes?)\b", _I)),
    (Intent.MEDICATION_QUERY, re.compile(r"\b(medications?|medicines?|meds|drugs?|tablets?|prescri\w*|dosage|doses?|insulin|metformin|acarbose|bolus|basal|injections?|hypoglyc(a)?emic agents?)\b", _I)),
    (Intent.VITAL_SIGNS, re.compile(r"\b(vitals?|vital signs|observations?|obs|spo2|sats|oxygen|saturations?|heart rate|pulse|blood pressure|bp|temperature|temp|respiratory rate|resp rate|breathing rate)\b", _I)),
    (Intent.PATIENT_HISTORY, re.compile(r"\b(history|(previous|prior|past|last) (visits?|encounters?|admissions?|attendances?|presentations?)|attend\w*|visits?|encounters?|admissions?|been (to|in) (the )?(ed|hospital|emergency))\b", _I)),
    (Intent.PATIENT_INFORMATION, re.compile(r"\b(age|how old|demographics?|profile|blood (group|type)|height|weight|bmi|chronic|conditions?|diagnos\w*|sex|gender|who is)\b", _I)),
    (Intent.WARD_OCCUPANCY, re.compile(r"\b(wards|which ward|ward[- ]level|ward occupancy|close to capacity|near(ly)? (full|capacity)|at capacity|over capacity)\b", _I)),
    (Intent.BED_AVAILABILITY, re.compile(r"\b(beds?|bed capacity|inpatient capacity)\b", _I)),
    (Intent.ED_STATUS, re.compile(r"\b(busy|crowded|congest\w*|waiting room|trolleys?|ambulances?|queue|wait(ing)? times?|how long .*wait|ed occupancy|ed traffic|patients waiting)\b", _I)),
]
# Naming the ED alone is an operational question only when no patient-level intent is present
# ("Has this patient attended the ED?" is history, not ED status).
ED_MENTION_RE = re.compile(r"\b(ed|a&e|a and e|emergency department|emergency room|er)\b", _I)
PATIENT_WAIT_RE = re.compile(r"\b(waiting|waited|wait so long|so long|taking so long|delay\w*|still (here|in|waiting)|been here|stuck)\b", _I)

# Diabetes-cohort (CGM) intents. The glucose rules apply only when the text is about glucose.
GLUCOSE_RE = re.compile(r"\b(glucose|sugars?|bg|cgm|glyc(a)?emi\w*|hypo\w*|hyper\w*|go(ing)? (low|high))\b", _I)
GLUCOSE_RULES: list[tuple[Intent, re.Pattern]] = [
    (Intent.GLYCEMIC_RISK, re.compile(r"\b(hypo\w*|hyper\w*|low blood sugar|high blood sugar|go(ing)? (low|high)|risk)\b", _I)),
    (Intent.GLUCOSE_FORECAST, re.compile(r"\b(predict\w*|forecast\w*|projected|expect\w*|\d+\s*(min|mins|minutes)|next (hour|half hour)|will\b.*\b(be|go|drop|rise|fall|reach)|going to)\b", _I)),
    (Intent.GLUCOSE_TREND, re.compile(r"\b(trend\w*|current\w*|now|rising|falling|dropping|going (up|down)|direction|trajectory|level|reading|latest)\b", _I)),
]
DM_RULES: list[tuple[Intent, re.Pattern]] = [
    (Intent.FORECAST_ACCURACY, re.compile(r"\b(accura\w*|reliab\w*|how good|confiden\w*|trust\w*|validat\w*|rmse|mae)\b", _I)),
    (Intent.MEAL_RESPONSE, re.compile(r"\b(meals?|ate|eat(ing)?|breakfast|lunch|dinner|food|postprandial|diet\w*)\b", _I)),
    (Intent.GLUCOSE_PATTERN_HISTORY, re.compile(r"\b(similar|patterns?|happened before|seen (this|that) before|like this before)\b", _I)),
    (Intent.MEMORY_RECALL, re.compile(r"\b(already (retrieved|looked up|checked|found|know)|retrieved (so far|already|in this conversation)|(what|which) (information|data|facts) (have|has)|so far in this conversation|from (the )?memory)\b", _I)),
]
# Broader glucose intents include the narrower ones' tools (risk -> forecast -> trend).
SUBSUMED_BY = {Intent.GLYCEMIC_RISK: {Intent.GLUCOSE_FORECAST, Intent.GLUCOSE_TREND, Intent.RISK_ASSESSMENT, Intent.PREDICTION},
               Intent.GLUCOSE_FORECAST: {Intent.GLUCOSE_TREND, Intent.PREDICTION},
               Intent.FORECAST_ACCURACY: {Intent.PREDICTION, Intent.GLUCOSE_FORECAST, Intent.GLUCOSE_TREND},
               Intent.MEMORY_RECALL: {Intent.PATIENT_INFORMATION}}

# A patient-level ED question already retrieves ED, bed and vitals context.
SUBSUMED_BY_PATIENT_ED = {Intent.ED_STATUS, Intent.BED_AVAILABILITY, Intent.VITAL_SIGNS}


def normalize_patient_id(raw: str | None) -> str | None:
    if not raw:
        return None
    if m := DM_ID_RE.search(raw):
        return f"DM{m.group(1)}"
    m = PATIENT_ID_RE.search(raw) or re.search(r"(\d{1,4})", raw)
    return f"P{int(m.group(1)):04d}" if m else None


def _as_of(text: str) -> str | None:
    m = AS_OF_RE.search(text)
    return f"{m.group(1)}T{int(m.group(2)):02d}:{m.group(3)}" if m else None


def _window(text: str) -> tuple[int | None, int | None]:
    m = WINDOW_RE.search(text)
    if m:
        n_raw = (m.group(1) or "1").lower()
        n = int(n_raw) if n_raw.isdigit() else NUMBER_WORDS.get(n_raw, 1)
        unit = m.group(2).lower()
        if unit.startswith(("hour", "hr")):
            return n, None
        if unit.startswith("day"):
            return None, n
        if unit.startswith("week"):
            return None, 7 * n
        if unit.startswith("month"):
            return None, 30 * n
        return None, 365 * n
    if re.search(r"\btoday\b|\bthis morning\b|\btonight\b", text, _I):
        return 24, None
    if re.search(r"\byesterday\b", text, _I):
        return 48, None
    return None, None


class RuleInterpreter:
    def parse(self, text: str) -> ParsedRequest:
        dm = DM_ID_RE.search(text)
        m = None if dm else (PATIENT_ID_RE.search(text) or PATIENT_NUM_RE.search(text))
        patient_id = f"DM{dm.group(1)}" if dm else (f"P{int(m.group(1)):04d}" if m else None)
        refers_to_self = bool(SELF_RE.search(text))
        refers_to_previous = patient_id is None and bool(PREVIOUS_PATIENT_RE.search(text))
        refers_to_context = patient_id is None and not refers_to_previous and bool(CONTEXT_PATIENT_RE.search(text))
        has_patient_ref = bool(patient_id or refers_to_self or refers_to_context or refers_to_previous)

        intents = [i for i, rx in DM_RULES if rx.search(text)] + [i for i, rx in INTENT_RULES if rx.search(text)]
        if GLUCOSE_RE.search(text):
            glucose = [i for i, rx in GLUCOSE_RULES if rx.search(text)]
            # A bare mention ("show me DM2041's glucose") means the current reading and trend.
            intents = (glucose or ([Intent.GLUCOSE_TREND] if not intents else [])) + intents
        for broad, narrower in SUBSUMED_BY.items():
            if broad in intents:
                intents = [i for i in intents if i not in narrower]
        if (ED_MENTION_RE.search(text) and Intent.ED_STATUS not in intents
                and not any(i in PATIENT_SCOPED_INTENTS for i in intents)):
            intents.append(Intent.ED_STATUS)
        if has_patient_ref and PATIENT_WAIT_RE.search(text):
            intents = [Intent.PATIENT_ED_STATUS] + [i for i in intents if i not in SUBSUMED_BY_PATIENT_ED]

        ward_m = WARD_RE.search(text)
        ward = WARDS[ward_m.group(1).lower()] if ward_m else None
        if ward and Intent.WARD_OCCUPANCY not in intents and Intent.BED_AVAILABILITY not in intents and not has_patient_ref:
            intents.append(Intent.BED_AVAILABILITY)
        if Intent.WARD_OCCUPANCY in intents and Intent.BED_AVAILABILITY in intents and not ward:
            intents.remove(Intent.BED_AVAILABILITY)

        if GENERAL_Q_RE.search(text) and not has_patient_ref and not any(
            i in intents for i in (Intent.ED_STATUS, Intent.BED_AVAILABILITY, Intent.WARD_OCCUPANCY, Intent.PREDICTION)
        ):
            intents = [Intent.GENERAL_MEDICAL_INFORMATION]

        hours, days = _window(text)
        return ParsedRequest(
            intents=intents or [Intent.UNKNOWN],
            patient_id=patient_id,
            refers_to_self=refers_to_self and patient_id is None,
            refers_to_context_patient=refers_to_context,
            refers_to_previous_patient=refers_to_previous,
            ward=ward,
            hours=hours,
            days=days,
            as_of=_as_of(text),
            parser="rules",
        )


INTERPRETER_INSTRUCTIONS = """You classify hospital assistant requests into structured intents.
Return only the schema. Use UNKNOWN when nothing fits; never invent a patient ID.
Intents:
- PATIENT_INFORMATION: demographics, chronic conditions of a patient
- PATIENT_HISTORY: previous encounters / ED attendances / admissions of a patient
- VITAL_SIGNS: observations (HR, RR, SpO2, temperature, BP) of a patient
- PATIENT_ED_STATUS: a specific patient's current ED stay or wait
- LAB_RESULTS, MEDICATION_QUERY, RISK_ASSESSMENT: patient labs, medicines, risk scores
- ED_STATUS: how busy the Emergency Department is, waits, trolleys, ambulances
- BED_AVAILABILITY: inpatient beds available; WARD_OCCUPANCY: which wards are full
- PREDICTION: forecasts of future demand
- GENERAL_MEDICAL_INFORMATION: general health questions not about a specific patient's record
Diabetes cohort (CGM, patient IDs like DM2045):
- GLUCOSE_TREND: current glucose and its direction; GLUCOSE_FORECAST: glucose in 30/60 min
- GLYCEMIC_RISK: risk of hypo- or hyperglycaemia; FORECAST_ACCURACY: how accurate the forecast is
- MEAL_RESPONSE: what happened after meals; GLUCOSE_PATTERN_HISTORY: similar past glucose patterns
- MEMORY_RECALL: what has already been retrieved in this conversation
Set refers_to_self when the user asks about themselves; refers_to_context_patient for 'this patient';
refers_to_previous_patient for 'the previous / other patient'."""


class LLMInterpreter:
    """Structured-output classifier (MAF Agent with response_format). Used only when rules fail."""

    def __init__(self, client):
        from agent_framework import Agent

        self._agent = Agent(client=client, name="request_interpreter", instructions=INTERPRETER_INSTRUCTIONS)

    async def parse(self, text: str) -> ParsedRequest:
        response = await self._agent.run(text, options={"response_format": LLMParsedRequest, "temperature": 0})
        out: LLMParsedRequest = response.value
        return ParsedRequest(
            intents=out.intents or [Intent.UNKNOWN],
            patient_id=normalize_patient_id(out.patient_id),
            refers_to_self=out.refers_to_self,
            refers_to_context_patient=out.refers_to_context_patient,
            refers_to_previous_patient=out.refers_to_previous_patient,
            ward=WARDS.get((out.ward or "").lower()),
            hours=out.hours,
            days=out.days,
            as_of=out.as_of,
            parser="llm",
        )
