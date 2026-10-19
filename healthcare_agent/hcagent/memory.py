"""Hierarchical conversation memory (memory tree), one tree per session.

    session (user, role)
     ├─ focus_patient_id
     ├─ patients/<patient_id>      only patients the user was granted access to
     │    ├─ facts                 cached tool results with source, data timestamp, origin request
     │    └─ turns                 questions answered about this patient
     ├─ operational                ED / bed facts and turns (no patient data)
     └─ general                    general, denied, clarify and not-found turns

Rules, all deterministic:
  * A patient branch is created only after access was granted; denied requests go to `general`.
  * The LLM sees turns from the target patient's branch only, never another patient's turns.
  * A cached fact is reused only for an identical tool call, under the same simulated clock and
    within its TTL, and only after the PolicyEngine has re-approved the call (see ToolExecutor).
  * The tree is not patient memory: the datastore stays the source of truth.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from .schemas import PATIENT_SCOPED_INTENTS, SharedState, ToolResult, utc_now
from .tools import TOOLS

# Seconds a cached result stays reusable (wall clock), by data source.
TTL_SECONDS = {"patients": 3600, "encounters": 300, "vitals": 300, "ed_status": 60, "beds": 60,
               # diabetes cohort: static per as_of, so reuse is safe for longer
               "dm_recordings": 3600, "dm_model_eval": 3600, "dm_cgm": 600, "dm_forecast_model": 600,
               "dm_medications": 600, "dm_meals": 600, "dm_cgm_history": 600}
MAX_TURNS_PER_BRANCH = 20
MAX_PATIENTS = 25
CONTEXT_TURNS = 3
NOT_GRANTED_ROUTES = ("denied", "clarify", "not_found")


def fact_key(tool: str, args: dict[str, Any]) -> str:
    return f"{tool}:{json.dumps(args, sort_keys=True, default=str)}"


TITLE_MAX = 60
_TITLE_LEAD = re.compile(r"^(please\s+|can you\s+|could you\s+|tell me\s+|show me\s+|give me\s+|what(?:'s| is| are| were| was)\s+)+", re.I)


def default_title(question: str) -> str:
    """Chat-list title from the first question, e.g. "Show me P0027's observations." -> "P0027's observations"."""
    t = " ".join(question.split()).rstrip("?.! ")
    t = _TITLE_LEAD.sub("", t) or t
    t = re.sub(r"^the\s+", "", t, flags=re.I) or t
    if len(t) > TITLE_MAX:
        t = t[:TITLE_MAX].rsplit(" ", 1)[0] + "…"
    return (t[:1].upper() + t[1:]) or "New chat"


class MemoryFact(BaseModel):
    key: str
    tool: str
    args: dict[str, Any]
    result: ToolResult
    request_id: str
    clock: str
    stored_at: float = Field(default_factory=time.time)

    def age_s(self) -> float:
        return time.time() - self.stored_at

    def fresh(self, clock: str) -> bool:
        return clock == self.clock and self.age_s() <= TTL_SECONDS.get(self.result.source, 0)


class TurnNode(BaseModel):
    turn_no: int
    request_id: str
    asked_at: str = Field(default_factory=utc_now)
    question: str
    answer: str
    intents: list[str]
    route: str | None
    patient_id: str | None
    tools: list[str] = Field(default_factory=list)
    cached_tools: list[str] = Field(default_factory=list)
    grounding: float | None = None


class Branch(BaseModel):
    facts: dict[str, MemoryFact] = Field(default_factory=dict)
    turns: list[TurnNode] = Field(default_factory=list)

    def add_turn(self, turn: TurnNode) -> None:
        self.turns.append(turn)
        del self.turns[:-MAX_TURNS_PER_BRANCH]


class PatientBranch(Branch):
    patient_id: str
    first_seen_turn: int
    last_seen_turn: int


class MemoryTree(BaseModel):
    session_id: str = Field(default_factory=lambda: "S-" + uuid.uuid4().hex[:10])
    user_id: str
    role: str
    created_at: str = Field(default_factory=utc_now)
    title: str | None = None           # chat-list title: set from the first question, renamable
    focus_patient_id: str | None = None
    turn_count: int = 0
    patients: dict[str, PatientBranch] = Field(default_factory=dict)
    operational: Branch = Field(default_factory=Branch)
    general: Branch = Field(default_factory=Branch)

    # ------------------------------------------------------------------ reads

    def previous_patient(self) -> str | None:
        """Most recently discussed patient other than the current focus."""
        others = [p for p in self.patients.values() if p.patient_id != self.focus_patient_id]
        return max(others, key=lambda p: p.last_seen_turn).patient_id if others else None

    def context_for(self, target_patient_id: str | None) -> list[dict[str, Any]]:
        """Turns the LLM may see. Patient context is isolated to the target patient's branch."""
        if target_patient_id:
            branch = self.patients.get(target_patient_id)
            turns = branch.turns if branch else []
        else:
            turns = sorted(self.operational.turns + self.general.turns, key=lambda t: t.turn_no)
        return [t.model_dump(include={"turn_no", "question", "answer", "patient_id"})
                for t in turns[-CONTEXT_TURNS:]]

    def lookup(self, tool: str, args: dict[str, Any], clock: str) -> MemoryFact | None:
        branch = self._branch_for(tool, args, create=False)
        fact = branch.facts.get(fact_key(tool, args)) if branch else None
        return fact if fact and fact.fresh(clock) else None

    # ------------------------------------------------------------------ writes

    def _branch_for(self, tool: str, args: dict[str, Any], create: bool) -> Branch | None:
        spec = TOOLS.get(tool)
        if spec is None:
            return None
        if spec.scope == "operational":
            return self.operational
        pid = args.get("patient_id")
        if pid not in self.patients and create:
            self.patients[pid] = PatientBranch(patient_id=pid, first_seen_turn=self.turn_count,
                                               last_seen_turn=self.turn_count)
        return self.patients.get(pid)

    def record(self, state: SharedState) -> TurnNode:
        """Write one completed request into the tree."""
        self.turn_count += 1
        if not self.title:
            self.title = default_title(state.message)
        granted = state.route not in NOT_GRANTED_ROUTES
        turn = TurnNode(
            turn_no=self.turn_count,
            request_id=state.request_id,
            question=state.message,
            answer=(state.answer or "")[:800],
            intents=[i.value for i in state.parsed.intents] if state.parsed else [],
            route=state.route,
            patient_id=state.target_patient_id,
            tools=[c.tool for c in state.tool_calls if c.status in ("ok", "error")],
            cached_tools=[c.tool for c in state.tool_calls if c.status == "cached"],
            grounding=state.grounding.score if state.grounding else None,
        )

        # Facts: only results actually executed (allowed by policy) in this request.
        for r in state.tool_results:
            if r.ok and not r.from_memory and TOOLS.get(r.tool) and TOOLS[r.tool].domain != "memory":
                branch = self._branch_for(r.tool, r.args, create=True)
                if branch is not None:
                    key = fact_key(r.tool, r.args)
                    branch.facts[key] = MemoryFact(key=key, tool=r.tool, args=r.args, result=r,
                                                   request_id=state.request_id, clock=state.clock)

        pid = state.target_patient_id
        patient_turn = granted and pid and (
            pid in self.patients or any(i in PATIENT_SCOPED_INTENTS for i in (state.parsed.intents if state.parsed else []))
        )
        if patient_turn:
            branch = self.patients.setdefault(pid, PatientBranch(
                patient_id=pid, first_seen_turn=self.turn_count, last_seen_turn=self.turn_count))
            branch.last_seen_turn = self.turn_count
            branch.add_turn(turn)
            self.focus_patient_id = pid
        elif granted and any(TOOLS[c.tool].scope == "operational" for c in state.tool_calls if c.tool in TOOLS):
            self.operational.add_turn(turn)
        else:
            self.general.add_turn(turn)

        if len(self.patients) > MAX_PATIENTS:
            oldest = min(self.patients.values(), key=lambda p: p.last_seen_turn)
            self.patients.pop(oldest.patient_id)
        return turn

    # ------------------------------------------------------------------ views

    def view(self) -> dict[str, Any]:
        def facts(b: Branch) -> list[dict[str, Any]]:
            return [{
                "tool": f.tool, "args": f.args, "source": f.result.source,
                "data_timestamp": f.result.data_timestamp, "request_id": f.request_id,
                "age_s": round(f.age_s()), "ttl_s": TTL_SECONDS.get(f.result.source, 0), "fresh": f.fresh(f.clock),
            } for f in b.facts.values()]

        def turns(b: Branch) -> list[dict[str, Any]]:
            return [t.model_dump(include={"turn_no", "question", "route", "patient_id", "tools", "cached_tools"})
                    for t in b.turns]

        patients = sorted(self.patients.values(), key=lambda p: -p.last_seen_turn)
        return {
            "session_id": self.session_id, "user_id": self.user_id, "role": self.role, "title": self.title,
            "turn_count": self.turn_count, "focus_patient_id": self.focus_patient_id,
            "previous_patient_id": self.previous_patient(),
            "patients": [{"patient_id": p.patient_id, "first_seen_turn": p.first_seen_turn,
                          "last_seen_turn": p.last_seen_turn, "facts": facts(p), "turns": turns(p)} for p in patients],
            "operational": {"facts": facts(self.operational), "turns": turns(self.operational)},
            "general": {"turns": turns(self.general)},
        }

    def render_text(self) -> str:
        v = self.view()
        lines = [f"session {v['session_id']}  user={v['user_id']} ({v['role']})  turns={v['turn_count']}",
                 f"├─ focus: {v['focus_patient_id'] or '—'}   previous: {v['previous_patient_id'] or '—'}"]
        for p in v["patients"]:
            lines.append(f"├─ patient {p['patient_id']}  (turns {p['first_seen_turn']}..{p['last_seen_turn']})")
            for f in p["facts"]:
                lines.append(f"│   ├─ fact {f['tool']} [{f['source']} @ {f['data_timestamp'] or 'n/a'}] "
                             f"from {f['request_id']} age {f['age_s']}s/{f['ttl_s']}s")
            for t in p["turns"]:
                cached = f" cached={t['cached_tools']}" if t["cached_tools"] else ""
                lines.append(f"│   └─ turn {t['turn_no']}: {t['question']}  [{t['route']}]{cached}")
        for name in ("operational", "general"):
            b = v[name]
            lines.append(f"├─ {name}")
            for f in b.get("facts", []):
                lines.append(f"│   ├─ fact {f['tool']} [{f['source']} @ {f['data_timestamp']}] age {f['age_s']}s/{f['ttl_s']}s")
            for t in b["turns"]:
                lines.append(f"│   └─ turn {t['turn_no']}: {t['question']}  [{t['route']}]")
        return "\n".join(lines)
