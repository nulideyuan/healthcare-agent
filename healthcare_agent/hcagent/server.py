"""Web frontend: FastAPI backend + single-page UI with a per-answer pipeline trace.

    python -m hcagent.server            # http://127.0.0.1:8010
    python -m hcagent.server --no-llm   # deterministic answers only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .system import DEMO_USERS, HealthcareAgentSystem, Session, resolve_user

WEB_DIR = Path(__file__).resolve().parent / "web"

EXAMPLES = {
    "patient": [
        "Why have I been waiting in the ED so long?",
        "What were my latest observations?",
        "Is the Emergency Department busy today?",
        "What were my latest blood test results?",
        "Show me patient P0005's medications.",
    ],
    "clinician": [
        "Why is P0002 still waiting in the ED?",
        "Show me P0027's recent observations and abnormal laboratory results.",
        "What is the age and sex of patient P0042?",
        "Has this patient attended the ED in the last six months?",
        "Show me the latest vital signs for P0200.",
    ],
    "operations": [
        "How busy is the ED right now?",
        "Which wards are close to capacity?",
        "How has ED occupancy changed over the last 12 hours?",
        "What is the predicted ED trolley demand over the next 24 hours?",
        "Show me the vital signs of patient P0109.",
    ],
}


DIABETES_EXAMPLES = [
    "What is DM2011's current glucose trend?",
    "What will the glucose level be in 30 or 60 minutes?",
    "Is there an increased risk of hypoglycaemia, and what happened after this patient's previous meals?",
    "How accurate is the prediction?",
    "What is this patient's HbA1c?",
    "What medications or insulin events have been recorded?",
    "Has the patient experienced similar glucose patterns before?",
    "What information has already been retrieved in this conversation?",
    "Forecast glucose for DM2041",
]
EXAMPLES_BY_USER = {
    "U201": DIABETES_EXAMPLES,
    "PAT-DM2011": ["What is my glucose doing right now?", "Am I going to go low soon?",
                   "What happened after my previous meals?", "What is my HbA1c?", "Show me DM2041's glucose."],
}


class AskBody(BaseModel):
    user_id: str
    session_id: str | None = None
    message: str


class RenameBody(BaseModel):
    user_id: str
    title: str


def create_app(use_llm: bool = True) -> FastAPI:
    system = HealthcareAgentSystem(use_llm=use_llm)
    sessions: dict[str, Session] = {}
    app = FastAPI(title="Controlled Healthcare Agent")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/info")
    def info():
        users = [{"user_id": u.user_id, "role": u.role.value, "name": u.display_name,
                  "scope": u.authorized_patients} for u in DEMO_USERS.values()]
        users += [{"user_id": f"PAT-{p}", "role": "patient", "name": f"Patient {p}", "scope": [p]}
                  for p in ("P0109", "P0002", "P0027") + (("DM2011", "DM2041") if system.diabetes else ())]
        if system.diabetes is None:
            users = [u for u in users if u["user_id"] != "U201"]
        return {"clock": system.clock, "llm": system.llm_enabled, "users": users, "examples": EXAMPLES,
                "persistent_memory": system.store is not None,
                "examples_by_user": EXAMPLES_BY_USER, "diabetes": system.diabetes is not None}

    def get_session(user, session_id: str | None) -> Session:
        """Live session, else resume a stored one (owner only, re-authorised), else a new one."""
        session = sessions.get(session_id or "")
        if session is not None and session.user.user_id == user.user_id:
            return session
        session = (system.resume_session(session_id, user) if session_id else None) or system.new_session(user)
        sessions[session.session_id] = session
        return session

    def history(session: Session) -> list[dict]:
        t = session.tree
        turns = [x for b in [*t.patients.values(), t.operational, t.general] for x in b.turns]
        return [x.model_dump(include={"turn_no", "asked_at", "question", "answer", "route", "patient_id"})
                for x in sorted(turns, key=lambda x: x.turn_no)]

    def user_or_400(user_id: str):
        try:
            return resolve_user(user_id)
        except KeyError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/ask")
    async def ask(body: AskBody):
        user = user_or_400(body.user_id)
        session = get_session(user, body.session_id)
        state = await system.ask(session, body.message.strip()[:2000])
        return {
            "session_id": session.session_id,
            "title": session.tree.title,
            "answer": state.answer,
            "state": state.model_dump(mode="json"),
            "memory": session.tree.view(),
        }

    @app.get("/api/session/{session_id}")
    def resume(session_id: str, user_id: str):
        user = user_or_400(user_id)
        session = sessions.get(session_id) if session_id in sessions else system.resume_session(session_id, user)
        if session is None or session.user.user_id != user.user_id:
            raise HTTPException(404, "No stored session for this user")
        sessions[session.session_id] = session
        return {"session_id": session.session_id, "created_at": session.tree.created_at, "title": session.tree.title,
                "history": history(session), "memory": session.tree.view()}

    @app.get("/api/sessions")
    def list_sessions(user_id: str):
        return system.list_sessions(user_or_400(user_id))

    @app.patch("/api/session/{session_id}")
    def rename(session_id: str, body: RenameBody):
        if not system.rename_session(session_id, user_or_400(body.user_id), body.title):
            raise HTTPException(404, "No stored session for this user")
        return {"ok": True}

    @app.delete("/api/session/{session_id}")
    def delete(session_id: str, user_id: str):
        user = user_or_400(user_id)
        live = sessions.get(session_id)
        if live is not None and live.user.user_id == user.user_id:
            sessions.pop(session_id)
        if not system.delete_session(session_id, user):
            raise HTTPException(404, "No stored session for this user")
        return {"ok": True}

    @app.post("/api/reset")
    def reset(session_id: str):
        session = sessions.pop(session_id, None)
        if session is not None:
            system.close_session(session)   # stays stored and resumable
        return {"ok": True}

    @app.post("/api/forget")
    def forget(user_id: str):
        user = user_or_400(user_id)
        for sid in [sid for sid, s in sessions.items() if s.user.user_id == user.user_id]:
            sessions.pop(sid)
        return {"deleted_sessions": system.forget_user(user)}

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    uvicorn.run(create_app(use_llm=not args.no_llm), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
