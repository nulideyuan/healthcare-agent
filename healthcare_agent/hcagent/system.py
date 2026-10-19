"""Public entry point: build the services once, then ask questions per session."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from . import config
from .agent_layer import LLMResponder, TitleWriter
from .data import ClinicalRepository
from .dm_data import DiabetesRepository
from .dm_model import GlucoseForecaster
from .dm_tools import DiabetesServices
from .executor import ToolExecutor
from .interpreter import LLMInterpreter, RuleInterpreter
from .memory import MemoryTree
from .memory_store import MemoryStore
from .policy import PolicyEngine
from .router import DeterministicRouter
from .schemas import Role, SharedState, UserContext
from .validator import ResultValidator
from .workflow import Services, build_workflow

# Demo identities. Authentication is out of scope; identity is fixed by login, never by the message.
DEMO_USERS: dict[str, UserContext] = {
    "U101": UserContext(user_id="U101", role=Role.CLINICIAN, display_name="Dr Lee (ED consultant)"),
    "U102": UserContext(user_id="U102", role=Role.CLINICIAN, display_name="Dr Shah (care team)",
                        authorized_patients=["P0109", "P0042", "P0001"]),
    "OPS1": UserContext(user_id="OPS1", role=Role.OPERATIONS, display_name="Bed manager"),
    # Diabetes clinic: care-team scope = the diabetes cohort (filled in when the cohort is loaded).
    "U201": UserContext(user_id="U201", role=Role.CLINICIAN, display_name="Dr Wang (diabetes clinic)",
                        authorized_patients=[]),
}


def patient_user(patient_id: str) -> UserContext:
    return UserContext(user_id=f"PAT-{patient_id}", role=Role.PATIENT, patient_id=patient_id,
                       display_name=f"Patient {patient_id}")


def resolve_user(user_id: str) -> UserContext:
    if user_id in DEMO_USERS:
        return DEMO_USERS[user_id]
    if user_id.upper().startswith("PAT-"):
        return patient_user(user_id[4:].upper())
    raise KeyError(f"Unknown user {user_id}. Known: {', '.join(DEMO_USERS)} or PAT-<patient_id>")


def load_diabetes() -> DiabetesServices | None:
    """The diabetes cohort is optional: the hospital system runs without it."""
    if not (config.DM_DB_PATH.exists() and config.DM_MODEL_PATH.exists()):
        return None
    return DiabetesServices(DiabetesRepository(config.DM_DB_PATH), GlucoseForecaster(config.DM_MODEL_PATH))


@dataclass
class Session:
    """One conversation. Its memory tree is registered in the system's memory store."""

    user: UserContext
    tree: MemoryTree

    @property
    def session_id(self) -> str:
        return self.tree.session_id


class HealthcareAgentSystem:
    def __init__(self, use_llm: bool = True, persist_memory: bool = True):
        repo = ClinicalRepository(config.DB_PATH)
        policy = PolicyEngine()
        self.memory: dict[str, MemoryTree] = {}
        self.store = (MemoryStore(config.MEMORY_DB_PATH, config.MEMORY_RETENTION_DAYS)
                      if persist_memory and config.MEMORY_DB_PATH else None)
        self.diabetes = load_diabetes()
        if self.diabetes is not None:
            DEMO_USERS["U201"].authorized_patients = self.diabetes.repo.patient_ids()
        executor = ToolExecutor(repo, policy, self.memory, self.diabetes, self.store)
        client = config.make_chat_client() if use_llm and config.llm_configured() else None
        self.llm_enabled = client is not None
        self.services = Services(
            repo=repo,
            policy=policy,
            router=DeterministicRouter(policy),
            executor=executor,
            validator=ResultValidator(repo, self.diabetes.forecaster.a["risk_rules_oof"] if self.diabetes else None),
            rules=RuleInterpreter(),
            llm_interpreter=LLMInterpreter(client) if client else None,
            responder=LLMResponder(client, policy, executor) if client else None,
            memory=self.memory,
            diabetes=self.diabetes,
        )
        self.client = client
        self.titler = TitleWriter(client) if client else None
        self._title_tasks: set[asyncio.Task] = set()
        self.clock = config.SIM_NOW or repo.latest_snapshot_time()
        self.workflow = build_workflow(self.services)

    def new_session(self, user: UserContext) -> Session:
        tree = MemoryTree(user_id=user.user_id, role=user.role.value)
        self.memory[tree.session_id] = tree
        return Session(user=user, tree=tree)

    def resume_session(self, session_id: str, user: UserContext) -> Session | None:
        """Reload a stored session for its owner. Patient branches are re-authorised on load."""
        if self.store is None:
            return None
        policy = self.services.policy
        tree = self.store.load(session_id, user.user_id, lambda pid: policy.check_patient_access(user, pid).allowed)
        if tree is None:
            return None
        self.memory[tree.session_id] = tree
        return Session(user=user, tree=tree)

    def list_sessions(self, user: UserContext) -> list[dict]:
        return self.store.list_sessions(user.user_id) if self.store else []

    def rename_session(self, session_id: str, user: UserContext, title: str) -> bool:
        """Owner-only. Updates the live tree too, so the next write-through keeps the new title."""
        title = " ".join(title.split())[:80]
        if not title:
            return False
        live = self.memory.get(session_id)
        if live is not None and live.user_id == user.user_id:
            live.title = title
        if self.store is not None:
            return self.store.rename(session_id, user.user_id, title)
        return live is not None and live.user_id == user.user_id

    def delete_session(self, session_id: str, user: UserContext) -> bool:
        live = self.memory.get(session_id)
        if live is not None and live.user_id == user.user_id:
            self.memory.pop(session_id)
        return bool(self.store.delete_session(session_id, user.user_id)) if self.store else live is not None

    def forget_user(self, user: UserContext) -> int:
        """Delete all stored and in-memory conversation memory for this user."""
        for sid in [sid for sid, t in self.memory.items() if t.user_id == user.user_id]:
            self.memory.pop(sid)
        return self.store.forget_user(user.user_id) if self.store else 0

    def close_session(self, session: Session) -> None:
        """Release the in-memory tree. A stored session stays resumable until retention expires."""
        self.memory.pop(session.session_id, None)

    async def ask(self, session: Session, message: str) -> SharedState:
        state = SharedState(clock=self.clock, user=session.user, message=message, session_id=session.session_id)
        result = await self.workflow.run(state)
        outputs = result.get_outputs()
        final: SharedState = outputs[-1] if outputs else state
        first_turn = session.tree.turn_count == 0
        session.tree.record(final)
        if self.store is not None:
            self.store.save(session.tree)
        if first_turn and self.titler is not None:
            task = asyncio.create_task(self._llm_title(session, message))
            self._title_tasks.add(task)
            task.add_done_callback(self._title_tasks.discard)
        return final

    async def _llm_title(self, session: Session, question: str) -> None:
        """Replace the first-question title with an LLM topic title, unless the user renamed it meanwhile."""
        fallback = session.tree.title
        try:
            title = await self.titler.title(question)
        except Exception as e:   # the chat keeps its first-question title
            logging.getLogger(__name__).warning("chat title failed: %s", e)
            return
        if title and session.tree.title == fallback:
            session.tree.title = title
            if self.store is not None:
                self.store.rename(session.session_id, session.user.user_id, title)
