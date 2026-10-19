"""Interactive console.

    python -m hcagent.cli --user U101                 # clinician (all patients)
    python -m hcagent.cli --user U102                 # clinician with care-team scope
    python -m hcagent.cli --user OPS1                 # operations
    python -m hcagent.cli --user PAT-P0109            # patient P0109 (currently in the ED)
    python -m hcagent.cli --user U101 --ask "Why is P0002 still waiting in the ED?" --trace

    python -m hcagent.cli --user U201 --session S-xxxxxxxxxx    # resume a stored conversation

Commands: /memory (memory tree), /sessions (stored conversations), /forget (delete my memory),
          /state (full shared state JSON), /trace, /quit
"""

from __future__ import annotations

import argparse
import asyncio

from .schemas import SharedState
from .system import DEMO_USERS, HealthcareAgentSystem, resolve_user


def print_trace(st: SharedState) -> None:
    p = st.parsed
    print(f"  ├ request   {st.request_id}  clock={st.clock}")
    print(f"  ├ intent    {[i.value for i in p.intents]} via {p.parser}  target={st.target_patient_id}")
    for d in st.permissions:
        print(f"  ├ policy    {'ALLOW' if d.allowed else 'DENY '} {d.rule}  {d.tool or 'request'}  {d.reason}")
    print(f"  ├ route     {st.route}: {st.route_reason}")
    for c in st.tool_calls:
        print(f"  ├ tool      [{c.caller}] {c.tool}({c.args}) -> {c.status}{' ' + c.error_code if c.error_code else ''}")
    if st.conversation:
        print(f"  ├ memory    context turns {[t['turn_no'] for t in st.conversation]}")
    for f in st.findings:
        print(f"  ├ finding   {f.category}/{f.severity} {f.code}")
    for s in st.sources:
        print(f"  ├ source    {s.source} @ {s.data_timestamp}  records={s.record_ids[:4]}{'…' if len(s.record_ids) > 4 else ''}")
    g = st.grounding
    if g:
        print(f"  ├ grounding {g.numbers_supported}/{g.numbers_checked} numbers supported"
              + (f"; unsupported {g.unsupported_numbers}" if g.unsupported_numbers else ""))
    for e in st.errors:
        print(f"  ├ error     {e}")
    print(f"  └ answer by {st.answer_source}; llm runs={st.llm_calls}; ms={st.timings_ms}")


async def main() -> None:
    ap = argparse.ArgumentParser(description="Controlled healthcare agent (synthetic data)")
    ap.add_argument("--user", default="U101", help=f"{', '.join(DEMO_USERS)} or PAT-<patient_id>")
    ap.add_argument("--ask", help="ask one question and exit")
    ap.add_argument("--no-llm", action="store_true", help="deterministic answers only")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--session", help="resume a stored conversation by its session ID")
    args = ap.parse_args()

    system = HealthcareAgentSystem(use_llm=not args.no_llm)
    user = resolve_user(args.user)
    session = (system.resume_session(args.session, user) if args.session else None) or system.new_session(user)
    if args.session and session.session_id != args.session:
        print(f"No stored session {args.session} for {user.user_id}; started a new one.")
    trace = args.trace
    print(f"Signed in as {user.display_name or user.user_id} ({user.role.value}). Simulated time: {system.clock}. "
          f"LLM: {'on' if system.llm_enabled else 'off'}. Session {session.session_id}"
          f"{' (resumed, ' + str(session.tree.turn_count) + ' earlier turns)' if session.tree.turn_count else ''}. "
          "Not for clinical use.")

    async def handle(q: str) -> SharedState:
        st = await system.ask(session, q)
        if trace:
            print_trace(st)
        print(f"\n{st.answer}\n")
        return st

    if args.ask:
        await handle(args.ask)
        return

    last: SharedState | None = None
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/trace":
            trace = not trace
            print(f"trace {'on' if trace else 'off'}")
            continue
        if q == "/sessions":
            for r in system.list_sessions(user):
                print(f"  {r['session_id']}  updated {r['updated_at']}  turns {r['turn_count']}  patients {r['patients'] or '-'}")
            continue
        if q == "/forget":
            print(f"deleted {system.forget_user(user)} stored session(s); starting a new conversation")
            session = system.new_session(user)
            continue
        if q == "/memory":
            print(session.tree.render_text())
            continue
        if q == "/state":
            print(last.model_dump_json(indent=2) if last else "No request yet.")
            continue
        last = await handle(q)


if __name__ == "__main__":
    asyncio.run(main())
