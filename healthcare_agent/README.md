# Controlled Healthcare Agent (MAF), v1 MVP

Implements the *Healthcare Context-Aware Agent v0.1* requirements on the synthetic dataset in
`../healthcare_agent_v1_package`. Built on **Microsoft Agent Framework 1.20** (Workflow, Agent,
FunctionTool, FunctionMiddleware). Routing, permissions, shared state, validation and clinical
thresholds are plain Python. The LLM only (a) classifies requests that the rules cannot, (b) calls the
tools it is handed, and (c) explains the evidence.

> Synthetic data for research only. Not for patient care.

## Quick start

```bash
cd healthcare_agent
PY=../maf-demo/.venv/bin/python      # existing venv with agent-framework installed
                                      # (or: python -m venv .venv && pip install -r requirements.txt)
# credentials: copy .env.example to .env, or leave it absent to reuse ../maf-demo/.env

$PY -m hcagent.server                             # web UI at http://127.0.0.1:8010 (--no-llm optional)

$PY -m hcagent.cli --user PAT-P0109 --trace       # patient currently waiting in the ED
$PY -m hcagent.cli --user U101 --trace            # clinician (all patients)
$PY -m hcagent.cli --user U102                    # clinician, care team = P0109, P0042, P0001
$PY -m hcagent.cli --user OPS1                    # operations
$PY -m hcagent.cli --user U101 --ask "Why is P0002 still waiting in the ED?" --trace
$PY -m hcagent.cli --user U101 --no-llm           # fully deterministic (no LLM)

$PY -m hcagent.cli --user U201 --trace            # diabetes clinic (ShanghaiT2DM cohort, see below)
$PY -m hcagent.cli --user PAT-DM2011              # diabetes patient whose glucose is falling

$PY -m unittest discover -s tests -t .            # 45 offline tests
$PY -m eval.run_eval --offline                    # proposed system, no LLM
$PY -m eval.run_eval --systems proposed,baseline_a,baseline_b --repeats 3
$PY -m eval.run_eval --cases eval/cases_diabetes.jsonl --systems proposed,baseline_a,baseline_b
```

**Simulated time.** The data ends at `2026-09-28T23:00`, which the system treats as "now"
(override with `HC_SIM_NOW`). All queries are time-consistent: records after "now" are invisible,
and the disposition and admission times of patients still in the ED are masked. At that time
P0109 (arrived 16:01, triage 3) and P0002 (arrived 22:04, triage 2) are in the ED.

## Architecture

```
interpret ─► policy ─┬─(denied / clarify / not_found)──────────────────────────► respond
                     └─► route ─┬─(unsupported)─────────────────────────────────► respond
                                ├─(deterministic)─► tools ─► validate ─► llm ─► guard
                                └─(general_llm / llm_fallback)──────────► llm ─► guard
```

This is a MAF `WorkflowBuilder` graph with switch-case edges ([workflow.py](hcagent/workflow.py)).
One `SharedState` object passes through every stage and is the single source of truth.

| Spec section | Component | File |
|---|---|---|
| §6 Request interpreter | Regex rules first; LLM structured output (`response_format`) only if the rules give `UNKNOWN` | [interpreter.py](hcagent/interpreter.py) |
| §7 Deterministic router | Intent → tool plan; second-stage calls depend on the encounter. The LLM router is a fallback | [router.py](hcagent/router.py) |
| §8 Tools | 5 tools, each with inputs (Pydantic), output, scope, allowed roles, errors and data source | [tools.py](hcagent/tools.py) |
| §9 Shared state | `SharedState`: user, parsed request, plan, patient, clinical/hospital context, findings, tool_calls, sources, permissions, grounding | [schemas.py](hcagent/schemas.py) |
| §10 Memory | Conversation memory: a per-session **memory tree** (see below). Patient context: the datastore only. Working memory: the shared state | [memory.py](hcagent/memory.py) |
| §11 Permissions | Rules R0–R4, checked before routing and again on every tool execution | [policy.py](hcagent/policy.py), [executor.py](hcagent/executor.py) |
| §12 Provenance | Every result carries source table, record IDs and data timestamp | [executor.py](hcagent/executor.py) |
| §14/§16 Clinical thresholds | Partial NEWS2 (SpO2 scale 2 for COPD), crowding levels, ward pressure, wait-attribution rules | [rules.py](hcagent/rules.py), [validator.py](hcagent/validator.py) |
| §17 Hallucination control | Evidence-first prompt + post-answer guard (numbers and dates must exist in evidence; foreign patient IDs blocked) | [agent_layer.py](hcagent/agent_layer.py), [grounding.py](hcagent/grounding.py) |
| §2 Baselines A/B | Pure LLM; LLM + all tools without router, policy or validation | [baselines.py](hcagent/baselines.py) |
| §18 Evaluation | Tool selection, arguments, permission safety, missing data, grounding, fact recall, consistency | [eval/run_eval.py](eval/run_eval.py) |

### Where the LLM is constrained, and how

| Concern | Enforced by |
|---|---|
| Who the user is | `UserContext` from login. The patient ID for "my/I" comes from the login, never from the text or the LLM |
| Which patient | The rule parser extracts and normalises IDs (`P042` → `P0042`). "This patient" resolves from conversation memory, and only if access was granted earlier |
| Whether access is allowed | `PolicyEngine`, before routing. Patient existence is checked only *after* authorisation, so it cannot be probed |
| Which tools run | The router's plan. The LLM is shown only the tools on that plan (fallback: tools its role may use) |
| LLM tool calls | `PolicyGuardMiddleware` (MAF `FunctionMiddleware`) checks the allowlist and the policy, fail-closed. The executor checks the policy again. State and identity reach tools through `function_invocation_kwargs`, which the LLM cannot see or set |
| Clinical thresholds / risk bands | `rules.py`; the LLM is told to use findings, not compute its own |
| Missing data | Labs, medications, predictions and risk models are recognised intents with no v1 source → fixed "not available" answer, no LLM |
| Leaks in the answer | Guard blocks answers that show a patient or operations user an unauthorised patient ID |

### Memory tree

Each session has one tree ([memory.py](hcagent/memory.py)). The web UI shows it in the "Memory tree" tab; in the CLI, type `/memory`.

```
session (user, role)
 ├─ focus_patient_id / previous_patient
 ├─ patients/<id>   facts (cached tool results + provenance) and turns. Created only after access was granted
 ├─ operational     ED / bed facts and turns
 └─ general         general, denied, clarify, not-found turns
```

* **Target resolution.** "This patient" resolves to the focus patient; "the previous patient" to the most recently discussed other patient. The policy engine still re-checks access.
* **Context isolation.** The LLM sees only the last 3 turns from the *target patient's* branch, never another patient's turns.
* **Fact reuse.** An identical tool call is served from memory only after the policy approves it again, under the same simulated clock and within a TTL (profile 1 h, encounters/vitals 5 min, ED/beds 1 min). The call is logged as `cached`, and the result keeps its source and originating request ID.
* **Persistence** ([memory_store.py](hcagent/memory_store.py), `data/memory.sqlite`):
  * Every turn writes the session's tree to disk. A conversation survives page refreshes and server
    restarts, and the web UI resumes the user's last conversation. The CLI resumes with
    `--session S-…` and lists stored conversations with `/sessions`.
  * "New conversation" starts an empty tree, but the old one stays stored.
  * Only the owner can resume a session.
  * On resume, every patient branch is re-authorised against the current policy. Branches the user
    may no longer access are dropped.
  * Cached results keep their TTL, so stale data is not reused after a restart.
* **Cross-session recall.** `recall_session_memory` also returns the *same user's* questions about the
  patient from earlier sessions, never another user's.
* **Retention and deletion.** Sessions untouched for `HC_MEMORY_RETENTION_DAYS` (default 30) are purged at
  startup. "Forget my history" (UI), `/forget` (CLI) or `POST /api/forget` deletes everything for a user.
  `HC_MEMORY_DB_PATH=""` keeps memory in RAM only.
* Tests and evaluations never touch the real store: they use a temporary database, or memory off.

### Tools (v1)

| Tool | Scope | Roles | Source |
|---|---|---|---|
| `get_patient_profile(patient_id)` | patient | patient (self), clinician | patients |
| `get_patient_encounters(patient_id, limit, days)` | patient | patient (self), clinician | encounters |
| `get_recent_vitals(patient_id, hours, encounter_id, limit)` | patient | patient (self), clinician | vitals |
| `get_ed_status(timestamp, trend_hours)` | operational | all | ed_status |
| `get_bed_capacity(ward, timestamp)` | operational | all | beds |

## Design notes and limitations

* **Partial NEWS2.** Consciousness level and oxygen therapy are not in the data, so scores are
  labelled partial. Scale 2 assumes room air.
* **The ED is always busy in this dataset** (median occupancy 102%). The validator therefore also
  reports a 30-day percentile, so "97%" can read as *typical* rather than exceptional.
* **Wait attribution** is a rule (`WAIT_DRIVERS_*`). It says "consistent with" and does not claim
  causation, and abnormal observations take priority over operational explanations.
* **A request mixing allowed and forbidden parts is denied as a whole**, e.g. an operations user
  asking for ED status *and* named patients' diagnoses. This is safer, but less helpful than a
  partial answer.
* **The grounding check is lexical.** It catches invented numbers and dates, not wrong claims made
  in words.
* **The eval cases were written alongside the rule interpreter**, so the proposed system's intent
  and tool scores are optimistic. Add held-out paraphrases before reporting results.
* The general-medical-information route uses the LLM with no data access, and only that route.

## Diabetes cohort: Glucose Prediction Agent + Patient Context Agent

Second data domain: the public **ShanghaiT2DM** CGM dataset (100 patients, 109 recordings of
2.6–13.9 days, 15-min CGM, meals, insulin, oral agents, labs). Patient IDs are `DM2000`–`DM2099`.

### Pipeline

```
~/Downloads/diabetes_datasets  (raw Excel)
   │  data_analysis.ipynb        data-quality report (in the dataset folder)
   ▼
diabetes/etl.py  ─────────────►  data/shanghai_t2dm.sqlite     (normalises 5 header layouts, parses doses/grams,
   │                                                             classifies insulin, flags suspected sensor bias)
   ▼
diabetes/train_forecaster.py ─►  data/models/glucose_forecaster_v1.json
                                  (patient-grouped 5-fold CV; one ridge model per fold; 90% intervals; risk-rule metrics)
   ▼
hcagent/dm_*.py  runtime: repository → tools → rules → two MAF specialist agents
```

Rebuild (analysis Python with pandas/xlrd/scikit-learn/LightGBM): `python3 diabetes/etl.py && python3 diabetes/train_forecaster.py`.
The agent runtime needs only numpy, because the deployed model is ridge regression in plain numpy.

### Data quality (from the notebook)

100 patients / 109 recordings / 112,462 CGM readings. Completeness 98.5–100%. Strict 15-min grid with 11 gaps
over 20 min (longest 212 min: the documented device change in 2029_0). Mean TIR 77.8%, TBR 2.4%, TAR 19.9%. 348
hypoglycaemia episodes in 63 recordings. Meals recorded in all recordings (~3/day). Insulin recorded in 65
recordings, oral agents in 92. Labs are often missing (2-h insulin 50%). Two recordings (2075_0, 2077_0) are
flagged as **suspected sensor bias** and excluded from training. The summary field *Hypoglycemia (yes/no)*
disagrees with CGM for 55 recordings, so it is treated as clinical history only.

### Forecasting model (out-of-fold, 98 patients, 109k samples)

| Model | +30 min RMSE | +30 Clarke A | +60 min RMSE | +60 Clarke A |
|---|---|---|---|---|
| No change (persistence) | 17.6 | 90.7% | 29.5 | 75.9% |
| Linear trend | 19.4 | 88.4% | 39.8 | 66.1% |
| **Ridge (deployed)** | **13.2** | **95.2%** | **24.2** | **80.8%** |
| LightGBM | 12.9 | 95.4% | 23.6 | 81.6% |

Each patient is served by the fold model that never saw them, so live forecasts are out-of-sample. The 90%
intervals have 90.0% empirical coverage. Point forecasts miss most lows (60-min hypoglycaemia sensitivity
25%), so **risk is rated from the interval**:

* **High:** the point forecast is below 70 (above 250).
* **Elevated:** the 90% bound crosses the threshold. For hypoglycaemia this catches 93% (30 min) and 84%
  (60 min) of lows, at 22% and 10% precision.

The agents quote these validated numbers.

### Agents and tools

| Agent | Answers | Tools (deterministic) |
|---|---|---|
| **Glucose Prediction Agent** | current trend, +30/+60 forecast, hypo/hyper risk, forecast accuracy | `dm_get_glucose_trend`, `dm_forecast_glucose`, `dm_get_forecast_accuracy` |
| **Patient Context & Analysis Agent** | HbA1c and labs, insulin/medication events, post-meal responses, similar past patterns, what this conversation already retrieved | `dm_get_patient_profile`, `dm_get_medication_events`, `dm_get_meal_responses`, `dm_find_similar_patterns`, `recall_session_memory` |

* **Assignment.** The router assigns agents from the tools' owners and gives each agent its own *task* (the
  intents it must answer). A mixed question ("is there a hypo risk, and what happened after meals?") is
  answered by both agents concurrently, each only about its own part.
* **Evidence.** Each agent sees only its own tool results and findings.
* **No LLM numbers.** The LLM never produces glucose values. Bands, trend arrows, risk levels and the HbA1c
  target come from [dm_rules.py](hcagent/dm_rules.py).
* **Replay clock.** Each patient's "now" is the end of their latest recording. Ask "... at 2020-11-10 07:30" to
  replay an earlier moment; tools never see data after it.
* **Access.** U201 (diabetes clinic) is scoped to the cohort; `PAT-DM2011` is a patient login. Hospital users
  with a care-team scope (U102) and operations cannot read the cohort.

### Limitations

* Event times are aligned to the 15-min CGM grid. Meal sizes are grams, not carbohydrates.
* The cohort is inpatients in one hospital, so results may not transfer to outpatients.
* The ridge model is a strong baseline, not the state of the art. LightGBM was 2.7% better at 60 min but
  needs an extra dependency at runtime.
* Similar-pattern search uses only the patient's own (≤ 4 weeks of) history.

## Next phases (spec §21)

Labs and medications (new tables + tools + intents already defined) → prediction models
(`PREDICTION`/`RISK_ASSESSMENT` intents are already routed to "unsupported") → knowledge graph if it
shows measurable benefit → larger held-out evaluation with repeats.
