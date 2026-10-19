"""Configuration: paths, simulated clock, and the MAF chat client factory."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = PROJECT_DIR.parent

# Project .env first; fall back to the existing maf-demo credentials (load_dotenv never overrides).
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(WORKSPACE_DIR / "maf-demo" / ".env")

DB_PATH = Path(
    os.getenv(
        "HC_DB_PATH",
        WORKSPACE_DIR / "healthcare_agent_v1_package" / "healthcare_agent_v1.sqlite",
    )
)

# Diabetes cohort (ShanghaiT2DM), built by diabetes/etl.py and diabetes/train_forecaster.py.
DM_DB_PATH = Path(os.getenv("HC_DM_DB_PATH", PROJECT_DIR / "data" / "shanghai_t2dm.sqlite"))
DM_MODEL_PATH = Path(os.getenv("HC_DM_MODEL_PATH", PROJECT_DIR / "data" / "models" / "glucose_forecaster_v1.json"))

# Persistent conversation memory (memory trees). Set HC_MEMORY_DB_PATH="" to keep memory in RAM only.
_mem = os.getenv("HC_MEMORY_DB_PATH", str(PROJECT_DIR / "data" / "memory.sqlite"))
MEMORY_DB_PATH: Path | None = Path(_mem) if _mem else None
MEMORY_RETENTION_DAYS = int(os.getenv("HC_MEMORY_RETENTION_DAYS", "30"))

# The synthetic data is historical (2026-07-01 .. 2026-09-28T23:00). The system treats
# HC_SIM_NOW as "now"; when unset it uses the latest ED status snapshot in the database.
SIM_NOW: str | None = os.getenv("HC_SIM_NOW") or None

# Upper bounds on what the LLM may do inside one request.
LLM_MAX_TOOL_CALLS = int(os.getenv("HC_LLM_MAX_TOOL_CALLS", "6"))
LLM_MAX_ITERATIONS = int(os.getenv("HC_LLM_MAX_ITERATIONS", "4"))


def llm_configured() -> bool:
    return all(
        os.getenv(k)
        for k in (
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_API_VERSION",
        )
    )


def make_chat_client():
    """Azure OpenAI chat client for MAF (same setup as maf-demo/main.py)."""
    from agent_framework.openai import OpenAIChatCompletionClient

    if not llm_configured():
        raise RuntimeError(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_DEPLOYMENT and AZURE_OPENAI_API_VERSION in healthcare_agent/.env"
        )
    return OpenAIChatCompletionClient(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        function_invocation_configuration={
            "max_iterations": LLM_MAX_ITERATIONS,
            "max_function_calls": LLM_MAX_TOOL_CALLS,
        },
    )
