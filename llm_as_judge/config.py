from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional local convenience
    def load_dotenv(*args, **kwargs):
        return False


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPERS_DIR = REPO_ROOT / "eval_paper2"
DEFAULT_RESULTS_DIR = REPO_ROOT / "llm_as_judge" / "results"


def load_eval_env() -> None:
    """Load root/backend env files without printing secrets."""
    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(REPO_ROOT / "backend" / ".env", override=False)
    load_dotenv(REPO_ROOT / "llm_as_judge" / ".env", override=False)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class EvalSettings:
    api_key: str
    base_url: str
    question_model: str
    judge_model: str
    rag_model: str
    bare_model: str
    embedding_model: str
    embedding_dimensions: int
    papermem_api_base: str
    papermem_top_k: int
    papermem_context_budget: int
    papermem_evidence_chars_per_unit: int
    papermem_use_source_hint: bool
    request_timeout_seconds: float
    rag_top_k: int
    rag_chunk_chars: int
    rag_chunk_overlap: int
    rag_max_context_chars: int
    paper_sample_chars: int

    @property
    def is_openrouter(self) -> bool:
        return "openrouter.ai" in self.base_url


def load_settings() -> EvalSettings:
    load_eval_env()
    base_url = (
        _env("EVAL_OPENAI_BASE_URL")
        or _env("OPENAI_BASE_URL")
        or _env("LLM_BASE_URL")
        or "https://api.openai.com/v1"
    )
    is_openrouter = "openrouter.ai" in base_url
    default_small_model = "openai/gpt-4o-mini" if is_openrouter else "gpt-4o-mini"
    default_gpt54 = "openai/gpt-5.4" if is_openrouter else "gpt-5.4"

    api_key = (
        _env("EVAL_OPENAI_API_KEY")
        or _env("OPENAI_API_KEY")
        or _env("LLM_API_KEY")
        or ""
    )

    return EvalSettings(
        api_key=api_key,
        base_url=base_url,
        question_model=_env("EVAL_QUESTION_MODEL", default_gpt54) or default_gpt54,
        judge_model=_env("EVAL_JUDGE_MODEL", default_gpt54) or default_gpt54,
        rag_model=_env("EVAL_RAG_MODEL", default_small_model) or default_small_model,
        bare_model=_env("EVAL_BARE_MODEL", default_small_model) or default_small_model,
        embedding_model=_env("EVAL_EMBEDDING_MODEL", _env("EMBEDDING_MODEL", "text-embedding-3-small"))
        or "text-embedding-3-small",
        embedding_dimensions=_env_int("EVAL_EMBEDDING_DIMENSIONS", _env_int("EMBEDDING_DIMENSIONS", 256)),
        papermem_api_base=_env("EVAL_PAPERMEM_API_BASE", "http://127.0.0.1:8000")
        or "http://127.0.0.1:8000",
        papermem_top_k=_env_int("EVAL_PAPERMEM_TOP_K", 16),
        papermem_context_budget=_env_int("EVAL_PAPERMEM_CONTEXT_BUDGET", 14000),
        papermem_evidence_chars_per_unit=_env_int("EVAL_PAPERMEM_EVIDENCE_CHARS_PER_UNIT", 1600),
        papermem_use_source_hint=(_env("EVAL_PAPERMEM_USE_SOURCE_HINT", "true") or "true").lower()
        in {"1", "true", "yes", "on"},
        request_timeout_seconds=_env_float("EVAL_REQUEST_TIMEOUT_SECONDS", 180.0),
        rag_top_k=_env_int("EVAL_RAG_TOP_K", 8),
        rag_chunk_chars=_env_int("EVAL_RAG_CHUNK_CHARS", 1800),
        rag_chunk_overlap=_env_int("EVAL_RAG_CHUNK_OVERLAP", 250),
        rag_max_context_chars=_env_int("EVAL_RAG_MAX_CONTEXT_CHARS", 14000),
        paper_sample_chars=_env_int("EVAL_PAPER_SAMPLE_CHARS", 18000),
    )


def require_api_key(settings: EvalSettings) -> None:
    if not settings.api_key:
        raise RuntimeError(
            "Missing API key. Set EVAL_OPENAI_API_KEY or OPENAI_API_KEY. "
            "If you want to reuse backend/.env, make sure LLM_API_KEY is set there."
        )
