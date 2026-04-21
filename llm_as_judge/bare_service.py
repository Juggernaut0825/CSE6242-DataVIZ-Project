from __future__ import annotations

import time
from typing import Any

from .config import EvalSettings
from .llm_client import EvalLLMClient


class BareGPTService:
    """No-RAG GPT-4o mini baseline."""

    def __init__(self, settings: EvalSettings) -> None:
        self.settings = settings
        self.client = EvalLLMClient(settings)

    async def answer(self, question: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        prompt = f"""Answer the paper-specific question below without retrieval context.
If you do not know the answer, state your uncertainty plainly.

Question:
{question["question"]}
"""
        answer, usage = await self.client.chat_text(
            model=self.settings.bare_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a baseline model with no access to the evaluation papers.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_completion_tokens=700,
        )
        elapsed = time.perf_counter() - started
        return {
            "system": "bare_gpt4o_mini",
            "question_id": question["id"],
            "answer": answer,
            "latency_seconds": elapsed,
            "usage": usage,
            "retrieved_context_chars": 0,
            "retrieved_chunks": [],
        }

