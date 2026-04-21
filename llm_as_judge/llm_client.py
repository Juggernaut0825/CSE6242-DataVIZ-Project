from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import APIError, AsyncOpenAI

from .config import EvalSettings, require_api_key


class EvalLLMClient:
    def __init__(self, settings: EvalSettings) -> None:
        require_api_key(settings)
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.request_timeout_seconds,
        )

    async def chat_text(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_completion_tokens: int = 1200,
    ) -> tuple[str, dict[str, int]]:
        response = await self._chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        content = response.choices[0].message.content or ""
        return content.strip(), self._usage(response)

    async def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_completion_tokens: int = 2000,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        response = await self._chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            parsed = json.loads(content[start : end + 1])
        return parsed, self._usage(response)

    async def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = [text.strip() or "(empty)" for text in texts[start : start + batch_size]]
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "model": self.settings.embedding_model,
            "input": batch,
        }
        if self.settings.embedding_dimensions > 0 and "text-embedding-3" in self.settings.embedding_model:
            kwargs["dimensions"] = self.settings.embedding_dimensions
        try:
            response = await self.client.embeddings.create(**kwargs)
        except Exception:
            kwargs.pop("dimensions", None)
            response = await self.client.embeddings.create(**kwargs)
        return [list(item.embedding) for item in response.data]

    async def _chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_completion_tokens: int,
        response_format: dict[str, str] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        try:
            return await self.client.chat.completions.create(**kwargs)
        except (TypeError, APIError):
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            return await self.client.chat.completions.create(**kwargs)

    @staticmethod
    def _usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }


async def gather_with_concurrency(limit: int, *coroutines):
    semaphore = asyncio.Semaphore(limit)

    async def run(coro):
        async with semaphore:
            return await coro

    tasks = [asyncio.create_task(run(coro)) for coro in coroutines]
    return await asyncio.gather(*tasks)
