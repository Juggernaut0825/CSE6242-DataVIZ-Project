from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from app.config import settings


class OpenRouterClient:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = settings.llm_model,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        stream: bool = False,
        enable_reasoning: bool = True,
        extra_body: Optional[Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Any:
        body = dict(extra_body or {})
        if enable_reasoning:
            body.setdefault("reasoning", {"enabled": True})
        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            response_format=response_format,
            extra_body=body if body else None,
        )
