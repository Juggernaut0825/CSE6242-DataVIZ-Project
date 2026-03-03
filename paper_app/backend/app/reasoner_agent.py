from typing import Any, AsyncGenerator, Dict, List, Optional

from app.openrouter_client import OpenRouterClient


class ReasonerAgent:
    def __init__(
        self,
        system_prompt: str,
        max_iterations: int = 1,
        model_provider: str = "gemini-3-flash",
    ) -> None:
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.model_provider = model_provider
        self.client = OpenRouterClient()

    async def stream(
        self, problem: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not self.system_prompt:
            raise ValueError("system_prompt不能为空，请确保传递了有效的custom_system_prompt")

        conversation: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem},
        ]

        for iteration in range(self.max_iterations):
            yield {"type": "iteration", "round": iteration + 1}
            reasoning_parts: List[str] = []
            content_parts: List[str] = []

            stream_response = await self.client.chat_completion(
                messages=conversation,
                model="google/gemini-3-flash-preview",
                temperature=0.2,
                max_tokens=8192,
                stream=True,
                enable_reasoning=True,
                extra_body={"reasoning": {"enabled": True}},
            )

            async for chunk in stream_response:
                if not chunk or not getattr(chunk, "choices", None):
                    continue

                delta = chunk.choices[0].delta

                reasoning_delta: Optional[str] = None
                if hasattr(delta, "reasoning_content") and getattr(
                    delta, "reasoning_content", None
                ):
                    reasoning_delta = delta.reasoning_content
                elif hasattr(delta, "reasoning") and getattr(delta, "reasoning", None):
                    reasoning_delta = delta.reasoning
                elif hasattr(delta, "reasoning_details") and getattr(
                    delta, "reasoning_details", None
                ):
                    reasoning_delta = str(delta.reasoning_details)

                if reasoning_delta:
                    reasoning_parts.append(str(reasoning_delta))
                    yield {"type": "reasoning", "content": str(reasoning_delta)}

                if hasattr(delta, "content") and delta.content:
                    content_parts.append(delta.content)
                    yield {"type": "content_chunk", "content": delta.content}

                if chunk.choices[0].finish_reason:
                    break

            if content_parts:
                conversation.append(
                    {"role": "assistant", "content": "".join(content_parts)}
                )
