import hashlib
import math
import re
from typing import List

from openai import AsyncOpenAI

from app.config import settings


class EmbeddingService:
    def __init__(self) -> None:
        self.dimensions = settings.embedding_dimensions
        self.client = None
        if settings.embedding_api_key:
            self.client = AsyncOpenAI(
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_base_url,
            )

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        cleaned = [text.strip() or "(empty)" for text in texts]
        if self.client:
            try:
                embeddings: List[List[float]] = []
                batch_size = max(1, settings.embedding_batch_size)
                for start in range(0, len(cleaned), batch_size):
                    response = await self.client.embeddings.create(
                        model=settings.embedding_model,
                        input=cleaned[start : start + batch_size],
                        dimensions=self.dimensions,
                    )
                    embeddings.extend(self._normalize_embedding(item.embedding) for item in response.data)
                return embeddings
            except Exception:
                pass
        return [self._fallback_embed(text) for text in cleaned]

    def _fallback_embed(self, text: str) -> List[float]:
        # Deterministic hash embedding keeps local development usable
        # when no remote embedding provider is configured.
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            slot = int(digest[:8], 16) % self.dimensions
            sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
            weight = 1.0 + (int(digest[10:14], 16) % 100) / 500.0
            vector[slot] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        vector = list(embedding[: self.dimensions])
        if len(vector) < self.dimensions:
            vector.extend([0.0] * (self.dimensions - len(vector)))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
