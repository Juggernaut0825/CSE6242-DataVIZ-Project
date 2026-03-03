from typing import Any, Dict

import httpx

from app.config import settings


async def post_extract(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{settings.memory_api_base}/extract"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


async def post_agentic_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{settings.memory_api_base}/agenticSearch"
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
