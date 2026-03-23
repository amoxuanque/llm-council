"""SiliconFlow API client for making LLM requests."""

import asyncio
import httpx
from typing import List, Dict, Any, Optional
from config import SILICONFLOW_API_KEY, SILICONFLOW_API_URL

# Shared client with connection pooling for all requests
_client: Optional[httpx.AsyncClient] = None


def get_client() -> httpx.AsyncClient:
    """Return the shared httpx client, creating it if needed."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            timeout=120.0,
        )
    return _client


async def close_client():
    """Close the shared client (call on app shutdown)."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via SiliconFlow API.

    Args:
        model: Model identifier (e.g., "deepseek-ai/DeepSeek-R1")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_content', or None if failed.
        For reasoning models (e.g. DeepSeek-R1), the thinking process is in 'reasoning_content'.
    """
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    try:
        client = get_client()
        response = await client.post(
            SILICONFLOW_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        data = response.json()
        message = data['choices'][0]['message']

        result = {
            'content': message.get('content') or '',
        }

        # Capture reasoning content from thinking models (DeepSeek-R1, etc.)
        # SiliconFlow returns this in 'reasoning_content' field
        reasoning = message.get('reasoning_content') or message.get('reasoning_details')
        if reasoning:
            result['reasoning_content'] = reasoning

        return result

    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    tasks = [query_model(model, messages) for model in models]
    responses = await asyncio.gather(*tasks)
    return {model: response for model, response in zip(models, responses)}
