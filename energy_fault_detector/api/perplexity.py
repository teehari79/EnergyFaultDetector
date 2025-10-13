"""Lightweight Perplexity API client used for optional root-cause retrieval."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class PerplexityClient:
    """Minimal client for the Perplexity chat completion endpoint."""

    api_url = "https://api.perplexity.ai/chat/completions"

    def __init__(self, api_key: Optional[str] = None, model: str = "sonar-small-chat") -> None:
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.model = model

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str) -> Optional[str]:
        if not self.is_configured():
            logger.info("Perplexity API key not configured; skipping web search.")
            return None

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an assistant that surfaces concise engineering root-cause insights.",
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network side effects
            logger.warning("Perplexity API request failed: %s", exc)
            return None

        try:
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                return None
            message = choices[0].get("message", {})
            return message.get("content")
        except (ValueError, KeyError):  # pragma: no cover - defensive parsing
            logger.warning("Unexpected Perplexity API response format.")
            return None


__all__ = ["PerplexityClient"]
