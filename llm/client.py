# -*- coding: utf-8 -*-
"""Minimal OpenAI-compatible client used by local analyst agents."""
import json
import logging
from typing import Any, Dict, Optional

import requests

from config import get_settings

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when LLM access is disabled or misconfigured."""


class OpenAICompatibleClient:
    """Small JSON-oriented client for OpenAI-compatible chat endpoints."""

    def __init__(self):
        self.settings = get_settings()
        self.cfg = self.settings.llm

    def is_available(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.api_key and self.cfg.base_url and self.cfg.model)

    def _should_enable_thinking(self) -> bool:
        if self.cfg.thinking_enabled:
            return True
        base_url = (self.cfg.base_url or "").lower()
        model = (self.cfg.model or "").lower()
        return "bigmodel.cn" in base_url and model.startswith("glm")

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise LLMUnavailableError("LLM is not configured")

        payload = {
            "model": self.cfg.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt[: self.cfg.max_prompt_chars]},
                {"role": "user", "content": user_prompt[: self.cfg.max_prompt_chars]},
            ],
        }
        if self._should_enable_thinking():
            payload["thinking"] = {"type": self.cfg.thinking_type}

        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        with requests.Session() as session:
            # Optional hard switch to ignore HTTP(S)_PROXY from environment.
            request_kwargs = {}
            if self.cfg.disable_proxy:
                session.trust_env = False
                session.proxies.clear()
                request_kwargs["proxies"] = {"http": None, "https": None}

            response = session.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.cfg.timeout_seconds,
                **request_kwargs,
            )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        return json.loads(self._extract_json_string(content))

    @staticmethod
    def _extract_json_string(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.startswith("```")]
            text = "\n".join(lines).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model response")
        return text[start : end + 1]
