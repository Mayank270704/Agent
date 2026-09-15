from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from openai import APIError, OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        *,
        provider: str,
        model_name: str,
        api_key: str = "",
        base_url: str = "http://localhost:11434",
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

        if self.provider == "openai":
            if not api_key or not api_key.strip():
                raise ValueError("Missing OPENAI_API_KEY configuration for the OpenAI provider.")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None

    def generate(self, messages: list[dict[str, str]]) -> str:
        if self.provider == "ollama":
            return self._generate_with_ollama(messages)

        if self.provider == "openai":
            return self._generate_with_openai(messages)

        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _generate_with_ollama(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }

        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama connection failed at {self.base_url}: {exc}") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON.") from exc

        try:
            return parsed["message"]["content"].strip()
        except KeyError as exc:
            raise RuntimeError("Ollama response did not include a message content field.") from exc

    def _generate_with_openai(self, messages: list[dict[str, str]]) -> str:
        if self.client is None:
            raise RuntimeError("OpenAI client is not configured.")

        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=messages,
            )
            return response.output_text.strip()
        except APIError as exc:
            logger.error("OpenAI API request failed: %s", exc)
            raise RuntimeError(f"OpenAI API request failed: {exc}") from exc
