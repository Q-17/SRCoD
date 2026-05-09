from __future__ import annotations

import base64
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout_sec: float = 600.0,
        max_retries: int = 2,
        retry_backoff_sec: float = 2.0,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.max_retries = int(max_retries)
        self.retry_backoff_sec = float(retry_backoff_sec)
        self.temperature = float(temperature)
        self.session = requests.Session()

    def chat(self, model: str, prompt: str, image_paths: list[str] | None = None) -> tuple[str | None, str | None]:
        images_b64: list[str] = []
        for image_path in image_paths or []:
            p = Path(image_path)
            if not p.exists():
                return None, f"image_not_found:{image_path}"
            images_b64.append(base64.b64encode(p.read_bytes()).decode("utf-8"))

        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [{"role": "user", "content": prompt, "images": images_b64}],
        }
        return _post_with_retries(
            session=self.session,
            url=f"{self.base_url}/api/chat",
            payload=payload,
            timeout_sec=self.timeout_sec,
            max_retries=self.max_retries,
            retry_backoff_sec=self.retry_backoff_sec,
            parser=lambda body: body.get("message", {}).get("content"),
        )


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_sec: float = 600.0,
        max_retries: int = 2,
        retry_backoff_sec: float = 2.0,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing API key. Pass --api-key or set {api_key_env}.")
        self.timeout_sec = float(timeout_sec)
        self.max_retries = int(max_retries)
        self.retry_backoff_sec = float(retry_backoff_sec)
        self.temperature = float(temperature)
        self.session = requests.Session()

    def chat(self, model: str, prompt: str, image_paths: list[str] | None = None) -> tuple[str | None, str | None]:
        image_contents, image_error = _openai_image_contents(image_paths or [])
        if image_error:
            return None, image_error
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are solving a multiple-choice question. Output exactly one uppercase option letter and nothing else.",
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}] + image_contents,
                },
            ],
            "temperature": self.temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        return _post_with_retries(
            session=self.session,
            url=f"{self.base_url}/chat/completions",
            payload=payload,
            timeout_sec=self.timeout_sec,
            max_retries=self.max_retries,
            retry_backoff_sec=self.retry_backoff_sec,
            parser=lambda body: body.get("choices", [{}])[0].get("message", {}).get("content"),
            headers=headers,
        )


def _openai_image_contents(image_paths: list[str]) -> tuple[list[dict[str, Any]], str | None]:
    out: list[dict[str, Any]] = []
    for image_path in image_paths:
        p = Path(image_path)
        if not p.exists():
            return [], f"image_not_found:{image_path}"
        mime = mimetypes.guess_type(str(p))[0] or "image/png"
        encoded = base64.b64encode(p.read_bytes()).decode("ascii")
        out.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
    return out, None


def _post_with_retries(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    timeout_sec: float,
    max_retries: int,
    retry_backoff_sec: float,
    parser,
    headers: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            response = session.post(url, json=payload, headers=headers, timeout=timeout_sec)
            if response.status_code != 200:
                last_error = f"http_{response.status_code}:{response.text[:500]}"
            else:
                body = response.json()
                content = parser(body)
                if content is None:
                    return None, "missing_message_content"
                return str(content), None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < max_retries:
            sleep_sec = retry_backoff_sec * (2**attempt)
            LOGGER.warning("Model call failed (attempt %s/%s): %s", attempt + 1, max_retries + 1, last_error)
            time.sleep(sleep_sec)
    return None, last_error
