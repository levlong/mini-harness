"""Thin LLM client: one chat call, with hand-written retry + exponential backoff."""

import random
import time

import openai
from openai import OpenAI

RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)


class LLM:
    def __init__(self, api_key: str, base_url: str, model: str, max_retries: int = 5):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = max_retries
        self.last_usage: dict | None = None  # token counts of the latest call, for run logs

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        """Return the assistant message (has .content and .tool_calls)."""
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools or None,
                )
                self.last_usage = resp.usage.model_dump() if resp.usage else None
                return resp.choices[0].message
            except openai.BadRequestError as e:
                # Show the provider's real reason instead of a bare "400"
                body = e.response.text if e.response is not None else e
                raise RuntimeError(f"400 from provider: {body}") from e
            except RETRYABLE as e:
                if attempt == self.max_retries:
                    raise
                # 1s, 2s, 4s, 8s... plus jitter so parallel agents don't retry in sync
                delay = 2**attempt + random.random()
                print(f"[llm] {type(e).__name__}, retry {attempt + 1} in {delay:.1f}s")
                time.sleep(delay)
