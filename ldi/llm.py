"""One small function to call an LLM, whichever provider is configured.

Groq, Gemini and Ollama all offer an OpenAI-compatible endpoint, so a single
dependency (the `openai` package) covers all three: only the base URL, the
API key and the model name change. Configure in .env:

    LLM_PROVIDER=groq | gemini | ollama
    LLM_MODEL=...            (optional, overrides the default below)
"""
import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
    },
    "ollama": {  # offline fallback: `ollama pull qwen2.5-coder:7b` first
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "model": "qwen2.5-coder:7b",
    },
}


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "groq")
        if self.provider not in PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {list(PROVIDERS)}")
        cfg = PROVIDERS[self.provider]
        self.model = model or os.getenv("LLM_MODEL") or cfg["model"]
        api_key = os.getenv(cfg["key_env"]) if cfg["key_env"] else "ollama"
        if not api_key:
            raise RuntimeError(f"Set {cfg['key_env']} in .env (see .env.example)")
        # max_retries: the client backs off and retries on rate limits (HTTP 429),
        # which free tiers hit quickly during an eval run.
        self.client = OpenAI(api_key=api_key, base_url=cfg["base_url"], max_retries=5)

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,  # same question -> (nearly) same answer; needed for evals
            **kwargs,
        )
        return response.choices[0].message.content or ""


def parse_json(text: str) -> dict:
    """Parse the model's JSON reply. Some models wrap it in ```json fences or
    add a sentence around it, so fall back to the outermost {...} block."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"LLM did not return JSON: {text[:200]!r}")
        return json.loads(match.group(0))
