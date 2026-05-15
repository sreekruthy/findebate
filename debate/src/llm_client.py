"""
FinDebate — Person 5
Unified LLM client: Gemini 2.5 Flash (free default) + OpenAI + Anthropic

WHY GEMINI 2.5 FLASH?
  - Best free tier available: 1,500 req/day, no credit card
  - Strong reasoning; listed as a benchmark model in the paper (Table 1)
  - Get a free key at: https://aistudio.google.com/apikey
"""

import time
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── lazy imports so a missing SDK for one provider won't break the others ─────

def _get_gemini_client(api_key: str):
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai
    except ImportError:
        raise ImportError(
            "google-generativeai not installed.\n"
            "Run: pip install google-generativeai"
        )

def _get_openai_client(api_key: str):
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError("openai not installed. Run: pip install openai")

def _get_anthropic_client(api_key: str):
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise ImportError("anthropic not installed. Run: pip install anthropic")


# ─────────────────────────────────────────────────────────────────────────────
class LLMClient:
    """
    Single interface for Gemini, OpenAI, and Anthropic.

    Usage:
        client = LLMClient(provider="gemini", api_key="AIza...",
                           model="gemini-2.0-flash")
        text = client.chat(system_prompt, user_prompt)
    """

    def __init__(self,
                 provider: str,
                 api_key: str,
                 model: str,
                 temperature: float = 0.6,
                 max_tokens: int = 6500,
                 top_p: float = 0.85,
                 frequency_penalty: float = 0.1,
                 max_retries: int = 5,
                 retry_delay: float = 15.0):

        self.provider          = provider.lower()
        self.model             = model
        self.temperature       = temperature
        self.max_tokens        = max_tokens
        self.top_p             = top_p
        self.frequency_penalty = frequency_penalty
        self.max_retries       = max_retries
        self.retry_delay       = retry_delay

        if self.provider == "gemini":
            self._client = _get_gemini_client(api_key)
            self._api_key = api_key          # needed to init GenerativeModel per call
        elif self.provider == "openai":
            self._client = _get_openai_client(api_key)
        elif self.provider == "anthropic":
            self._client = _get_anthropic_client(api_key)
        else:
            raise ValueError(
                f"Unknown provider: {provider!r}. "
                "Use 'gemini', 'openai', or 'anthropic'."
            )

    # ── public API ────────────────────────────────────────────────────────────
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM and return the response text. Retries on transient errors."""
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.provider == "gemini":
                    return self._call_gemini(system_prompt, user_prompt)
                elif self.provider == "openai":
                    return self._call_openai(system_prompt, user_prompt)
                else:
                    return self._call_anthropic(system_prompt, user_prompt)
            except Exception as exc:
                wait = self.retry_delay * attempt   # linear back-off
                logger.warning(
                    f"[{self.provider}] attempt {attempt}/{self.max_retries} "
                    f"failed: {exc}. Waiting {wait}s..."
                )
                if attempt < self.max_retries:
                    time.sleep(wait)
                else:
                    raise

    # ── private helpers ───────────────────────────────────────────────────────
    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig

        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt,
        )
        gen_cfg = GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        response = model.generate_content(
            user_prompt,
            generation_config=gen_cfg,
        )
        return response.text.strip()

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text.strip()


# ─────────────────────────────────────────────────────────────────────────────
def build_client(agent_name: str, config) -> LLMClient:
    """Factory: read provider from config and return the right LLMClient."""
    provider = config.DEFAULT_PROVIDER   # "gemini" | "openai" | "anthropic"

    if provider == "gemini":
        api_key = config.GEMINI_API_KEY
        model   = config.GEMINI_MODEL
    elif provider == "anthropic":
        api_key = config.ANTHROPIC_API_KEY
        model   = config.ANTHROPIC_MODEL
    else:
        api_key = config.OPENAI_API_KEY
        model   = config.OPENAI_MODEL

    if not api_key:
        raise ValueError(
            f"API key for provider '{provider}' is empty.\n"
            f"Set the environment variable and try again:\n"
            f"  gemini    → export GEMINI_API_KEY='AIza...'\n"
            f"  openai    → export OPENAI_API_KEY='sk-...'\n"
            f"  anthropic → export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    return LLMClient(
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        top_p=config.TOP_P,
        frequency_penalty=config.FREQUENCY_PENALTY,
        max_retries=config.MAX_RETRIES,
        retry_delay=config.RETRY_DELAY_SEC,
    )


# ─────────────────────────────────────────────────────────────────────────────
def safe_parse_json(text: str) -> Optional[dict]:
    """
    Parse JSON from LLM output. Handles markdown fences (```json ... ```)
    and locates the outermost { } block if extra text surrounds it.
    Returns None if no valid JSON is found.
    """
    text = text.strip()

    # Strip ``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find outermost { ... }
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None
