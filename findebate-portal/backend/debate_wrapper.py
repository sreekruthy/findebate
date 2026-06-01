import asyncio
import json
import os
import re
import sys
import time

from dotenv import load_dotenv


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


PIPELINE_AVAILABLE = False
PIPELINE_ERROR = ""

P5_PATHS = [
    "/Users/sreekruthyreddy/Documents/GitHub/findebate/Debate",
    "/Users/sreekruthyreddy/Documents/GitHub/findebate/Debate/p5_debate",
    os.path.expanduser("~/findebate/p5_debate/p5_debate"),
    "/Users/sreekruthyreddy/Downloads/p5_debate/p5_debate",
]

for path in P5_PATHS:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

try:
    from src.algorithm1 import run_safe_debate
    import configs.config as p5_config

    PIPELINE_AVAILABLE = True
except Exception as exc:
    PIPELINE_ERROR = str(exc)


class PortalDebateClient:
    def __init__(self, role: str):
        self.role = role
        self.provider = os.getenv("FINDEBATE_P5_PROVIDER", "gemini").strip().lower()
        self.model = os.getenv(
            "FINDEBATE_P5_MODEL",
            "gemini-2.5-flash" if self.provider == "gemini" else "llama-3.1-8b-instant",
        )
        self.temperature = float(os.getenv("FINDEBATE_P5_TEMPERATURE", str(p5_config.TEMPERATURE)))
        self.max_tokens = int(os.getenv("FINDEBATE_P5_MAX_TOKENS", "1200"))
        self.top_p = float(os.getenv("FINDEBATE_P5_TOP_P", str(p5_config.TOP_P)))
        self.max_retries = int(os.getenv("FINDEBATE_P5_MAX_RETRIES", "2"))
        self.retry_delay = float(os.getenv("FINDEBATE_P5_RETRY_DELAY_SEC", "8"))
        self._client = None
        self._types = None

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                prompt = self._fit_prompt_for_provider(user_prompt)
                if self.provider == "groq":
                    text = self._chat_groq(system_prompt, prompt)
                else:
                    text = self._chat_gemini(system_prompt, prompt)
                if self.role == "leader":
                    return self._ensure_json(text)
                return text
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if self.provider == "gemini" and (
                    "429" in message or "quota" in message or "resource_exhausted" in message
                ) and os.getenv("GROQ_API_KEY"):
                    self.provider = "groq"
                    self.model = os.getenv("FINDEBATE_P5_GROQ_MODEL", "llama-3.3-70b-versatile")
                    self._client = None
                    continue
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
        raise RuntimeError(f"{self.role} debate client failed: {last_error}")

    def _fit_prompt_for_provider(self, user_prompt: str) -> str:
        if self.provider != "groq":
            return user_prompt
        max_chars = int(os.getenv("FINDEBATE_P5_GROQ_MAX_PROMPT_CHARS", "5200"))
        if len(user_prompt) <= max_chars:
            return user_prompt
        head = user_prompt[: max_chars // 2]
        tail = user_prompt[-max_chars // 2 :]
        return (
            f"{head}\n\n[...middle debate context compacted for Groq token budget; "
            "preserve recommendations and use the final constraints below...]\n\n"
            f"{tail}"
        )

    def _parse_json_candidate(self, text: str) -> dict | None:
        clean = text.strip().replace("```json", "").replace("```", "").strip()
        candidates = [clean]
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            candidates.append(clean[start : end + 1])
        for candidate in candidates:
            candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return None

    def _ensure_json(self, text: str) -> str:
        parsed = self._parse_json_candidate(text)
        if parsed is not None:
            return json.dumps(parsed, ensure_ascii=False)
        repair_prompt = """Convert the following Leader Agent response into one valid JSON object.
Return ONLY JSON. Preserve the investment stance, recommendations, conviction levels, evidence,
risk additions, and conclusion. Do not add markdown fences.

LEADER RESPONSE:
""" + text[:12000]
        repaired = (
            self._chat_groq("You repair malformed JSON into valid JSON only.", repair_prompt)
            if self.provider == "groq"
            else self._chat_gemini("You repair malformed JSON into valid JSON only.", repair_prompt, force_json=True)
        )
        parsed = self._parse_json_candidate(repaired)
        if parsed is None:
            return text
        return json.dumps(parsed, ensure_ascii=False)

    def _chat_gemini(self, system_prompt: str, user_prompt: str, force_json: bool = False) -> str:
        if self._client is None:
            from google import genai
            from google.genai import types

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is missing")
            self._client = genai.Client(api_key=api_key)
            self._types = types
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        config_kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_tokens,
            "http_options": self._types.HttpOptions(timeout=120000),
        }
        if force_json or self.role == "leader":
            config_kwargs["response_mime_type"] = "application/json"
        response = self._client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config=self._types.GenerateContentConfig(**config_kwargs),
        )
        return (response.text or "").strip()

    def _chat_groq(self, system_prompt: str, user_prompt: str) -> str:
        if self._client is None:
            from groq import Groq

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY is missing")
            self._client = Groq(api_key=api_key)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()


async def run_debate(synthesis: dict, p3_data: dict, p4_data: dict) -> tuple[dict, dict]:
    if not PIPELINE_AVAILABLE:
        raise RuntimeError(f"P5 debate unavailable: {PIPELINE_ERROR}")
    trust_client = PortalDebateClient("trust")
    skeptic_client = PortalDebateClient("skeptic")
    leader_client = PortalDebateClient("leader")
    loop = asyncio.get_event_loop()
    optimized, debate_log = await loop.run_in_executor(
        None,
        run_safe_debate,
        synthesis,
        p3_data,
        p4_data,
        trust_client,
        skeptic_client,
        leader_client,
    )
    return optimized, debate_log
