"""Structured intent, risk, and emotion judgment through an OpenAI-compatible LLM."""

from __future__ import annotations

import json
import re
import urllib.parse

import userconfig
from emotions import (
    EMOTIONS,
    EMOTION_INTENSITY_LEVELS,
    EMOTION_TRENDS,
    normalize_emotion_fields,
)
from generate import _endpoint, _extra_params, http_post_json
from judge import ACTION_MAP, INTENTS


DEFAULT_BASE = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen3.5:4b"
TIMEOUT = 90
MAX_CONTEXT_CHARS = 6_000
CONTEXT_HEAD_CHARS = 2_000
OLLAMA_KEEP_ALIVE = "30m"


def configured() -> bool:
    return userconfig.get("JEV_JUDGE_BACKEND").strip().lower() in {
        "openai", "ollama", "openjev",
    }


def _number(value, default: float, low: float, high: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if result > 1.0 and high == 1.0 and result <= 100.0:
        result /= 100.0
    return min(high, max(low, result))


def _choice(value, choices, default: str) -> str:
    text = str(value or "").strip()
    if text in choices:
        return text
    for choice in choices:
        if choice in text:
            return choice
    return default


def _json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(),
                     flags=re.IGNORECASE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge response did not contain JSON")
    value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge response was not an object")
    return value


def _bounded_context(context: str) -> str:
    if len(context) <= MAX_CONTEXT_CHARS:
        return context
    marker = "\n（中间较早上下文已压缩省略）\n"
    tail_chars = MAX_CONTEXT_CHARS - CONTEXT_HEAD_CHARS - len(marker)
    return (
        context[:CONTEXT_HEAD_CHARS]
        + marker
        + context[-tail_chars:]
    )


class OpenAIJudge:
    """Use one local text model for Jev-shaped decisions without a hidden fallback."""

    name = "OpenJev-style Qwen3.5-4B (Ollama)"
    shares_generation_model = True
    ranks_candidates = False

    def __init__(self, base: str | None = None, key: str | None = None,
                 model: str | None = None, timeout: int = TIMEOUT):
        self.base = (
            base
            or userconfig.get("JEV_JUDGE_BASE_URL")
            or userconfig.get("OPENAI_BASE_URL")
            or DEFAULT_BASE
        ).rstrip("/")
        self.key = (
            key
            or userconfig.get("JEV_JUDGE_API_KEY")
            or userconfig.get("OPENAI_API_KEY")
            or "ollama"
        )
        self.model = (
            model
            or userconfig.get("JEV_JUDGE_MODEL")
            or userconfig.get("OPENAI_MODEL")
            or DEFAULT_MODEL
        )
        self.timeout = timeout
        self._last_url = ""

    def judge(self, message: str, context: str | None = None) -> dict:
        context = _bounded_context((context or "").strip())
        intents = "；".join(f"{name}={desc}" for name, desc in INTENTS.items())
        emotions = "；".join(
            f"{name}={desc}" for name, desc in EMOTIONS.items())
        trends = "；".join(
            f"{name}={desc}" for name, desc in EMOTION_TRENDS.items())
        prompt = (
            f"结合完整对话判断最后一句。\n"
            f"意图定义：{intents}\n"
            f"情绪定义：{emotions}\n"
            f"趋势定义：{trends}\n"
            f"对话：{context or '无'}\n"
            f"最后一句：{message}\n"
            "只输出 JSON："
            '{"i":"意图名","c":0到100,"r":0到9,"e":"情绪名",'
            '"ec":0到100,"s":0到4,"t":"趋势名"}。'
        )
        body = {
            "model": self.model,
            "max_tokens": 120,
            "temperature": 0,
            "stream": False,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是 OpenJev 风格的结构化判断器。"
                        "只输出合法 JSON，不解释，不补充事实。"
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }
        body.update(_extra_params())
        data = self._post(body)
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("judge response had no choices")
        content = ((choices[0].get("message") or {}).get("content") or "")
        result = _json_object(content)

        intent = _choice(
            result.get("i", result.get("intent")), INTENTS, "闲聊")
        confidence = _number(
            result.get("c", result.get("confidence")), 0.0, 0.0, 1.0)
        risk = _number(
            result.get("r", result.get("risk")), 0.0, 0.0, 9.0)
        emotion = _choice(
            result.get("e", result.get("emotion")), EMOTIONS, "平静")
        trend = _choice(
            result.get("t", result.get("emotion_trend")),
            EMOTION_TRENDS,
            "稳定",
        )
        emotion_fields = normalize_emotion_fields(
            emotion=emotion,
            confidence=_number(
                result.get("ec", result.get("emotion_confidence")),
                0.0,
                0.0,
                1.0,
            ),
            intensity=_number(
                result.get("s", result.get("emotion_intensity")),
                0.0,
                0.0,
                float(len(EMOTION_INTENSITY_LEVELS) - 1),
            ),
            trend=trend,
        )
        return {
            "intent": intent,
            "confidence": confidence,
            "intent_probs": {intent: confidence},
            "risk": round(risk, 1),
            "risk_probs": {},
            "actions": ACTION_MAP.get(intent, []),
            "message": message,
            "backend": f"openjev-style/{self.model}",
            **emotion_fields,
        }

    def rank_candidates(self, message: str, intent: str,
                        candidates: list[str]) -> list[dict]:
        """Keep generation order and avoid a second large-model pass."""
        if not candidates:
            return []
        probability = 1.0 / len(candidates)
        return [
            {"text": candidate, "prob": probability}
            for candidate in candidates
        ]

    def _post(self, body: dict) -> dict:
        self._last_url = _endpoint(self.base, "openai")
        return http_post_json(
            self._last_url,
            {
                "content-type": "application/json",
                "authorization": f"Bearer {self.key}",
            },
            body,
            self.timeout,
        )

    def _ollama_generate_url(self) -> str:
        parsed = urllib.parse.urlsplit(self.base)
        if (
            parsed.scheme in {"http", "https"}
            and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            and parsed.port == 11434
            and parsed.path.rstrip("/") in {"", "/v1"}
        ):
            return urllib.parse.urlunsplit(
                (parsed.scheme, parsed.netloc, "/api/generate", "", ""))
        return ""

    def keep_warm(self) -> bool:
        url = self._ollama_generate_url()
        if not url:
            return False
        http_post_json(
            url,
            {"content-type": "application/json"},
            {
                "model": self.model,
                "prompt": "",
                "stream": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
            },
            self.timeout,
        )
        return True

    def warm(self) -> bool:
        if not self.keep_warm():
            return False
        try:
            self.judge("预热", context="无")
        finally:
            self.keep_warm()
        return True
