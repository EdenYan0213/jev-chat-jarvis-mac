"""Structured intent, risk, and emotion judgment through an OpenAI-compatible LLM."""

from __future__ import annotations

import json
import re

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
MAX_CONTEXT_CHARS = 12_000
CONTEXT_HEAD_CHARS = 4_000


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
        prompt = {
            "conversation_context": context or "无",
            "latest_message": message,
            "intent_options": INTENTS,
            "emotion_options": EMOTIONS,
            "emotion_trend_options": EMOTION_TRENDS,
            "task": (
                "判断最新消息的意图、直接回复风险、主情绪、情绪强度和相对趋势。"
                "只依据对话，不补充事实。risk 为 0 到 9，emotion_intensity 为 0 到 4。"
            ),
            "output_schema": {
                "intent": "必须是 intent_options 的一个键",
                "confidence": "0 到 1",
                "risk": "0 到 9",
                "emotion": "必须是 emotion_options 的一个键",
                "emotion_confidence": "0 到 1",
                "emotion_intensity": "0 到 4",
                "emotion_trend": "必须是 emotion_trend_options 的一个键",
            },
        }
        body = {
            "model": self.model,
            "max_tokens": 320,
            "temperature": 0.1,
            "stream": False,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是 OpenJev 风格的结构化判断器。"
                        "不要解释推理过程，只输出一个符合要求的 JSON 对象。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=False),
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

        intent = _choice(result.get("intent"), INTENTS, "闲聊")
        confidence = _number(result.get("confidence"), 0.0, 0.0, 1.0)
        risk = _number(result.get("risk"), 0.0, 0.0, 9.0)
        emotion = _choice(result.get("emotion"), EMOTIONS, "平静")
        trend = _choice(
            result.get("emotion_trend"), EMOTION_TRENDS, "稳定")
        emotion_fields = normalize_emotion_fields(
            emotion=emotion,
            confidence=_number(
                result.get("emotion_confidence"), 0.0, 0.0, 1.0),
            intensity=_number(
                result.get("emotion_intensity"), 0.0, 0.0,
                float(len(EMOTION_INTENSITY_LEVELS) - 1)),
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

    def warm(self) -> None:
        return None
