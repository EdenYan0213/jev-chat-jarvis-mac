"""Structured intent, risk, and emotion judgment through an OpenAI-compatible LLM."""

from __future__ import annotations

import json
import re
import urllib.parse

import userconfig
from emotions import (
    EMOTIONS,
    EMOTION_CHOICE_CRITERIA,
    EMOTION_INTENSITY_LEVELS,
    EMOTION_TRENDS,
    normalize_emotion_fields,
)
from generate import _endpoint, _extra_params, http_post_json
from judge import (
    ACTION_MAP,
    CHAT_SCENES,
    INTENTS,
    INTENT_CHOICE_CRITERIA,
    INTENT_CHOICE_INSTRUCTION,
    SCENE_BOUNDARY_EXAMPLES,
    normalize_chat_scene,
    normalize_intent_for_scene,
)


DEFAULT_BASE = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen3.5:4b"
TIMEOUT = 90
MAX_CONTEXT_CHARS = 1_200
CONTEXT_HEAD_CHARS = 400
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


def _indexed_choice(value, choices, default: str) -> str:
    names = list(choices)
    if not isinstance(value, bool):
        try:
            index = int(float(value))
        except (TypeError, ValueError):
            pass
        else:
            if 0 <= index < len(names):
                return names[index]
    return _choice(value, choices, default)


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
        intents = "；".join(
            f"{name}={desc}" for name, desc in INTENT_CHOICE_CRITERIA.items())
        emotions = "；".join(
            f"{name}={desc}" for name, desc in EMOTION_CHOICE_CRITERIA.items())
        trends = "；".join(
            f"{name}={desc}" for name, desc in EMOTION_TRENDS.items())
        scenes = "；".join(
            f"{name}={desc}" for name, desc in CHAT_SCENES.items())
        prompt = (
            "结合会话判断“对方最新消息”。会话里的“我”是 App 用户，"
            "其他姓名或“对方”是发消息的人。\n"
            f"意图规则：{INTENT_CHOICE_INSTRUCTION}\n"
            f"场景：{scenes}\n"
            f"{SCENE_BOUNDARY_EXAMPLES}\n"
            f"意图：{intents}\n"
            f"情绪：{emotions}\n"
            f"趋势：{trends}\n"
            "强制边界：意图指对方这句话正在对我做什么，不是聊天主题。"
            "只要没有要求我执行任务，就绝不能选“派活”；"
            "提到工作、项目或很累都不等于派活。"
            "“我今天真的累坏了”是倾诉求安慰；"
            "“你今天感觉怎么样”是关心问候。\n"
            f"对话：{context or '无'}\n"
            f"对方最新消息：{message}\n"
            "只输出紧凑 JSON，不要字段解释："
            '{"v":["场景名","意图名",意图置信度0到100,风险0到9,'
            '"情绪名",情绪置信度0到100,强度0到4,"趋势名"]}。'
        )
        body = {
            "model": self.model,
            "max_tokens": 64,
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
        if self._ollama_generate_url():
            body["reasoning_effort"] = "none"
        body.update(_extra_params())
        data = self._post(body)
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("judge response had no choices")
        content = ((choices[0].get("message") or {}).get("content") or "")
        result = _json_object(content)

        values = result.get("v")
        compact = values if isinstance(values, list) and len(values) >= 7 else None
        if compact is not None:
            scene_value, intent_value, confidence_value, risk_value = compact[:4]
            emotion_value, emotion_confidence_value = compact[4:6]
            intensity_value = compact[6]
            trend_value = compact[7] if len(compact) > 7 else "稳定"
        else:
            compact_scene = (
                values[0]
                if isinstance(values, list) and values else None
            )
            scene_value = result.get(
                "p", result.get("scene", compact_scene))
            intent_value = result.get("i", result.get("intent"))
            confidence_value = result.get("c", result.get("confidence"))
            risk_value = result.get("r", result.get("risk"))
            emotion_value = result.get("e", result.get("emotion"))
            emotion_confidence_value = result.get(
                "ec", result.get("emotion_confidence"))
            intensity_value = result.get(
                "s", result.get("emotion_intensity"))
            trend_value = result.get("t", result.get("emotion_trend"))

        intent = _indexed_choice(intent_value, INTENTS, "闲聊")
        if isinstance(scene_value, (int, float)) and not isinstance(
                scene_value, bool):
            scene_value = _indexed_choice(
                scene_value, CHAT_SCENES, "不确定")
        scene = normalize_chat_scene(scene_value)
        intent = normalize_intent_for_scene(intent, scene)
        confidence = _number(
            confidence_value, 0.0, 0.0, 1.0)
        risk = _number(
            risk_value, 0.0, 0.0, 9.0)
        emotion = _indexed_choice(
            emotion_value, EMOTIONS, "平静")
        trend = _indexed_choice(
            trend_value,
            EMOTION_TRENDS,
            "稳定",
        )
        emotion_fields = normalize_emotion_fields(
            emotion=emotion,
            confidence=_number(
                emotion_confidence_value,
                0.0,
                0.0,
                1.0,
            ),
            intensity=_number(
                intensity_value,
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
            "scene": scene,
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
        # Loading the model is enough. A dummy structured judgment competes with the
        # first real message on Ollama's single request slot and doubles startup latency.
        return self.keep_warm()
