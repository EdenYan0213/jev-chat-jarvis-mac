"""Shared emotion taxonomy and normalization for every judge backend."""
from __future__ import annotations

import math


EMOTIONS = {
    "开心": "明确表达愉快、满意、轻松或兴奋",
    "平静": "语气中性稳定，没有明显情绪波动",
    "期待": "对后续结果、回应或行动抱有正向期待",
    "困惑": "不理解、需要澄清或无法判断下一步",
    "焦虑": "担心结果、时间、风险或失去控制",
    "委屈": "感到被误解、被忽视、不公平或没有被照顾",
    "生气": "表达不满、责备、敌意或明显急躁",
    "失望": "期待落空，对人或结果感到不满意",
    "悲伤": "表达难过、失落、痛苦或告别感",
    "疲惫": "表达精力耗尽、厌倦、无力或想暂停",
}

# Laya's native classifier reserves a fixed 256-token question head. Keep the
# Jev-facing choice descriptions compact enough that all ten labels survive
# exactly; the local decoder prompt can continue using the richer definitions.
EMOTION_CHOICE_CRITERIA = {
    "开心": "愉快满意",
    "平静": "中性稳定",
    "期待": "正向期待",
    "困惑": "不理解或要澄清",
    "焦虑": "担心失控",
    "委屈": "被误解或不公平",
    "生气": "不满责备",
    "失望": "期待落空",
    "悲伤": "难过失落",
    "疲惫": "厌倦无力",
}

EMOTION_INTENSITY_LEVELS = [
    "没有明显情绪",
    "轻微情绪，可以平常回应",
    "情绪清晰，需要照顾语气",
    "情绪较强，回复需要谨慎",
    "情绪非常强烈，优先安抚或暂停冲突",
]

EMOTION_TRENDS = {
    "缓和": "相比前文，负面情绪减弱或正面状态恢复",
    "稳定": "相比前文，情绪方向和强度基本没有变化",
    "升温": "相比前文，情绪强度增加或冲突正在升级",
}


def default_emotion_fields() -> dict:
    return {
        "emotion": "平静",
        "emotion_confidence": 0.0,
        "emotion_intensity": 0.0,
        "emotion_trend": "稳定",
    }


def _number(value, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clamp(value, low: float, high: float, default: float) -> float:
    return min(high, max(low, _number(value, default)))


def normalize_emotion_fields(*, emotion=None, confidence=None,
                             intensity=None, trend=None) -> dict:
    fields = default_emotion_fields()
    if emotion in EMOTIONS:
        fields["emotion"] = emotion
        fields["emotion_confidence"] = _clamp(
            confidence, 0.0, 1.0, 0.0)
    fields["emotion_intensity"] = _clamp(
        intensity, 0.0, 4.0, 0.0)
    if trend in EMOTION_TRENDS:
        fields["emotion_trend"] = trend
    return fields


def ensure_emotion_fields(verdict: dict) -> dict:
    result = dict(verdict)
    result.update(normalize_emotion_fields(
        emotion=result.get("emotion"),
        confidence=result.get("emotion_confidence"),
        intensity=result.get("emotion_intensity"),
        trend=result.get("emotion_trend"),
    ))
    return result


def format_emotion(verdict: dict) -> str:
    fields = ensure_emotion_fields(verdict)
    intensity = int(round(fields["emotion_intensity"]))
    return (
        f"{fields['emotion']} {intensity}/4"
        f" · 情绪{fields['emotion_trend']}")
