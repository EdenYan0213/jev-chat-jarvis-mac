"""Judge: local Jev-shaped model reads a Chinese message and returns intent + risk.

One forward pass answers every slot (decider's documented multi-question layout:
append further `Question k: ... Answer k: (` blocks and read logits at each slot).
"""

from __future__ import annotations

import threading

import numpy as np

from emotions import (
    EMOTIONS,
    EMOTION_INTENSITY_LEVELS,
    EMOTION_TRENDS,
    ensure_emotion_fields,
    normalize_emotion_fields,
)

# The descriptions deliberately state both the relationship and the requested response.
# Without those anchors, a personal favor gets mislabeled as 派活 and every emotional
# message collapses into 闲聊. judge_zh_test.py imports this exact mapping so additions
# can be evaluated against the prompt the app really sends.
INTENTS = {
    "派活": "在明确的职场或项目语境中，对方要我执行任务或承担交付",
    "催进度": "对方在催促我尽快完成某个已在办的事",
    "问进度": "对方在询问某件事的进展或状态",
    "批评": "对方对我的做法、言语、工作或结果表达不满、指出错误",
    "要解释": "对方要求我说明原因或给出解释",
    "闲聊": "普通寒暄或分享，没有更具体的求助、倾诉、问候、邀约等意图",
    "约会议": "对方想围绕明确的工作事项安排会议或工作通话",
    "夸奖": "对方在肯定、称赞我的表现、能力或成果",
    "求助帮忙": "对方请我处理生活或私人请托，即使对方本身是同事",
    "征求建议": "对方想听我的看法、建议或如何选择，不是要我替他完成",
    "倾诉求安慰": "对方在表达难过、委屈、焦虑或疲惫，希望被理解和陪伴",
    "关心问候": "对方在关心我的近况、状态、安全或身体感受",
    "朋友邀约": "对方想为非工作目的约我吃饭、见面、出游、娱乐或私人通话",
    "玩笑调侃": "对方在开玩笑、接梗或善意打趣，不是在认真批评",
    "道歉和解": "对方在道歉、缓和矛盾或尝试修复关系",
    "感谢": "对方在感谢我的帮助、陪伴或付出",
}

# Keep this compact for structured Jev/Laya choice heads with a limited question budget.
INTENT_CHOICE_CRITERIA = {
    "派活": "职场任务",
    "催进度": "催已办事项",
    "问进度": "问进展状态",
    "批评": "不满或指错",
    "要解释": "追问原因",
    "闲聊": "无具体诉求的寒暄分享",
    "约会议": "工作会议通话",
    "夸奖": "肯定称赞",
    "求助帮忙": "生活私人请托",
    "征求建议": "询问看法选择",
    "倾诉求安慰": "负面感受求理解",
    "关心问候": "关心近况安全",
    "朋友邀约": "非工作见面娱乐",
    "玩笑调侃": "开玩笑接梗",
    "道歉和解": "道歉修复关系",
    "感谢": "表达感谢",
}

INTENT_SELECTION_INSTRUCTION = (
    "先判断是工作协作还是朋友社交，再结合完整会话选择最具体的意图。"
    "只有明确的工作任务分配才选“派活”，朋友请托选“求助帮忙”；"
    "工作会议选“约会议”，吃饭出游等选“朋友邀约”；"
    "能选倾诉、问候、建议、调侃、道歉或感谢时不要笼统选“闲聊”。"
)

INTENT_CHOICE_INSTRUCTION = (
    "结合上下文选最具体意图；工作任务才是派活，生活请托是求助帮忙，"
    "工作会议与朋友邀约分开，具体社交意图优先于闲聊。"
)

CHAT_SCENES = {
    "工作": "这句话在推进工作任务、项目或业务",
    "朋友": "这句话在处理私人日常、情感或社交",
}

SCENE_BOUNDARY_EXAMPLES = (
    "场景按事情而不是联系人身份判断，同事聊生活也属于朋友场景。"
    "边界示例：“同事下班帮带咖啡”=朋友+求助帮忙；"
    "“帮我取一下快递”=朋友+求助帮忙；"
    "“这个需求今天跟一下”=工作+派活；"
    "“周末一起吃饭吗”=朋友+朋友邀约；"
    "“下午三点开会同步”=工作+约会议。"
)


def normalize_chat_scene(value) -> str:
    text = str(value or "").strip()
    if text in CHAT_SCENES:
        return text
    if any(word in text for word in ("朋友", "私人", "社交", "家人", "伴侣")):
        return "朋友"
    if any(word in text for word in ("工作", "职场", "同事", "老板", "客户")):
        return "工作"
    return "不确定"


def normalize_intent_for_scene(intent: str, scene: str) -> str:
    """Repair work-shaped labels that small models emit for private messages."""
    if scene == "朋友":
        return {
            "派活": "求助帮忙",
            "约会议": "朋友邀约",
        }.get(intent, intent)
    return intent


RISK_LEVELS = [
    "完全没风险，怎么回都行",
    "基本没风险",
    "平淡，正常回就好",
    "需要稍微留神",
    "有点敏感，措辞注意",
    "需要谨慎，可能被挑刺",
    "比较危险，容易得罪人或踩坑",
    "很危险，说错要出问题",
    "非常危险，涉及责任或利益",
    "极度危险，先别回，想清楚再说",
]

# V0: actions are a static derivation, no generation involved
ACTION_MAP = {
    "派活": ["接住", "问清交付标准和期限", "先给个时间点"],
    "催进度": ["先给当前状态", "给明确的完成时间", "别解释太多"],
    "问进度": ["直接说事实", "给下个节点", "有卡点就说卡点"],
    "批评": ["先认下来", "别急着辩解", "给补救方案"],
    "要解释": ["说清原因", "别找借口", "给改进措施"],
    "闲聊": ["轻松回应", "可以互动", "不用当真"],
    "约会议": ["确认时间", "说清议程", "准备好材料"],
    "夸奖": ["接住并感谢", "别过度谦虚", "可以顺带提下一步"],
    "求助帮忙": ["确认具体需要", "能帮就说明怎么帮", "做不到就给替代办法"],
    "征求建议": ["先理解顾虑", "给明确看法", "把决定权留给对方"],
    "倾诉求安慰": ["先共情", "别急着讲道理", "问对方想被陪伴还是要建议"],
    "关心问候": ["自然回应近况", "接住对方关心", "顺势关心对方"],
    "朋友邀约": ["明确愿不愿意", "确认时间地点", "不方便就给替代时间"],
    "玩笑调侃": ["顺着接梗", "保持分寸", "别误判成冲突"],
    "道歉和解": ["接住道歉", "说清感受和边界", "愿意就给和解下一步"],
    "感谢": ["自然接住", "不用过度客套", "回应彼此关系"],
}


class Judge:
    """Wraps a decoder-only decision model; lazy-loads on first use."""

    name = "本地 decider-2b"
    shares_generation_model = False
    ranks_candidates = True

    def __init__(self, repo: str = "Mapika/decider-2b", device: str | None = None):
        import torch

        self.torch = torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.repo = repo
        self.temperature = 1.3
        self._loaded = False
        # RLock, not Lock: warm() holds it across the whole dummy forward, and judge()
        # inside that same call re-enters _load(). One lock guards both the load and the
        # first forward, so a warm-up and a real judgment can never run a forward at the
        # same time — they queue up instead.
        self._load_lock = threading.RLock()

    def _load(self):
        if self._loaded:
            return
        # Double-checked: the warm-up thread and the first real message can both get here
        # at once, and two concurrent from_pretrained calls would load the model twice.
        # The loser of the race just waits on the lock until the winner is done.
        with self._load_lock:
            if self._loaded:
                return
            from transformers import AutoModelForCausalLM, AutoTokenizer

            t = self.torch
            self.tok = AutoTokenizer.from_pretrained(self.repo)
            # float16, not bfloat16: MPS takes the slow path for bf16 (limited op coverage) and
            # it costs exactly 2x here — measured on this model, same prompt, three runs each:
            # bf16 1352/1393/1467 ms vs fp16 734/745/827 ms. The judge is the single biggest
            # steady-state cost in the pipeline, so this is the difference between a ~3 s and a
            # ~4 s reply. CPU has no fp16 win, so it stays fp32.
            dtype = t.float16 if self.device == "mps" else t.float32
            self.model = AutoModelForCausalLM.from_pretrained(self.repo, dtype=dtype).to(self.device).eval()
            self._letters = [self.tok.encode(c, add_special_tokens=False)[0]
                             for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
            self._loaded = True

    def warm(self) -> bool:
        """Load the model and run one real-shaped forward, so no real message pays for it.

        decider-2b's first load costs 9-15 s and lands inside whichever judge() call gets
        there first — the HUD starts this in the background right after launch, so that
        call is ours, not the user's first message. The whole thing runs under the load
        lock: if a real message arrives mid-warm-up, its judge() blocks here until the
        warm-up is done, then runs at steady state.
        """
        with self._load_lock:
            self._load()
            self.judge("预热")
        return True

    def _slot_probs(self, logits_by_slot: list, n_options: int, slot: int) -> np.ndarray:
        logits = logits_by_slot[slot]
        ids = self._letters[:n_options]
        probs = self.torch.softmax(logits[ids].float() / self.temperature, -1)
        return probs.cpu().numpy()

    def _forward(self, prompt: str, n_slots: int):
        """One forward pass; returns (logits, [token index per 'Answer: (' slot]).

        logits[i] is the distribution for position i+1, so reading at the token that
        contains "(" gives the letter distribution for that slot.
        """
        import re

        ids = self.tok(prompt, return_tensors="pt", return_offsets_mapping=True).to(self.device)
        offsets = ids.pop("offset_mapping")[0].tolist()
        with self.torch.no_grad():
            out = self.model(**ids)
        slot_token_idx = []
        for m in re.finditer(r"Answer: \(", prompt):
            char_pos = m.start() + len("Answer: ")
            for i, (s, e) in enumerate(offsets):
                if s <= char_pos < e:
                    slot_token_idx.append(i)
                    break
        if len(slot_token_idx) < n_slots:
            raise RuntimeError(f"expected {n_slots} answer slots, found {len(slot_token_idx)}")
        return out.logits[0], slot_token_idx

    def rank_candidates(self, message: str, intent: str,
                        candidates: list[str]) -> list[dict]:
        """Rank reply candidates by asking which one fits best.

        The candidates are the options, so one forward pass yields the distribution the
        phone demo shows as 89% / 9% / 2%.
        """
        self._load()
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        prompt = f"Context:\n收到：「{message}」\n判断出的意图：{intent}\n\n"
        prompt += "Question: 哪一条回复最合适？\nOptions:\n"
        for i, c in enumerate(candidates):
            prompt += f"({letters[i]}) {c}\n"
        prompt += "Answer: ("

        logits, slots = self._forward(prompt, 1)
        probs = self._slot_probs([logits[slots[0]]], len(candidates), 0)
        ranked = sorted(
            ({"text": c, "prob": float(p)} for c, p in zip(candidates, probs)),
            key=lambda r: -r["prob"])
        return ranked

    def judge(self, message: str, context: str | None = None) -> dict:
        self._load()
        intents = list(INTENTS)
        emotions = list(EMOTIONS)
        trends = list(EMOTION_TRENDS)
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        prompt = f"Context:\n{context + chr(10) + chr(10) if context else ''}{message}\n\n"
        # slot 0: intent
        prompt += f"Question: {INTENT_SELECTION_INSTRUCTION}\nOptions:\n"
        for i, name in enumerate(intents):
            prompt += f"({letters[i]}) {name} - {INTENTS[name]}\n"
        prompt += "Answer: ("
        # slot 1: risk
        prompt += "\n\nQuestion: 如果直接回复这句话，风险有多大？\nOptions:\n"
        for i, lv in enumerate(RISK_LEVELS):
            prompt += f"({letters[i]}) {lv}\n"
        prompt += "Answer: ("
        # slot 2: primary emotion
        prompt += "\n\nQuestion: 结合完整会话，这句话的主情绪是什么？\nOptions:\n"
        for i, name in enumerate(emotions):
            prompt += f"({letters[i]}) {name} - {EMOTIONS[name]}\n"
        prompt += "Answer: ("
        # slot 3: emotion intensity
        prompt += "\n\nQuestion: 这句话的情绪强度有多高？\nOptions:\n"
        for i, level in enumerate(EMOTION_INTENSITY_LEVELS):
            prompt += f"({letters[i]}) {level}\n"
        prompt += "Answer: ("
        # slot 4: emotion trend relative to prior turns
        prompt += "\n\nQuestion: 相比会话前文，这句话的情绪趋势是什么？\nOptions:\n"
        for i, name in enumerate(trends):
            prompt += f"({letters[i]}) {name} - {EMOTION_TRENDS[name]}\n"
        prompt += "Answer: ("
        # slot 5: relationship scene, used to repair the two most common cross-scene
        # confusions without a second model call.
        prompt += (
            "\n\nQuestion: 最后一句是在推进工作，还是在处理私人社交？"
            f"{SCENE_BOUNDARY_EXAMPLES}\nOptions:\n"
        )
        for i, name in enumerate(CHAT_SCENES):
            prompt += f"({letters[i]}) {name} - {CHAT_SCENES[name]}\n"
        prompt += "Answer: ("

        logits, slot_token_idx = self._forward(prompt, 6)

        intent_probs = self._slot_probs([logits[slot_token_idx[0]]], len(intents), 0)
        risk_probs = self._slot_probs([logits[slot_token_idx[1]]], len(RISK_LEVELS), 0)
        emotion_probs = self._slot_probs(
            [logits[slot_token_idx[2]]], len(emotions), 0)
        intensity_probs = self._slot_probs(
            [logits[slot_token_idx[3]]],
            len(EMOTION_INTENSITY_LEVELS), 0)
        trend_probs = self._slot_probs(
            [logits[slot_token_idx[4]]], len(trends), 0)
        scene_probs = self._slot_probs(
            [logits[slot_token_idx[5]]], len(CHAT_SCENES), 0)

        intent_idx = int(np.argmax(intent_probs))
        scene_idx = int(np.argmax(scene_probs))
        scene = list(CHAT_SCENES)[scene_idx]
        intent = normalize_intent_for_scene(intents[intent_idx], scene)
        risk_value = float((np.arange(len(RISK_LEVELS)) * risk_probs).sum())
        emotion_idx = int(np.argmax(emotion_probs))
        intensity_value = float((
            np.arange(len(EMOTION_INTENSITY_LEVELS))
            * intensity_probs).sum())
        trend_idx = int(np.argmax(trend_probs))
        emotion_fields = normalize_emotion_fields(
            emotion=emotions[emotion_idx],
            confidence=float(emotion_probs[emotion_idx]),
            intensity=intensity_value,
            trend=trends[trend_idx],
        )

        return {
            "intent": intent,
            "confidence": float(intent_probs[intent_idx]),
            "intent_probs": {n: float(p) for n, p in zip(intents, intent_probs)},
            "risk": round(risk_value, 1),
            "risk_probs": {str(i): float(p) for i, p in enumerate(risk_probs)},
            "actions": ACTION_MAP.get(intent, []),
            "message": message,
            "scene": scene,
            **emotion_fields,
        }


if __name__ == "__main__":
    import json
    import sys

    j = Judge()
    msg = sys.argv[1] if len(sys.argv) > 1 else "这个需求你今天跟一下"
    print(json.dumps(j.judge(msg), ensure_ascii=False, indent=1))


class FallbackJudge:
    """Prefer the official Jev API; drop to the local model if it fails.

    A judgment layer that dies because a key expired or a gateway hiccuped would take the
    whole panel down, so the first failure switches permanently to the local model and the
    verdict carries which backend produced it.
    """

    def __init__(self):
        import judge_jev
        self.primary = judge_jev.JevJudge()
        self.local = None
        self.fell_back = False
        self.reason = ""
        self.name = "TypeSafe Jev"
        self.shares_generation_model = False
        self.ranks_candidates = True

    def _fallback(self):
        if self.local is None:
            self.local = Judge()
        return self.local

    def judge(self, message: str, context: str | None = None) -> dict:
        if not self.fell_back:
            try:
                return ensure_emotion_fields(
                    self.primary.judge(message, context))
            except Exception as e:
                self.fell_back = True
                self.reason = f"{type(e).__name__}: {str(e)[:80]}"
        out = self._fallback().judge(message, context)
        out["backend"] = f"local (Jev 不可用: {self.reason})"
        return ensure_emotion_fields(out)

    def rank_candidates(self, message: str, intent: str, candidates: list[str]) -> list[dict]:
        if not self.fell_back:
            try:
                return self.primary.rank_candidates(message, intent, candidates)
            except Exception as e:
                self.fell_back = True
                self.reason = f"{type(e).__name__}: {str(e)[:80]}"
        return self._fallback().rank_candidates(message, intent, candidates)

    def warm(self) -> bool:
        return False


def make_judge():
    """Select the configured judgment backend."""
    import judge_openai
    if judge_openai.configured():
        # An explicitly selected shared model must never silently load decider-2b.
        return judge_openai.OpenAIJudge()
    try:
        import judge_jev
        if judge_jev.jev_configured():
            return FallbackJudge()
    except Exception:
        pass
    return Judge()
