"""Chinese intent-judgment test across workplace and everyday social messages.

Zero-shot, no training. The decider path calls the production Judge directly so scene
normalization and all prompt changes are covered instead of being reimplemented here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from judge import (
    CHAT_SCENES,
    INTENT_CHOICE_CRITERIA,
    INTENT_CHOICE_INSTRUCTION,
    Judge,
    normalize_chat_scene,
    normalize_intent_for_scene,
)

# (message text, gold intent) — includes the live-captured ones
CASES: list[tuple[str, str]] = [
    ("这个需求你今天跟一下", "派活"),
    ("顺手把这个需求文档补一下", "派活"),
    ("明天把这个方案给客户发过去吧", "派活"),
    ("那个东西做完了吗", "催进度"),
    ("那个东西什么时候能好？", "催进度"),
    ("这块还没动呢？抓紧点", "催进度"),
    ("现在进度怎么样了", "问进度"),
    ("上线了吗", "问进度"),
    ("客户那边反馈如何", "问进度"),
    ("这个逻辑不对啊，你再看下", "批评"),
    ("怎么又出问题了", "批评"),
    ("这做的什么玩意", "批评"),
    ("为什么用这个方案？", "要解释"),
    ("你当时怎么想的", "要解释"),
    ("这个数据从哪来的", "要解释"),
    ("哈哈哈太搞笑了", "闲聊"),
    ("我周末去爬山了", "闲聊"),
    ("牛啊这也能写出来", "闲聊"),
    ("下午三点开个会同步一下", "约会议"),
    ("方便的话我们语音聊十分钟", "约会议"),
    ("这个做得不错，继续", "夸奖"),
    ("牛逼，这个思路好", "夸奖"),
    ("能帮我取一下快递吗", "求助帮忙"),
    ("我搬家那天能来搭把手吗", "求助帮忙"),
    ("你觉得我该不该换工作", "征求建议"),
    ("这两件衣服你觉得哪件好看", "征求建议"),
    ("今天真的好累，感觉没人理解我", "倾诉求安慰"),
    ("我最近心里特别难受，想找个人说说", "倾诉求安慰"),
    ("到家了吗，路上还顺利吗", "关心问候"),
    ("你感冒好点没有", "关心问候"),
    ("周末一起吃饭吗", "朋友邀约"),
    ("晚上要不要出来散散步", "朋友邀约"),
    ("你这个表情包也太像你了哈哈", "玩笑调侃"),
    ("可以啊你，现在都会放我鸽子了", "玩笑调侃"),
    ("刚才是我语气不好，对不起", "道歉和解"),
    ("别生气了，是我没考虑你的感受", "道歉和解"),
    ("谢谢你今天一直陪着我", "感谢"),
    ("多亏你帮忙，不然我真搞不定", "感谢"),
]


def run_decider() -> dict:
    """Mapika/decider-2b through the exact production prompt and normalization."""
    judge = Judge()
    t0 = time.perf_counter()
    results = []
    for text, gold in CASES:
        verdict = judge.judge(text)
        results.append({
            "text": text,
            "gold": gold,
            "pred": verdict["intent"],
            "conf": verdict["confidence"],
            "scene": verdict["scene"],
        })
    return {"model": "decider-2b", "elapsed_s": time.perf_counter() - t0, "results": results}


def run_laya_multilingual() -> dict:
    """convaiinnovations/laya multilingual checkpoint (Chinese-capable per its card)."""
    import laya

    agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
    question = {
        "intent": {"type": "choice",
                   "instructions": INTENT_CHOICE_INSTRUCTION,
                   "criteria": INTENT_CHOICE_CRITERIA},
        "scene": {"type": "choice",
                  "instructions": "这句话是在推进工作还是处理私人社交？",
                  "criteria": CHAT_SCENES},
    }

    t0 = time.perf_counter()
    results = []
    for text, gold in CASES:
        try:
            out = agent.predict(text, question)
            ans = out["answers"]["intent"]
            scene = normalize_chat_scene(
                (out["answers"].get("scene") or {}).get("choice"))
            pred = normalize_intent_for_scene(ans.get("choice"), scene)
            conf = float(ans.get("confidence", 0.0))
        except Exception as e:
            pred, conf = f"ERR:{type(e).__name__}", 0.0
        results.append({"text": text, "gold": gold, "pred": pred, "conf": conf})
    return {"model": "laya-multilingual", "elapsed_s": time.perf_counter() - t0,
            "results": results}


def summarize(run: dict) -> dict:
    res = run["results"]
    n = len(res)
    correct = sum(1 for r in res if r["pred"] == r["gold"])
    by_gold: dict[str, list[bool]] = {}
    for r in res:
        by_gold.setdefault(r["gold"], []).append(r["pred"] == r["gold"])
    return {
        "model": run["model"],
        "acc": correct / n,
        "n": n,
        "elapsed_s": round(run["elapsed_s"], 1),
        "per_intent": {k: round(sum(v) / len(v), 2) for k, v in by_gold.items()},
        "majority_baseline": round(max(
            sum(1 for r in res if r["gold"] == g) for g in set(r["gold"] for r in res)) / n, 3),
    }


def main() -> None:
    out_path = Path("results/judge_zh.json")
    out_path.parent.mkdir(exist_ok=True)
    report = {}

    for name, fn in (("decider", run_decider), ("laya_ml", run_laya_multilingual)):
        print(f"\n===== {name} =====", flush=True)
        try:
            run = fn()
            s = summarize(run)
            report[name] = {"summary": s, "results": run["results"]}
            print(f"acc={s['acc']:.3f}  n={s['n']}  elapsed={s['elapsed_s']}s  "
                  f"majority_baseline={s['majority_baseline']}")
            print("per-intent:", s["per_intent"])
            for r in run["results"]:
                flag = "OK " if r["pred"] == r["gold"] else "XX "
                print(f"  {flag}{r['text'][:22]:24s} gold={r['gold']:5s} "
                      f"pred={str(r['pred']):6s} conf={r['conf']:.2f}")
        except Exception as e:
            import traceback
            report[name] = {"error": f"{type(e).__name__}: {e}"}
            print("FAILED:", e)
            traceback.print_exc()

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
