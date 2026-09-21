"""Built-in 话术 presets — one label + one instruction per tone.

These are what the generation prompt asks the model to write in; the user picks up to
MAX_SLOTS of them from the HUD. Keeping them here (rather than inline in the prompt) means
one place to add a tone, and lets the parser strip any label the model echoes back.

Each `prompt` is written as a direct instruction to the model: say what the tone IS, and
what it must not become. Vague one-word tones ("幽默") produce generic replies — the
useful part is the constraint.
"""

from __future__ import annotations

import re

# How many candidates one generation call can produce. The HUD shows exactly this many
# dropdowns; the panel's candidate area is built for this many rows.
MAX_SLOTS = 3

# label -> instruction. Order here is the order shown in the dropdowns.
PRESETS: dict[str, str] = {
    "高情商话术": (
        "高情商：先接住对方的情感和诉求，再把难处说成客观情况而不是你的态度；"
        "拒绝也要给出替代方案和一个明确的下一步，给对方留台阶。不谄媚、不硬扛。"
    ),
    "贴吧老哥 v1.0": (
        "贴吧老哥：像贴吧/论坛里说话，口语化、可以自嘲和玩梗、有话直说不装，"
        "带一点痞气但不冒犯。禁止书面语和客套话，要有网感。"
    ),
    "拒绝加班": (
        "拒绝加班：明确说今天做不完、时间上排不进去，语气平和但不留继续压的空间；"
        "必须给一个可接受的替代（比如明早第一时间、或一个具体时间点）。不过度道歉。"
    ),
    "卑微乙方": (
        "卑微乙方：姿态放到极低，极度客气、随叫随到、把问题都算在自己头上，"
        "像随时怕甲方不高兴的乙方。适度夸张，让人一看就懂这个梗，但话本身仍然能直接发出去。"
    ),
    "稳如老狗": (
        "沉稳：不解释、不辩解、不铺垫，只给事实、结论和一个明确时间点；"
        "语气平静，让对方觉得事情稳了。"
    ),
    "已读乱回": (
        "敷衍但不失礼：用最短的话把对方接住，不承诺任何事、不展开细节，"
        "让对方觉得你回了、但又没法接着追问。"
    ),
    "职场黑话": (
        "职场黑话：用互联网黑话把简单的事说得很专业（对齐、抓手、闭环、颗粒度、"
        "拉通、复盘、赋能、沉淀），但整句要能看懂，不要堆砌到不知所云。"
    ),
    "阴阳怪气": (
        "阴阳怪气：表面礼貌客气，实际带着软刺和反问，让对方不好发作又不能说你没礼貌。"
        "这是高风险选项，克制一点，别变成直接骂人。"
    ),
    "理科直男": (
        "理科直男：只回答被问到的问题本身，零寒暄、零情绪、零修饰，"
        "短到不能再短，像一个不太会说话但很靠谱的工程师。"
    ),
}

# What the panel starts with: two tones, not three — a third slot defaults to 不用.
DEFAULT_SLOTS: list[str] = ["高情商话术", "贴吧老哥 v1.0"]
NONE_LABEL = "不用"          # the third dropdown's way of saying "only two candidates"

def _label_alternation() -> str:
    """The labels as one regex alternative, with spaces made optional.

    "贴吧老哥 v1.0" is written with a space in the dropdown but the model may echo it
    without one ("贴吧老哥v1.0：") — matching the space loosely costs nothing and avoids a
    label that leaks through only sometimes.
    """
    escaped = (re.escape(k).replace(r"\ ", r"\s*") for k in sorted(PRESETS, key=len, reverse=True))
    return "|".join(escaped)


_LABEL_RE = re.compile(
    rf"^[*_#\s]*(?:{_label_alternation()})[^，。！？；、,.!?;：:]{{0,4}}[*_#\s]*[:：]\s*")


def labels() -> list[str]:
    """All preset labels, in dropdown order."""
    return list(PRESETS)


def resolve(ids: list[str]) -> list[str]:
    """Turn dropdown selections into a usable list: known labels only, no duplicates.

    Duplicates matter because the three dropdowns are independent — picking the same tone
    twice would otherwise ask the model for two replies in one voice.
    """
    out: list[str] = []
    for i in ids:
        if i in PRESETS and i not in out:
            out.append(i)
    return out


def prompt_block(ids: list[str]) -> str:
    """The numbered style list that goes into the generation prompt."""
    return "\n".join(f"{n}. {label}：{PRESETS[label]}" for n, label in enumerate(ids, 1))


def strip_label(line: str) -> str:
    """Remove a leading preset label the model echoed back, e.g. "贴吧老哥 v1.0：好的哥".

    The model is told not to label its lines, and usually complies — but the prompt itself
    shows it these labels, so now and then it echoes one. Because the labels are data, they
    are stripped from the same place they are defined: adding a tone here cannot silently
    break the parser the way a hardcoded list would.
    """
    return _LABEL_RE.sub("", line)
