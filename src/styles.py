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

import userconfig

# How many candidates one generation call can produce. The HUD shows exactly this many
# dropdowns; the panel's candidate area is built for this many rows.
MAX_SLOTS = 3

# Candidates per tone. Local Ollama serves one request at a time by default, so one concise
# candidate per selected voice keeps the panel responsive while still giving distinct
# choices across slots.
PER_TONE = 1

# Careful, context-sensitive voices need less sampling variance than the playful presets.
# Unlisted tones retain Generator's established 0.9 default.
TONE_TEMPERATURES: dict[str, float] = {
    "高情商幽默朋友": 0.35,
}

INTENT_REPLY_GUIDANCE: dict[str, str] = {
    "派活": "从我的角度回应是否能接；没有明确信息就说需要先确认，不能直接承诺。",
    "催进度": "先接住对方在等我，再只说上下文确认过的进度或我现在会确认什么。",
    "问进度": "直接回答我掌握的进展；没有进展信息就坦白需要确认，不编状态。",
    "批评": "回应对方对我的不满，承认已知问题但不虚构过错，也不反过来责怪对方。",
    "要解释": "以我的立场说明上下文已有的原因；原因未知就明确说需要确认。",
    "闲聊": "只接对方最新分享，前文仅用于理解关系，不能抢答已经过去的话题。",
    "约会议": "从我的日程和立场回应能否参加；未知时只问一个必要的时间或议题信息。",
    "夸奖": "这是对方在夸我；由我接住称赞和谢意，不能反过来夸成对方被表扬。",
    "求助帮忙": "这是对方在请我帮忙；从我的能力和边界回应能否帮，不能替对方回答。",
    "征求建议": "这是对方在问我的看法；直接给我的判断和理由，把选择留给对方。",
    "倾诉求安慰": (
        "这是对方在讲自己的感受；优先回应最新一句里的对方，"
        "用「你」指向对方，不能把对方的情绪写成我的。"
    ),
    "关心问候": (
        "这是对方在问我的近况；第一小句必须直接回答我现在怎么样，"
        "不能先去安慰前文里的对方，也不能只反问对方。"
        "上下文没写明我的当前状态时，只能用「我还好」「还行」这类中性回答，"
        "不得补充我刚做完什么、正在忙什么、睡觉或后续安排，"
        "也不新增约饭、见面或稍后再聊。"
    ),
    "朋友邀约": "从我的角度先回应想不想去或是否方便；信息不足再问一个必要细节。",
    "玩笑调侃": "我是被对方调侃或接梗的人；从我的视角回应，笑点不转成攻击对方。",
    "道歉和解": "这是对方向我道歉；由我回应是否收到和真实感受，不能替对方接受道歉。",
    "感谢": "这是对方在感谢我；由我自然接住谢意，不能改成我向对方道谢。",
}

TONE_INTENT_GUIDANCE: dict[str, dict[str, str]] = {
    "高情商幽默朋友": {
        "派活": (
            "先明确我是否能接；上下文没有确认时，只说明需要先看一下，"
            "不能擅自答应完成时间，也不能把任务推回给对方。"
        ),
        "催进度": (
            "先承认让对方久等了，再说上下文已经确认的状态或时间；"
            "没有可靠进展就说我现在确认，不能反过来嫌对方催，禁止说「别催」。"
        ),
        "问进度": (
            "直接从我的角度说已确认的进展；没有进展信息就坦白正在确认，"
            "不虚构完成比例、卡点或时间。"
        ),
        "批评": (
            "第一小句必须以「是我……」开头，复述上下文里已经发生的行为；"
            "第二小句承认这会让对方不舒服；最后只表达「我现在认真听你说」这一类当下关注，"
            "不描述未经上下文确认的动作。不要用问句，不解释原因，不谈以后。"
        ),
        "要解释": "只解释上下文已有的原因；原因不清楚就坦白并问清，不编理由。",
        "闲聊": (
            "紧跟对方当下分享的事实和情绪，笑点停在已经发生的事情上；"
            "明显是成果时，把肯定落在结果和坚持上。回复到此结束，"
            "不提出任何后续动作、请求或新活动。"
        ),
        "夸奖": "自然接住称赞，别过度谦虚；可以用轻自嘲让气氛更亲近。",
        "求助帮忙": (
            "先按上下文明确能帮、不能帮或需要确认什么；"
            "未给出的口味、时间、地点等细节不能猜，也不能只开玩笑。"
        ),
        "征求建议": "先给明确看法和一个理由，再把决定权留给对方；幽默只碰选择困境。",
        "倾诉求安慰": (
            "点出这件事具体难在哪里并给陪伴；只描述对方的感受，"
            "禁止说「我也累」「我也难受」或把对方经历写成我的，"
            "除非上下文明确我有相同经历；情绪明显时不要加笑点或给对方贴标签。"
        ),
        "关心问候": "先如实回应自己的近况，再自然关心回去；不夸大、不卖惨。",
        "朋友邀约": (
            "根据上下文先说能不能去，只向对方问时间或地点中的一个；"
            "如果两者都未知就优先只问时间，不能自己给出具体日期、时段或地点，也不新增其他安排。"
        ),
        "玩笑调侃": "顺着对方原话轻轻接梗，笑点落在处境或自己，不嘲笑对方。",
        "道歉和解": (
            "第一小句明确说收到了对方的道歉或诚意；第二小句保留刚才的真实感受；"
            "最后邀请现在继续沟通。不能直接宣布事情已经过去，也不自行结束、推迟或改约。"
        ),
        "感谢": "自然接住谢意，回应彼此关系，不用客套话；可以轻松地说这是朋友该做的。",
    },
}

# label -> instruction. Order here is the order shown in the dropdowns.
#
# Each entry is written as a *persona plus its verbal tics*, not as a description of a mood.
# "语气放松、带一点幽默" gives the model nothing to hold on to and every tone drifts toward
# the same bland helpfulness; naming who is talking and which words they reach for is what
# actually separates the voices. The trailing constraint matters as much as the rest: a tone
# with no ceiling slides back into generic politeness by the second line.
BUILTIN: dict[str, str] = {
    "高情商话术": (
        "像公司里那个谁都说好的老同事：先接住对方情绪（「我理解」「确实」），再说事实和下一步，"
        "拒绝也带替代方案加一个具体时间点。不说教、不绕圈子、句尾不堆「呢/哦/啦」。"
    ),
    "暖心阳光男孩": (
        "像真诚、温暖又有分寸的阳光男生：先用一句自然的关心接住情绪，再给积极但不空洞的回应，"
        "能帮忙时顺手给一个具体行动或陪伴。口语轻松、可靠，偶尔带一点小幽默；"
        "不油腻、不暧昧、不爹味，不强行灌鸡汤，也不堆感叹号和表情。"
    ),
    "幽默男孩": (
        "像反应快、会接梗但有分寸的男生：先回应正事，再用轻巧反转、生活化比喻或适度自嘲增添幽默，"
        "整句话仍然自然、能直接发送，不能为了搞笑回避问题。"
        "不硬造网络梗、不嘲笑对方，不冒犯、不油腻暧昧、不讲低俗玩笑；"
        "对方严肃、难过或正在冲突时收住玩笑，优先认真回应。"
    ),
    "高情商幽默朋友": (
        "像情商高、会接梗的熟朋友，不像客服或段子手。先贴着原话处理对方最重要的感受或诉求，"
        "给清楚态度；只有气氛轻松时，末尾才加半句幽默。幽默只用消息里已经出现的词和处境做轻反差、"
        "温和自嘲或顺梗，笑点不指向对方。只用上下文事实，不新增原因、安排、细节、共同经历、"
        "昵称或承诺；微信口语，一句能直接发送。"
    ),
    "贴吧老哥 v1.0": (
        "贴吧老哥：一口网感口语，「有一说一」「绷不住了」「搁这」「这就去整」随手就来，"
        "自称我、管对方叫「哥/兄弟」，可以自嘲玩梗甚至摆烂，但不骂人。"
        "禁止「您好」「感谢」这类书面客套。"
    ),
    "拒绝加班": (
        "态度平和但把话说死：明确今天做不完，**不给**「我尽量」「看情况」这种会被继续压的口子；"
        "必须给一个具体替代时间（比如「明早九点前」），并说清不用等今晚。"
        "道歉不超过一句，理由不超过一句。"
    ),
    "卑微乙方": (
        "极度卑微的乙方：「好的好的」「收到收到」「实在抱歉」「麻烦您了」张口就来，全程称「您」，"
        "任何问题先认在自己头上，随叫随到。夸张到一眼看出是梗，但整句仍然能直接发出去。"
    ),
    "稳如老狗": (
        "十年老工程师那种稳：不解释、不铺垫、不道歉，只给结论加一个时间点，句子短、"
        "主语是事不是情绪（「三点前给你」「已确认，没问题」），让对方觉得事情已经稳了。"
    ),
    "已读乱回": (
        "敷衍但不失礼：一到六个字把对方接住（「在忙，你说」「嗯嗯」「好」），"
        "不承诺、不展开、不给时间点，让对方觉得回了又没法接着追问。"
    ),
    "鱼塘主": (
        "海王海后式回消息：我是塘主，对方只是鱼塘里的一条鱼。先推后拉——先淡淡降一句、"
        "再轻轻给个甜头；惜字如金，不解释、不道歉、不讨好；事情不说死、留点悬念，"
        "收尾自带先撤感（「先这样」）。嘴甜心硬，不主动不拒绝不负责——不揽活、不否认、不背锅。"
        "分寸在高冷从容，不油腻、不暧昧，不是撩。"
    ),
    "职场黑话": (
        "把简单的事说得很专业：对齐、抓手、闭环、颗粒度、拉通、复盘、赋能、沉淀、打法轮着用，"
        "一句话里至少两个；但整句要能看懂，不要堆到不知所云。"
    ),
    "阴阳怪气": (
        "表面客气、话里带刺：多用「哦」「呢」「那就」「辛苦你了」配反问或夸张的客气，"
        "让对方不好发作又不能说你没礼貌。不要升级成直接骂人或人身攻击。"
    ),
    "理科直男": (
        "只回答被问到的：零寒暄、零情绪、零修饰、零表情，能两个字说清就不用五个字，"
        "像一个不太会说话但很靠谱的工程师。不做任何延伸，也不表示关心。"
    ),
}

# What the panel starts with: two useful voices, one candidate each.
DEFAULT_SLOTS: list[str] = ["高情商话术", "暖心阳光男孩"]
NONE_LABEL = "不用"          # the third dropdown's way of saying "only two candidates"

CUSTOM_VAR = "JEV_TONES"     # env var holding user-defined tones


def _custom_tones() -> dict[str, str]:
    """Tones the user defined in their env file, as `名字=说明` entries separated by `|`.

        export JEV_TONES="摸鱼大师=像个资深摸鱼选手，把活推得很得体|孙子兵法=用兵法比喻说话"

    A same-named entry overrides the built-in one, so the shipped wording can be tuned
    without touching this file. A tone called 不用 is dropped: that label is the panel's
    sentinel for "this slot is switched off", and letting a tone shadow it would make a
    slot impossible to switch off.
    """
    raw = userconfig.get(CUSTOM_VAR)
    out: dict[str, str] = {}
    for part in (raw or "").split("|"):
        name, sep, desc = part.partition("=")
        name, desc = name.strip(), desc.strip()
        if sep and name and desc and name != NONE_LABEL:
            out[name] = desc
    return out


CUSTOM: dict[str, str] = _custom_tones()
PRESETS: dict[str, str] = {**BUILTIN, **CUSTOM}


def guidance_for(tone: str, intent: str = "") -> str:
    """Combine the universal reply direction with the selected tone's finer rule."""
    parts = [
        INTENT_REPLY_GUIDANCE.get(intent, ""),
        TONE_INTENT_GUIDANCE.get(tone, {}).get(intent, ""),
    ]
    return "；".join(part for part in parts if part)


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


def strip_label(line: str) -> str:
    """Remove a leading preset label the model echoed back, e.g. "贴吧老哥 v1.0：好的哥".

    The model is told not to label its lines, and usually complies — but the prompt itself
    shows it these labels, so now and then it echoes one. Because the labels are data, they
    are stripped from the same place they are defined: adding a tone here cannot silently
    break the parser the way a hardcoded list would.
    """
    return _LABEL_RE.sub("", line)
