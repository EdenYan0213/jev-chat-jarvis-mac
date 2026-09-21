"""Candidate reply generation via a fast Chinese LLM API.

Why an API instead of a local model: a 3B local model costs ~6 GB of disk, ~2-3 s per
generation on MPS and writes noticeably worse Chinese than the hosted fast tier. The
judgment half stays local (decider-2b, ~0.5 s, no network) — only reply writing goes out.

Two API shapes are supported, because providers disagree:
    openai     POST {base}/v1/chat/completions   Authorization: Bearer   -> choices[0].message.content
    anthropic  POST {base}/v1/messages           x-api-key + version    -> content[].text
推 most providers (DeepSeek, 通义, Moonshot, SiliconFlow, Ollama, vLLM, OpenRouter) only
speak the OpenAI shape; 智谱 and a few gateways offer both. The shape is inferred.
one, otherwise it is inferred from the base URL (a path containing "anthropic" => anthropic).

Nothing is ever written back, and the key is never logged. Run
`uv run python src/generate.py --check` to see which source is in use (key masked).

Privacy: the boss's message text is sent to the provider. That is the one place this app
leaves the machine — swap in a local model if that matters more than reply quality.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from pathlib import Path

import userconfig
import styles

DEFAULT_MODEL = "glm-4-flash"
# any Anthropic-compatible /v1/messages endpoint works; this one is a cheap, fast
# Chinese-native option and is what the project was tested against
DEFAULT_BASE = "https://open.bigmodel.cn/api/anthropic"
DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_BASE = "https://api.anthropic.com"
MISSING_HINT = ("未配置生成层 Key：候选回复需要它，判断/风险不需要。"
                "设置 OPENAI_API_KEY（或 ANTHROPIC_API_KEY）后重启，见 README 配置章节。")

# {n} appears twice on purpose: the style list and the "exactly n lines" demand have to agree,
# or the model pads the answer with a line of its own.
PROMPT = """刚收到一条微信消息，你要帮我回。

消息：「{message}」
{intent_line}
请写 {n} 条风格不同的回复候选：
{styles}

硬性要求：
- 每条不超过 30 个字，是直接能发出去的口吻，不要客套话、不要解释
- 只说人话，像微信里打字的语气
- 只输出 {n} 行，每行一条，不要编号、不要引号、不要任何前后缀
- 不要写出风格名（不要写「贴吧老哥 v1.0：」这类前缀），直接从回复内容开始"""


# The model is told not to label its lines, and usually complies — but "usually" is exactly
# why these exist. Seen for real: "轻松型：" (the intended echo), "轻松的回复：", "轻松版：",
# "轻松一点：", "**轻松型**：", and behind numbering ("1. 轻松型：" — the numbering is
# stripped first, leaving the bare label). So: a style word, then up to a few characters of
# filler that may not contain sentence punctuation, then the colon.
_STYLE_LABEL = re.compile(
    r"^[*_#\s]*(稳妥|轻松|简短|简洁)[^，。！？；、,.!?;：:]{0,5}[*_#\s]*[:：]\s*")
# One-character style words match only when the colon follows almost immediately: a loose
# filler here would eat a legitimate reply like "简单说：我先确认一下".
_STYLE_LABEL_SHORT = re.compile(r"^[*_#\s]*(简|稳|轻)\s*(型|洁)?[*_#\s]*[:：]\s*")
# wrapping quotes, straight or CJK — applied before the label strip, and again after it,
# because either order can expose the other ("「轻松型：xxx」")
_QUOTES = re.compile(r"""^["“”「『'‘]+|["”」』'’]+$""")


def _strip_style_label(s: str) -> str:
    s = _STYLE_LABEL.sub("", s)
    s = _STYLE_LABEL_SHORT.sub("", s)
    return styles.strip_label(s)


def _strip_quotes(s: str) -> str:
    return _QUOTES.sub("", s)


def _endpoint(base: str, api: str) -> str:
    """Compose the request URL, tolerating both base-URL conventions.

    Providers disagree about whether the version segment belongs to the base:
        https://api.deepseek.com             -> /v1/chat/completions
        https://api.deepseek.com/v1          -> /v1/chat/completions
        https://open.bigmodel.cn/api/paas/v4 -> /v4/chat/completions
    So: if the base already ends in a version segment, append only the path.
    """
    b = (base or "").rstrip("/")
    last = b.rsplit("/", 1)[-1].lower()
    has_version = bool(re.fullmatch(r"v\d+[a-z]*", last))
    if api == "anthropic":
        return b + ("/messages" if has_version else "/v1/messages")
    return b + ("/chat/completions" if has_version else "/v1/chat/completions")


def pick_api_format(base: str, configured: str | None) -> str:
    """Explicit setting wins; otherwise infer from the URL.

    A base path containing "anthropic" means the Anthropic shape. Every other endpoint is
    assumed OpenAI-shaped, which is what most providers and local servers expose.
    """
    if configured:
        c = configured.strip().lower()
        if c in ("openai", "anthropic"):
            return c
    return "anthropic" if "anthropic" in (base or "").lower() else "openai"


def load_credentials() -> tuple[str, str, str, str, str]:
    """Returns (base_url, api_key, model, source, api_format). Never raises.

    Names are the conventional ones (src/userconfig.py), so whatever you already export
    for other tools works here:
        OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL         the common case
        ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL / ANTHROPIC_MODEL
    The API shape is inferred from the endpoint (a path containing "anthropic" means the
    Anthropic /v1/messages format; everything else is assumed OpenAI-shaped).
    """
    oai = userconfig.provider("OPENAI")
    anth = userconfig.provider("ANTHROPIC")

    if oai["key"]:
        base = oai["base"] or DEFAULT_OPENAI_BASE
        return base, oai["key"], oai["model"] or DEFAULT_MODEL, oai["source"], pick_api_format(base, None)
    if anth["key"]:
        base = anth["base"] or DEFAULT_ANTHROPIC_BASE
        return base, anth["key"], anth["model"] or DEFAULT_MODEL, anth["source"], pick_api_format(base, None)

    base = oai["base"] or anth["base"] or DEFAULT_OPENAI_BASE
    model = oai["model"] or anth["model"] or DEFAULT_MODEL
    return base, "", model, "none", pick_api_format(base, None)


def credential_status() -> str:
    """Human-readable state for --check; the key itself is never printed."""
    base, key, model, source, api = load_credentials()
    shape = ("Anthropic 格式 /v1/messages" if api == "anthropic"
             else "OpenAI 格式 /v1/chat/completions")
    home = str(Path.home())
    if not key:
        return (f"❌ 未配置 API Key\n"
                f"   端点: {base}  ({shape})\n"
                f"   模型: {model}\n"
                f"   {MISSING_HINT}")
    return (f"✅ 凭据来源: {source.replace(home, '~')}\n"
            f"   端点: {base}\n"
            f"   接口: {shape}\n"
            f"   模型: {model}\n"
            f"   Key : {key[:6]}…{key[-4:]}  ({len(key)} chars)")


class Generator:
    def __init__(self, model: str | None = None, timeout: int = 30,
                 api: str | None = None):
        self.model_override = model
        self.api_override = api if api in ("openai", "anthropic") else None
        self.timeout = timeout
        self._creds: tuple[str, str, str] | None = None
        self._last_url = ""

    def _creds_or_load(self):
        if self._creds is None:
            base, key, model, _src, _api = load_credentials()
            self._creds = (base, key, self.model_override or model)
        return self._creds

    def _call(self, prompt: str) -> str:
        base, key, model, _src, api = load_credentials()
        # the constructor's overrides win — without this the `model` argument was accepted
        # and silently ignored, so the request went out with whatever the config named
        if self.model_override:
            model = self.model_override
        if self.api_override:
            api = self.api_override
        if api == "anthropic":
            url = _endpoint(base, "anthropic")
            body = {"model": model, "max_tokens": 300, "temperature": 0.9,
                    "messages": [{"role": "user", "content": prompt}]}
            headers = {"content-type": "application/json", "x-api-key": key,
                       "anthropic-version": "2023-06-01"}
            data = self._post(url, headers, body)
            parts = data.get("content") or []
            return "".join(p.get("text", "") for p in parts if isinstance(p, dict))

        url = _endpoint(base, "openai")
        body = {"model": model, "max_tokens": 300, "temperature": 0.9,
                "messages": [{"role": "user", "content": prompt}]}
        headers = {"content-type": "application/json", "authorization": f"Bearer {key}"}
        data = self._post(url, headers, body)
        choices = data.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content") or ""

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        self._last_url = url
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    @staticmethod
    def _parse(raw: str) -> list[str]:
        out = []
        for line in raw.splitlines():
            s = line.strip()
            if not s:
                continue
            s = re.sub(r"^[\d]+[.、)．]\s*", "", s)   # "1." / "2、" numbering
            s = _strip_quotes(s)
            s = _strip_style_label(s)
            s = _strip_quotes(s)                     # quotes the label removal exposed
            if s:
                out.append(s.strip())
        return out

    def generate(self, message: str, intent: str = "",
                 tones: list[str] | None = None) -> dict:
        """intent is optional so the caller can start this in parallel with judging.

        `tones` is the 话术 the user picked in the panel; one candidate comes back per tone,
        so the number of candidates follows the selection rather than a fixed 3.
        """
        ids = styles.resolve(tones if tones is not None else styles.DEFAULT_SLOTS)
        if not ids:
            return {"candidates": [], "error": "没有选择任何话术", "elapsed_s": 0.0}
        n = len(ids)
        intent_line = f"判断出的意图：{intent}\n" if intent else ""
        prompt = PROMPT.format(message=message, intent_line=intent_line, n=n,
                               styles=styles.prompt_block(ids))
        t0 = time.perf_counter()
        if not self._creds_or_load()[1]:
            return {"candidates": [], "error": MISSING_HINT, "elapsed_s": 0.0}
        try:
            raw = self._call(prompt)
            texts = self._parse(raw)[:n]
        except urllib.error.HTTPError as e:
            detail = e.read()[:160].decode(errors="replace")
            return {"candidates": [],
                    "error": f"HTTP {e.code} @ {self._last_url} — {detail}",
                    "elapsed_s": time.perf_counter() - t0}
        except Exception as e:
            return {"candidates": [], "error": f"{type(e).__name__}: {e}",
                    "elapsed_s": time.perf_counter() - t0}

        seen, uniq = set(), []
        for t in texts:
            if t not in seen:
                seen.add(t)
                uniq.append({"text": t})
        base, key, model = self._creds_or_load()  # noqa: F841
        return {"candidates": uniq, "model": model,
                "elapsed_s": time.perf_counter() - t0}


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        print(credential_status())
        raise SystemExit(0 if load_credentials()[1] else 1)

    g = Generator()
    msg = sys.argv[1] if len(sys.argv) > 1 else "这个需求你今天跟一下"
    intent = sys.argv[2] if len(sys.argv) > 2 else "派活"
    print(json.dumps(g.generate(msg, intent), ensure_ascii=False, indent=1))
