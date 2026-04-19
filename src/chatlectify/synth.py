from __future__ import annotations

import json
import os
import random
import re
from importlib.resources import files

from .schemas import Message, StyleFeatures

DEFAULT_MODELS = {"anthropic": "claude-sonnet-4-6", "openai": "gpt-4o-mini"}

_PROMPT_TMPL = """You are building a writing-style skill file. Output ONLY a SKILL.md with YAML frontmatter.

<user_style_features>
{features_json}
</user_style_features>

<user_exemplars>
{exemplars}
</user_exemplars>

<anti_exemplars>
{anti}
</anti_exemplars>

Produce SKILL.md with:
1. YAML frontmatter: name (snake_case), description (1 sentence, <200 chars)
2. ## Style Rules — 10-15 imperative rules derived from features (concrete, measurable)
3. ## Exemplars — 5 best user messages verbatim
4. ## Anti-patterns — bullet list of "do NOT" derived from anti-exemplars contrast
5. ## Quantified Targets — table: metric | target value

Output ONLY the SKILL.md content. No preamble. No code fences around the whole document.
"""

_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_SECTION_RE = re.compile(r"^## ", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def stratified_exemplars(msgs: list[Message], n_short=10, n_mid=20, n_long=10) -> list[Message]:
    rng = random.Random(42)
    short = [m for m in msgs if m.word_count < 10]
    mid = [m for m in msgs if 10 <= m.word_count <= 50]
    long = [m for m in msgs if m.word_count > 50]

    def pick(pool, k):
        if not pool:
            return []
        if len(pool) <= k:
            return list(pool)
        return rng.sample(pool, k)

    return pick(short, n_short) + pick(mid, n_mid) + pick(long, n_long)


def _load_antiexemplars(k=5) -> list[str]:
    try:
        text = files("chatlectify.assets").joinpath("antiexemplars.txt").read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[:k]


def _features_json_truncated(feats: StyleFeatures) -> str:
    d = feats.model_dump()
    d["top_unigrams"] = d["top_unigrams"][:30]
    d["top_bigrams"] = d["top_bigrams"][:20]
    d["top_sent_starters"] = d["top_sent_starters"][:10]
    return json.dumps(d, indent=2, default=str)


def _validate(skill_md: str) -> list[str]:
    errs = []
    fm = _FRONTMATTER_RE.match(skill_md)
    if not fm:
        errs.append("missing YAML frontmatter")
    else:
        fm_text = fm.group(1)
        name_match = re.search(r"^name:\s*(.+)$", fm_text, re.MULTILINE)
        if not name_match or not _NAME_RE.match(name_match.group(1).strip()):
            errs.append("invalid name in frontmatter")
        if not re.search(r"^description:\s*.+$", fm_text, re.MULTILINE):
            errs.append("missing description")
    sections = _SECTION_RE.findall(skill_md)
    if len(sections) < 4:
        errs.append(f"expected ≥4 sections, got {len(sections)}")
    return errs


def _fallback(feats: StyleFeatures, exemplars: list[Message]) -> str:
    picks = exemplars[:5]
    examples = "\n\n".join(f"> {m.text}" for m in picks)
    return f"""---
name: user_voice
description: Writing style derived from {feats.msg_count} user messages.
---

## Style Rules

- Keep average sentence length near {feats.avg_sent_len:.0f} words.
- Target TTR of {feats.ttr:.2f}.
- Use contractions at ~{feats.contraction_rate:.1f} per 100 words.
- Open questions at ~{feats.question_rate:.0%} of sentences.
- Favor imperative phrasing at ~{feats.imperative_rate:.0%}.
- Keep bullets rare unless needed.
- Avoid emoji beyond {feats.emoji_rate:.2f}/100 words.
- Prefer short messages (~{feats.avg_msg_words:.0f} words).
- Use hedges sparingly (~{feats.hedge_rate:.1f}/100 words).
- Keep capitalization consistent with {feats.cap_start_ratio:.0%} cap starts.

## Exemplars

{examples}

## Anti-patterns

- Do NOT open with "Certainly!" or "Great question!".
- Do NOT summarize with "In conclusion" or "Overall".
- Do NOT pad with filler hedges.
- Do NOT use corporate boilerplate.

## Quantified Targets

| metric | target |
| --- | --- |
| avg_sent_len | {feats.avg_sent_len:.1f} |
| ttr | {feats.ttr:.2f} |
| contraction_rate | {feats.contraction_rate:.2f} |
| question_rate | {feats.question_rate:.2%} |
| imperative_rate | {feats.imperative_rate:.2%} |
"""


def _call_llm(prompt: str, provider: str, model: str) -> str:
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        r = client.messages.create(
            model=model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text
    elif provider == "openai":
        import openai
        client = openai.OpenAI()
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )
        return r.choices[0].message.content
    raise ValueError(f"unknown provider: {provider}")


def synthesize(
    feats: StyleFeatures,
    msgs: list[Message],
    provider: str = "anthropic",
    model: str | None = None,
    llm_fn=None,
) -> str:
    model = model or DEFAULT_MODELS[provider]
    exemplars = stratified_exemplars(msgs)
    ex_text = "\n\n".join(f"{i+1}. {m.text}" for i, m in enumerate(exemplars))
    anti = "\n\n".join(_load_antiexemplars())
    prompt = _PROMPT_TMPL.format(
        features_json=_features_json_truncated(feats),
        exemplars=ex_text,
        anti=anti,
    )
    # cost gate
    tok_est = len(prompt) // 4
    if tok_est > 50_000:
        raise RuntimeError(f"input tokens ~{tok_est} > 50k")

    call = llm_fn or (lambda p: _call_llm(p, provider, model))
    try:
        out = call(prompt)
        errs = _validate(out)
        if errs:
            out = call(prompt + f"\n\nPrevious attempt failed: {errs}. Fix and retry.")
            if _validate(out):
                return _fallback(feats, exemplars)
        return out
    except Exception:
        if os.environ.get("CHATLECTIFY_STRICT"):
            raise
        return _fallback(feats, exemplars)
