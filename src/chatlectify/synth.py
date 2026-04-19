
import json
import os
import random
import re
from importlib.resources import files

from .schemas import Message, StyleFeatures

DEFAULT_MODELS = {"anthropic": "claude-sonnet-4-6", "openai": "gpt-4o-mini"}

_PROMPT = """You are building a writing-style skill file. Output ONLY a SKILL.md with YAML frontmatter.

<user_style_features>
{feats}
</user_style_features>

<user_exemplars>
{exemplars}
</user_exemplars>

<anti_exemplars>
{anti}
</anti_exemplars>

Produce SKILL.md with: (1) YAML frontmatter with name (snake_case) and description (<200 chars); (2) ## Style Rules (10-15 imperative rules from features); (3) ## Exemplars (5 best user messages verbatim); (4) ## Anti-patterns (do-NOT list derived from anti-exemplars contrast); (5) ## Quantified Targets (metric|target table). Output ONLY the SKILL.md. No preamble. No outer code fences.
"""

_NAME = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def stratified_exemplars(msgs: list[Message], n_short=10, n_mid=20, n_long=10) -> list[Message]:
    rng = random.Random(42)
    buckets = [
        ([m for m in msgs if m.word_count < 10], n_short),
        ([m for m in msgs if 10 <= m.word_count <= 50], n_mid),
        ([m for m in msgs if m.word_count > 50], n_long),
    ]
    out: list[Message] = []
    for pool, k in buckets:
        out.extend(pool if len(pool) <= k else rng.sample(pool, k))
    return out


def _anti(k=5) -> list[str]:
    try:
        text = files("chatlectify.assets").joinpath("antiexemplars.txt").read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()][:k]


def _feats_json(feats: StyleFeatures) -> str:
    d = feats.model_dump()
    d["top_unigrams"] = d["top_unigrams"][:30]
    d["top_bigrams"] = d["top_bigrams"][:20]
    d["top_sent_starters"] = d["top_sent_starters"][:10]
    return json.dumps(d, indent=2, default=str)


def _validate(md: str) -> list[str]:
    errs = []
    fm = _FM.match(md)
    if not fm:
        errs.append("missing frontmatter")
    else:
        name_m = re.search(r"^name:\s*(.+)$", fm.group(1), re.MULTILINE)
        if not name_m or not _NAME.match(name_m.group(1).strip()):
            errs.append("bad name")
        if not re.search(r"^description:\s*.+$", fm.group(1), re.MULTILINE):
            errs.append("no description")
    if len(re.findall(r"^## ", md, re.MULTILINE)) < 4:
        errs.append("too few sections")
    return errs


def _fallback(f: StyleFeatures, exemplars: list[Message]) -> str:
    ex = "\n\n".join(f"> {m.text}" for m in exemplars[:5])
    rules = [
        f"Keep average sentence length near {f.avg_sent_len:.0f} words.",
        f"Target type-token ratio ~{f.ttr:.2f}.",
        f"Use contractions at ~{f.contraction_rate:.1f}/100 words.",
        f"Ask questions at ~{f.question_rate:.0%} of sentences.",
        f"Use imperatives at ~{f.imperative_rate:.0%}.",
        f"Avoid emoji beyond {f.emoji_rate:.2f}/100 words.",
        f"Keep messages ~{f.avg_msg_words:.0f} words.",
        f"Use hedges sparingly (~{f.hedge_rate:.1f}/100 words).",
        f"Keep cap-starts at {f.cap_start_ratio:.0%}.",
        "Keep bullet/header usage consistent with source.",
    ]
    targets = [("avg_sent_len", f"{f.avg_sent_len:.1f}"), ("ttr", f"{f.ttr:.2f}"),
               ("contraction_rate", f"{f.contraction_rate:.2f}"),
               ("question_rate", f"{f.question_rate:.2%}"),
               ("imperative_rate", f"{f.imperative_rate:.2%}")]
    return ("---\nname: user_voice\n"
            f"description: Writing style derived from {f.msg_count} user messages.\n---\n\n"
            "## Style Rules\n\n" + "\n".join(f"- {r}" for r in rules) +
            f"\n\n## Exemplars\n\n{ex}\n\n"
            '## Anti-patterns\n\n- Do NOT open with "Certainly!" or "Great question!".\n'
            '- Do NOT summarize with "In conclusion" or "Overall".\n'
            "- Do NOT pad with filler hedges or corporate boilerplate.\n\n"
            "## Quantified Targets\n\n| metric | target |\n| --- | --- |\n" +
            "\n".join(f"| {k} | {v} |" for k, v in targets) + "\n")


def _call_llm(prompt: str, provider: str, model: str) -> str:
    if provider == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=model, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text
    if provider == "openai":
        import openai
        r = openai.OpenAI().chat.completions.create(
            model=model, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content
    raise ValueError(provider)


def synthesize(feats: StyleFeatures, msgs: list[Message], provider="anthropic",
               model: str | None = None, llm_fn=None) -> str:
    model = model or DEFAULT_MODELS[provider]
    exemplars = stratified_exemplars(msgs)
    prompt = _PROMPT.format(
        feats=_feats_json(feats),
        exemplars="\n\n".join(f"{i+1}. {m.text}" for i, m in enumerate(exemplars)),
        anti="\n\n".join(_anti()),
    )
    if len(prompt) // 4 > 50_000:
        raise RuntimeError("prompt >50k tokens")
    call = llm_fn or (lambda p: _call_llm(p, provider, model))
    try:
        out = call(prompt)
        if _validate(out):
            out = call(prompt + "\n\nPrevious attempt failed validation. Fix and retry.")
            if _validate(out):
                return _fallback(feats, exemplars)
        return out
    except Exception:
        if os.environ.get("CHATLECTIFY_STRICT"):
            raise
        return _fallback(feats, exemplars)
