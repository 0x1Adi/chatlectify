from __future__ import annotations

import hashlib
import re

import numpy as np

from .schemas import Message

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_WS = re.compile(r"\s+")
_PROSE_PUNCT = re.compile(r"[.!?,:;]")


def _strip_indented_code(text: str) -> str:
    out = []
    for line in text.splitlines():
        if line.startswith("    ") and not _PROSE_PUNCT.search(line):
            continue
        out.append(line)
    return "\n".join(out)


def _clean_text(raw: str) -> str:
    t = _FENCE.sub("", raw)
    t = _strip_indented_code(t)
    t = _URL.sub("", t)
    t = _EMAIL.sub("", t)
    t = _WS.sub(" ", t).strip()
    return t


def _is_paste(text: str, wc: int, p95_wc: float) -> bool:
    if wc == 0:
        return False
    lines = text.count("\n") + 1
    total = len(text)
    if total == 0:
        return False
    alnum = sum(c.isalnum() or c.isspace() for c in text)
    alnum_ratio = alnum / total
    unique_ratio = len(set(text)) / total
    return any([wc > p95_wc, lines > 20, alnum_ratio < 0.6, unique_ratio < 0.15])


def clean(msgs: list[Message]) -> tuple[list[Message], list[Message], dict]:
    """Returns (clean_msgs, paste_msgs, stats)."""
    raw_n = len(msgs)
    cleaned: list[Message] = []
    for m in msgs:
        ct = _clean_text(m.text)
        assert len(ct) <= len(m.text), "clean grew text"
        wc = len(ct.split())
        if wc < 5:
            continue
        cleaned.append(m.model_copy(update={"text": ct, "word_count": wc}))
    post_clean = len(cleaned)

    # dedupe
    seen: set[str] = set()
    deduped: list[Message] = []
    for m in cleaned:
        key = hashlib.md5(m.text[:200].lower().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)

    # paste detection after dedupe
    wcs = np.array([m.word_count for m in deduped]) if deduped else np.array([0])
    p95 = float(np.percentile(wcs, 95)) if len(wcs) > 0 else 0.0
    pastes: list[Message] = []
    kept: list[Message] = []
    for m in deduped:
        if _is_paste(m.text, m.word_count, p95):
            pastes.append(m)
        else:
            kept.append(m)

    stats = {
        "raw": raw_n,
        "post_clean": post_clean,
        "deduped": len(deduped),
        "pastes_flagged": len(pastes),
        "kept": len(kept),
        "paste_contamination_pct": (len(pastes) / len(deduped)) if deduped else 0.0,
    }
    if raw_n and post_clean / raw_n < 0.3:
        raise RuntimeError(f"over-aggressive cleaning: {post_clean}/{raw_n}")
    return kept, pastes, stats
