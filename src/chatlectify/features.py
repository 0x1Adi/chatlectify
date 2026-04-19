from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from importlib.resources import files

import numpy as np
import regex as re2

from .schemas import Message, StyleFeatures

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_WORD_RE = re2.compile(r"\b\w+\b")
_EMOJI_RE = re2.compile(r"\p{Emoji_Presentation}|\p{Extended_Pictographic}")
_CONTRACTIONS = ["n't", "'re", "'s", "'ve", "'ll", "'d", "'m"]
_PUNCT_CHARS = [".", ",", "!", "?", ";", ":", "—", "-", "(", ")", '"', "'"]
_HEADER_RE = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+", re.MULTILINE)
_FENCE_RE = re.compile(r"```")
_CAP_PRONOUNS = {"I", "We", "You", "He", "She", "It", "They"}


@lru_cache(maxsize=1)
def _load_asset(name: str) -> set[str]:
    try:
        text = files("chatlectify.assets").joinpath(name).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        return set()
    return {w.strip().lower() for w in text.splitlines() if w.strip()}


def _sentences(text: str) -> list[str]:
    sents: list[str] = []
    for para in text.splitlines():
        para = para.strip()
        if not para:
            continue
        parts = _SENT_SPLIT.split(para)
        sents.extend(p.strip() for p in parts if p.strip())
    return sents


def extract(msgs: list[Message]) -> StyleFeatures:
    assert msgs, "need at least one message"
    texts = [m.text for m in msgs]
    full = "\n".join(texts)
    lower = full.lower()
    words = _WORD_RE.findall(lower)
    assert words, "no words found"
    total_words = len(words)
    total_chars = max(len(full), 1)

    # lexical
    unique = len(set(words))
    ttr = unique / total_words
    avg_word_len = float(np.mean([len(w) for w in words]))
    wc = Counter(words)
    top_unigrams = [(w, c) for w, c in wc.most_common(100)]
    bigrams = Counter(zip(words, words[1:]))
    top_bigrams = [(f"{a} {b}", c) for (a, b), c in bigrams.most_common(50)]
    contraction_hits = sum(lower.count(c) for c in _CONTRACTIONS)
    contraction_rate = 100.0 * contraction_hits / total_words

    # syntactic
    sents: list[str] = []
    for t in texts:
        sents.extend(_sentences(t))
    sent_word_counts = [len(_WORD_RE.findall(s)) for s in sents if s]
    sent_word_counts = [n for n in sent_word_counts if n > 0]
    if sent_word_counts:
        avg_sent_len = float(np.mean(sent_word_counts))
        sent_len_std = float(np.std(sent_word_counts))
    else:
        avg_sent_len = sent_len_std = 0.0
    punct_hist = {p: 1000.0 * full.count(p) / total_chars for p in _PUNCT_CHARS}
    cap_start = sum(1 for s in sents if s and s[0].isupper())
    cap_start_ratio = cap_start / max(len(sents), 1)

    # structural
    msg_count = len(msgs)
    bullet_rate = len(_BULLET_RE.findall(full)) / msg_count
    header_rate = len(_HEADER_RE.findall(full)) / msg_count
    fence_rate = len(_FENCE_RE.findall(full)) / (2 * msg_count)
    avg_line_breaks = float(np.mean([t.count("\n") for t in texts]))

    # markers
    starter_counter: Counter = Counter()
    for s in sents:
        tok = _WORD_RE.findall(s.lower())
        if tok:
            starter_counter[tok[0]] += 1
    top_sent_starters = [(w, c) for w, c in starter_counter.most_common(20)]

    hedges = _load_asset("hedges.txt")
    hedge_hits = 0
    for h in hedges:
        hedge_hits += len(re.findall(rf"\b{re.escape(h)}\b", lower))
    hedge_rate = 100.0 * hedge_hits / total_words

    emoji_hits = len(_EMOJI_RE.findall(full))
    emoji_rate = 100.0 * emoji_hits / total_words

    dict_words = _load_asset("dict_words.txt")
    if dict_words:
        checkable = [w for w in words if len(w) >= 3 and not any(c.isdigit() for c in w)]
        unknown = sum(1 for w in checkable if w not in dict_words)
        typo_rate = unknown / max(len(checkable), 1)
    else:
        typo_rate = 0.0

    verbs = _load_asset("verbs_top500.txt")
    imp_count = 0
    for s in sents:
        toks_raw = s.split()
        if not toks_raw:
            continue
        first_raw = toks_raw[0].strip(".,!?;:")
        first_low = first_raw.lower()
        if first_low in verbs and first_raw not in _CAP_PRONOUNS:
            imp_count += 1
    imperative_rate = imp_count / max(len(sents), 1)
    question_rate = sum(1 for s in sents if s.endswith("?")) / max(len(sents), 1)

    msg_words = [m.word_count for m in msgs]
    avg_msg_words = float(np.mean(msg_words))
    avg_msg_words_std = float(np.std(msg_words))

    feats = StyleFeatures(
        ttr=ttr,
        avg_word_len=avg_word_len,
        top_unigrams=top_unigrams,
        top_bigrams=top_bigrams,
        contraction_rate=contraction_rate,
        avg_sent_len=avg_sent_len,
        sent_len_std=sent_len_std,
        punct_hist=punct_hist,
        cap_start_ratio=cap_start_ratio,
        bullet_rate=bullet_rate,
        header_rate=header_rate,
        fence_rate=fence_rate,
        avg_line_breaks=avg_line_breaks,
        top_sent_starters=top_sent_starters,
        hedge_rate=hedge_rate,
        emoji_rate=emoji_rate,
        typo_rate=typo_rate,
        imperative_rate=imperative_rate,
        question_rate=question_rate,
        avg_msg_words=avg_msg_words,
        avg_msg_words_std=avg_msg_words_std,
        msg_count=msg_count,
    )
    # invariants
    for v in (
        feats.ttr,
        feats.contraction_rate,
        feats.hedge_rate,
        feats.emoji_rate,
        feats.typo_rate,
        feats.imperative_rate,
        feats.question_rate,
        feats.avg_word_len,
        feats.avg_sent_len,
    ):
        assert np.isfinite(v), f"non-finite feature: {v}"
    assert 0.0 <= feats.ttr <= 1.0
    assert 0.0 <= feats.typo_rate <= 1.0
    assert feats.top_unigrams, "empty unigrams"
    return feats
