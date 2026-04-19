
import re
from collections import Counter
from functools import lru_cache
from importlib.resources import files

import numpy as np
import regex as re2

from .schemas import Message, StyleFeatures

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_WORD = re2.compile(r"\b\w+\b")
_EMOJI = re2.compile(r"\p{Emoji_Presentation}|\p{Extended_Pictographic}")
_CONTRACT = ["n't", "'re", "'s", "'ve", "'ll", "'d", "'m"]
_PUNCT = [".", ",", "!", "?", ";", ":", "—", "-", "(", ")", '"', "'"]
_HDR = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_BUL = re.compile(r"^\s*([-*+]|\d+\.)\s+", re.MULTILINE)
_CAP_PRON = {"I", "We", "You", "He", "She", "It", "They"}


@lru_cache(maxsize=4)
def _asset(name: str) -> set[str]:
    try:
        t = files("chatlectify.assets").joinpath(name).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        return set()
    return {w.strip().lower() for w in t.splitlines() if w.strip()}


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for para in text.splitlines():
        p = para.strip()
        if p:
            out.extend(x.strip() for x in _SENT.split(p) if x.strip())
    return out


def extract(msgs: list[Message]) -> StyleFeatures:
    assert msgs, "need at least one message"
    texts = [m.text for m in msgs]
    full = "\n".join(texts)
    lower = full.lower()
    words = _WORD.findall(lower)
    assert words, "no words found"
    tw = len(words)
    tc = max(len(full), 1)

    wc = Counter(words)
    bigrams = Counter(zip(words, words[1:]))
    sents = [s for t in texts for s in _sentences(t)]
    swc = [len(_WORD.findall(s)) for s in sents]
    swc = [n for n in swc if n > 0]

    starter: Counter = Counter()
    for s in sents:
        tok = _WORD.findall(s.lower())
        if tok:
            starter[tok[0]] += 1

    hedges = _asset("hedges.txt")
    hedge_hits = sum(len(re.findall(rf"\b{re.escape(h)}\b", lower)) for h in hedges)

    dict_words = _asset("dict_words.txt")
    if dict_words:
        checkable = [w for w in words if len(w) >= 3 and not any(c.isdigit() for c in w)]
        unknown = sum(1 for w in checkable if w not in dict_words)
        typo_rate = unknown / max(len(checkable), 1)
    else:
        typo_rate = 0.0

    verbs = _asset("verbs_top500.txt")
    imp = 0
    for s in sents:
        toks = s.split()
        if not toks:
            continue
        first = toks[0].strip(".,!?;:")
        if first.lower() in verbs and first not in _CAP_PRON:
            imp += 1

    msg_words = [m.word_count for m in msgs]
    feats = StyleFeatures(
        ttr=len(set(words)) / tw,
        avg_word_len=float(np.mean([len(w) for w in words])),
        top_unigrams=[(w, c) for w, c in wc.most_common(100)],
        top_bigrams=[(f"{a} {b}", c) for (a, b), c in bigrams.most_common(50)],
        contraction_rate=100.0 * sum(lower.count(c) for c in _CONTRACT) / tw,
        avg_sent_len=float(np.mean(swc)) if swc else 0.0,
        sent_len_std=float(np.std(swc)) if swc else 0.0,
        punct_hist={p: 1000.0 * full.count(p) / tc for p in _PUNCT},
        cap_start_ratio=sum(1 for s in sents if s and s[0].isupper()) / max(len(sents), 1),
        bullet_rate=len(_BUL.findall(full)) / len(msgs),
        header_rate=len(_HDR.findall(full)) / len(msgs),
        fence_rate=full.count("```") / (2 * len(msgs)),
        avg_line_breaks=float(np.mean([t.count("\n") for t in texts])),
        top_sent_starters=[(w, c) for w, c in starter.most_common(20)],
        hedge_rate=100.0 * hedge_hits / tw,
        emoji_rate=100.0 * len(_EMOJI.findall(full)) / tw,
        typo_rate=typo_rate,
        imperative_rate=imp / max(len(sents), 1),
        question_rate=sum(1 for s in sents if s.endswith("?")) / max(len(sents), 1),
        avg_msg_words=float(np.mean(msg_words)),
        avg_msg_words_std=float(np.std(msg_words)),
        msg_count=len(msgs),
    )
    for v in (feats.ttr, feats.contraction_rate, feats.hedge_rate, feats.emoji_rate,
              feats.typo_rate, feats.imperative_rate, feats.question_rate,
              feats.avg_word_len, feats.avg_sent_len):
        assert np.isfinite(v), f"non-finite: {v}"
    assert 0.0 <= feats.ttr <= 1.0 and 0.0 <= feats.typo_rate <= 1.0
    assert feats.top_unigrams
    return feats
