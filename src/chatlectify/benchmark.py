
import random
import re

import numpy as np

from .features import extract
from .schemas import BenchmarkReport, Message
from .synth import DEFAULT_MODELS

BASELINE = "You are a helpful assistant. Answer in 1-3 sentences."
_SENT = re.compile(r"(?<=[.!?])\s+")


def _split(msgs: list[Message], ratio=0.8):
    rng = random.Random(42)
    pools = [
        [m for m in msgs if m.word_count < 10],
        [m for m in msgs if 10 <= m.word_count <= 50],
        [m for m in msgs if m.word_count > 50],
    ]
    train, test = [], []
    for pool in pools:
        pool = list(pool)
        rng.shuffle(pool)
        cut = int(len(pool) * ratio)
        train.extend(pool[:cut])
        test.extend(pool[cut:])
    return train, test


def ngram_auc(user_texts: list[str], ai_texts: list[str]) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X = user_texts + ai_texts
    y = [1] * len(user_texts) + [0] * len(ai_texts)
    Xv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=5000).fit_transform(X)
    return float(cross_val_score(
        LogisticRegression(max_iter=1000, random_state=42), Xv, y, cv=5, scoring="roc_auc"
    ).mean())


def _fv(f) -> np.ndarray:
    return np.array([f.ttr, f.avg_word_len, f.contraction_rate, f.avg_sent_len,
                     f.sent_len_std, f.cap_start_ratio, f.bullet_rate, f.header_rate,
                     f.fence_rate, f.avg_line_breaks, f.hedge_rate, f.emoji_rate,
                     f.typo_rate, f.imperative_rate, f.question_rate, f.avg_msg_words], dtype=float)


def _cos(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 1.0 if na == 0 or nb == 0 else float(1.0 - np.dot(a, b) / (na * nb))


def _m(t, i):
    return Message(msg_id=f"g{i}", conv_id="gen", role="human", text=t, word_count=len(t.split()))


def _call(system: str, user: str, provider: str, model: str) -> str:
    from .llm import call
    return call(provider, user, system=system, model=model, max_tokens=150)


def run_benchmark(skill_body: str, msgs: list[Message], provider="anthropic",
                  model: str | None = None, n: int = 100, llm_fn=None) -> BenchmarkReport:
    model = model or DEFAULT_MODELS[provider]
    _, test = _split(msgs)
    if len(test) < 30:
        raise RuntimeError(f"too few test samples: {len(test)}")
    prompts = [(_SENT.split(m.text, maxsplit=1)[0].strip() or m.text[:80]) for m in test[:n]]
    call = llm_fn or (lambda s, u: _call(s, u, provider, model))
    base_out, skill_out = [], []
    for p in prompts:
        b, s = call(BASELINE, p), call(skill_body, p)
        if not b or not s:
            raise RuntimeError("empty LLM response")
        base_out.append(b)
        skill_out.append(s)
    user_texts = [m.text for m in test[: len(base_out)]]
    auc_b, auc_s = ngram_auc(user_texts, base_out), ngram_auc(user_texts, skill_out)
    assert 0.0 <= auc_b <= 1.0 and 0.0 <= auc_s <= 1.0
    fu = _fv(extract([_m(t, i) for i, t in enumerate(user_texts)]))
    fb = _fv(extract([_m(t, i) for i, t in enumerate(base_out)]))
    fs = _fv(extract([_m(t, i) for i, t in enumerate(skill_out)]))
    db, ds = _cos(fu, fb), _cos(fu, fs)
    pass_ng = auc_s < 0.75 and (auc_b - auc_s) > 0.10
    pass_ft = ds < db and (db - ds) / max(db, 1e-9) > 0.20
    return BenchmarkReport(
        ngram_auc_baseline=auc_b, ngram_auc_skill=auc_s, ngram_auc_delta=auc_b - auc_s,
        feature_dist_baseline=db, feature_dist_skill=ds,
        feature_dist_improvement=(db - ds) / max(db, 1e-9),
        pass_criteria={"ngram": bool(pass_ng), "feature": bool(pass_ft)},
        overall_pass=bool(pass_ng and pass_ft),
    )
