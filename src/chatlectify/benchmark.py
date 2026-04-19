from __future__ import annotations

import random
import re

import numpy as np

from .features import extract
from .schemas import BenchmarkReport, Message
from .synth import DEFAULT_MODELS

BASELINE_PROMPT = "You are a helpful assistant. Answer in 1-3 sentences."

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_train_test(msgs: list[Message], ratio=0.8):
    rng = random.Random(42)
    short = [m for m in msgs if m.word_count < 10]
    mid = [m for m in msgs if 10 <= m.word_count <= 50]
    long = [m for m in msgs if m.word_count > 50]
    train: list[Message] = []
    test: list[Message] = []
    for pool in (short, mid, long):
        pool = list(pool)
        rng.shuffle(pool)
        cut = int(len(pool) * ratio)
        train.extend(pool[:cut])
        test.extend(pool[cut:])
    return train, test


def _topic_prompt(text: str) -> str:
    first = _SENT_SPLIT.split(text, maxsplit=1)[0].strip()
    return first or text[:80]


def ngram_auc(user_texts: list[str], ai_texts: list[str]) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = user_texts + ai_texts
    y = [1] * len(user_texts) + [0] * len(ai_texts)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=5000)
    Xv = vec.fit_transform(X)
    return float(
        cross_val_score(
            LogisticRegression(max_iter=1000, random_state=42),
            Xv, y, cv=min(5, len(set(y)) * 2), scoring="roc_auc",
        ).mean()
    )


def _feat_vec(feats) -> np.ndarray:
    return np.array([
        feats.ttr, feats.avg_word_len, feats.contraction_rate,
        feats.avg_sent_len, feats.sent_len_std, feats.cap_start_ratio,
        feats.bullet_rate, feats.header_rate, feats.fence_rate,
        feats.avg_line_breaks, feats.hedge_rate, feats.emoji_rate,
        feats.typo_rate, feats.imperative_rate, feats.question_rate,
        feats.avg_msg_words,
    ], dtype=float)


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def _to_msg(text: str, i: int) -> Message:
    return Message(msg_id=f"g{i}", conv_id="gen", role="human", text=text, word_count=len(text.split()))


def _call_llm(system: str, user: str, provider: str, model: str) -> str:
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        r = client.messages.create(
            model=model, max_tokens=150, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return r.content[0].text
    elif provider == "openai":
        import openai
        client = openai.OpenAI()
        r = client.chat.completions.create(
            model=model, max_tokens=150, temperature=0.7,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return r.choices[0].message.content
    raise ValueError(provider)


def run_benchmark(
    skill_body: str,
    msgs: list[Message],
    provider: str = "anthropic",
    model: str | None = None,
    n: int = 100,
    llm_fn=None,
) -> BenchmarkReport:
    model = model or DEFAULT_MODELS[provider]
    _, test = _split_train_test(msgs)
    if len(test) < 30:
        raise RuntimeError(f"too few test samples: {len(test)}")
    prompts = [_topic_prompt(m.text) for m in test[:n]]

    call = llm_fn or (lambda sys_p, usr: _call_llm(sys_p, usr, provider, model))

    baseline_out: list[str] = []
    skill_out: list[str] = []
    for p in prompts:
        b = call(BASELINE_PROMPT, p)
        s = call(skill_body, p)
        if not b or not s:
            raise RuntimeError("empty LLM response")
        baseline_out.append(b)
        skill_out.append(s)

    user_texts = [m.text for m in test[: len(baseline_out)]]

    auc_baseline = ngram_auc(user_texts, baseline_out)
    auc_skill = ngram_auc(user_texts, skill_out)
    assert 0.0 <= auc_baseline <= 1.0 and 0.0 <= auc_skill <= 1.0

    f_user = _feat_vec(extract([_to_msg(t, i) for i, t in enumerate(user_texts)]))
    f_base = _feat_vec(extract([_to_msg(t, i) for i, t in enumerate(baseline_out)]))
    f_skill = _feat_vec(extract([_to_msg(t, i) for i, t in enumerate(skill_out)]))
    d_base = _cos_dist(f_user, f_base)
    d_skill = _cos_dist(f_user, f_skill)

    pass_ngram = auc_skill < 0.75 and (auc_baseline - auc_skill) > 0.10
    pass_feat = d_skill < d_base and (d_base - d_skill) / max(d_base, 1e-9) > 0.20
    overall = bool(pass_ngram and pass_feat)

    return BenchmarkReport(
        ngram_auc_baseline=auc_baseline,
        ngram_auc_skill=auc_skill,
        ngram_auc_delta=auc_baseline - auc_skill,
        feature_dist_baseline=d_base,
        feature_dist_skill=d_skill,
        feature_dist_improvement=(d_base - d_skill) / max(d_base, 1e-9),
        pass_criteria={"ngram": bool(pass_ngram), "feature": bool(pass_feat)},
        overall_pass=overall,
    )
