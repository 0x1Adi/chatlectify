import random

from chatlectify.benchmark import _cos, _split, ngram_auc, run_benchmark
from chatlectify.schemas import Message


def _msgs(n=200):
    rng = random.Random(0)
    pool_words = ["lets", "go", "check", "the", "data", "pipeline", "quickly", "now",
                  "i", "dont", "think", "that", "works", "for", "us", "honestly"]
    out = []
    for i in range(n):
        wc = 5 + rng.randint(0, 20)
        toks = [rng.choice(pool_words) for _ in range(wc)]
        t = " ".join(toks).capitalize() + "."
        out.append(Message(msg_id=str(i), conv_id="c", role="human", text=t, word_count=wc))
    return out


def test_split():
    train, test = _split(_msgs(200))
    assert len(train) > 0 and len(test) > 0
    assert abs(len(train) / 200 - 0.8) < 0.1


def test_ngram_auc():
    u = ["hey whats up man", "yo lets go now", "ok sounds good to me", "alright gotcha"] * 5
    a = ["Certainly! Let me help you with that task today.",
         "As an AI, I would like to note the following points.",
         "Here is a comprehensive overview of the situation.",
         "I hope this clarifies things for you moving forward."] * 5
    auc = ngram_auc(u, a)
    assert auc > 0.7


def test_cos_dist():
    import numpy as np
    assert _cos(np.array([1.0, 0.0]), np.array([1.0, 0.0])) < 1e-6
    assert _cos(np.array([1.0, 0.0]), np.array([0.0, 1.0])) > 0.9


def test_benchmark_mocked():
    msgs = _msgs(200)

    def mock(system_p, user_p):
        if "helpful assistant" in system_p:
            return "Certainly! As an AI, here is a comprehensive and thorough overview of that topic."
        return "yo " + user_p[:30]

    report = run_benchmark("skill body", msgs, n=50, llm_fn=mock)
    assert 0 <= report.ngram_auc_baseline <= 1
    assert 0 <= report.ngram_auc_skill <= 1
    assert isinstance(report.overall_pass, bool)
