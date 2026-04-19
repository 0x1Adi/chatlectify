from chatlectify.gates import run_gates
from chatlectify.schemas import Message, StyleFeatures


def _feats(n=250, ttr=0.5, avg_msg_words=10.0):
    return StyleFeatures(
        ttr=ttr, avg_word_len=4.0, top_unigrams=[("the", 10)], top_bigrams=[],
        contraction_rate=1.0, avg_sent_len=10.0, sent_len_std=3.0,
        punct_hist={".": 10.0}, cap_start_ratio=0.9, bullet_rate=0.0,
        header_rate=0.0, fence_rate=0.0, avg_line_breaks=1.0,
        top_sent_starters=[], hedge_rate=0.5, emoji_rate=0.0,
        typo_rate=0.01, imperative_rate=0.1, question_rate=0.1,
        avg_msg_words=avg_msg_words, avg_msg_words_std=2.0, msg_count=n,
    )


def _msgs(n, avg_len):
    body = "x" * avg_len
    return [Message(msg_id=str(i), conv_id="c", role="human", text=body, word_count=10) for i in range(n)]


def test_pass():
    r = run_gates(_msgs(250, 100), _feats(250), 0.1)
    assert r.passed
    assert r.warnings  # 250 < 500 triggers low sample warning


def test_fail_min_msgs():
    r = run_gates(_msgs(50, 500), _feats(50), 0.0)
    assert not r.passed
    assert any("min_msgs" in e for e in r.errors)


def test_fail_paste():
    r = run_gates(_msgs(250, 200), _feats(250), 0.7)
    assert not r.passed


def test_ttr_warning():
    r = run_gates(_msgs(600, 100), _feats(600, ttr=0.9), 0.0)
    assert r.passed
    assert any("TTR" in w for w in r.warnings)
