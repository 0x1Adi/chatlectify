from chatlectify.schemas import Message, StyleFeatures
from chatlectify.synth import _fallback, _validate, stratified_exemplars, synthesize


def _feats():
    return StyleFeatures(
        ttr=0.5, avg_word_len=4.0, top_unigrams=[("the", 10)], top_bigrams=[],
        contraction_rate=2.0, avg_sent_len=12.0, sent_len_std=3.0,
        punct_hist={".": 10.0}, cap_start_ratio=0.9, bullet_rate=0.1,
        header_rate=0.0, fence_rate=0.0, avg_line_breaks=1.0,
        top_sent_starters=[], hedge_rate=0.5, emoji_rate=0.0, typo_rate=0.01,
        imperative_rate=0.2, question_rate=0.1, avg_msg_words=12.0,
        avg_msg_words_std=3.0, msg_count=200,
    )


def _msgs(n):
    out = []
    for i in range(n):
        wc = 5 + (i % 30)
        t = " ".join(["word"] * wc)
        out.append(Message(msg_id=str(i), conv_id="c", role="human", text=t, word_count=wc))
    return out


def test_stratified():
    picks = stratified_exemplars(_msgs(200))
    assert 1 <= len(picks) <= 40


def test_validate_ok():
    good = """---
name: my_style
description: A test.
---

## Style Rules
- Be concise.

## Exemplars
> hi

## Anti-patterns
- do not ramble

## Quantified Targets
| a | b |
"""
    assert _validate(good) == []


def test_validate_bad():
    assert _validate("no frontmatter here") != []


def test_fallback_valid():
    out = _fallback(_feats(), _msgs(200))
    assert _validate(out) == []


def test_synth_with_mock_llm():
    captured = {}

    def mock(prompt):
        captured["p"] = prompt
        return """---
name: user_voice
description: Mocked.
---

## Style Rules
- rule

## Exemplars
> x

## Anti-patterns
- do not x

## Quantified Targets
| a | b |
"""
    out = synthesize(_feats(), _msgs(50), llm_fn=mock)
    assert "user_voice" in out
    assert "user_style_features" in captured["p"]


def test_synth_fallback_on_bad_output():
    def mock(prompt):
        return "garbage"
    out = synthesize(_feats(), _msgs(50), llm_fn=mock)
    assert _validate(out) == []
