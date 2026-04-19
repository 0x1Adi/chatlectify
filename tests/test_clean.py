from chatlectify.clean import clean
from chatlectify.schemas import Message


def _m(i, text):
    return Message(msg_id=f"m{i}", conv_id="c", role="human", text=text, word_count=len(text.split()))


def test_strip_code_and_urls():
    raw = "Check this code:\n```\nprint('hi')\n```\nSee https://example.com for info."
    msgs = [_m(1, raw)] * 1 + [_m(2, "A totally different message here now.")]
    kept, pastes, stats = clean(msgs)
    assert all("```" not in m.text for m in kept)
    assert all("https://" not in m.text for m in kept)


def test_dedupe():
    msgs = [_m(i, "This is exactly the same message text.") for i in range(5)]
    msgs.append(_m(99, "A totally unique message appears here."))
    kept, _, stats = clean(msgs)
    # 5 dups collapse to 1, plus the unique
    assert stats["deduped"] == 2


def test_discard_short():
    msgs = [_m(1, "hi"), _m(2, "also short one"), _m(3, "this message is long enough to keep here")]
    kept, _, _ = clean(msgs)
    assert len(kept) == 1


def test_paste_detection():
    long = ("word " * 500).strip()
    msgs = [_m(i, f"normal message number {i} is fine here") for i in range(20)]
    msgs.append(_m(100, long))
    kept, pastes, stats = clean(msgs)
    assert stats["pastes_flagged"] >= 1
    assert any(m.msg_id == "m100" for m in pastes)
