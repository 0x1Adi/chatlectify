from chatlectify.features import extract
from chatlectify.schemas import Message


def _m(i, text):
    return Message(msg_id=f"m{i}", conv_id="c", role="human", text=text, word_count=len(text.split()))


def test_basic_features():
    msgs = [
        _m(1, "Hello world. This is a test."),
        _m(2, "I don't think that's correct."),
        _m(3, "Write a Python function that sorts a list."),
        _m(4, "What is your favorite color?"),
    ]
    f = extract(msgs)
    assert f.msg_count == 4
    assert 0 < f.ttr <= 1
    assert f.avg_word_len > 0
    assert f.contraction_rate > 0  # don't, that's
    assert f.question_rate > 0  # 'What is your favorite color?'
    assert len(f.top_unigrams) > 0


def test_punct_and_structure():
    msgs = [
        _m(1, "# Header\n- bullet 1\n- bullet 2\nSome prose here, with commas."),
        _m(2, "Normal message with period."),
    ]
    f = extract(msgs)
    assert f.bullet_rate > 0
    assert f.header_rate > 0
    assert f.punct_hist["."] > 0
