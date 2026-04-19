from pathlib import Path

import pytest

from chatlectify.ingest import ingest

FIX = Path(__file__).parent / "fixtures"


def test_claude_ingest():
    msgs = ingest(FIX / "claude_export.json")
    assert len(msgs) == 8  # 8 human messages
    assert all(m.role == "human" for m in msgs)
    assert msgs[0].text.startswith("Hey")
    assert msgs[0].word_count > 0


def test_chatgpt_ingest():
    msgs = ingest(FIX / "chatgpt_export.json")
    assert len(msgs) == 2
    assert msgs[0].text.startswith("Hello")


def test_empty_raises(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text('[{"uuid":"x","chat_messages":[]}]')
    with pytest.raises(ValueError):
        ingest(p)


def test_unknown_format(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"foo": "bar"}')
    with pytest.raises(ValueError):
        ingest(p)
