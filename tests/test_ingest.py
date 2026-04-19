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


def test_text_file(tmp_path):
    p = tmp_path / "essay.md"
    p.write_text("First paragraph here.\n\nSecond paragraph here.\n\nThird one.\n")
    msgs = ingest(p)
    assert len(msgs) == 3
    assert msgs[1].text == "Second paragraph here."


def test_text_folder(tmp_path):
    (tmp_path / "a.md").write_text("Post one, para one.\n\nPost one, para two.")
    (tmp_path / "b.txt").write_text("Post two.")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "c.md").write_text("Nested post.")
    msgs = ingest(tmp_path)
    assert len(msgs) == 4


def test_unknown_format(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"foo": "bar"}')
    with pytest.raises(ValueError):
        ingest(p)
