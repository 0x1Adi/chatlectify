
import json
import re
from datetime import datetime
from pathlib import Path

from .schemas import Message


def _ts(v):
    if not v:
        return None
    try:
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(float(v))
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _wc(t: str) -> int:
    return len(t.split())


def _detect(path: Path, data) -> str:
    if path.suffix.lower() in (".html", ".htm"):
        return "gemini"
    candidates = data if isinstance(data, list) else [data]
    first = candidates[0] if candidates else {}
    if isinstance(first, dict):
        if "chat_messages" in first:
            return "claude"
        if "mapping" in first:
            return "chatgpt"
    keys = list(first.keys()) if isinstance(first, dict) else []
    raise ValueError(f"cannot detect format; keys={keys}")


def _claude(data) -> list[Message]:
    convs = data if isinstance(data, list) else [data]
    out: list[Message] = []
    for conv in convs:
        cid = conv.get("uuid") or conv.get("name") or "conv"
        for m in conv.get("chat_messages", []):
            if m.get("sender") not in ("human", "user"):
                continue
            text = m.get("text") or ""
            if not text and "content" in m:
                text = "".join(p.get("text", "") for p in (m.get("content") or []) if isinstance(p, dict))
            if not text:
                continue
            out.append(Message(
                msg_id=m.get("uuid", f"{cid}-{len(out)}"), conv_id=cid,
                timestamp=_ts(m.get("created_at")), role="human",
                text=text, word_count=_wc(text),
            ))
    return out


def _chatgpt(data) -> list[Message]:
    convs = data if isinstance(data, list) else [data]
    out: list[Message] = []
    for conv in convs:
        cid = conv.get("id") or conv.get("title") or "conv"
        for node_id, node in conv.get("mapping", {}).items():
            m = node.get("message") if isinstance(node, dict) else None
            if not m or (m.get("author") or {}).get("role") != "user":
                continue
            parts = (m.get("content") or {}).get("parts") or []
            text = "\n".join(p for p in parts if isinstance(p, str))
            if not text:
                continue
            out.append(Message(
                msg_id=m.get("id", node_id), conv_id=cid,
                timestamp=_ts(m.get("create_time")), role="human",
                text=text, word_count=_wc(text),
            ))
    return out


_TEXT_EXTS = {".txt", ".md", ".markdown", ".rst"}


def _text_file_msgs(fp: Path, cid: str, start: int) -> list[Message]:
    """Split one text file into messages on blank-line paragraphs."""
    raw = fp.read_text(encoding="utf-8", errors="ignore")
    paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    out: list[Message] = []
    for i, p in enumerate(paras):
        out.append(Message(msg_id=f"{cid}-{start+i}", conv_id=cid, role="human",
                           text=p, word_count=_wc(p)))
    return out


def _text(path: Path) -> list[Message]:
    """Ingest a plaintext file or a folder of them. Each paragraph = one message."""
    out: list[Message] = []
    if path.is_dir():
        files = sorted(f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in _TEXT_EXTS)
        if not files:
            raise ValueError(f"no text files (.txt/.md/.markdown/.rst) in {path}")
        for fp in files:
            out.extend(_text_file_msgs(fp, cid=fp.stem, start=len(out)))
    else:
        out = _text_file_msgs(path, cid=path.stem, start=0)
    return out


def _gemini(path: Path) -> list[Message]:
    text = re.sub(r"<[^>]+>", "\n", path.read_text(encoding="utf-8", errors="ignore"))
    out: list[Message] = []
    for i, line in enumerate(text.splitlines()):
        s = line.strip()
        if s.startswith("You:"):
            body = s[4:].strip()
            if body:
                out.append(Message(msg_id=f"g-{i}", conv_id="gemini", role="human",
                                   text=body, word_count=_wc(body)))
    return out


def ingest(path: str | Path, fmt: str = "auto") -> list[Message]:
    path = Path(path)
    if fmt == "auto":
        if path.is_dir():
            fmt = "text"
        elif path.suffix.lower() in (".html", ".htm"):
            fmt = "gemini"
        elif path.suffix.lower() in _TEXT_EXTS:
            fmt = "text"
        elif path.suffix.lower() == ".json":
            fmt = _detect(path, json.loads(path.read_text(encoding="utf-8")))
        else:
            raise ValueError(f"cannot detect format for {path}")
    if fmt in ("claude", "chatgpt"):
        data = json.loads(path.read_text(encoding="utf-8"))
        msgs = _claude(data) if fmt == "claude" else _chatgpt(data)
    elif fmt == "gemini":
        msgs = _gemini(path)
    elif fmt == "text":
        msgs = _text(path)
    else:
        raise ValueError(f"unknown format: {fmt}")
    if not msgs:
        raise ValueError("parsed 0 messages")
    return msgs
