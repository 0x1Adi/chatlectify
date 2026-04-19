
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
                raise ValueError("empty claude message")
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
    data = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else None
    if fmt == "auto":
        fmt = "gemini" if path.suffix.lower() in (".html", ".htm") else _detect(path, data)
    if fmt == "claude":
        msgs = _claude(data)
    elif fmt == "chatgpt":
        msgs = _chatgpt(data)
    elif fmt == "gemini":
        msgs = _gemini(path)
    else:
        raise ValueError(f"unknown format: {fmt}")
    if not msgs:
        raise ValueError("parsed 0 messages")
    return msgs
