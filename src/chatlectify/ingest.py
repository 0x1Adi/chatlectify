from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from .schemas import Message


def _parse_ts(v) -> datetime | None:
    if not v:
        return None
    try:
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(float(v))
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _wc(text: str) -> int:
    return len(text.split())


def detect_format(path: Path, data) -> str:
    if path.suffix.lower() in (".html", ".htm"):
        return "gemini"
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if "chat_messages" in data[0]:
            return "claude"
        if "mapping" in data[0]:
            return "chatgpt"
    if isinstance(data, dict):
        if "chat_messages" in data:
            return "claude"
        if "mapping" in data:
            return "chatgpt"
    keys = list(data.keys()) if isinstance(data, dict) else (list(data[0].keys()) if data else [])
    raise ValueError(f"Cannot detect format; keys={keys}")


def _parse_claude(data) -> list[Message]:
    convs = data if isinstance(data, list) else [data]
    msgs: list[Message] = []
    for conv in convs:
        cid = conv.get("uuid") or conv.get("name") or "conv"
        for m in conv.get("chat_messages", []):
            sender = m.get("sender", "")
            if sender not in ("human", "user"):
                continue
            text = m.get("text") or ""
            if not text and "content" in m:
                parts = m.get("content") or []
                text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
            if text is None or text == "":
                raise ValueError("empty message text in claude export")
            msgs.append(
                Message(
                    msg_id=m.get("uuid", f"{cid}-{len(msgs)}"),
                    conv_id=cid,
                    timestamp=_parse_ts(m.get("created_at")),
                    role="human",
                    text=text,
                    word_count=_wc(text),
                )
            )
    return msgs


def _parse_chatgpt(data) -> list[Message]:
    convs = data if isinstance(data, list) else [data]
    msgs: list[Message] = []
    for conv in convs:
        cid = conv.get("id") or conv.get("conversation_id") or conv.get("title") or "conv"
        mapping = conv.get("mapping", {})
        for node_id, node in mapping.items():
            m = node.get("message") if isinstance(node, dict) else None
            if not m:
                continue
            role = (m.get("author") or {}).get("role")
            if role != "user":
                continue
            content = m.get("content") or {}
            parts = content.get("parts") or []
            text = "\n".join(p for p in parts if isinstance(p, str))
            if not text:
                continue
            msgs.append(
                Message(
                    msg_id=m.get("id", node_id),
                    conv_id=cid,
                    timestamp=_parse_ts(m.get("create_time")),
                    role="human",
                    text=text,
                    word_count=_wc(text),
                )
            )
    return msgs


def _parse_gemini(path: Path) -> list[Message]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    stripped = re.sub(r"<[^>]+>", "\n", text)
    msgs: list[Message] = []
    for i, line in enumerate(stripped.splitlines()):
        line = line.strip()
        if line.startswith("You:"):
            body = line[4:].strip()
            if body:
                msgs.append(
                    Message(
                        msg_id=f"g-{i}",
                        conv_id="gemini",
                        role="human",
                        text=body,
                        word_count=_wc(body),
                    )
                )
    return msgs


def ingest(path: str | Path, fmt: str = "auto") -> list[Message]:
    path = Path(path)
    data = None
    if path.suffix.lower() in (".json",):
        data = json.loads(path.read_text(encoding="utf-8"))
    if fmt == "auto":
        if path.suffix.lower() in (".html", ".htm"):
            fmt = "gemini"
        else:
            fmt = detect_format(path, data)
    if fmt == "claude":
        msgs = _parse_claude(data)
    elif fmt == "chatgpt":
        msgs = _parse_chatgpt(data)
    elif fmt == "gemini":
        msgs = _parse_gemini(path)
    else:
        raise ValueError(f"unknown format: {fmt}")
    if not msgs:
        raise ValueError("parsed 0 messages")
    return msgs
