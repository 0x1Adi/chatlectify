"""Regenerate bundled assets from public sources.

Usage: python scripts/bundle_assets.py

Fetches SCOWL 50k wordlist and writes src/chatlectify/assets/dict_words.txt.
Other asset files (hedges, verbs, antiexemplars, baseline_prompts) are
curated and shipped as-is; this script leaves them untouched.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "src" / "chatlectify" / "assets"

SCOWL_URL = "https://raw.githubusercontent.com/en-wl/wordlist/master/alt12dicts/2of12inf.txt"


def fetch_dict() -> None:
    print(f"fetching {SCOWL_URL}...")
    with urllib.request.urlopen(SCOWL_URL, timeout=30) as r:
        raw = r.read().decode("utf-8", errors="ignore")
    words: set[str] = set()
    for line in raw.splitlines():
        w = line.strip().strip("%!").lower()
        if w.isalpha() and 1 <= len(w) <= 25:
            words.add(w)
    out = ASSETS / "dict_words.txt"
    out.write_text("\n".join(sorted(words)) + "\n")
    print(f"wrote {len(words)} words -> {out}")


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    try:
        fetch_dict()
    except Exception as e:
        print(f"failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
