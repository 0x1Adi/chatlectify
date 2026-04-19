# chatlectify

> Compile your AI chat history into a writing-style skill.

[![CI](https://github.com/0x1Adi/chatlectify/actions/workflows/ci.yml/badge.svg)](https://github.com/0x1Adi/chatlectify/actions/workflows/ci.yml)
[![CodeQL](https://github.com/0x1Adi/chatlectify/actions/workflows/codeql.yml/badge.svg)](https://github.com/0x1Adi/chatlectify/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)

`chatlectify` turns your exported AI conversations (Claude, ChatGPT, Gemini) or
any corpus of your own writing into a portable **`SKILL.md`** file. Drop that
file into a Claude skill folder, paste the `system_prompt.txt` into any LLM,
and the model writes like *you* — not like the factory-default assistant.

It's a small, local CLI. Your chat data never leaves your machine except when
it makes the single LLM call that distills the style file.

---

## Why

Every large model speaks the same corporate-cheerful dialect out of the box:
"Certainly!", "Great question!", "In conclusion...". If you've spent a year
chatting with it, *you* have a voice and it has no idea. `chatlectify` mines
that voice from your own messages and hands you back a reusable style artifact.

## What you get

Running the pipeline on one export produces:

| File | Purpose |
| --- | --- |
| `SKILL.md` | Claude-compatible skill file (frontmatter + rules + exemplars + anti-patterns + quantified targets) |
| `system_prompt.txt` | Drop-in system prompt for any LLM |
| `style_metrics.json` | Raw stylometric features (TTR, sentence length, punctuation histograms, …) |
| `exemplars.json` | Stratified sample of your best messages |
| `gate_report.json` | Quality-gate pass/fail reasons |
| `pipeline_report.json` | End-to-end run summary |
| `benchmark_report.json` | (optional) Fidelity measurement vs. a baseline prompt |

---

## Install

```bash
pip install chatlectify
```

Or from source:

```bash
git clone https://github.com/0x1Adi/chatlectify
cd chatlectify
pip install -e ".[dev]"
```

Requires **Python 3.11+**.

## Auth

The synth step needs *one* LLM call. `chatlectify` auto-detects either:

- An **API key**: `ANTHROPIC_API_KEY` (default) or `OPENAI_API_KEY`.
- A local **CLI**: the `claude` CLI (default) or `codex` CLI.

If neither is available, the pipeline still runs and emits a deterministic
fallback skill using your extracted features.

---

## Quick start

### 1. Export your chat history

- **Claude** — Settings → Privacy → Export data. Unzip; you need `conversations.json`.
- **ChatGPT** — Settings → Data Controls → Export. Unzip; you need `conversations.json`.
- **Gemini** — Takeout; the `.html` file works.
- **Your own writing** — any folder of `.txt` / `.md` / `.markdown` / `.rst` files.

### 2. Build the skill

```bash
chatlectify all ./conversations.json --out-dir ./skill
```

That's it. Open `./skill/SKILL.md`.

### 3. Use the skill

**With Claude Code / Claude Desktop:**
Copy the folder into `~/.claude/skills/<skill_name>/`.

**With any other LLM:**
Paste `./skill/system_prompt.txt` as the system prompt.

---

## CLI

```
chatlectify ingest <input>    Normalize an export into messages.
chatlectify features <input>  Extract stylometric features only.
chatlectify build <input>     Build SKILL.md (skip benchmark).
chatlectify benchmark ...     Measure fidelity vs. baseline.
chatlectify all <input>       Full pipeline.
```

Common options:

- `--out-dir PATH` — where to write artifacts.
- `--provider {anthropic,openai}` — default `anthropic`.
- `--model MODEL` — override the default model (applies to synth + benchmark).
- `--bench-model MODEL` — use a separate (cheaper) model for the benchmark only. Falls back to `--model`.
- `--benchmark` — opt in to fidelity measurement (makes `2N` extra LLM calls).
- `--n N` — benchmark sample size (default 100).
- `--force` — bypass quality gates (use at your own risk).

Run `chatlectify <cmd> --help` for full flags.

---

## Pipeline

```
  ingest  ->  clean  ->  features  ->  gates  ->  synth  ->  (benchmark)  ->  emit
```

- **ingest** — parses export formats (Claude, ChatGPT, Gemini, plaintext) into a normalized `Message[]`.
- **clean** — strips code fences, URLs, emails; dedupes near-duplicates; flags pastes.
- **features** — computes 20+ stylometric metrics (lexical, syntactic, structural).
- **gates** — blocks low-quality runs: `<200` messages, `<20k` characters, `>50%` pastes.
- **synth** — asks one LLM call to distill features + exemplars + anti-patterns into a `SKILL.md`. Falls back to a deterministic template if the call fails or produces invalid output.
- **benchmark** *(optional)* — generates `N` pairs of baseline vs. skill-prompted completions and measures (a) char-n-gram ROC-AUC between your text and each (lower = more indistinguishable from you) and (b) feature-distance reduction.
- **emit** — writes all artifacts to `--out-dir`.

---

## Privacy

- Everything runs locally. The only egress is the single synth call (and optional benchmark calls) to your chosen provider.
- Default `.gitignore` excludes `conversations.json`, `SKILL.md`, and all intermediate JSON — your data and voice stay yours.
- Nothing is telemetered; no analytics; no cloud component.

---

## Development

```bash
pip install -e ".[dev]"
ruff check src/ tests/
pytest -ra
pytest --cov=chatlectify --cov-report=term-missing
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for contribution guidelines and
[SECURITY.md](./SECURITY.md) to report vulnerabilities.

## License

MIT — see [LICENSE](./LICENSE).
