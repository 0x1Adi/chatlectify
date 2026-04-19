# chatlectify

> Compile your AI chat history into a writing-style skill.

Ingest an AI chat export → emit a Claude `SKILL.md`, a portable system prompt, and a benchmark report proving style fidelity. Local-first, BYO API key.

## Install

```bash
pip install chatlectify
# or from source
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Usage

```bash
export ANTHROPIC_API_KEY=sk-...   # or OPENAI_API_KEY

chatlectify ingest   export.json --out normalized.json
chatlectify features normalized.json --out features.json
chatlectify build    export.json --out-dir ./skill/
chatlectify benchmark --skill ./skill/SKILL.md --user-samples normalized.json
chatlectify all      export.json --out-dir ./skill/            # synth only
chatlectify all      export.json --out-dir ./skill/ --benchmark # + fidelity test
```

`--format` auto-detects Claude, ChatGPT, or Gemini exports. Use `--provider openai` to use GPT.

## Outputs (`--out-dir`)

- `SKILL.md` — Claude skill file with YAML frontmatter + rules + exemplars.
- `system_prompt.txt` — portable prompt usable in any LLM chat.
- `style_metrics.json` — extracted stylometric features.
- `gate_report.json` — sanity-gate results.
- `benchmark_report.json` — A/B test report (only in `all`).
- `exemplars.json`, `pastes.json`, `pipeline_report.json`.

## Gate thresholds

`build` aborts unless all hard gates pass (override with `--force`):

- min **200** messages
- min **20,000** characters
- paste-contamination ≤ **50%**

Soft warnings fire at `<500` msgs, `>30%` paste, `<3` avg words/msg, TTR `>0.8`.

## Accuracy expectations

Surface style (vocab, punctuation, sentence length, formatting): **75–90% match**.
Deep voice / rhetorical patterns: **50–65%**.

The same ceiling applies to all prompt-based style-transfer tools. The benchmark module exposes exact numbers per run so you can verify.

## Troubleshooting

- **"missing ANTHROPIC_API_KEY"** — export the key for your chosen provider.
- **"min_msgs=200 failed"** — your export is too small; rerun with `--force` at lower fidelity.
- **"too few test samples"** in benchmark — use `--skip-benchmark` or supply a larger export.
- **over-aggressive cleaning** — raise an issue with a redacted fixture; cleaning keeps ≥30% of raw messages.

## Non-goals

No GUI, no cloud storage, no OAuth to Claude.ai/ChatGPT, no live scraping, no fine-tuning. English-only (v1).

## License

MIT.
