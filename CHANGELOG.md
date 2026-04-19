# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-04-20

### Added
- `--bench-model` option on `all` to run the fidelity benchmark against a cheaper/faster model than synth (e.g. Sonnet for synth, Haiku for benchmark).
- `--model` option on the standalone `benchmark` subcommand.
- End-to-end integration test (`tests/test_e2e.py`) driving the full CLI pipeline with a mocked LLM.
- Unit tests for `emit.py`.
- `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, `LICENSE`.
- GitHub Actions security hardening: CodeQL, `pip-audit`, Dependabot, pinned permissions, multi-version Python matrix.
- PyPI packaging metadata: author, classifiers, keywords, project URLs.

### Fixed
- `pip-audit` CI job: dropped invalid `--disable-pip` flag and skip the editable package itself.

## [0.1.0] - 2026-04-19

### Added
- Initial release.
- Ingesters for Claude, ChatGPT, and Gemini exports.
- Plaintext/folder ingester for `.txt` / `.md` / `.markdown` / `.rst` corpora.
- Clean pipeline: code/URL/email stripping, dedupe, paste detection.
- Stylometric feature extraction (TTR, sentence length, punctuation, bullets, headers, hedges, emoji, typos, imperative/question rates, …).
- Quality gates (`min_msgs=200`, `min_chars=20000`, `paste_contamination<=50%`).
- Synth step producing a Claude-compatible `SKILL.md` with deterministic fallback.
- Fidelity benchmark: char-n-gram ROC-AUC + stylometric feature-distance reduction.
- Unified LLM caller with API key → local CLI fallback (`claude` / `codex`).
- Typer CLI: `ingest`, `features`, `build`, `benchmark`, `all`.

[Unreleased]: https://github.com/0x1Adi/chatlectify/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/0x1Adi/chatlectify/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/0x1Adi/chatlectify/releases/tag/v0.1.0
