# Contributing to chatlectify

Thanks for considering a contribution. This project is small and opinionated;
keep PRs focused and they'll merge fast.

## Ground rules

- **One concern per PR.** Bug fix, feature, refactor — pick one.
- **Tests required** for behavioral changes. If you add an ingest format, add a fixture + a test.
- **No LLM calls in the test suite.** Mock `llm.call` (see `tests/test_synth.py` and `tests/test_e2e.py` for patterns).
- **Ruff clean.** `ruff check src/ tests/` must pass.
- **Keep the public CLI surface stable** across minor versions.

## Getting set up

```bash
git clone https://github.com/0x1Adi/chatlectify
cd chatlectify
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -ra
```

## Running the checks CI runs

```bash
ruff check src/ tests/
pytest -ra --cov=chatlectify --cov-report=term-missing
pip-audit --strict
python -m build
```

## Project layout

```
src/chatlectify/
  cli.py         Typer entrypoint
  ingest.py      Format detectors + parsers (Claude / ChatGPT / Gemini / plaintext)
  clean.py       Strip code/URLs, dedupe, paste detection
  features.py    Stylometric feature extraction
  gates.py       Quality gates (min msgs / min chars / paste ratio)
  synth.py       LLM call to build SKILL.md + deterministic fallback
  benchmark.py   Fidelity measurement (n-gram AUC + feature distance)
  emit.py        Writes artifacts to --out-dir
  llm.py         Unified provider caller (API key OR local CLI)
  schemas.py     Pydantic models (Message, StyleFeatures, GateReport, BenchmarkReport)
  assets/        Word lists used by feature extraction
tests/
  fixtures/      Tiny export samples
  test_*.py      Unit tests per module + test_e2e.py for the full pipeline
```

## Conventions

- **No docstrings or comments that narrate the code.** Only explain *why* when a reader would be surprised.
- **No `print`** in library code — use `typer.echo(..., err=True)` or stderr.
- **Determinism**: any sampling should seed `random.Random(42)`.
- **LLM calls** go through `chatlectify.llm.call` — never import `anthropic`/`openai` directly elsewhere.

## Adding a new ingest format

1. Add a parser `_<format>(...)` in `ingest.py` returning `list[Message]`.
2. Extend `_detect` and the `ingest()` dispatcher.
3. Add a fixture under `tests/fixtures/`.
4. Add a `test_<format>_ingest()` case to `tests/test_ingest.py`.

## Commit messages

Short, lowercase, scoped: `ingest: accept plaintext folders`, `cli: add --n to benchmark`.

## Reporting bugs

Open an issue with:

- `chatlectify --version` (or commit SHA)
- Python version and OS
- The exact command you ran
- What you expected vs. what happened
- Minimal reproducer if possible

## Security

Do **not** file security issues in the public tracker. See [SECURITY.md](./SECURITY.md).

## License

By contributing you agree your contributions are licensed under the MIT
License — same as the rest of the project.
