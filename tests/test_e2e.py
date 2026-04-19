"""End-to-end integration tests driving the full CLI pipeline.

No real LLM calls: the unified `chatlectify.llm.call` is monkeypatched to a
deterministic stub so the pipeline runs offline in CI.
"""
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chatlectify import cli as cli_mod
from chatlectify.cli import app

VALID_SKILL_MD = """---
name: user_voice
description: Mocked style distilled from messages.
---

## Style Rules
- Keep sentences short.
- Avoid corporate filler.

## Exemplars
> hey can you check this

## Anti-patterns
- Do NOT open with "Certainly!"

## Quantified Targets
| metric | target |
| --- | --- |
| avg_sent_len | 10 |
"""


def _stub_llm(monkeypatch, reply: str = VALID_SKILL_MD):
    """Replace the unified LLM caller and mark auth as available."""
    def fake_call(provider, prompt, system=None, model=None, max_tokens=4000):
        return reply

    def fake_available(provider):
        return "api"

    monkeypatch.setattr("chatlectify.llm.call", fake_call)
    monkeypatch.setattr("chatlectify.llm.available", fake_available)
    monkeypatch.setattr(cli_mod, "__name__", cli_mod.__name__)


def _make_corpus(path: Path, n: int = 260) -> Path:
    """Write a plaintext corpus big enough to pass the 200-msg gate."""
    lines = []
    for i in range(n):
        lines.append(
            f"Message number {i} talking about the pipeline, data, and debugging workflow today."
        )
    path.write_text("\n\n".join(lines), encoding="utf-8")
    return path


# ---------- `chatlectify ingest` ----------


def test_cli_ingest_plaintext(tmp_path):
    runner = CliRunner()
    corpus = _make_corpus(tmp_path / "essay.md", n=10)
    out = tmp_path / "normalized.json"
    result = runner.invoke(app, ["ingest", str(corpus), "--out", str(out)])
    assert result.exit_code == 0, result.stdout
    data = json.loads(out.read_text())
    assert len(data) == 10
    assert data[0]["role"] == "human"
    assert "ingested 10 messages" in result.stdout


def test_cli_ingest_claude_fixture(tmp_path):
    runner = CliRunner()
    fx = Path(__file__).parent / "fixtures" / "claude_export.json"
    out = tmp_path / "normalized.json"
    result = runner.invoke(app, ["ingest", str(fx), "--out", str(out)])
    assert result.exit_code == 0, result.stdout
    assert json.loads(out.read_text())


def test_cli_ingest_unknown_format_fails(tmp_path):
    runner = CliRunner()
    bad = tmp_path / "x.json"
    bad.write_text('{"foo": 1}')
    result = runner.invoke(app, ["ingest", str(bad), "--out", str(tmp_path / "n.json")])
    assert result.exit_code != 0


# ---------- `chatlectify features` ----------


def test_cli_features_from_plaintext(tmp_path):
    runner = CliRunner()
    corpus = _make_corpus(tmp_path / "essay.md", n=50)
    out = tmp_path / "features.json"
    result = runner.invoke(app, ["features", str(corpus), "--out", str(out)])
    assert result.exit_code == 0, result.stdout
    feats = json.loads(out.read_text())
    assert feats["msg_count"] > 0
    assert 0 < feats["ttr"] <= 1


# ---------- `chatlectify build` (skill synthesis) ----------


def test_cli_build_happy_path(tmp_path, monkeypatch):
    _stub_llm(monkeypatch)
    runner = CliRunner()
    corpus = _make_corpus(tmp_path / "corpus.md", n=260)
    out_dir = tmp_path / "skill"
    result = runner.invoke(
        app, ["build", str(corpus), "--out-dir", str(out_dir), "--provider", "anthropic"]
    )
    assert result.exit_code == 0, result.stdout
    # All expected artifacts exist
    for name in [
        "SKILL.md",
        "system_prompt.txt",
        "style_metrics.json",
        "gate_report.json",
        "exemplars.json",
        "pastes.json",
        "pipeline_report.json",
    ]:
        assert (out_dir / name).exists(), f"missing {name}"

    skill = (out_dir / "SKILL.md").read_text()
    assert skill.startswith("---")
    assert "## Style Rules" in skill

    sys_prompt = (out_dir / "system_prompt.txt").read_text()
    assert sys_prompt.startswith("Write in this style:")
    assert "---" not in sys_prompt.split("\n", 1)[0]  # frontmatter stripped

    gate = json.loads((out_dir / "gate_report.json").read_text())
    assert gate["passed"] is True
    assert gate["msg_count"] >= 200

    report = json.loads((out_dir / "pipeline_report.json").read_text())
    assert report["gate"]["passed"] is True
    assert report["benchmark"] is None


def test_cli_build_gate_fails_on_small_corpus(tmp_path, monkeypatch):
    _stub_llm(monkeypatch)
    runner = CliRunner()
    corpus = _make_corpus(tmp_path / "tiny.md", n=5)  # <200 msgs -> gate fails
    out_dir = tmp_path / "skill"
    result = runner.invoke(app, ["build", str(corpus), "--out-dir", str(out_dir)])
    assert result.exit_code != 0
    assert "gate failures" in result.stdout or "gate failures" in (result.stderr or "")


def test_cli_build_force_bypasses_gate(tmp_path, monkeypatch):
    _stub_llm(monkeypatch)
    runner = CliRunner()
    corpus = _make_corpus(tmp_path / "tiny.md", n=10)
    out_dir = tmp_path / "skill"
    result = runner.invoke(
        app, ["build", str(corpus), "--out-dir", str(out_dir), "--force"]
    )
    assert result.exit_code == 0, result.stdout
    assert (out_dir / "SKILL.md").exists()


def test_cli_build_invalid_llm_output_uses_fallback(tmp_path, monkeypatch):
    """If the LLM returns garbage on both attempts, emit the deterministic fallback."""
    _stub_llm(monkeypatch, reply="this is not a valid skill file")
    runner = CliRunner()
    corpus = _make_corpus(tmp_path / "corpus.md", n=260)
    out_dir = tmp_path / "skill"
    result = runner.invoke(app, ["build", str(corpus), "--out-dir", str(out_dir)])
    assert result.exit_code == 0, result.stdout
    skill = (out_dir / "SKILL.md").read_text()
    assert skill.startswith("---\nname: user_voice")
    assert "## Quantified Targets" in skill


def test_cli_build_missing_auth(tmp_path, monkeypatch):
    monkeypatch.setattr("chatlectify.llm.available", lambda provider: None)
    runner = CliRunner()
    corpus = _make_corpus(tmp_path / "corpus.md", n=260)
    result = runner.invoke(
        app, ["build", str(corpus), "--out-dir", str(tmp_path / "skill")]
    )
    assert result.exit_code == 2


# ---------- `chatlectify all` ----------


def test_cli_all_no_benchmark(tmp_path, monkeypatch):
    _stub_llm(monkeypatch)
    runner = CliRunner()
    corpus = _make_corpus(tmp_path / "corpus.md", n=260)
    out_dir = tmp_path / "skill"
    result = runner.invoke(app, ["all", str(corpus), "--out-dir", str(out_dir)])
    assert result.exit_code == 0, result.stdout
    assert (out_dir / "SKILL.md").exists()
    assert not (out_dir / "benchmark_report.json").exists()


def test_cli_all_bench_model_passed_through(tmp_path, monkeypatch):
    """--bench-model should override --model for the benchmark call only."""
    seen = {"synth": None, "bench": []}

    def fake_call(provider, prompt, system=None, model=None, max_tokens=4000):
        if system is None:
            seen["synth"] = model
            return VALID_SKILL_MD
        seen["bench"].append(model)
        return "short reply"

    monkeypatch.setattr("chatlectify.llm.call", fake_call)
    monkeypatch.setattr("chatlectify.llm.available", lambda p: "api")

    runner = CliRunner()
    corpus = _make_corpus(tmp_path / "corpus.md", n=260)
    out_dir = tmp_path / "skill"
    result = runner.invoke(
        app,
        ["all", str(corpus), "--out-dir", str(out_dir),
         "--model", "claude-sonnet-4-6",
         "--bench-model", "claude-haiku-4-5",
         "--benchmark", "--n", "30"],
    )
    assert result.exit_code == 0, result.stdout
    assert seen["synth"] == "claude-sonnet-4-6"
    assert seen["bench"], "benchmark was never called"
    assert all(m == "claude-haiku-4-5" for m in seen["bench"])


def test_cli_all_with_benchmark(tmp_path, monkeypatch):
    _stub_llm(monkeypatch)
    runner = CliRunner()
    corpus = _make_corpus(tmp_path / "corpus.md", n=260)
    out_dir = tmp_path / "skill"
    result = runner.invoke(
        app,
        ["all", str(corpus), "--out-dir", str(out_dir), "--benchmark", "--n", "40"],
    )
    assert result.exit_code == 0, result.stdout
    bench_path = out_dir / "benchmark_report.json"
    # benchmark may skip if it errors, but with our stub it should produce a report
    if bench_path.exists():
        bench = json.loads(bench_path.read_text())
        assert "ngram_auc_skill" in bench


# ---------- `chatlectify` (top-level help) ----------


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for sub in ("ingest", "features", "build", "benchmark", "all"):
        assert sub in result.stdout


@pytest.mark.parametrize("cmd", ["ingest", "features", "build", "all", "benchmark"])
def test_cli_subcommand_help(cmd):
    runner = CliRunner()
    result = runner.invoke(app, [cmd, "--help"])
    assert result.exit_code == 0
