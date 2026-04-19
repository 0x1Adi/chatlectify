from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Compile AI chat history into a writing-style skill.")


def _require_key(provider: str) -> str:
    env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    key = os.environ.get(env)
    if not key:
        typer.echo(f"error: missing {env}", err=True)
        raise typer.Exit(code=2)
    return key


@app.command()
def ingest(
    input_path: Path = typer.Argument(..., exists=True),
    format: str = typer.Option("auto", "--format"),
    out: Path = typer.Option(Path("normalized.json"), "--out"),
):
    from .ingest import ingest as _ing
    msgs = _ing(input_path, format)
    out.write_text(json.dumps([m.model_dump(mode="json") for m in msgs], indent=2))
    typer.echo(f"ingested {len(msgs)} messages -> {out}")


@app.command()
def features(
    input: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(Path("features.json"), "--out"),
):
    from .clean import clean
    from .features import extract
    from .ingest import ingest as _ing
    from .schemas import Message

    if input.suffix == ".json":
        try:
            data = json.loads(input.read_text())
            if isinstance(data, list) and data and "word_count" in data[0]:
                msgs = [Message(**m) for m in data]
            else:
                msgs = _ing(input)
        except Exception:
            msgs = _ing(input)
    else:
        msgs = _ing(input)
    kept, _, stats = clean(msgs)
    feats = extract(kept)
    out.write_text(feats.model_dump_json(indent=2))
    typer.echo(f"features -> {out} (kept={stats['kept']})")


@app.command()
def build(
    input: Path = typer.Argument(..., exists=True),
    out_dir: Path = typer.Option(..., "--out-dir"),
    provider: str = typer.Option("anthropic", "--provider"),
    model: str = typer.Option("", "--model"),
    force: bool = typer.Option(False, "--force"),
):
    from .clean import clean
    from .emit import write_outputs
    from .features import extract
    from .gates import run_gates
    from .ingest import ingest as _ing
    from .synth import synthesize

    _require_key(provider)
    msgs = _ing(input)
    kept, pastes, stats = clean(msgs)
    feats = extract(kept)
    report = run_gates(kept, feats, stats["paste_contamination_pct"])
    if not report.passed and not force:
        typer.echo(f"gate failures: {report.errors}", err=True)
        raise typer.Exit(code=1)
    skill_md = synthesize(feats, kept, provider=provider, model=model or None)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(out_dir, skill_md, feats, report, kept, pastes, stats, benchmark=None)
    typer.echo(f"skill written -> {out_dir}")


@app.command()
def benchmark(
    skill: Path = typer.Option(..., "--skill", exists=True),
    user_samples: Path = typer.Option(..., "--user-samples", exists=True),
    provider: str = typer.Option("anthropic", "--provider"),
    n: int = typer.Option(100, "--n"),
):
    from .benchmark import run_benchmark
    from .schemas import Message

    _require_key(provider)
    data = json.loads(user_samples.read_text())
    msgs = [Message(**m) for m in data]
    skill_body = skill.read_text()
    report = run_benchmark(skill_body, msgs, provider=provider, n=n)
    typer.echo(report.model_dump_json(indent=2))


@app.command("all")
def all_cmd(
    input: Path = typer.Argument(..., exists=True),
    out_dir: Path = typer.Option(..., "--out-dir"),
    provider: str = typer.Option("anthropic", "--provider"),
    model: str = typer.Option("", "--model"),
    skip_benchmark: bool = typer.Option(False, "--skip-benchmark"),
    force: bool = typer.Option(False, "--force"),
):
    from .benchmark import run_benchmark
    from .clean import clean
    from .emit import write_outputs
    from .features import extract
    from .gates import run_gates
    from .ingest import ingest as _ing
    from .synth import synthesize

    _require_key(provider)
    msgs = _ing(input)
    kept, pastes, stats = clean(msgs)
    feats = extract(kept)
    report = run_gates(kept, feats, stats["paste_contamination_pct"])
    if not report.passed and not force:
        typer.echo(f"gate failures: {report.errors}", err=True)
        raise typer.Exit(code=1)
    skill_md = synthesize(feats, kept, provider=provider, model=model or None)
    bench = None
    if not skip_benchmark:
        try:
            bench = run_benchmark(skill_md, kept, provider=provider, model=model or None)
        except Exception as e:
            typer.echo(f"benchmark skipped: {e}", err=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(out_dir, skill_md, feats, report, kept, pastes, stats, benchmark=bench)
    typer.echo(f"done -> {out_dir}")


def main():
    app()


if __name__ == "__main__":
    sys.exit(main())
