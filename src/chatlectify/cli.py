
import json
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Compile AI chat history into a writing-style skill.")


def _key(provider: str) -> None:
    from .llm import available
    mode = available(provider)
    if mode is None:
        hint = ("ANTHROPIC_API_KEY or `claude` CLI" if provider == "anthropic"
                else "OPENAI_API_KEY or `codex` CLI")
        typer.echo(f"error: no auth for {provider}; need {hint}", err=True)
        raise typer.Exit(2)
    typer.echo(f"[{provider}] using {mode}", err=True)


def _load_msgs(path: Path):
    from .ingest import ingest as _ing
    from .schemas import Message
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list) and data and "word_count" in data[0]:
                return [Message(**m) for m in data]
        except Exception:
            pass
    return _ing(path)


@app.command()
def ingest(input_path: Path = typer.Argument(..., exists=True),
           format: str = typer.Option("auto", "--format"),
           out: Path = typer.Option(Path("normalized.json"), "--out")):
    from .ingest import ingest as _ing
    msgs = _ing(input_path, format)
    out.write_text(json.dumps([m.model_dump(mode="json") for m in msgs], indent=2))
    typer.echo(f"ingested {len(msgs)} messages -> {out}")


@app.command()
def features(input: Path = typer.Argument(..., exists=True),
             out: Path = typer.Option(Path("features.json"), "--out")):
    from .clean import clean
    from .features import extract
    msgs = _load_msgs(input)
    kept, _, stats = clean(msgs)
    feats = extract(kept)
    out.write_text(feats.model_dump_json(indent=2))
    typer.echo(f"features -> {out} (kept={stats['kept']})")


def _pipeline(input: Path, provider: str, model: str, force: bool, do_bench: bool, n: int = 100):
    from .benchmark import run_benchmark
    from .clean import clean
    from .features import extract
    from .gates import run_gates
    from .synth import synthesize
    msgs = _load_msgs(input)
    kept, pastes, stats = clean(msgs)
    feats = extract(kept)
    report = run_gates(kept, feats, stats["paste_contamination_pct"])
    if not report.passed and not force:
        typer.echo(f"gate failures: {report.errors}", err=True)
        raise typer.Exit(1)
    skill_md = synthesize(feats, kept, provider=provider, model=model or None)
    bench = None
    if do_bench:
        try:
            bench = run_benchmark(skill_md, kept, provider=provider, model=model or None, n=n)
        except Exception as e:
            typer.echo(f"benchmark skipped: {e}", err=True)
    return skill_md, feats, report, kept, pastes, stats, bench


@app.command()
def build(input: Path = typer.Argument(..., exists=True),
          out_dir: Path = typer.Option(..., "--out-dir"),
          provider: str = typer.Option("anthropic", "--provider"),
          model: str = typer.Option("", "--model"),
          force: bool = typer.Option(False, "--force")):
    from .emit import write_outputs
    _key(provider)
    skill_md, feats, report, kept, pastes, stats, _ = _pipeline(input, provider, model, force, False)
    write_outputs(out_dir, skill_md, feats, report, kept, pastes, stats)
    typer.echo(f"skill written -> {out_dir}")


@app.command()
def benchmark(skill: Path = typer.Option(..., "--skill", exists=True),
              user_samples: Path = typer.Option(..., "--user-samples", exists=True),
              provider: str = typer.Option("anthropic", "--provider"),
              n: int = typer.Option(100, "--n")):
    from .benchmark import run_benchmark
    from .schemas import Message
    _key(provider)
    msgs = [Message(**m) for m in json.loads(user_samples.read_text())]
    report = run_benchmark(skill.read_text(), msgs, provider=provider, n=n)
    typer.echo(report.model_dump_json(indent=2))


@app.command("all")
def all_cmd(input: Path = typer.Argument(..., exists=True),
            out_dir: Path = typer.Option(..., "--out-dir"),
            provider: str = typer.Option("anthropic", "--provider"),
            model: str = typer.Option("", "--model"),
            benchmark: bool = typer.Option(False, "--benchmark", help="run fidelity benchmark (2N LLM calls)"),
            force: bool = typer.Option(False, "--force"),
            n: int = typer.Option(100, "--n", help="benchmark sample size")):
    from .emit import write_outputs
    _key(provider)
    skill_md, feats, report, kept, pastes, stats, bench = _pipeline(
        input, provider, model, force, benchmark, n=n)
    write_outputs(out_dir, skill_md, feats, report, kept, pastes, stats, benchmark=bench)
    typer.echo(f"done -> {out_dir}")


def main():
    app()


if __name__ == "__main__":
    sys.exit(main())
