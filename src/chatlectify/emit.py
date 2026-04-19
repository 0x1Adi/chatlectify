
import json
import re
from pathlib import Path

from .schemas import BenchmarkReport, GateReport, Message, StyleFeatures

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def write_outputs(
    out_dir: Path,
    skill_md: str,
    feats: StyleFeatures,
    gate: GateReport,
    msgs: list[Message],
    pastes: list[Message],
    clean_stats: dict,
    benchmark: BenchmarkReport | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SKILL.md").write_text(skill_md)

    body = _FRONTMATTER_RE.sub("", skill_md, count=1).strip()
    (out_dir / "system_prompt.txt").write_text("Write in this style:\n\n" + body)
    (out_dir / "style_metrics.json").write_text(feats.model_dump_json(indent=2))
    (out_dir / "gate_report.json").write_text(gate.model_dump_json(indent=2))
    if benchmark is not None:
        (out_dir / "benchmark_report.json").write_text(benchmark.model_dump_json(indent=2))

    from .synth import stratified_exemplars
    ex = stratified_exemplars(msgs)
    (out_dir / "exemplars.json").write_text(
        json.dumps([m.model_dump(mode="json") for m in ex], indent=2)
    )
    (out_dir / "pastes.json").write_text(
        json.dumps([m.model_dump(mode="json") for m in pastes], indent=2)
    )

    pipeline = {
        "clean": clean_stats,
        "features": {"msg_count": feats.msg_count},
        "gate": {"passed": gate.passed, "errors": gate.errors, "warnings": gate.warnings},
        "benchmark": benchmark.model_dump() if benchmark else None,
    }
    (out_dir / "pipeline_report.json").write_text(json.dumps(pipeline, indent=2, default=str))
