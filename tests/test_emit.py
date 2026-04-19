"""Unit tests for emit.write_outputs."""
import json

from chatlectify.emit import write_outputs
from chatlectify.schemas import BenchmarkReport, GateReport, Message, StyleFeatures


def _msgs(n: int = 10) -> list[Message]:
    return [
        Message(msg_id=str(i), conv_id="c", role="human",
                text=f"message {i} text body here", word_count=5)
        for i in range(n)
    ]


def _feats(n: int = 10) -> StyleFeatures:
    return StyleFeatures(
        ttr=0.5, avg_word_len=4.0, top_unigrams=[("the", 3)], top_bigrams=[],
        contraction_rate=1.0, avg_sent_len=10.0, sent_len_std=2.0,
        punct_hist={".": 5.0}, cap_start_ratio=0.9, bullet_rate=0.0,
        header_rate=0.0, fence_rate=0.0, avg_line_breaks=1.0,
        top_sent_starters=[], hedge_rate=0.0, emoji_rate=0.0,
        typo_rate=0.0, imperative_rate=0.1, question_rate=0.1,
        avg_msg_words=5.0, avg_msg_words_std=1.0, msg_count=n,
    )


def _gate() -> GateReport:
    return GateReport(
        passed=True, msg_count=10, char_count=1000,
        paste_contamination_pct=0.0, warnings=[], errors=[],
    )


SKILL_WITH_FRONTMATTER = """---
name: my_voice
description: test.
---

## Style Rules
- Be brief.
"""


def test_write_outputs_all_artifacts(tmp_path):
    out = tmp_path / "skill"
    msgs = _msgs(10)
    write_outputs(out, SKILL_WITH_FRONTMATTER, _feats(), _gate(), msgs, [], {"kept": 10})

    for name in ("SKILL.md", "system_prompt.txt", "style_metrics.json",
                 "gate_report.json", "exemplars.json", "pastes.json",
                 "pipeline_report.json"):
        assert (out / name).exists(), f"missing {name}"
    assert not (out / "benchmark_report.json").exists()


def test_write_outputs_strips_frontmatter_from_system_prompt(tmp_path):
    out = tmp_path / "skill"
    write_outputs(out, SKILL_WITH_FRONTMATTER, _feats(), _gate(), _msgs(), [], {})
    sys_prompt = (out / "system_prompt.txt").read_text()
    assert sys_prompt.startswith("Write in this style:")
    assert "---" not in sys_prompt
    assert "name: my_voice" not in sys_prompt
    assert "## Style Rules" in sys_prompt


def test_write_outputs_with_benchmark(tmp_path):
    out = tmp_path / "skill"
    bench = BenchmarkReport(
        ngram_auc_baseline=0.9, ngram_auc_skill=0.6, ngram_auc_delta=0.3,
        feature_dist_baseline=0.5, feature_dist_skill=0.2,
        feature_dist_improvement=0.6,
        pass_criteria={"ngram": True, "feature": True}, overall_pass=True,
    )
    write_outputs(out, SKILL_WITH_FRONTMATTER, _feats(), _gate(), _msgs(), [],
                  {"kept": 10}, benchmark=bench)
    loaded = json.loads((out / "benchmark_report.json").read_text())
    assert loaded["overall_pass"] is True
    pipeline = json.loads((out / "pipeline_report.json").read_text())
    assert pipeline["benchmark"]["overall_pass"] is True


def test_write_outputs_creates_nested_dir(tmp_path):
    out = tmp_path / "a" / "b" / "c"
    write_outputs(out, SKILL_WITH_FRONTMATTER, _feats(), _gate(), _msgs(), [], {})
    assert (out / "SKILL.md").exists()
