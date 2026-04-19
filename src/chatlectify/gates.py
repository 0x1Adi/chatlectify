
from .schemas import GateReport, Message, StyleFeatures


def run_gates(msgs: list[Message], features: StyleFeatures, paste_pct: float) -> GateReport:
    errors: list[str] = []
    warnings: list[str] = []
    char_count = sum(len(m.text) for m in msgs)

    if features.msg_count < 200:
        errors.append("min_msgs=200 failed")
    if char_count < 20_000:
        errors.append("min_chars=20000 failed")
    if paste_pct > 0.5:
        errors.append(f"paste_contamination {paste_pct:.1%} > 50%")

    if 0.3 < paste_pct <= 0.5:
        warnings.append(f"paste_contamination {paste_pct:.1%}")
    if features.msg_count < 500:
        warnings.append("low sample; fidelity may degrade")
    if features.avg_msg_words < 3:
        warnings.append("mostly very short msgs")
    if features.ttr > 0.8:
        warnings.append("unusually high TTR")

    return GateReport(
        passed=len(errors) == 0,
        msg_count=features.msg_count,
        char_count=char_count,
        paste_contamination_pct=paste_pct,
        warnings=warnings,
        errors=errors,
    )
