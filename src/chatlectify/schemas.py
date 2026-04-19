from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    msg_id: str
    conv_id: str
    timestamp: datetime | None = None
    role: Literal["human", "assistant"]
    text: str
    word_count: int


class StyleFeatures(BaseModel):
    ttr: float
    avg_word_len: float
    top_unigrams: list[tuple[str, int]]
    top_bigrams: list[tuple[str, int]]
    contraction_rate: float
    avg_sent_len: float
    sent_len_std: float
    punct_hist: dict[str, float]
    cap_start_ratio: float
    bullet_rate: float
    header_rate: float
    fence_rate: float
    avg_line_breaks: float
    top_sent_starters: list[tuple[str, int]]
    hedge_rate: float
    emoji_rate: float
    typo_rate: float
    imperative_rate: float
    question_rate: float
    avg_msg_words: float
    avg_msg_words_std: float
    msg_count: int


class GateReport(BaseModel):
    passed: bool
    msg_count: int
    char_count: int
    paste_contamination_pct: float
    warnings: list[str]
    errors: list[str]


class BenchmarkReport(BaseModel):
    ngram_auc_baseline: float
    ngram_auc_skill: float
    ngram_auc_delta: float
    feature_dist_baseline: float
    feature_dist_skill: float
    feature_dist_improvement: float
    pass_criteria: dict[str, bool]
    overall_pass: bool
