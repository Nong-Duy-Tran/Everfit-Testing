"""Offline unit tests for the deterministic eval metrics."""

from __future__ import annotations

from app.eval.dataset import EvalCase
from app.eval.metrics import (
    attribution_present,
    status_correct,
    values_grounded,
)


def test_status_correct_matches_refusal_category():
    case = EvalCase(id="x", category="adversarial", endpoint="/ask",
                    question="q", expect_status="refused",
                    expect_refusal_category="injury_rehab")
    good = {"status": "refused", "refusal_category": "injury_rehab"}
    bad = {"status": "refused", "refusal_category": "eating_disorder"}
    assert status_correct(case, good).passed is True
    assert status_correct(case, bad).passed is False


def test_attribution_requires_used_source_for_rag():
    case = EvalCase(id="x", category="rag", endpoint="/ask", question="q")
    assert attribution_present(case, {"sources": [{"used": True}]}).passed is True
    assert attribution_present(case, {"sources": [{"used": False}]}).passed is False


def test_values_grounded_flags_fabricated_numbers():
    case = EvalCase(id="x", category="analysis", endpoint="/analyze",
                    user_id="u", question="q")
    grounded = {
        "summary": {"balance": {"push_to_pull_ratio": 8.67}},
        "data_points_used": ["push_to_pull_ratio 8.67"],
    }
    fabricated = {
        "summary": {"balance": {"push_to_pull_ratio": 8.67}},
        "data_points_used": ["push_to_pull_ratio 99.9"],
    }
    assert values_grounded(case, grounded).passed is True
    assert values_grounded(case, fabricated).passed is False


def test_metrics_return_na_when_not_applicable():
    case = EvalCase(id="x", category="rag", endpoint="/ask", question="q")
    assert values_grounded(case, {}).passed is None  # not an analysis case
