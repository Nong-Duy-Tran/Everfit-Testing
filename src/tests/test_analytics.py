"""Deterministic analytics — the pre-processing the brief grades most heavily.

These assert the computed numbers directly (no LLM), so a regression in unit
handling, e1RM, or balance is caught here rather than surfacing as a wrong-but-
plausible insight downstream.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analysis.analytics import InsufficientData, build_summary

SAMPLE = Path(__file__).resolve().parents[2] / "sample-data" / "workout-history.json"


def load(user_id: str) -> list[dict]:
    return json.loads(SAMPLE.read_text())["users"][user_id]["workouts"]


def test_empty_history_raises():
    with pytest.raises(InsufficientData):
        build_summary([])


def test_units_are_normalised_to_kg():
    """A 100 lb set must become ~45.4 kg, not stay 100. Without this, user B's
    110 lb opening bench reads heavier than his kg sessions and inverts the trend."""
    s = build_summary(
        [{"date": "2026-01-01", "exercise": "Bench Press",
          "sets": [{"reps": 10, "weight": 100, "unit": "lb"}]}]
    )
    bench = s.exercises[0]
    # 100 lb = 45.36 kg; e1RM = 45.36 * (1 + 10/30) = 60.5
    assert bench.best_weight_kg == pytest.approx(45.36, abs=0.1)
    assert bench.best_e1rm == pytest.approx(60.5, abs=0.5)


def test_mixed_units_raise_a_warning():
    s = build_summary(load("user_b"))
    assert any("kg and lb" in w for w in s.warnings)


def test_epley_1rm_matches_knowledge_base_example():
    """KB 17-one-rep-max: 80 kg x 8 reps -> 101 kg (Epley)."""
    s = build_summary(
        [{"date": "2026-01-01", "exercise": "Bench Press",
          "sets": [{"reps": 8, "weight": 80, "unit": "kg"}]}]
    )
    assert s.exercises[0].best_e1rm == pytest.approx(101.3, abs=0.5)


def test_progressive_overload_reads_as_increasing():
    s = build_summary(load("user_a"))
    bench = next(e for e in s.exercises if e.exercise == "Bench Press")
    assert bench.direction == "increasing"
    assert bench.e1rm_change_pct is not None and bench.e1rm_change_pct > 10


def test_chest_dominant_user_flagged_as_imbalanced():
    """User B is the answer key for 'overtraining chest vs back' and 'neglecting'."""
    s = build_summary(load("user_b"))
    balance = s.balance.as_dict()
    assert balance["chest_to_back_ratio"] > 3
    assert balance["push_to_pull_ratio"] > 3
    assert "pull" in s.neglected_movements
    assert "legs" in s.neglected_movements


def test_balanced_user_not_flagged_as_neglecting():
    s = build_summary(load("user_a"))
    assert s.neglected_movements == []


def test_single_session_has_no_trend():
    s = build_summary(
        [{"date": "2026-01-01", "exercise": "Squat",
          "sets": [{"reps": 5, "weight": 100, "unit": "kg"}]}]
    )
    assert s.exercises[0].direction == "insufficient_data"
    assert s.exercises[0].e1rm_change_pct is None


def test_unknown_exercise_surfaced_not_silently_bucketed():
    s = build_summary(
        [{"date": "2026-01-01", "exercise": "Zercher Carry",
          "sets": [{"reps": 5, "weight": 60, "unit": "kg"}]}]
    )
    assert "Zercher Carry" in s.unknown_exercises
    # Unknown exercise must not distort movement balance.
    assert sum(s.balance.by_movement.values()) == 0


def test_high_rep_e1rm_is_flagged_as_less_reliable():
    s = build_summary(
        [{"date": "2026-01-01", "exercise": "Leg Press",
          "sets": [{"reps": 20, "weight": 200, "unit": "kg"}]}]
    )
    assert s.exercises[0].high_rep_estimate is True


def test_longest_gap_detected():
    s = build_summary(
        [
            {"date": "2026-01-01", "exercise": "Squat", "sets": [{"reps": 5, "weight": 100, "unit": "kg"}]},
            {"date": "2026-01-15", "exercise": "Squat", "sets": [{"reps": 5, "weight": 100, "unit": "kg"}]},
        ]
    )
    assert s.longest_gap_days == 14


def test_pull_only_history_has_null_push_pull_ratio():
    """No pull volume must not divide-by-zero; ratio is null, not inf."""
    s = build_summary(
        [{"date": "2026-01-01", "exercise": "Bench Press",
          "sets": [{"reps": 5, "weight": 60, "unit": "kg"}]}]
    )
    assert s.balance.as_dict()["push_to_pull_ratio"] is None
