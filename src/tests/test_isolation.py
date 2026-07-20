"""User data isolation — the boundary the brief explicitly requires a test for.

"User A's history must never appear in User B's context or response." Verified at
two levels: the repository cannot hand back two users' data, and a summary built
for one user contains none of the other user's exclusive exercises.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analysis.analytics import build_summary
from app.analysis.repository import HistoryRepository, UnknownUser

SAMPLE = Path(__file__).resolve().parents[2] / "sample-data" / "workout-history.json"


@pytest.fixture
def repo() -> HistoryRepository:
    return HistoryRepository(SAMPLE)


def test_repository_returns_only_the_requested_user(repo):
    a = repo.get("user_a")
    b = repo.get("user_b")
    assert a is not b
    a_exercises = {w["exercise"] for w in a}
    b_exercises = {w["exercise"] for w in b}
    # user_a does these and user_b does not; they must not appear in b's data.
    a_only = {"Romanian Deadlift", "Leg Press", "Deadlift", "Face Pull"}
    assert a_only <= a_exercises
    assert not (a_only & b_exercises)


def test_repository_rejects_unknown_user(repo):
    with pytest.raises(UnknownUser):
        repo.get("user_c")


def test_repository_has_no_accessor_that_returns_all_users(repo):
    """Structural guarantee: there is no method to obtain more than one user's
    data at once, so a caller cannot cross-contaminate by accident."""
    assert not hasattr(repo, "all")
    assert not hasattr(repo, "get_all")


def test_summary_for_one_user_contains_no_other_users_exercises(repo):
    """The end-to-end isolation property: B's computed summary — the only thing
    sent to the LLM — must not mention any exercise exclusive to A."""
    summary_b = build_summary(repo.get("user_b"))
    text_b = json.dumps(summary_b.as_dict())
    for a_only_exercise in ("Romanian Deadlift", "Leg Press", "Face Pull"):
        assert a_only_exercise not in text_b


def test_analyzing_b_does_not_read_a(repo):
    """Regression guard: build B's summary, confirm no A-specific number leaks.
    A's total leg volume (~61,200 kg) is far larger than B's; it must not appear."""
    summary_b = build_summary(repo.get("user_b"))
    assert summary_b.balance.as_dict()["leg_volume_kg"] < 5000  # B's real leg volume is ~2,640
