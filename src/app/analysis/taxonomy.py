"""Exercise taxonomy — movement pattern and muscle group per exercise.

Covers the 13 exercises in the sample data. Two questions the brief asks about
history depend on this mapping: "which exercises am I neglecting?" (movement
balance) and "am I overtraining chest compared to back?" (muscle group).

`chest` and `back` are broken out separately from the push/pull movement split,
because "chest vs back" is a narrower comparison than "push vs pull" — triceps
and shoulders are push but not chest.
"""

from __future__ import annotations

from dataclasses import dataclass

PUSH = "push"
PULL = "pull"
LEGS = "legs"


@dataclass(frozen=True)
class ExerciseInfo:
    movement: str  # push | pull | legs
    muscle_group: str  # chest | back | shoulders | legs | arms
    compound: bool


EXERCISES: dict[str, ExerciseInfo] = {
    "Bench Press": ExerciseInfo(PUSH, "chest", True),
    "Incline Dumbbell Press": ExerciseInfo(PUSH, "chest", True),
    "Overhead Press": ExerciseInfo(PUSH, "shoulders", True),
    "Lateral Raise": ExerciseInfo(PUSH, "shoulders", False),
    "Tricep Pushdown": ExerciseInfo(PUSH, "arms", False),
    "Barbell Row": ExerciseInfo(PULL, "back", True),
    "Pull-Up": ExerciseInfo(PULL, "back", True),
    "Face Pull": ExerciseInfo(PULL, "back", False),
    "Bicep Curl": ExerciseInfo(PULL, "arms", False),
    "Squat": ExerciseInfo(LEGS, "legs", True),
    "Deadlift": ExerciseInfo(LEGS, "legs", True),
    "Romanian Deadlift": ExerciseInfo(LEGS, "legs", True),
    "Leg Press": ExerciseInfo(LEGS, "legs", True),
}


def classify(exercise: str) -> ExerciseInfo | None:
    """Return taxonomy for an exercise, or None if it isn't recognised.

    Unknown exercises are surfaced explicitly by the analytics layer rather than
    silently bucketed, so a future exercise doesn't quietly skew balance ratios.
    """
    return EXERCISES.get(exercise)
