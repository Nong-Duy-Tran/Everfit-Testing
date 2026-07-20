"""Deterministic pre-processing of workout history.

The brief is explicit: parse and analyse the data before the LLM sees it — do
not dump raw JSON into the prompt. This module is that layer. It takes the raw
history and produces a compact, computed summary (trends, e1RM, volume, balance,
gaps); only that summary is passed to the model.

Everything here is pure and deterministic, so the numbers in a response are
reproducible and testable independent of the LLM.

Unit handling is a correctness gate, not an edge case: the sample data mixes kg
and lb within one user (a 110 lb bench read as 110 kg would invert the trend).
All weights are normalised to kg at parse time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from app.analysis.taxonomy import LEGS, PULL, PUSH, classify

LB_TO_KG = 0.453592


class InsufficientData(Exception):
    """Raised when there is not enough history to analyse at all (empty)."""


@dataclass
class SetStat:
    reps: int
    weight_kg: float

    @property
    def volume(self) -> float:
        return self.reps * self.weight_kg

    @property
    def epley_1rm(self) -> float:
        # Epley, straight from the knowledge base (17-one-rep-max). Most accurate
        # for <=10 reps; flagged in the summary when best sets exceed that.
        return self.weight_kg * (1 + self.reps / 30)


@dataclass
class SessionStat:
    day: date
    sets: list[SetStat]

    @property
    def top_set(self) -> SetStat:
        # "Top" by estimated 1RM rather than raw weight, so a heavy low-rep set
        # and a lighter high-rep set are compared on the same footing.
        return max(self.sets, key=lambda s: s.epley_1rm)

    @property
    def volume(self) -> float:
        return sum(s.volume for s in self.sets)


@dataclass
class ExerciseTrend:
    exercise: str
    sessions: int
    first_day: date
    last_day: date
    first_e1rm: float
    last_e1rm: float
    best_e1rm: float
    best_weight_kg: float
    total_volume: float
    high_rep_estimate: bool  # best set > 10 reps → e1RM less reliable

    @property
    def e1rm_change_pct(self) -> float | None:
        if self.sessions < 2 or self.first_e1rm == 0:
            return None
        return (self.last_e1rm - self.first_e1rm) / self.first_e1rm * 100

    @property
    def direction(self) -> str:
        pct = self.e1rm_change_pct
        if pct is None:
            return "insufficient_data"
        if pct >= 2.5:
            return "increasing"
        if pct <= -2.5:
            return "decreasing"
        return "flat"

    def as_dict(self) -> dict[str, object]:
        return {
            "exercise": self.exercise,
            "sessions": self.sessions,
            "date_range": f"{self.first_day.isoformat()} to {self.last_day.isoformat()}",
            "first_e1rm_kg": round(self.first_e1rm, 1),
            "last_e1rm_kg": round(self.last_e1rm, 1),
            "best_e1rm_kg": round(self.best_e1rm, 1),
            "best_weight_kg": round(self.best_weight_kg, 1),
            "e1rm_change_pct": (
                None if self.e1rm_change_pct is None else round(self.e1rm_change_pct, 1)
            ),
            "trend": self.direction,
            "total_volume_kg": round(self.total_volume),
            "e1rm_note": (
                "best set above 10 reps, e1RM less reliable"
                if self.high_rep_estimate
                else None
            ),
        }


@dataclass
class BalanceStat:
    by_movement: dict[str, float]  # push/pull/legs -> volume
    by_muscle_group: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        total = sum(self.by_movement.values()) or 1.0
        push = self.by_movement.get(PUSH, 0.0)
        pull = self.by_movement.get(PULL, 0.0)
        legs = self.by_movement.get(LEGS, 0.0)
        return {
            "volume_share_pct": {
                "push": round(push / total * 100),
                "pull": round(pull / total * 100),
                "legs": round(legs / total * 100),
            },
            "push_to_pull_ratio": (
                round(push / pull, 2) if pull else None  # None => no pull volume at all
            ),
            "chest_to_back_ratio": self._ratio("chest", "back"),
            "leg_volume_kg": round(legs),
        }

    def _ratio(self, a: str, b: str) -> float | None:
        va = self.by_muscle_group.get(a, 0.0)
        vb = self.by_muscle_group.get(b, 0.0)
        if vb == 0:
            return None
        return round(va / vb, 2)


@dataclass
class HistorySummary:
    total_sessions: int
    date_range: tuple[date, date]
    exercises: list[ExerciseTrend]
    balance: BalanceStat
    neglected_movements: list[str]
    unknown_exercises: list[str]
    longest_gap_days: int
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "total_sessions": self.total_sessions,
            "date_range": f"{self.date_range[0].isoformat()} to {self.date_range[1].isoformat()}",
            "longest_gap_days": self.longest_gap_days,
            "per_exercise": [e.as_dict() for e in self.exercises],
            "balance": self.balance.as_dict(),
            "neglected_movements": self.neglected_movements,
            "unknown_exercises": self.unknown_exercises,
            "warnings": self.warnings,
        }


def _to_kg(weight: float, unit: str) -> float:
    unit = (unit or "kg").lower()
    if unit == "lb":
        return weight * LB_TO_KG
    return weight


def _parse_day(value: str) -> date:
    return date.fromisoformat(value)


def build_summary(workouts: list[dict]) -> HistorySummary:
    """Turn a raw workout-history array into a computed summary.

    `workouts` is a list of {date, exercise, sets:[{reps, weight, unit}]}.
    Raises InsufficientData if there is nothing to analyse.
    """
    if not workouts:
        raise InsufficientData("history is empty")

    warnings: list[str] = []
    mixed_units = {
        (s.get("unit") or "kg").lower()
        for w in workouts
        for s in w.get("sets", [])
    }
    if "lb" in mixed_units and "kg" in mixed_units:
        warnings.append("history mixes kg and lb; all weights normalised to kg")

    # Group sessions per exercise, keyed by day.
    per_exercise_days: dict[str, dict[date, SessionStat]] = defaultdict(dict)
    all_days: set[date] = set()
    unknown: set[str] = set()

    for entry in workouts:
        exercise = entry.get("exercise", "").strip()
        try:
            day = _parse_day(entry["date"])
        except (KeyError, ValueError):
            warnings.append(f"skipped entry with invalid date: {entry.get('date')!r}")
            continue

        sets = [
            SetStat(reps=int(s["reps"]), weight_kg=_to_kg(float(s["weight"]), s.get("unit", "kg")))
            for s in entry.get("sets", [])
            if s.get("reps") is not None and s.get("weight") is not None
        ]
        if not sets:
            continue

        all_days.add(day)
        if classify(exercise) is None:
            unknown.add(exercise)

        # A day may appear once per exercise; merge sets if repeated.
        if day in per_exercise_days[exercise]:
            per_exercise_days[exercise][day].sets.extend(sets)
        else:
            per_exercise_days[exercise][day] = SessionStat(day=day, sets=sets)

    if not all_days:
        raise InsufficientData("no analysable sets in history")

    trends = [_exercise_trend(ex, days) for ex, days in per_exercise_days.items()]
    trends.sort(key=lambda t: t.sessions, reverse=True)

    balance = _balance(per_exercise_days)
    neglected = _neglected(balance)
    longest_gap = _longest_gap(sorted(all_days))

    return HistorySummary(
        total_sessions=len(all_days),
        date_range=(min(all_days), max(all_days)),
        exercises=trends,
        balance=balance,
        neglected_movements=neglected,
        unknown_exercises=sorted(unknown),
        longest_gap_days=longest_gap,
        warnings=warnings,
    )


def _exercise_trend(exercise: str, days: dict[date, SessionStat]) -> ExerciseTrend:
    ordered = [days[d] for d in sorted(days)]
    top_sets = [s.top_set for s in ordered]
    return ExerciseTrend(
        exercise=exercise,
        sessions=len(ordered),
        first_day=ordered[0].day,
        last_day=ordered[-1].day,
        first_e1rm=top_sets[0].epley_1rm,
        last_e1rm=top_sets[-1].epley_1rm,
        best_e1rm=max(s.epley_1rm for s in top_sets),
        best_weight_kg=max(s.weight_kg for s in top_sets),
        total_volume=sum(s.volume for s in ordered),
        high_rep_estimate=any(s.reps > 10 for s in top_sets),
    )


def _balance(per_exercise_days: dict[str, dict[date, SessionStat]]) -> BalanceStat:
    by_movement: dict[str, float] = defaultdict(float)
    by_muscle: dict[str, float] = defaultdict(float)
    for exercise, days in per_exercise_days.items():
        info = classify(exercise)
        if info is None:
            continue  # unknown exercises don't skew balance ratios
        vol = sum(s.volume for s in days.values())
        by_movement[info.movement] += vol
        by_muscle[info.muscle_group] += vol
    return BalanceStat(by_movement=dict(by_movement), by_muscle_group=dict(by_muscle))


def _neglected(balance: BalanceStat) -> list[str]:
    """Movement categories with no volume, or under 10% of total."""
    total = sum(balance.by_movement.values()) or 1.0
    neglected = []
    for movement in (PUSH, PULL, LEGS):
        share = balance.by_movement.get(movement, 0.0) / total
        if share < 0.10:
            neglected.append(movement)
    return neglected


def _longest_gap(sorted_days: list[date]) -> int:
    if len(sorted_days) < 2:
        return 0
    return max(
        (sorted_days[i + 1] - sorted_days[i]).days for i in range(len(sorted_days) - 1)
    )
