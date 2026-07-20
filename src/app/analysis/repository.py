"""User workout-history store, and the data-isolation boundary.

The brief requires that one user's history never appears in another user's
context or response. That guarantee is enforced here, structurally: `get()`
returns only the requested user's workouts, and there is no method that returns
more than one user's data at once. Analysis is pure over whatever list `get()`
hands back, so a summary built for user B can only contain user B's numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings


class UnknownUser(Exception):
    pass


class HistoryRepository:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or get_settings().workout_history_path
        self._users: dict[str, list[dict]] = {}
        self._load()

    def _load(self) -> None:
        data = json.loads(self._path.read_text(encoding="utf-8"))
        for user_id, payload in data.get("users", {}).items():
            self._users[user_id] = payload.get("workouts", [])

    def user_ids(self) -> list[str]:
        return sorted(self._users)

    def get(self, user_id: str) -> list[dict]:
        """Return one user's workouts. Raises UnknownUser if absent.

        This is the only accessor. There is deliberately no `all()` — the
        isolation guarantee is that a caller cannot obtain two users' data
        through this repository at once.
        """
        if user_id not in self._users:
            raise UnknownUser(user_id)
        return self._users[user_id]
