"""Generate a data-backed insight from a computed history summary.

Only the computed summary (trends, ratios, gaps) is sent to the model — never
the raw set-by-set JSON. The model's job is to interpret numbers that are
already correct, and to reference them, not to do arithmetic on raw data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.analysis.analytics import HistorySummary, InsufficientData, build_summary
from app.config import Settings, get_settings
from app.llm.client import LLMClient, Usage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a strength coach analysing a client's training history. You are given a \
PRE-COMPUTED summary of their data — trends, estimated 1RMs, volume shares, and \
balance ratios — not the raw logs.

Rules:
- Base every claim on the numbers in the summary. Reference specific figures \
(percentages, ratios, dates, kg) so the client can see the evidence.
- Do not invent data or compute new numbers the summary doesn't contain.
- If the summary flags a warning (e.g. mixed units, insufficient data), respect \
it — don't over-claim precision it doesn't support.
- Be direct and practical, like a coach reviewing a log with a client. Answer \
the specific question asked.
- Note in `data_points_used` the specific figures you relied on."""

INSIGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "insight": {
            "type": "string",
            "description": "The answer to the question, referencing specific numbers.",
        },
        "data_points_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The specific figures cited, e.g. 'push_to_pull_ratio 8.67'.",
        },
    },
    "required": ["insight", "data_points_used"],
    "additionalProperties": False,
}

EMPTY_HISTORY_MESSAGE = (
    "There's no workout history to analyse yet. Once a few sessions are logged, "
    "I can look at strength trends, training balance, and where to focus next."
)


@dataclass
class InsightResult:
    status: str  # answered | insufficient_data
    insight: str
    data_points_used: list[str]
    summary: dict | None  # the computed summary, returned for transparency
    usage: Usage

    def as_dict(self, settings: Settings) -> dict[str, object]:
        return {
            "status": self.status,
            "insight": self.insight,
            "data_points_used": self.data_points_used,
            "summary": self.summary,
            "usage": self.usage.as_dict(settings),
        }


class HistoryAnalyzer:
    def __init__(
        self, *, client: LLMClient, settings: Settings | None = None
    ) -> None:
        self._client = client
        self._settings = settings or get_settings()

    async def analyze(self, workouts: list[dict], question: str) -> InsightResult:
        usage = Usage()

        try:
            summary: HistorySummary = build_summary(workouts)
        except InsufficientData as exc:
            logger.info("insufficient history: %s", exc)
            return InsightResult(
                status="insufficient_data",
                insight=EMPTY_HISTORY_MESSAGE,
                data_points_used=[],
                summary=None,
                usage=usage,
            )

        summary_dict = summary.as_dict()
        raw = await self._client.structured(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Computed summary:\n{json.dumps(summary_dict, indent=2)}\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
            INSIGHT_SCHEMA,
            schema_name="history_insight",
            usage=usage,
        )

        try:
            payload = json.loads(raw)
            insight = str(payload["insight"]).strip()
            points = [str(p) for p in payload.get("data_points_used", [])]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("insight payload unparseable, returning raw text")
            insight, points = raw.strip(), []

        return InsightResult(
            status="answered",
            insight=insight,
            data_points_used=points,
            summary=summary_dict,
            usage=usage,
        )
