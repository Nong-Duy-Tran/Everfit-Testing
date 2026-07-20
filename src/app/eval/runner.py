"""Execute the eval set against the live app and aggregate metric results."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from fastapi.testclient import TestClient

from app.eval.dataset import DATASET, EvalCase
from app.eval.metrics import RULE_METRICS, judge
from app.llm.client import LLMClient, Usage

logger = logging.getLogger(__name__)


@dataclass
class CaseReport:
    case_id: str
    category: str
    question: str
    answer: str
    metrics: dict[str, Any] = field(default_factory=dict)
    judge_reasoning: str = ""

    @property
    def rule_pass(self) -> bool:
        applicable = [
            m for m in self.metrics.values()
            if m["type"] == "rule" and m["passed"] is not None
        ]
        return all(m["passed"] for m in applicable) if applicable else True

    @property
    def faithfulness(self) -> int | None:
        m = self.metrics.get("faithfulness")
        return m["score"] if m else None


def _call_endpoint(client: TestClient, case: EvalCase) -> dict:
    if case.endpoint == "/ask":
        return client.post("/ask", json={"question": case.question}).json()
    if case.endpoint == "/analyze":
        body: dict[str, Any] = {"question": case.question}
        if case.user_id is not None:
            body["user_id"] = case.user_id
        else:
            body["workouts"] = case.workouts or []
        return client.post("/analyze", json=body).json()
    if case.endpoint == "/agent":
        return client.post("/agent", json={"question": case.question}).json()
    raise ValueError(f"unknown endpoint {case.endpoint}")


async def run_evaluation(app, llm: LLMClient) -> dict[str, Any]:
    judge_usage = Usage()
    reports: list[CaseReport] = []

    with TestClient(app) as client:
        for case in DATASET:
            response = _call_endpoint(client, case)
            answer = str(response.get("answer") or response.get("insight") or "")

            report = CaseReport(
                case_id=case.id,
                category=case.category,
                question=case.question,
                answer=answer,
            )

            for metric_fn in RULE_METRICS:
                result = metric_fn(case, response)
                report.metrics[result.name] = {
                    "type": "rule",
                    "passed": result.passed,
                    "detail": result.detail,
                }

            faith, tone, reasoning = await judge(
                case, response, client=llm, usage=judge_usage
            )
            for result in (faith, tone):
                report.metrics[result.name] = {
                    "type": "judge",
                    "passed": result.passed,
                    "score": result.score,
                    "detail": result.detail,
                }
            report.judge_reasoning = reasoning
            reports.append(report)

    return _aggregate(reports, judge_usage)


def _aggregate(reports: list[CaseReport], judge_usage: Usage) -> dict[str, Any]:
    from app.config import get_settings

    settings = get_settings()

    # Per-metric pass rate across applicable cases.
    metric_names = ["status_correct", "attribution_present", "values_grounded",
                    "keywords_present", "tools_correct", "faithfulness", "tone"]
    per_metric: dict[str, Any] = {}
    for name in metric_names:
        applicable = [r for r in reports if r.metrics.get(name, {}).get("passed") is not None]
        passed = [r for r in applicable if r.metrics[name]["passed"]]
        per_metric[name] = {
            "applicable": len(applicable),
            "passed": len(passed),
            "pass_rate": round(len(passed) / len(applicable), 3) if applicable else None,
        }

    faith_scores = [r.faithfulness for r in reports if r.faithfulness]
    by_category: dict[str, dict[str, int]] = {}
    for r in reports:
        c = by_category.setdefault(r.category, {"total": 0, "rule_pass": 0})
        c["total"] += 1
        c["rule_pass"] += int(r.rule_pass)

    return {
        "n_cases": len(reports),
        "per_metric": per_metric,
        "avg_faithfulness": round(sum(faith_scores) / len(faith_scores), 2) if faith_scores else None,
        "by_category": by_category,
        "judge_cost": judge_usage.as_dict(settings),
        "cases": [
            {
                "id": r.case_id,
                "category": r.category,
                "question": r.question,
                "answer": r.answer,
                "rule_pass": r.rule_pass,
                "faithfulness": r.faithfulness,
                "tone": r.metrics.get("tone", {}).get("score"),
                "metrics": r.metrics,
                "judge_reasoning": r.judge_reasoning,
            }
            for r in reports
        ],
    }
