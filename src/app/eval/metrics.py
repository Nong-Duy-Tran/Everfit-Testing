"""Evaluation metrics: rule-based checks plus an LLM-as-judge.

Rule-based (deterministic, no model):
  - status_correct:      the outcome matches the expected behaviour
  - attribution_present: answers cite their sources / tools
  - values_grounded:     analysis insights cite only numbers that exist in the
                         computed summary (no fabricated figures)
  - keywords_present:    expected key facts appear in the answer

LLM-as-judge (nxchat scoring, structured output):
  - faithfulness (1-5):  answer is consistent with the reference, no contradiction
                         or fabricated specifics
  - tone (1-5):          appropriate, practical coach voice; refusals redirect

The rule-based metrics catch mechanical failures cheaply and deterministically;
the judge catches semantic ones (hallucination, wrong emphasis, bad tone) that
substring checks can't. Reporting both is the point — a green rule-based row with
a low faithfulness score is exactly the kind of finding worth surfacing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.eval.dataset import EvalCase
from app.llm.client import LLMClient, Usage


@dataclass
class MetricResult:
    name: str
    passed: bool | None  # None => not applicable to this case
    score: float | None = None  # for graded (judge) metrics
    detail: str = ""


# ---------------- rule-based ----------------


def status_correct(case: EvalCase, response: dict) -> MetricResult:
    actual = response.get("status") or ("answered" if response.get("answer") else "unknown")
    # /agent has no status field; its expected status is always "answered".
    if case.endpoint == "/agent":
        actual = "answered"
    passed = actual == case.expect_status
    if case.expect_refusal_category and passed:
        passed = response.get("refusal_category") == case.expect_refusal_category
    return MetricResult(
        "status_correct",
        passed,
        detail=f"expected={case.expect_status}"
        + (f"/{case.expect_refusal_category}" if case.expect_refusal_category else "")
        + f" actual={actual}/{response.get('refusal_category')}",
    )


def attribution_present(case: EvalCase, response: dict) -> MetricResult:
    """RAG answers cite sources; agent answers make tool calls. N/A otherwise."""
    if case.endpoint == "/ask" and case.expect_status == "answered":
        used = [s for s in response.get("sources", []) if s.get("used")]
        return MetricResult(
            "attribution_present", len(used) >= 1, detail=f"{len(used)} sources cited"
        )
    if case.endpoint == "/agent":
        calls = response.get("tool_calls", [])
        return MetricResult(
            "attribution_present", len(calls) >= 1, detail=f"{len(calls)} tool calls"
        )
    return MetricResult("attribution_present", None, detail="n/a")


def values_grounded(case: EvalCase, response: dict) -> MetricResult:
    """Analysis: every number cited in data_points_used must appear in the summary.

    This directly tests Feature 2's core claim — the model interprets computed
    numbers, it doesn't invent them.
    """
    if case.endpoint != "/analyze" or case.expect_status != "answered":
        return MetricResult("values_grounded", None, detail="n/a")

    summary_text = json.dumps(response.get("summary") or {})
    summary_numbers = set(re.findall(r"-?\d+\.?\d*", summary_text))
    cited = response.get("data_points_used", [])

    ungrounded = []
    for point in cited:
        nums = re.findall(r"-?\d+\.?\d*", str(point))
        for n in nums:
            # Allow the number, or its integer rounding, to appear in the summary.
            if n not in summary_numbers and _round_variants(n) & summary_numbers == set():
                ungrounded.append(f"{point} ({n})")
    return MetricResult(
        "values_grounded",
        len(ungrounded) == 0,
        detail="all cited numbers grounded"
        if not ungrounded
        else f"ungrounded: {ungrounded}",
    )


def keywords_present(case: EvalCase, response: dict) -> MetricResult:
    if not case.must_include:
        return MetricResult("keywords_present", None, detail="n/a")
    text = _answer_text(response).lower()
    missing = [k for k in case.must_include if k.lower() not in text]
    return MetricResult(
        "keywords_present",
        len(missing) == 0,
        detail="all present" if not missing else f"missing: {missing}",
    )


def tools_correct(case: EvalCase, response: dict) -> MetricResult:
    if case.endpoint != "/agent" or not case.must_use_tools:
        return MetricResult("tools_correct", None, detail="n/a")
    used = {c["name"] for c in response.get("tool_calls", [])}
    missing = [t for t in case.must_use_tools if t not in used]
    return MetricResult(
        "tools_correct",
        len(missing) == 0,
        detail=f"used={sorted(used)}" + (f" missing={missing}" if missing else ""),
    )


RULE_METRICS = [
    status_correct,
    attribution_present,
    values_grounded,
    keywords_present,
    tools_correct,
]


# ---------------- LLM-as-judge ----------------

JUDGE_SYSTEM = """\
You are a strict evaluator of a fitness assistant's answers. You are given the \
user's question, a reference describing the correct/expected response, and the \
assistant's actual answer. Score two dimensions from 1 to 5.

faithfulness: Is the answer consistent with the reference and free of fabricated \
specifics or contradictions? 5 = fully consistent and grounded; 3 = mostly right \
but with an unsupported or vague claim; 1 = contradicts the reference or invents \
facts. If the reference says the answer should refuse or report no data, a \
correct refusal / no-data response scores 5.

tone: Is the voice appropriate — practical and clear like a coach, and (for \
refusals) redirecting rather than dismissive? 5 = well-judged; 1 = inappropriate.

Be critical. Reserve 5 for genuinely strong answers."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "tone": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "reasoning": {"type": "string"},
    },
    "required": ["faithfulness", "tone", "reasoning"],
    "additionalProperties": False,
}


async def judge(
    case: EvalCase, response: dict, *, client: LLMClient, usage: Usage
) -> tuple[MetricResult, MetricResult, str]:
    answer = _answer_text(response)
    raw = await client.structured(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {case.question}\n\n"
                    f"Reference (expected): {case.reference}\n\n"
                    f"Assistant answer: {answer}"
                ),
            },
        ],
        JUDGE_SCHEMA,
        schema_name="judgement",
        usage=usage,
        max_tokens=400,
    )
    try:
        payload = json.loads(raw)
        f, t = int(payload["faithfulness"]), int(payload["tone"])
        reasoning = str(payload.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, ValueError):
        f, t, reasoning = 0, 0, "judge output unparseable"

    return (
        MetricResult("faithfulness", f >= 4, score=f, detail=reasoning),
        MetricResult("tone", t >= 4, score=t, detail=reasoning),
        reasoning,
    )


# ---------------- helpers ----------------


def _answer_text(response: dict) -> str:
    return str(response.get("answer") or response.get("insight") or "")


def _round_variants(num: str) -> set[str]:
    try:
        f = float(num)
    except ValueError:
        return set()
    return {str(int(round(f))), str(round(f, 1)), str(round(f))}
