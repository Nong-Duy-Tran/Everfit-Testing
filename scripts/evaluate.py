"""Run the evaluation suite and write results.

Run:  PYTHONPATH=src python scripts/evaluate.py
Writes eval_results/results.json and prints a summary table.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.eval.runner import run_evaluation
from app.llm.client import LLMClient
from app.main import app

OUT_DIR = Path(__file__).resolve().parents[1] / "eval_results"


async def main() -> int:
    for n in ("httpx", "httpcore", "openai", "chromadb"):
        logging.getLogger(n).setLevel(logging.CRITICAL)
    logging.basicConfig(level=logging.WARNING)

    settings = get_settings()
    llm = LLMClient(settings)
    try:
        results = await run_evaluation(app, llm)
    finally:
        await llm.aclose()

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'='*64}\nEVALUATION SUMMARY  ({results['n_cases']} cases)\n{'='*64}")
    print("\nPer-metric pass rate:")
    for name, m in results["per_metric"].items():
        if m["pass_rate"] is None:
            continue
        print(f"  {name:20} {m['passed']}/{m['applicable']}  ({m['pass_rate']*100:.0f}%)")
    print(f"\nAvg faithfulness (1-5): {results['avg_faithfulness']}")
    print("\nBy category (rule-based pass):")
    for cat, c in results["by_category"].items():
        print(f"  {cat:14} {c['rule_pass']}/{c['total']}")
    print(f"\nJudge cost: {results['judge_cost']}")

    print("\nPer-case:")
    for case in results["cases"]:
        flag = "OK " if case["rule_pass"] and (case["faithfulness"] or 0) >= 4 else "XX "
        print(f"  {flag} {case['id']:16} rules={case['rule_pass']!s:5} faith={case['faithfulness']} tone={case['tone']}")

    print(f"\nWrote {OUT_DIR / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
