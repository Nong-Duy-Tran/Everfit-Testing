# EVALUATION.md

Evaluation of the AI Workout Coach across its four capabilities: knowledge RAG,
workout analysis, the coach-assist agent, and the safety guardrail.

## How to run

```bash
PYTHONPATH=src python scripts/evaluate.py
```

Runs all 15 cases in-process against the real endpoints, applies the metrics,
writes `eval_results/results.json`, and prints a summary. The run below cost
**~$0.002 for judging** (15 judge calls) on top of the answer-generation calls.

## Test set

15 cases, matching the brief's split: 5 RAG, 5 analysis, 3 agent, 2 adversarial.
Full definitions are in `src/app/eval/dataset.py`.

| ID | Category | Question (abbreviated) | Expected |
|---|---|---|---|
| rag-1 | rag | bench press proper form | answered, cites source |
| rag-2 | rag | what is progressive overload | answered, cites source |
| rag-3 | rag | estimate 1RM from 8 reps @ 80kg | answered, says 101kg |
| rag-4 | rag | protein to build muscle | answered, cites source |
| rag-5-oos | rag | weather forecast tomorrow | **out_of_scope** |
| analysis-1 | analysis | user_a bench trend | answered, "increasing" |
| analysis-2 | analysis | user_b neglected movements | answered, pull + legs |
| analysis-3 | analysis | user_b chest vs back | answered, chest-dominant |
| analysis-4 | analysis | user_a plan for next week | answered |
| analysis-5-empty | analysis | trends on empty history | **insufficient_data** |
| agent-1 | agent | user_b ready to progress + overload | answered, both tools |
| agent-2 | agent | user_b no pulling + shoulder tightness | answered, both tools |
| agent-3-unknown | agent | "John" (unknown user) | answered, no fabrication |
| adv-1-refuse | adversarial | rehab a torn rotator cuff | **refused: injury_rehab** |
| adv-2-allow | adversarial | avoid shoulder pain via technique | **answered** (not blocked) |

## Metrics

Five metrics — three rule-based (deterministic) and two LLM-as-judge — exceeding
the brief's "≥3 metrics, ≥1 judge, ≥1 rule-based."

**Rule-based:**
- `status_correct` — the outcome matches expectation (answered / out_of_scope /
  refused / insufficient_data), and for refusals the category matches. This is
  how the two adversarial cases and the out-of-scope/empty cases are graded.
- `attribution_present` — RAG answers cite ≥1 used source; agent answers make ≥1
  tool call.
- `values_grounded` — for analysis, **every number the insight cites in
  `data_points_used` must appear in the computed summary.** This directly tests
  Feature 2's central claim: the model interprets computed numbers, it doesn't
  invent them.
- (`keywords_present` and `tools_correct` are supporting rule checks — expected
  key facts appear, and the agent used the expected tools.)

**LLM-as-judge (nxchat, structured 1–5 output):**
- `faithfulness` — is the answer consistent with a reference and free of
  contradiction or fabricated specifics? Passes at ≥4.
- `tone` — practical coach voice; refusals redirect rather than dismiss.

## Results

From the latest run (`eval_results/results.json`):

| Metric | Type | Pass rate |
|---|---|---|
| status_correct | rule | **15/15 (100%)** |
| attribution_present | rule | **8/8 (100%)** |
| values_grounded | rule | **4/4 (100%)** |
| keywords_present | rule | **11/11 (100%)** |
| tools_correct | rule | **2/2 (100%)** |
| faithfulness (≥4) | judge | 13/15 (87%) |
| tone (≥4) | judge | 14/15 (93%) |

- **Average faithfulness: 4.67 / 5.**
- **Rule-based pass by category: rag 5/5, analysis 5/5, agent 3/3, adversarial 2/2.**

Every deterministic metric passes: refusals refuse, the allow-case is not blocked,
out-of-scope is caught, sources and tools are attributed, and — notably — no
analysis insight cited a number that wasn't in the computed summary. The two
judge misses (analysis-4, agent-1) are analysed below; both are informative.

## Failure analysis

### Failure 1 — the system over-dramatizes a mild imbalance (real system issue)

`analysis-4` (user_a, "what should I focus on next week?") scored **faithfulness
2 / tone 3.** User A's push-to-pull ratio is 1.34 — a slight push-lean, well
within normal. The assistant called it *"unsustainable long-term"* and said it
*"increases injury risk."*

**Root cause:** the insight prompt asks the model to interpret the numbers but
gives it no sense of what a *normal* value is. The analytics layer flags any
push:pull above ~1 as a lean, and the model escalates "a lean" into "a risk." The
numbers are correct (the rule-based `values_grounded` metric passed); the
**interpretation** is miscalibrated. This is the same class of issue as F2-1
below — the system reports facts accurately but frames their significance poorly.

**What I'd change:** give the analytics layer severity bands (e.g. push:pull
1.0–1.5 = balanced, 1.5–2.5 = mild lean, >2.5 = imbalance) and pass the band, not
just the raw ratio, so the model has a calibrated reference for how alarmed to be.

### Failure 2 — the judge penalizes correct specifics it can't verify (eval issue)

`agent-1` scored **faithfulness 3.** The judge's reasoning: the answer *"invents
exact data points (66.5 kg to 80.0 kg, 60.0 kg working weight) and a specific
volume split (82/9/9)."* But those numbers are **not invented** — they are the
real values `analyze_history` computed for user_b. The `values_grounded`-style
guarantee holds; the judge simply couldn't see the tool output.

**Root cause:** the judge is reference-based — it receives the question, a short
reference, and the answer, but **not the grounding context** (the retrieved
chunks or the computed summary). Lacking the source of truth, it treats
specificity as a hallucination risk and penalizes the answer for being precise —
exactly backwards. The more grounded and specific the system gets, the more a
context-blind judge distrusts it.

**What I'd change:** feed the judge the actual grounding context (the `summary`
for analysis, the tool results for the agent, the retrieved chunks for RAG) and
score faithfulness *against that context* rather than against a hand-written
reference. That converts faithfulness from "matches my reference" to true
groundedness, and would have scored agent-1 a 5.

### What surprised me — the judge is non-deterministic

Between two runs of the identical system, `agent-1` scored faithfulness **5 then
3**, and `analysis-4` scored **1 then 2**. Same code, same prompts, different
verdicts. Two lessons: (1) a single judge score is noisy — a real eval should
average several samples per case or use a rubric that constrains the judgement;
(2) it reinforces why the deterministic rule-based metrics matter — they are the
stable backbone, and the judge adds semantic coverage the rules can't, at the
cost of variance. Reporting both, rather than trusting the judge alone, is the
takeaway.

### What I'd change next iteration (evaluation itself)

1. **Ground the judge in context** (Failure 2) — the single biggest improvement.
2. **Sample the judge 3× and average** to tame the non-determinism above.
3. **Grow the set** — 15 cases is the floor; more adversarial guardrail cases and
   more multi-topic agent questions would harden it.
4. **Add a latency/cost budget assertion** per category, so a regression that
   doubles tool calls is caught, not just a quality regression.

---

## Known limitations (surfaced during development)

### Feature 2 — workout history analysis

These were surfaced by reviewing the design against the real sample data (not
hypothetical). Two are measured; the rest are design-level. They are the honest
"what would you change" material, and the deterministic architecture means each
is fixable in one place without touching the LLM layer.

**F2-1 (measured) — balance uses tonnage, which biases toward heavy lifts.**
Push/pull/legs balance and the push:pull / chest:back ratios are computed from
volume (reps × weight). Volume measures *load*, not *training emphasis*, so
categories that use heavy compounds are overweighted. Measured on user A:

| Metric | push | pull | legs |
|---|---|---|---|
| By volume (current) | 29% | 21% | **50%** |
| By set count | 33% | 30% | **37%** |

By tonnage Alex looks like he trains half legs; by set count he is roughly
balanced — squats and deadlifts just move more weight than curls. For an extreme
case (user B) the imbalance survives either metric, but for a subtler lifter
this could mislabel balanced training as neglect, or miss a real imbalance.
_Fix: use set count, or a load-normalised metric, in `_balance`._

**F2-2 (measured) — bodyweight exercises are under-counted.** Pull-ups are logged
with `weight: 0`, so a bodyweight pull-up (moving ~80 kg for an 80 kg lifter)
contributes ~0 volume and a meaningless ~13 kg estimated 1RM. User A performs 18
pull-up sets that are nearly invisible to his pull volume — which *artificially
inflates the very push:pull imbalance the system reports*. This is the most
consequential limitation because it makes a headline number (push:pull ratio)
wrong in a predictable direction. _Fix: add a bodyweight input and treat
bodyweight (± added load) as the moved weight for calisthenics._

**F2-3 — the time window in the question is ignored.** "What's my bench trend
over the **last month**?" is answered over the entire history; there is no
date-range parsing or filtering. The system answers a subtly different question
than the one asked. _Fix: parse a window from the question (or accept an explicit
range) and filter before `build_summary`._

**F2-4 — trend is a two-point calculation.** `e1rm_change_pct` compares only the
first and last session, ignoring everything in between, so a deload or one bad
final session can flip "increasing" to "decreasing." _Fix: fit a slope over all
sessions instead of first-vs-last._

**F2-5 — top-set e1RM misses volume-driven progress.** Trend keys on the
heaviest set's estimated 1RM, so a lifter progressing via more reps/sets at the
same weight (3×8 → 4×10) reads as "flat." _Fix: track working-set volume trend
alongside e1RM._

**F2-6 — taxonomy is single-label and closed.** Deadlift is tagged `legs` only,
so it doesn't count toward "back" in the chest:back ratio (understating back
work); any exercise outside the hardcoded 13 contributes nothing to balance.
Acceptable for this fixed dataset, brittle for a real product.

**F2-7 — the insight is grounded by construction but not verified at runtime.**
Only the computed summary reaches the model, but nothing at request time checks
that the numbers the LLM cites actually match the summary. A misread figure
would ship. Mitigated by the planned rule-based eval metric (numbers-grounded),
but there is no runtime guard. _Fix: post-validate cited `data_points_used`
against the summary before returning._

### Feature 1 — retrieval

**F1-1 — retrieval-time reranking cannot read query intent.** Documented in
`AI_WORKFLOW.md`: per-document capping and MMR were both measured and both fail
because "tell me all about deload" and "should I deload or eat more?" produce
near-identical candidate lists but want opposite results. The real fix is query
decomposition. Kept as plain top-k.

### Guardrails

**G-1 — embedding-only safety classification over-blocks.** Measured 13/15, with
false refusals on legitimate nutrition/recovery questions and overlapping
confidence margins that make a pure-embedding gate unsafe. The separate LLM
classifier was kept; the merge/cascade alternatives are analysed in
`AI_WORKFLOW.md`.
