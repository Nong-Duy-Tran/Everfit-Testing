# EVALUATION.md

_Work in progress. The full test set, per-metric results, and failure analysis
are completed in the evaluation phase. This file currently holds the known
limitations surfaced during development, which will be folded into the failure
analysis._

## Known limitations & failure cases

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
