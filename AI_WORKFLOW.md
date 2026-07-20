# AI_WORKFLOW.md

A running log of how AI tools were used to build this project — written during
development, not reconstructed afterwards. Each phase appends to it.

## Tools used

| Stage | Tool | Purpose |
|---|---|---|
| Spec analysis | Claude (agentic CLI) | Read the PDF and follow user explaination to generate specs |
| Architecture | Claude | Draft the phase plan, argue tradeoffs with user (framework vs. raw tool-calling, chunking strategy, where to enforce user isolation) |
| Implementation | Claude | Quickly construct codebase; Write modules|

---

## Phase 0 — scaffold & capability discovery

### Correction 1: I designed against the wrong LLM provider entirely

**What the AI produced.** My initial plan specified Anthropic's API with a
two-tier model routing strategy — Haiku 4.5 for the high-volume RAG and analysis
paths, Opus 4.8 for agent reasoning and the LLM-judge — plus a cost table built
from Anthropic's published per-token rates. I also proposed running
`sentence-transformers` (`all-MiniLM-L6-v2`) locally for embeddings, reasoning
that Anthropic has no embeddings endpoint and a local model keeps
`docker compose up` self-contained.

**Why it was wrong.** All of it was inference from the task description rather
than from the repo. The project's `src/.env.samples` pointed at
`https://api.ntq.ai/` with `LLM_MODEL_NAME=nxchat` and
`TEXT_EMBEDDING_MODEL_NAME=nx-text-embedding` — an OpenAI-compatible internal
gateway with a single chat model and a hosted embedding model. There was no
model tier to route between, no Anthropic pricing to apply, and no need for a
local embedding model.

**How it was corrected.** I read `src/.env.samples` before writing any code and
rebuilt the stack around `AsyncOpenAI` with `base_url` pointed at the gateway.
The cost section changed shape as a result: rather than multiplying guessed
token counts by published rates, `Usage` in `src/app/llm/client.py` accumulates
the `usage` block returned by every call, so the README's cost-per-query figures
are measured and the price-per-token is a single declared constant in
`config.py`.

**The transferable lesson:** the AI will confidently design a whole stack from
the problem statement while ignoring configuration files sitting in the repo.
Read the project's own config before accepting an architecture.

### Correction 2: an assumed embedding dimension that would have failed at ingest

**What the AI produced.** Working from "assume `text-embedding-3-small`", the
natural default is a 1536-dimension vector, and a Chroma collection would have
been configured for it.

**Why it was wrong.** `nx-text-embedding` returns **1024** dimensions. This is
the kind of error that surfaces as a runtime dimension-mismatch during ingest —
after the pipeline is written — rather than as a design-review finding.

**How it was corrected.** Rather than assume, I wrote a throwaway probe script
against the live gateway and measured it. `embedding_dim: int = 1024` in
`config.py` now carries a comment recording that it was verified, not guessed.

### Rejected suggestion: treating tool-calling support as a given

**The suggestion.** Proceed straight to implementing Feature 3's agent as an
OpenAI-native function-calling loop, since the gateway is OpenAI-compatible.

**Why I rejected it.** "OpenAI-compatible" is a spectrum. Many gateways and
self-hosted proxies implement `/chat/completions` faithfully but return
`tool_calls: null`, reject `response_format: {"type": "json_schema"}`, or only
emit one tool call per turn. Feature 3 is the single highest-risk feature in the
exercise, and discovering a tool-calling gap in Phase 4 would have forced a
rewrite into brittle ReAct-style text parsing.

**What I did instead.** I spent ~10 minutes on a capability probe before
committing to the design, testing five things against the live endpoint:
model listing, basic chat, native tool-calling, *parallel* tool-calling, strict
`json_schema` structured outputs, and embeddings. All passed — `nxchat` returns
`finish_reason=tool_calls`, emits two parallel calls when the question warrants
it, and honours strict JSON schemas. That result de-risked the agent design and
also let Feature 2 and the Feature 4 judge rely on schema-validated JSON instead
of regex-parsing prose.

The probe started as a throwaway script but was kept as
`scripts/probe_gateway.py` — it doubles as a pre-flight check that fails loudly
if the gateway's behaviour drifts, and it documents *why* the agent is a plain
tool-calling loop rather than a framework.

### Prompting strategy

Full context upfront, then narrow. The opening turn supplied the exercise PDF,
the knowledge base, and the sample data together, and asked for *analysis and a
plan* rather than code. That surfaced two things a code-first prompt would have
missed: that the sample data is an answer key (User B's 4:1 push/pull ratio and
2 leg sessions in 3 months are the intended answers to the spec's example
questions), and that mixed `kg`/`lb` units in User B's history is a correctness
gate rather than an edge case.

The strategy evolved once implementation started: instead of one large "build
the RAG pipeline" prompt, each phase is scoped to a layer with an explicit
verification step, so errors surface against a running system rather than
accumulating.

---

## Phase 1 — RAG pipeline

### Rejected suggestion (twice): retrieval-time diversity reranking

This is the phase's main story, and it is a case of the AI proposing a fix,
measuring it, and being talked out of it by the data — across two rounds.

**The problem.** Section-level chunks are small, so one document can win every
retrieval slot. "Should I deload **or eat more**?" returned five `10-deload`
sections and never surfaced nutrition or recovery — the model literally could
not answer half the question.

**Round 1 — the AI proposed a per-document cap.** I added a `max_chunks_per_doc`
setting: over-fetch, then keep at most N sections per document. Measured it at
N=2/3 against the real index. N=3 fixed the crowding with no regression on
single-topic queries, and I set it as the default with a confident comment.

**The user pushed back with one question:** *"what if the user asks 'tell me all
about deload'?"* I tested it instead of defending the design, and the cap was
wrong — it evicted two genuine deload sections to make room for an off-topic
deadlift chunk. So the cap traded one failure (multi-topic) for another
(single-topic deep-dive).

**Round 2 — the AI proposed MMR** as the "proper" version of the same idea:
penalise a candidate by its similarity to already-selected chunks, comparing
chunk-to-chunk instead of counting filenames. Before writing it into the store,
I simulated it over the real vectors at several λ values. It **also failed**:
λ=0.85 nailed the deep-dive and failed the multi-topic question; λ=0.7 did the
reverse. No single λ is right for both.

**The conclusion, and why it's the honest one.** Both questions produce a nearly
identical all-deload candidate list. The signal that separates them — *does the
user want one topic exhaustively, or several topics together?* — is **query
intent**, which lives in the question, not in the chunks or their vectors. No
retrieval-time reranking can recover a signal that was never in the retrieval
scores. I reverted to plain top-k, kept the whole investigation, and it becomes
the headline finding in EVALUATION.md: the real fix is query decomposition,
deferred as out of scope for Feature 1.

**The transferable lesson:** the AI defaults to solving a retrieval-shaped
problem with a retrieval-shaped tool, and will keep escalating within that frame
(cap → MMR) rather than stepping out of it. A single concrete counter-example
from the user collapsed two rounds of plausible engineering. Measuring the
proposal on a real adversarial query — before committing to it — is what caught
both.

### Correction: threshold cannot double as a medical guardrail

While tuning the out-of-scope cutoff, I checked adversarial inputs against real
ones and found that "How do I rehab my torn rotator cuff?" scores **0.434** —
statistically identical to the legitimate "How much protein should I eat?" at
0.439. The naive design would have leaned on the relevance threshold to catch
both out-of-scope *and* unsafe questions. It can't: a medical question is
*topically* on-scope. Refusal for safety has to be a separate intent-classifying
layer, not a similarity cutoff. Banked as the central constraint for Phase 2.

### What the AI got right without correction

- **Measure before choosing.** The chunking strategy was picked only after
  counting the corpus (111 sections, 18-152 words), which is what ruled out
  fixed-size chunking — it would have split the Epley/Brzycki formulas and the
  %1RM table. The title-prefix-per-chunk decision came from noticing that a bare
  "Common Mistakes" body is near-identical across bench/squat/deadlift.
- **Fail loud at the seams.** Ingest asserts the returned embedding dimension
  matches config, so a gateway model swap fails at ingest with a clear message
  rather than as a silent dimension mismatch at query time.

---

## Phase 2 — guardrails

### Guardrails reflection (the reflection the brief asks for)

**Did AI tools help or hinder my thinking on what the system should refuse?**
Both, in a useful order. The *hinder* came first and was caught in Phase 1: the
naive design the AI reached for would have leaned on the relevance threshold to
do double duty as a safety filter — refuse "what's the weather?" and "how do I
rehab my rotator cuff?" with the same mechanism. That is the trap. It only
surfaced because I measured similarity scores on adversarial inputs and saw the
rehab question (0.434) land on top of the legitimate protein question (0.439).
The AI's tidy instinct — "one refusal path for everything not answerable" — was
exactly wrong, because an unsafe question *is* answerable and *is* on-topic.

The *help* came once the problem was framed correctly. I asked whether the
gateway model could separate prevention from treatment on the genuinely hard
pairs, and probed it on ten borderline cases before writing any classifier code
("avoid shoulder pain" vs "shoulder clicks and hurts"; normal DOMS vs "felt a
pop"). 10/10. That measurement is what justified an LLM intent classifier over a
keyword list, rather than me asserting it would work.

**Was there a moment the AI-generated refusal logic was too broad, too narrow,
or missed the point?** Yes — the default failure behaviour. My first instinct on
"what if the classifier returns malformed JSON?" was to fail *closed* (refuse),
because that reads as the safe choice. It is the wrong choice here: the brief's
sharpest guardrail requirement is *not over-blocking*, and failing closed on
every transient parse error would refuse legitimate questions for a
model/gateway hiccup. The system now fails **open** — a malformed classification
allows the question, which is then answered by the grounded, cited pipeline that
has its own out-of-scope gate. That is a deliberate trade of a little safety
recall for far fewer false refusals, and it is documented as such rather than
hidden.

### Design choices worth noting

- **Safety runs before relevance, and concurrently with embedding.** Ordering is
  a correctness requirement (a medical question passes the similarity gate);
  running the classifier concurrently with the query embedding via
  `asyncio.gather` means it adds no latency on the happy path. A test asserts the
  ordering directly (`test_guardrail_runs_before_relevance_check`).
- **Refusals redirect, not dead-end.** Each names the right professional
  (physio / doctor / registered dietitian) and points back to what the assistant
  can still help with — verified by a test that every refusal message contains
  one of those referrals.
- **Two adversarial cases are carried into the eval set**, one guarding each
  direction of the boundary: an allow-case that must not be blocked, and a
  diagnosis-phrased-as-training case that must be. See `docs/GUARDRAILS.md`.

### Researching the "no second LLM call" question

The user asked whether the guardrail could avoid a dedicated classification call.
Rather than answer from intuition, I researched it: I built exemplar embeddings
per category and tested classifying the query by nearest-exemplar using the
vector already computed for retrieval — a genuinely zero-extra-cost approach. It
scored 13/15, and both failures were false refusals (over-blocking "what should
I eat before a workout?" as eating-disorder content). Crucially, the confidence
margins overlapped between right and wrong answers — a truly unsafe "felt a pop
in my knee" was *less* confident than a false-positive — which proves no margin
threshold can make embedding-only classification safe. That measurement turned a
"maybe we can drop the call" hunch into a documented finding, and the user chose
to keep the LLM classifier with the tradeoff understood. The merge-vs-cascade
options are written up for EVALUATION.md.

---

## Phase 3 — Feature 2: workout history analysis

### The design was driven by reading the data as an answer key

Before writing the analytics layer, I profiled both sample users. This is where
the "full context upfront" prompting paid off again: the data is engineered so
that the brief's four example questions have verifiable answers. User A shows
real progressive overload (bench e1RM +17.9%, balanced push/pull/legs); user B
is chest-dominant (push:pull 8.67, chest:back 10.34), skips legs, and — the
detail that shaped the whole module — mixes kg and lb *within one user*. So the
analytics layer was built to the data, not to a generic spec, and the tests
assert the specific answer-key numbers rather than just "returns a number".

### The correctness gate that would have silently produced wrong answers

The AI's default when building trend analysis is to treat `weight` as a bare
number. On this data that is a latent bug: user B's opening bench is 110 **lb**.
Left unconverted it reads as 110 kg — heavier than his later kg sessions — so
his single most-improved lift would report as *declining*. The insight would be
fluent, cited, and wrong. Unit normalisation to kg at parse time is therefore a
correctness gate, not an edge case, and it has a dedicated test
(`test_units_are_normalised_to_kg`) plus a runtime warning surfaced in the
summary. This is the kind of error that never appears in a design review and
only shows up if you actually run the numbers against the real file.

### Enforcing isolation structurally rather than by convention

The brief requires that one user's history never appears in another's context.
The tempting implementation is a store with an `all()` accessor and a discipline
of "only pass the right slice." I made it structural instead: the repository's
only accessor is `get(user_id)`, there is deliberately no `all()`/`get_all()`,
and analysis is pure over whatever list `get()` returns. A test asserts the
absence of a bulk accessor, so the guarantee can't be eroded by a future
convenience method. That is a stronger guarantee than a passing data-flow test,
because it removes the capability to leak rather than checking one path doesn't.

### Pre-processing keeps the LLM's job small — and honest

Only the computed summary reaches the model (trends, ratios, gaps), never the
raw sets. The model interprets numbers that are already correct and lists them
in `data_points_used`; it does no arithmetic. This is the brief's "aggregation
and trend detection vs raw dump" requirement, and it has a second benefit for
Feature 4: because the numbers are deterministic, a rule-based eval metric can
check that the insight's cited figures actually exist in the summary.
