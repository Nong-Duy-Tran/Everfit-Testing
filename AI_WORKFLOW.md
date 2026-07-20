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

### Guardrails reflection

_To be completed in Phase 2 (guardrails implementation)._

---

## Phase 1+ — _in progress_

_Appended as each phase completes._
