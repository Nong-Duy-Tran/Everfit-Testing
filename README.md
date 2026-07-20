# AI Workout Coach

Everfit AI Engineer take-home. An assistant that answers fitness questions from a
knowledge base, analyses a user's training history, and helps coaches with
multi-step questions through a tool-calling agent — with guardrails for a
health-adjacent context.

> **Status:** Feature 1 (knowledge RAG + guardrails) and Feature 2 (workout
> history analysis) complete. Feature 3 (coach-assist agent) and Feature 4
> (evaluation) land in subsequent phases.

## Time estimate

Stated before starting: **9–11 hours**, against the brief's 6–8. The delta is
deliberate — Feature 4 (evaluation + honest failure analysis) and the three
required documents are graded as heavily as the pipeline code, and the brief
weights AI-adoption evidence equally with technical output.

## Stack

| Concern | Choice | Why |
|---|---|---|
| API | FastAPI | Async-native (the gateway client is `AsyncOpenAI`), typed request/response models via Pydantic, free OpenAPI docs |
| LLM | `nxchat` via `https://api.ntq.ai/v1` | Project-provided gateway; OpenAI-compatible, so accessed with the official `openai` SDK rather than raw HTTP |
| Embeddings | `nx-text-embedding` (1024-dim) | Same gateway — no second credential, no local model weights in the image |
| Vector DB | Chroma (persistent, local) | Embedded — no extra service in `docker-compose`, which keeps `docker compose up` a single command. At ~100 chunks the scaling ceiling is irrelevant |
| Deploy | `docker compose up` | Single command, as preferred by the brief |

Gateway capabilities were **verified by probe**, not assumed — native
tool-calling, parallel tool-calling, and strict `json_schema` structured outputs
all work. See `AI_WORKFLOW.md`.

## Setup

```bash
cp src/.env.samples src/.env   # then fill in LLM_API_KEY
docker compose up --build
curl localhost:8000/health
```

Local development without Docker:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/uvicorn app.main:app --reload
```

API keys are read from the environment only — never hardcoded. `src/.env` is
gitignored.

## Layout

```
src/app/
  config.py           # env-driven settings (pydantic-settings)
  main.py             # FastAPI entrypoint
  llm/client.py       # async gateway wrapper + token/cost accounting
  rag/                # Feature 1: chunking, store, ingest, grounded answers
  guardrail/          # safety intent classifier
  analysis/           # Feature 2: taxonomy, analytics, insight, user repository
  api/                # routes + request/response schemas
knowledge-base/       # 20 markdown fitness documents
sample-data/          # 3 months of workout history for 2 users
```

## Endpoints

| Endpoint | Feature | Purpose |
|---|---|---|
| `POST /ask` | 1 | Answer a fitness question from the knowledge base, with citations. Refuses out-of-scope and medical-advice questions. |
| `POST /analyze` | 2 | Analyse a user's workout history (`user_id` or inline `workouts`) and answer a question about it, backed by computed stats. |
| `POST /agent` | 3 | Answer a multi-step coaching question by deciding which tools to call (`rag_search`, `analyze_history`) and in what order. |
| `GET /health` | — | Liveness + indexed-chunk count. |

Interactive API docs at `/docs` when the server is running.

## Documents

- [`AI_WORKFLOW.md`](AI_WORKFLOW.md) — how AI tools were used, what they got wrong, and how it was corrected
- [`docs/GUARDRAILS.md`](docs/GUARDRAILS.md) — safety refusal strategy: triggers, messages, and how over-restriction is avoided
- `EVALUATION.md` — test set, metric results, failure analysis _(Phase 5)_

## Coach-assist agent (Feature 3)

A registry-driven tool-calling loop over `rag_search` (Feature 1) and
`analyze_history` (Feature 2). No agent framework: for two tools and one level of
delegation, native function-calling via the gateway is enough and keeps the
reasoning visible. Native + parallel tool-calling was verified against the
gateway before building (see `AI_WORKFLOW.md`). The loop is tool-agnostic — it
asks the model which tools to call, executes them from the registry, feeds
results back, and repeats until the model stops or the iteration cap is hit. It
does not hardcode the call order.

The three questions the brief asks about the agent:

**What happens if the agent calls the wrong tool first?** Nothing breaks. Every
tool result is fed back into the conversation, so if the model calls the wrong
tool it sees a result that doesn't answer the question and re-plans on the next
iteration — the loop working as designed, not a failure. A tool that finds no
usable data returns an explicit `status` (`unknown_user`, `insufficient_data`)
rather than an empty result, so the model is told what went wrong instead of
inferring an answer from nothing.

**How would you add a third tool without rewriting the agent logic?** Register
one more `Tool` (name, function schema, async callable) in `build_registry`. The
loop iterates the registry and dispatches by name, so it needs zero changes — it
never references a specific tool. That is the reason the registry exists.

**What's the failure mode you're most worried about in production?** Not wrong
tool selection — that self-corrects. It's a tool returning *plausible but
insufficient* data that the model treats as sufficient: `analyze_history` on a
user with two sessions returns real-looking numbers, and the model gives
confident advice on a weak signal. The mitigation is the explicit-status contract
at the tool boundary (`insufficient_data` is a distinct status the prompt is told
to respect), but a determined model can still over-read thin data. This is the
same class of risk flagged for Feature 2 in `EVALUATION.md`.

## Design decisions

_Expanded per phase. Architecture diagram and the cost-per-query estimate at
1,000 queries/day land in the final documentation pass._
