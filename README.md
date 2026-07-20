# AI Workout Coach

Everfit AI Engineer take-home. An assistant that answers fitness questions from a
knowledge base, analyses a user's training history, and helps coaches with
multi-step questions through a tool-calling agent — with guardrails for a
health-adjacent context.

> **Status:** Phase 0 complete (scaffold, config, gateway client).
> Features 1–4 land in subsequent phases.

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
  config.py        # env-driven settings (pydantic-settings)
  main.py          # FastAPI entrypoint
  llm/client.py    # async gateway wrapper + token/cost accounting
knowledge-base/    # 20 markdown fitness documents
sample-data/       # 3 months of workout history for 2 users
```

## Documents

- [`AI_WORKFLOW.md`](AI_WORKFLOW.md) — how AI tools were used, what they got wrong, and how it was corrected
- `EVALUATION.md` — test set, metric results, failure analysis _(Phase 5)_

## Design decisions

_Expanded per phase. Architecture diagram, API reference, guardrail strategy,
and the cost-per-query estimate at 1,000 queries/day land as their features do._
