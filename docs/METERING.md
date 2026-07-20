# Bonus — usage metering layer

_How Everfit could charge coaches per AI query. Architecture only; ~200 words._

**What to meter.** Not raw tokens — coaches can't reason about those. Meter a
**billable query** (one `/ask`, `/analyze`, or `/agent` call), recording tokens
underneath for cost reconciliation. The system already measures this: every
response carries a `Usage` object. One `/agent` call is one billable query even
though it fans out to sub-calls — merged usage already rolls those up.

**Where to meter and enforce.** A middleware wrapping the routes, keyed by an
authenticated `coach_id`. It (1) checks remaining quota *before* the handler
runs — returning `429` with `Retry-After` and the reset time, so no model spend
happens once the cap is hit — then (2) atomically increments usage from the
response's `Usage`. Counters live in Redis (`INCR` on a `coach_id:period` key
with a reset-window TTL): one fast op, correct across instances. A periodic job
flushes to Postgres for invoicing and audit.

**Hitting the limit mid-session.** Check quota once at the start of a request,
not before each internal tool call, so a coach is never cut off half-way through
one answer — the query they started always completes. The `429` then applies to
their *next* query, with a clear message and upgrade path, rather than a partial,
broken response.
