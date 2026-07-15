# 04 — Batch vs online

## The dichotomy is softer than it looks

Everything the score needs is count-based sufficient statistics (per-item usage,
pairwise co-usage, residual errors). ΔL for any proposal is O(1) from these
counters; **"re-evaluate on the whole batch" never happens in either version.**
"Batch" here means only: the dataset is fixed and in memory, and we make many
passes of *structural* changes over it. The real batch-vs-online question
reduces to a single operation: **retro-re-encoding**.

## Retro-re-encoding

When concept `c = {a,b}` is accepted, objects currently using `a` and `b`
separately should switch to using `c` — that is where c's saving actually comes
from, and it updates the usage counts that make c's own future correlations
visible.

- **Batch:** just do it. Item→objects index makes it cheap; cost ∝ support(c),
  not N; exact.
- **Strict online:** refuse to look back. Encode each arriving object with the
  current dictionary, update counters, accept a new concept when a pair's
  accumulated ΔL bound crosses zero, never revisit old encodings.

## Why the online version is beautiful

It is exactly **prequential MDL** (Dawid, Rissanen): the total codelength of the
stream under the evolving model is the accumulated predictive log-loss, and
`L(G)` disappears as an explicit term — it is paid implicitly through early
mispredictions before each concept exists. One pass, O(active-items²) per
object. It is also, in neural terms, a **Hebbian coincidence detector with an
MDL recruitment threshold**: a unit is allocated when a suspicious coincidence
has accumulated enough evidence to pay its description cost.

## Why we don't start with it: stale statistics

After `c` is born, counters are a mixture of pre-c encodings (a,b co-fire) and
post-c encodings (c fires). The a–b correlation fades only as fresh data dilutes
old counts, so the model "can't see clearly" for a while, and concepts higher up
— which depend on c's usage statistics — are slow to become discoverable.
**Depth accrues one statistical settling time per level.**

Standard fixes: exponential decay on all counters (fading factors; also buys
adaptation to non-stationary streams), or periodic **rejuvenation** (re-encode a
reservoir sample / random slice of history with the current dictionary) — at
which point you've reinvented mini-batching from the online side.

## Decision

**Start batch with incremental counters.** First datasets are planted-DAG
synthetics (10³–10⁵ objects, in memory); batch re-encoding is exact, has no
settling-time artifact, and makes debugging sane: every accepted move provably
decreases one global number, and counters can be asserted against a from-scratch
recount in tests. Multiple structural accepts happen per pass, so we don't pay a
full sweep per concept anyway.

The online variant is a small delta on the same code — same counters, same ΔL
formulas, plus decayed counts, a per-object update path, and no
retro-re-encoding. Build later, both as the scaling story and because
prequential codelength on held-out order is a very clean model-comparison
statistic even for batch-trained models.

## The design commitment that keeps the door open

Make the counter layer a **self-contained object** with
increment/decrement/decay operations, and make the scorer talk **only to it**,
never to raw data. Then batch and online are two drivers over the same core,
not two implementations.
