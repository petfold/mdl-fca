# mdl-fca

Learning "good" concept DAGs from binary data by minimum description length.
An MDL/information-theoretic reworking of Formal Concept Analysis: instead of
building the full concept lattice (which invents a concept for every random
coincidence), we build a small DAG of concepts that *compresses* the data.

## What this project is

Given a binary object×attribute matrix `X`, learn a DAG whose nodes are
**concepts** (latent) and **attributes** (observed sinks). Activating a subset
of concepts for an object generates, by downward closure, a predicted attribute
set; the data is explained as that prediction plus sparse noise. The best DAG is
the one with the shortest total codelength `L(G) + L(usage) + L(residual)`.

This reframes FCA as **noisy Boolean matrix factorization with a hierarchy**,
and the learning algorithm as **Krimp/Slim generalized from a flat code table to
a DAG with closure semantics** — which as far as we know does not exist in the
literature, so even the simple version is novel.

## Where to start reading

1. `docs/01-background-and-goal.md` — the problem, why trees and plain FCA don't
   work, the connection to sparse/factorial codes and Barlow's suspicious
   coincidences. Read this first for the *why*.
2. `docs/02-model-and-codelength.md` — the generative model and the exact
   codelength decomposition. This is the intellectual core; the optimizer is
   swappable.
3. `docs/03-algorithm.md` — the greedy pair-merge constructor (the agreed
   starting algorithm), proposal types, prune/collapse sweeps.
4. `docs/04-batch-vs-online.md` — why we start batch with incremental counters,
   and how the online (prequential) variant is a small delta on the same core.
5. `docs/05-prototype-spec.md` — the concrete build order for THIS session.
6. `docs/06-roadmap.md` — everything deliberately deferred (deep multi-layer
   noisy-OR, Bayesian sampling, product-of-DAGs / factorial codes, alternatives).

## The one-sentence summary

Barlow's "create a unit for a suspicious coincidence" made constructive and
exact: repeatedly introduce the concept whose creation most shortens the
description length, evaluated incrementally from count-based sufficient
statistics, with periodic pruning so later concepts can retire the ones they
make redundant.

## Immediate task for this session

Build the prototype described in `docs/05-prototype-spec.md`, in this order:
planted-DAG generator → counter store → codelength scorer → greedy constructor →
recovery test. Get the planted-hierarchy recovery test passing before adding
anything from the roadmap.

## Non-negotiable design commitments (agreed in design discussion)

- **The scorer talks only to the counter store, never to raw data.** This is
  what keeps batch and online as two drivers over one core. Do not let ΔL
  computation iterate over X.
- **Every accepted structural move must provably decrease one global scalar**
  (total codelength). Add a test asserting incremental counters equal a
  from-scratch recount.
- **"Correlation" = ΔL itself.** Do not introduce a separate significance
  threshold or p-value; the model cost is the threshold. Rank candidate pairs by
  the cheap `n_ab * PMI` upper bound, evaluate exact ΔL lazily (lazy-greedy).
- **Use KT / prequential (add-1/2) estimators** for all Bernoulli codes so there
  are no hand-tuned probabilities.
- Object codes `S_n` are **antichains** (no node in S_n reachable from another).
- Allow **direct (bare) attribute activation** with the same usage-based pricing,
  so idiosyncratic features don't spawn junk concepts.
- Maintain acyclicity + an incremental topological order on every edge add.

## Environment

- Python ≥ 3.11, stdlib + numpy only for the prototype (keep dependencies thin;
  this is meant to be readable and hackable). Add scipy/numba only if a hotspot
  demands it.
- `src/` layout, package name `mdlfca`. Tests with pytest.
