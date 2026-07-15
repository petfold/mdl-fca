# 02 — Model and codelength

This is the core of the project. The optimizer is swappable; the score is not.
The measure of "goodness" of a DAG is simply the total **codelength** `L(G, X)`;
the search minimizes it, and goodness = compression achieved over the null model.

## Model space

A DAG `G` with two node kinds:

- **Attribute nodes** (the `A` observed columns of `X`) — sinks.
- **Concept nodes** (`K` latent) — edges point from concepts to their children
  (concepts or attributes).

**Semantics (v1): deterministic downward closure.** Activating a concept
activates everything reachable from it. Each object `n` has an explicit
activation set `S_n` ⊆ nodes (concepts and, optionally, bare attributes); its
predicted attribute vector `Ŷ_n` is the union of attributes reachable from
`S_n`. All stochasticity lives in an observation-noise layer:

- `ε⁺` — false-positive rate (attribute on though not generated)
- `ε⁻` — false-negative rate (attribute generated but observed off)

Noisy propagation *inside* the DAG (true noisy-OR belief network) is a roadmap
item, not v1. Deterministic closure keeps the combinatorics clean and we likely
won't miss internal noise at first.

**Constraints:**
- `G` acyclic; maintain an incremental topological order.
- `S_n` is an **antichain**: no node in `S_n` reachable from another node in
  `S_n`. Free to enforce in the greedy encoder; removes an identifiability wart
  (redundant activations).
- **Bare attribute activation is allowed**: an object may activate an attribute
  directly, priced by the same usage coding. This gives idiosyncratic features
  a cheap home and prevents junk concepts being invented for them.

## Codelength decomposition

```
L(G, X) = L(G) + Σ_n L(S_n | G) + Σ_n L(X_n | S_n, G)
```

### L(G) — structure cost

- Code `K` with Rissanen's universal integer code (or log-uniform up to a cap).
- Code the edge set: each node's child set with a **global Bernoulli edge
  probability**, estimated by the KT/prequential estimator (add-1/2), so edge
  density is adaptive and nothing is hand-tuned. Edges are cheap when the DAG is
  dense-ish, expensive when sparse.

### L(S_n | G) — usage cost

For each dictionary item `k` (concept or bare attribute), code the column
"is `k` explicitly active in object `n`" across objects with a **per-item
KT-estimated Bernoulli**.

This is the term that makes concepts **pay rent**: a concept used by many
objects gets a cheap per-use price; a concept invented for one coincidence costs
nearly as much as coding its attributes raw, so it dies. This is exactly the
mechanism FCA lacks.

### L(X_n | S_n, G) — residual cost

Count false positives and false negatives of `X_n` against `Ŷ_n`; code them
with KT-estimated `ε⁺`, `ε⁻` (equivalently, NML-ish coding of two binomials
plus bit positions).

## The three degenerate baselines the score automatically punishes

- **Full FCA lattice**: zero residual, monstrous `L(G)`.
- **One concept per distinct object**: likewise.
- **Empty model**: zero `L(G)`, full residual.

The ontology lives in the minimum between these.

## Key identity: "correlation" IS ΔL

For binary variables, the saving from introducing concept `c` covering the
co-use of dictionary items `a` and `b` is approximately

```
n_ab · [−log q̂_a − log q̂_b + log q̂_c]  −  overhead
```

where the overhead includes: structure bits for the new node + its two edges,
the new usage column, **and the increased per-use cost of `a` and `b`** (their
usage frequencies drop when `c` absorbs their co-occurrences — naive scores miss
this re-pricing). To first order the gain is `N · I(a;b) − overhead`: mutual
information weighted by data size, thresholded by model cost.

So "add the strongest correlation" and "greedy MDL" are the same algorithm; the
MDL form gets the threshold and tie-breaking right for free and needs **no
significance parameter**.

## Sufficient statistics are counts

Everything the score needs is a set of counters:

- per-item usage counts (over objects)
- pairwise co-usage counts
- residual error counts (FP, FN totals)

ΔL for any proposal is computed from these in O(1), **never by touching the
data**. This is the fact that makes both batch and online efficient, and it is
why the scorer must talk only to the counter store (see doc 04 and CLAUDE.md).

## MDL ↔ Bayes

The MDL score ≈ negative log marginal likelihood with particular priors (KT =
Jeffreys-prior Bayes mixture). Running Metropolis at temperature 1 with the
same proposals (plus proposal-ratio corrections) samples an approximate
posterior over DAGs — the Bayesian version comes almost free later.
