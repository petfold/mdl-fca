# 05 — Prototype spec (build this first)

Language: Python ≥ 3.11, numpy + stdlib only. `src/` layout, package `mdlfca`,
pytest. Keep it small and readable; this is a research prototype.

## Modules and build order

### 1. `mdlfca/generator.py` — planted-DAG generator
Ground truth for validation. Must exist before anything else so every later
module has data to run against.

- `PlantedDAG`: sample a random DAG — parameters: number of levels (default 3),
  K concepts (default ~15), edge density, number of attributes A.
- Sample objects: activate a **sparse random antichain** of concepts per object,
  apply downward closure to get the attribute set, then flip cells with noise
  rates ε⁺, ε⁻.
- Return: X (N×A uint8 array), plus the planted structure (node children,
  closures, per-object true activation sets) for evaluation.
- Also provide the codelength of the *planted* model on its own data (the
  target to match or beat).

### 2. `mdlfca/counters.py` — counter store
The self-contained sufficient-statistics object (see doc 04).

- Per-item usage counts; pairwise co-usage counts (dict keyed by frozenset or
  sorted tuple; sparse — never a dense K×K matrix); residual FP/FN totals.
- Operations: `add_object_code`, `remove_object_code` (increment/decrement all
  affected counters), later `decay(γ)` for the online variant.
- Invariant test hook: `recount_from_scratch(codes)` must equal incremental
  state.

### 3. `mdlfca/codelength.py` — the scorer
Implements doc 02 exactly. Talks ONLY to the counter store + structure object.

- KT/prequential Bernoulli codelength helper:
  `L_KT(n1, n0)` (log-gamma form, natural log internally, report bits).
- `L_structure(G)`: Rissanen integer code for K; KT-coded edge indicators.
- `L_usage(counters)`: per-item KT columns.
- `L_residual(counters)`: KT-coded FP and FN binomials + positions.
- `total_codelength()` and, critically, **incremental ΔL functions**:
  - `delta_add_concept(a, b)` — including the re-pricing of a and b
    (their usage drops by n_ab), new node/edge structure bits, new usage column.
  - `delta_remove_concept(c)` — uses re-routed to children.
  - Exactness contract: ΔL functions must agree with
    `total_codelength(after) − total_codelength(before)` in tests
    (small instances, brute-force check).

### 4. `mdlfca/dag.py` — structure object
- Nodes (attribute | concept), children sets, cached closures
  (concept → reachable attribute set), incremental topological order,
  acyclicity check on edge add.
- Closure invalidation: on structure edit, recompute closures only for
  ancestors of touched nodes.
- Item → objects inverted index for local re-encoding.

### 5. `mdlfca/encoder.py` — per-object greedy encoder
Given G and an object row x, produce antichain code S:
repeatedly add the item whose closure most reduces the object's codelength
(covering 1s at ε⁻ price, introducing 0-violations at ε⁺ price, plus the item's
usage price); stop when no addition helps. Bare attributes are candidate items.

### 6. `mdlfca/learner.py` — the greedy pair-merge loop
Doc 03 main loop:
- candidate priority queue keyed by upper bound `n_ab · PMI(a,b)` from co-usage
  counters; lazy-greedy exact ΔL on pop.
- on accept: create concept, local re-encode (only objects with both a,b
  active, via inverted index), update counters incrementally.
- every m accepts: prune-and-collapse sweep (marginal ΔL per node; collapse a
  node whose usage is almost entirely as child of one parent).
- stop when best bound ≥ 0.
- Log: total codelength trajectory (must be monotone decreasing on accepts —
  assert it), dictionary size, sweep actions.

### 7. `tests/`
- `test_counters.py`: incremental == from-scratch recount after random
  add/remove sequences.
- `test_codelength.py`: ΔL functions == brute-force before/after difference on
  tiny instances; KT helper against hand-computed values.
- `test_encoder.py`: encoder output is an antichain; encoding a noise-free
  closure of a single concept recovers that concept.
- `test_recovery.py` (the headline test): generate a small planted hierarchy
  (e.g. 2 levels, K=6, A=20, N=2000, ε=0.02); run the learner; assert
  (a) learned codelength ≤ planted codelength × 1.05,
  (b) concept recovery: each planted concept matched by a learned one with
      extent Jaccard ≥ 0.8 (match by best overlap),
  (c) order recovery: precision/recall of the learned reachability relation
      against the planted partial order, thresholds to be tuned once running.
- A baseline script (not a test): full FCA concept count on the same data
  (simple closed-itemset enumeration is fine at this size) to display the
  "what we saved you from" number.

## Deliberately NOT in the prototype
Deep multi-layer noisy-OR generation, Bayesian sampling, product-of-DAGs,
online/prequential driver, split/add-edge/merge proposals beyond pair-merge and
prune/collapse, real datasets. See docs/06-roadmap.md.

## Definition of done for the first Claude Code session
`pytest` green including `test_recovery.py`, plus a small
`examples/run_planted.py` that prints the codelength trajectory, planted vs
learned codelength, FCA lattice size, and the recovered DAG (as an indented
text rendering with attribute names per concept).
