# 06 — Roadmap (deferred, in rough order)

Everything here was discussed and deliberately deferred. Nothing should be lost.

## 1. Richer proposal set + refinement search
Add to the greedy constructor: merge (overlapping extents/intents), split
(bimodal user clusters), add-edge k→j (j active in most objects where k is —
this refactors shared attributes upward into parents and is the main
depth-discovery move beyond pair-merge ladders), remove by lowest usage, and
add-concept from **closed itemsets of the residual context** (FCA as proposal
distribution). Refinement phase: late-acceptance hill climbing or basin-hopping
with restarts (simpler to tune than an SA schedule); greedy-first always.

## 2. Online / prequential driver
Same counter store + ΔL formulas; add counter decay (fading factors) and/or
reservoir rejuvenation; per-object update path; no retro-re-encoding. Prequential
codelength on held-out order as a model-comparison statistic (useful even for
batch-trained models). Interpretation: Hebbian coincidence detection with an MDL
recruitment threshold — worth developing explicitly for the neural-coding
framing of the eventual writeup.

## 3. Bayesian version (nearly free)
Metropolis at temperature 1 over the same proposals with proposal-ratio
corrections ⇒ approximate posterior over DAGs. KT codes = Jeffreys-prior Bayes
mixtures, so the MDL score is already ≈ −log marginal likelihood.

## 4. Deep generative version (the original "explicit generative DAG" route)
Replace deterministic closure with **stacked noisy-OR layers** (noisy-OR belief
network; sigmoid belief nets are the sibling): parent activation
probabilistically drives children. Nonparametric structure prior: **Cascading
Indian Buffet Process** (Adams, Wallach & Ghahramani 2010) — unbounded width and
depth, multiple parents. Related: Pachinko Allocation (DAG-structured LDA),
nested HDP, Deep Exponential Families (variational-inference-friendly umbrella).
Design question flagged early: downward-closed codes (FCA-faithful, DAG means
"is-a") vs freer belief-net propagation (easier inference). Plan: prototype
belief-net style, *measure* how downward-closed learned codes turn out; if
closure emerges, add it as a hard constraint for interpretability.

Flat-model relatives for this stage: noisy-OR component analysis, Boolean factor
analysis with IBP priors, variational EM / Gibbs with beta-Bernoulli priors
(fixed K first, IBP later). Route 1 from the very first discussion — learn a
flat noisy factorization, then read the partial order off approximate inclusion
of extents/intents — remains a cheap strong baseline.

## 5. Product of DAGs / factorial codes
Several small independent DAGs whose activations combine (product lattices =
independent factors of variation). Formally: block structure on the loading
matrix, or mutual-information penalties between concept groups. Factorial
noisy-OR is underexplored — possibly the genuinely novel headline contribution
beyond the DAG-Krimp core. Connects directly to redundancy reduction, factorial
codes, sparse coding (Barlow, Atick, Földiák).

## 6. Baselines to implement for the paper
- Full FCA lattice size (already in prototype as a display number); iceberg
  lattices and Kuznetsov's **concept stability** as FCA-native pruning baselines.
- Krimp and Slim (flat code tables) on the same data.
- Asso / MDL4BMF (Boolean matrix factorization + MDL model selection).
- CorEx (layer-wise total correlation) for the information-theoretic comparison.
- Bayesian hierarchical clustering (to demonstrate the tree limitation
  empirically).

## 7. Evaluation beyond planted recovery
Held-out predictive likelihood; description length on real datasets; structural
recovery metrics (edge precision/recall up to node matching by extent overlap;
order agreement of learned reachability vs ground truth). Candidate real data:
any binary object×attribute context with known ontology (e.g. taxonomy-annotated
datasets), and eventually **binarized sparse neural codes** — the original
motivation: interpreting a learned sparse representation as an ontology.

## 8. Known theoretical caveats to state in any writeup
- Pairwise proposals are blind to parity-like higher-order structure with zero
  pairwise signal (non-issue for is-a data; residual closed-itemset proposals
  catch stragglers).
- Deterministic closure (v1) vs noisy propagation (v4): the v1 noise model puts
  all stochasticity at the observation layer.

## 9. Performance and scaling
Profiling (cProfile, 4000×40, 3-level) shows the batch **E-step**
(`encode_object`) is ~94% of runtime; numpy is only at the I/O edges, so the hot
path is pure-Python — the worry is real for large real data. Priorities, cheapest
first; **optimise only after confirming greedy suffices** (§1/§3: restart-variance
and gap-to-reference study — a rugged landscape or a scoring limit like the
overlap rent is not fixed by a faster inner loop).

**Done (low-hanging, no new deps):**
- Cached **concept-reachability bitmask** (`DAG._reach_concepts`) makes
  `reachable()` an O(1) bit test instead of a per-call DFS — it was ~45% of
  runtime. Invalidated with the closure cache on every edge add / concept remove.
- Hoisted the per-object **activation price and closure masks** out of the
  encoder's inner loop (constant while one object is encoded), and inlined
  `is_attribute` (87M calls) as `k < n_attrs` in the hot paths.
- Result: ~1.7× on medium planted problems with **identical** output (same L,
  concepts, moves); `reachable` no longer dominates. Remaining cost is the
  encoder's inner item×round arithmetic — the boundary for the next tier.

**Vectorisation — measured, and the axis that matters.** The E-step's big
dimension is the **objects (rows)**, not the candidate items (columns): objects
are independent given the fixed DAG in a pass, so they can be encoded in
**lock-step blocks** (all objects do greedy round 1 together, round 2 together,
with a boolean "still improving" mask retiring objects as they finish). We
benchmarked three forms against the scalar loop, all producing *identical* output:
- *Per-item vectorisation within one object* (wrong axis): **slower** everywhere
  (0.5–0.8×) — numpy dispatch overhead per object dominates.
- *Row-batched, dense boolean* `(B, items, attrs)`: ~1.3× up to ~60 attributes,
  then **regresses below 1×** by 100 attributes — the dense array is O(B·items·attrs)
  memory-bound.
- *Row-batched, bit-packed* (uint8 words + popcount LUT): ~1.2–1.6× and holds at
  scale (1.37× at 160 attributes).

Key finding: the payoff is only a modest **constant factor (~1.4×)**, because the
scalar baseline is *not* naive Python arithmetic — closure/coverage run on
CPython's C-level arbitrary-precision integer bit operations. numpy is competing
with compiled code, so it wins only modestly. Order-of-magnitude speed needs a
compiled kernel and/or multiple cores, below. Not shipped: the ~1.4× doesn't
justify the added complexity yet, and the row-block is really the unit for
multicore. Benchmarks kept for the eventual kernel/parallel work.

**Next tiers (deferred) — where the real speed is:**
- **Multicore over row-blocks (likely the biggest practical lever).** The
  vectorisation study confirmed objects are independent within a pass, so a pass
  splits into row-blocks that encode fully in parallel and merge their sufficient
  statistics — near-linear in cores. GIL means threads won't help pure Python;
  use multiprocessing/joblib now (broadcast DAG, map over object shards, reduce
  counts), or a GIL-releasing compiled kernel for threads. This is orthogonal to,
  and larger than, the ~1.4× SIMD win.
- **Compile the inner kernel** (`encode_object` + counter updates, a few hundred
  lines) in Cython/numba or Rust via PyO3, keeping orchestration in Python:
  packed-bitset ops with hardware POPCNT (no per-round 3D materialisation) plus a
  freed GIL for real threads — this is where order-of-magnitude gains live, not in
  pure numpy. A full Rust rewrite is premature and would cost the "readable and
  hackable" property.
- **Distributed:** the batch E-step is a textbook map-reduce over **additive
  sufficient statistics** — the "scorer talks only to the counter store"
  commitment is exactly what enables it: broadcast the DAG, each worker encodes
  its object shard and returns summed usage/pair counts, the driver reduces and
  runs the (serial) structural search. Parallelism is within a greedy round, not
  across the sequence. Watch the pairwise co-usage store (O(items²) sparse) as the
  communication/shuffle cost at scale.

## Positioning sentence for the eventual paper
Slim/Krimp generalized from a flat code table to a DAG with closure semantics —
equivalently, a probabilistic/MDL Formal Concept Analysis that only posits a
concept when it pays rent — with a constructive algorithm that is Barlow's
suspicious-coincidence unit recruitment made exact.
