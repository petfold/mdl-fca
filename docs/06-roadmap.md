# 06 — Roadmap (deferred, in rough order)

Everything here was discussed and deliberately deferred. Nothing should be lost.
Items are ticked when they land; each section keeps the reasoning that decided
it, since the reasoning is the point of the deferral.

## Status at a glance

- [x] §9 low-hanging performance work (DONE 2026-07-18): cached concept
      reachability, hoisted encoder work, ~1.7× with identical output.
- [x] §9 vectorisation study (DONE 2026-07-18): measured, written up, and
      **deliberately not shipped** — the ~1.4× does not justify the complexity.
- [x] §5 product-of-DAGs note, experiment and regression test (DONE
      2026-07-18).
- [ ] §1 richer proposal set + refinement search — the prerequisite for the
      rest, since §9 says optimise only after confirming greedy suffices.
- [ ] §2 online / prequential driver
- [ ] §3 Bayesian version
- [ ] §4 deep generative version
- [ ] §6 baselines for the paper
- [ ] §7 evaluation beyond planted recovery
- [ ] §8 theoretical caveats written up
- [ ] §9 next tiers: multicore over row-blocks, then a compiled kernel
- [ ] §10 learning the code first: `from_code`, three starts compared
- [ ] §11 real ontologies from OntoDAG as test data

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

**Result (2026-07-18): factorisation is emergent, no penalty needed.** Over
*disjoint* attribute alphabets, the coproduct (disjoint union) of two closure
DAGs generates *exactly* the direct product of the lattices they generate
individually — additive generator (m+n nodes), multiplicative lattice (m·n
concepts). Proved decoder-agnostically (any *separable* decoder factors) in
`note/product-of-dags-note.tex`. So the current learner already returns the
"product of DAGs": handed two independent planted hierarchies it returns the two
factors as disconnected components with **zero cross-block concepts**, carrying
the product in the object codes (`tests/test_product.py`,
`note/product-experiment.py`). No block-structure prior or MI penalty is
required — under independence a cross-factor concept has PMI ≈ 0 and cannot pay
its usage rent. Keep two conditions distinct: disjoint alphabets is *algebra*
(needed for the identity); statistical independence is *statistics* (needed only
for the learner to *prefer* the coproduct — dependent factors → a *partial*
product with a few bridge concepts, the open case to characterise). Classical
framing: apposition lattice = subdirect product (Ganter–Wille); factoring a
built lattice = Birkhoff subdirect decomposition (hard); building factors direct
sidesteps it. Follow-up: emit the factors as first-class output (connected
components of the loading graph — a linear-time post-step). Overlap *within* a
block is the genuinely hard case and coincides with the §1 / docs 02 overlap
limitation.

**A consumer for the factors (2026-11-12).** OntoDAG's
`docs/plans/CODES_AND_FACTORS.md` §7 would use them: to keep a derived index
small (one code per factor, no code for the combinations) and to group what a
browser offers. Its stored graph is already the note's coproduct: an item's
parents are its antichain code, and no node stands for a combination unless
someone names one. §11 below uses its stores as test data.

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
- [x] Cached **concept-reachability bitmask** (`DAG._reach_concepts`) makes
  `reachable()` an O(1) bit test instead of a per-call DFS — it was ~45% of
  runtime. Invalidated with the closure cache on every edge add / concept remove.
- [x] Hoisted the per-object **activation price and closure masks** out of the
  encoder's inner loop (constant while one object is encoded), and inlined
  `is_attribute` (87M calls) as `k < n_attrs` in the hot paths.
- [x] Result: ~1.7× on medium planted problems with **identical** output (same L,
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
- [ ] **Multicore over row-blocks (likely the biggest practical lever).** The
  vectorisation study confirmed objects are independent within a pass, so a pass
  splits into row-blocks that encode fully in parallel and merge their sufficient
  statistics — near-linear in cores. GIL means threads won't help pure Python;
  use multiprocessing/joblib now (broadcast DAG, map over object shards, reduce
  counts), or a GIL-releasing compiled kernel for threads. This is orthogonal to,
  and larger than, the ~1.4× SIMD win.
- [ ] **Compile the inner kernel** (`encode_object` + counter updates, a few hundred
  lines) in Cython/numba or Rust via PyO3, keeping orchestration in Python:
  packed-bitset ops with hardware POPCNT (no per-round 3D materialisation) plus a
  freed GIL for real threads — this is where order-of-magnitude gains live, not in
  pure numpy. A full Rust rewrite is premature and would cost the "readable and
  hackable" property.
- [ ] **Distributed:** the batch E-step is a textbook map-reduce over **additive
  sufficient statistics** — the "scorer talks only to the counter store"
  commitment is exactly what enables it: broadcast the DAG, each worker encodes
  its object shard and returns summed usage/pair counts, the driver reduces and
  runs the (serial) structural search. Parallelism is within a greedy round, not
  across the sequence. Watch the pairwise co-usage store (O(items²) sparse) as the
  communication/shuffle cost at scale.

## 10. Learning the code first (added 2026-11-12)

**The question.** Learning a binary code from data is a far more explored
field than learning a DAG. Would it be easier to learn the code and convert
it?

**What stands already.** `note/product-of-dags-note.tex` shows the choice is
not either/or: the DAG is the decoder, an object's antichain is its code, and
the factors are the DAG's connected components.

**What that leaves open: there are two codes, and an outside learner gives
neither.**

- The **antichain code** `S_n` holds only the most specific active concepts.
  The encoder emits it and the usage term prices it. In OntoDAG it is an
  item's stored parents.
- The **closed code** holds every concept at or above an active one. It is
  FCA's intent, and in OntoDAG a node's ancestors.

Given the DAG, each gives the other: closure one way, transitive reduction the
other. Without the DAG, a learner rewarded for sparseness drifts toward the
antichain code: the unit for a general concept stops firing where a more
specific unit covers the case (in sparse autoencoders this is reported as
feature absorption; Chanin et al., 2024). A learner with no such reward
drifts toward the closed code. So the order among units is read differently
in the two cases:

- from a closed code, by inclusion between the units' **extents** (the code's
  columns);
- from an antichain code, the extents do not nest: a general unit and its
  specific ones never fire together. The order is in the **decoder**: unit
  `v` is the more specific, and gets an edge to `u`, when the attributes `u`
  switches on are among those `v` switches on.

**The route.** `from_code(U, B)` takes an object×unit code and a unit×attribute
decoder, as any Boolean factorisation `X ≈ U ∘ B` gives them:

1. order the units by inclusion of their decoder rows, reduce, and add the
   attribute edges each unit still needs;
2. encode every object afresh with the project's encoder, which yields
   antichain codes whatever `U` was;
3. score with the same `L(G, X)`, run prune-and-collapse, and go on with the
   greedy moves.

The result is a **starting model for the existing search**, never an answer.
The score does not change (doc 02), and no tolerance is tuned by hand: an edge
that does not pay is pruned by the same rule as any other.

**Where the codes come from**, cheapest first:

1. Boolean matrix factorisation: Asso, or GreConD (Belohlavek & Vychodil),
   whose factors are formal concepts, so its decoder rows are intents and
   nest by construction. Both belong on §6's baseline list anyway.
2. The learner's own flat output: merges among attributes only, then the
   order.
3. A binary sparse code, or the features of a sparse autoencoder, cut at a
   threshold: the original motive, reading a learned representation as an
   ontology (§7). These need more than numpy, so they stay outside the
   package, as an example.

**What to expect.** A unit proposes a whole concept where pair-merging builds
a ladder of four merges and collapses it, so width should come faster. Depth
should not: a flat factoriser cannot see correlations among latent concepts
(doc 01), and the levels must still come from the greedy moves. This is §1's
"add-concept from closed itemsets" with another miner.

**The experiment that decides.** On the planted generator, three starts: the
empty model, a GreConD code, and the planted concepts themselves as a ceiling.
Compare codelength reached, recovery (doc 01's criteria) and time. It belongs
with §1's study of whether greedy suffices, as one more starting point.

## 11. Real ontologies from OntoDAG (added 2026-11-12)

OntoDAG (`../ontodag`) holds hand-built ontologies in a canonical reduced
form. Its shipped vocabulary has 11,643 nodes and 12,995 edges; 11% of the
nodes have several parents, a node has 5.3 ancestors on average and 19 at
most, and there are 16 top-level categories (`experiments/codes_probe.py`
there prints these). Step 1 of its plan `docs/plans/CODES_AND_FACTORS.md`
is built (2026-11-12): `experiments/context_export.py` there writes a
store's context as three files, `OUT.cxt` (Burmeister), `OUT.npy` (the
table as bytes, one row an object) and `OUT.json` (the names, the store's
own edges among them, and the knowledge id the table was made from). For
the whole shipped vocabulary that is 8,795 objects by 2,847 attributes
with 39,522 crosses and 2,944 different rows; below `place` alone, 439 by
183 with 1,601 crosses, 175 different rows and 31 attributes with
several parents:

    python3 experiments/context_export.py /tmp/place --packs all --below place

What that gives this project:

1. **Recovery against a known DAG that nobody planted.** The reference is the
   ontology's own reduction. How much of it the learner returns, and which
   categories it declines because they do not pay rent, is the measurement.
   The second half is a question in its own right: do the concepts MDL keeps
   agree with the ones people name? (OntoDAG's paper, direction D8.)
2. **Depth and several parents together.** The 1,279 nodes with several
   parents are overlap cases from real use, for the limitation in doc 02 and
   in `note/concept-dag-note.tex`.
3. **The product on a real vocabulary.** File items under one category from
   each of several top-level categories, as OntoDAG's probe does. The learner
   should return the top-level branches as separate components with no
   concept across them (§5). The 694 nodes that stand under several top-level
   categories are real bridge concepts: the partial product §5 leaves open.
4. **Scale.** 8,795 × 2,847 is far past the planted problems (§9 profiled
   4000 × 40). Start with one top-level branch (`place`, 439 × 183) and
   let it decide which of §9's next tiers comes first.

Flipping bits at a known rate tests the residual terms on real structure.

**A first run (2026-11-12),** `python examples/run_context.py /tmp/place`:
439 objects by 183 attributes, learned in 1.1 s. 39 concepts and 91 edges;
the codelength falls from 9,132 to 7,241 bits, 79% of the empty model's.
14 of the 162 categories whose code has more than one attribute come back
exactly. The most used concepts are upper combinations
(`{geographical-area, region}`, `{artifact, building, structure}`), several
of them a category's code less one attribute, which exact matching does not
count; doc 01's matching by extent overlap would. With 439 leaves in 175
different rows, most categories have two or three objects and cannot pay
rent: a vocabulary's leaves are thin data, and a store with items filed
under it is the fairer test. The whole vocabulary, 8,795 by 2,847, did not
finish in ten minutes, so it waits for §9's next tier.

**A second run, on real items (2026-11-12).** ontodag-core's
`tools/instances.py place OUT.od` files 9,651 things Wikidata calls
instances of the place categories under them (890 under two or more), and
`context_export.py OUT --store OUT.od --below place` makes the table:
9,876 objects by 397 attributes, 53,369 crosses, 802 different rows.

| | leaves only | with real items |
|---|---|---|
| objects by attributes | 439 by 183 | 9,876 by 397 |
| time | 1.5 s | 138 s |
| concepts, edges | 39, 91 | 337, 849 |
| codelength against the empty model | 79% | 45% |
| categories found exactly | 14 of 162 | 74 of 366 |
| found to within one attribute | 36 of 162 | 291 of 366 |

With enough items to pay their rent, four categories in five come back
to within one attribute, and one in five exactly. Of the 217 that are one
attribute off, 210 lack one, and it is the same one every time:
`physical-object`, the top-level attribute most of these places share
(seven have one too many). The concept for `mountain` holds `landform`,
`natural-elevation` and `natural-object`, and stops one short of the top,
which each object switches on by itself. So the learner finds the
hierarchy and declines to hang it under its broadest node: the overlap
rent of doc 02 on real data, and the first thing to look at. "Within one
attribute" asks that the concept hold the category's own attribute, so a
parent's code does not pass for its child's.

**The interface is files.** mdl-fca reads the `.npy` table (or the
Burmeister `.cxt`) and the reference edges in the `.json`. It does not
import ontodag, and ontodag never imports mdl-fca. What the learner finds
goes back as a list of concepts with their attributes; putting one into a
store is a person's act there, never an automatic merge.

## Positioning sentence for the eventual paper
Slim/Krimp generalized from a flat code table to a DAG with closure semantics —
equivalently, a probabilistic/MDL Formal Concept Analysis that only posits a
concept when it pays rent — with a constructive algorithm that is Barlow's
suspicious-coincidence unit recruitment made exact.
