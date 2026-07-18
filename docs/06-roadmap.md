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

## Positioning sentence for the eventual paper
Slim/Krimp generalized from a flat code table to a DAG with closure semantics —
equivalently, a probabilistic/MDL Formal Concept Analysis that only posits a
concept when it pays rent — with a constructive algorithm that is Barlow's
suspicious-coincidence unit recruitment made exact.
