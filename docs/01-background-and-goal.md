# 01 — Background and goal

## Origin of the idea

The starting observation comes from artificial neural network codes and sparse,
explicit representations: the *internal structure* of a neural code is a good way
to represent ontologies (semantic structure). Formal Concept Analysis (FCA) is a
well-developed mathematical theory that fits this task: it maps a neural code
(FCA calls it a **formal context**) into a DAG (FCA prefers to talk about
**lattices**) of concepts, and back. This is an appealing way both to interpret a
neural representation and to embed an ontology into an artificial neural code.

## The problem with plain FCA

FCA is **not a probabilistic model**. It introduces a concept even for a random
combination of features, so the resulting lattice is far too big. We want a
**probabilistic / information-theoretic** version that only posits a concept when
it pays for itself.

## Why not the obvious neighbours

- **Hierarchical clustering (even Bayesian):** derives hierarchy but yields a
  **tree**. Trees are usually the wrong structure for concepts, because a tree
  node has exactly one parent, whereas concepts have **many parents**.
- **Topic modelling (CRP, stick-breaking, HDP):** allows multiple topics to
  explain one item (multiple parents, good) but **lacks the hierarchical
  structure** of hierarchical clustering.

We want the best of both: multiple parents **and** hierarchy — i.e. a genuine
**DAG**, not a tree and not a flat set of topics.

## The target object

A DAG that can **activate a subset of its nodes to generate the input**. "Good"
is defined in terms of probability / information theory / MDL. We might call this
**probabilistic FCA**, though it can be simpler than FCA — the real goal is just
to build good DAGs. Possibly even a **product of DAGs** (in the sense of product
lattices), which connects to **redundancy reduction, factorial codes, and sparse
coding**.

## Theoretical anchors (why this is coherent, not ad hoc)

- **FCA ≡ Boolean matrix factorization.** Belohlavek & Vychodil: formal concepts
  are exactly the optimal factors for *exact* Boolean matrix factorization.
  "Probabilistic FCA" = **noisy** Boolean factorization (noisy-OR). A concept is
  worth positing only if it explains more likelihood than it costs in
  prior/description length — this is precisely what kills the coincidence
  explosion.
- **Barlow's suspicious coincidences.** Create a unit when two things co-occur
  more than independence predicts. Our constructive algorithm is this idea made
  exact, with an MDL recruitment threshold. The online variant is literally a
  Hebbian coincidence detector that allocates a unit when a coincidence has
  accumulated enough evidence to pay its description cost.
- **Redundancy reduction / factorial codes (Barlow, Atick).** The
  product-of-DAGs extension = independent factors of variation = factorial code.

## Relation to existing MDL/pattern-mining work

- **Krimp / Slim** (Vreeken, Smets): select a small set of itemsets that
  compress the data (two-part MDL). Slim builds its code table by proposing
  **pairwise unions of current dictionary elements** — structurally the same move
  as our agreed starting algorithm. What we add: **DAG semantics** (a new node
  sits *above* its constituents, with closure) and explicit **devaluation
  dynamics** (a new concept can make earlier ones redundant).
- **Asso / MDL4BMF** (Miettinen, Vreeken): Boolean factorization from the
  pairwise association matrix with an MDL stopping rule. Good baseline/initializer;
  can't see correlations among *latent* concepts, so it struggles with depth.
- **CorEx** (Ver Steeg & Galstyan): layer-wise total-correlation maximization.
  The multivariate (not just pairwise) information-theoretic cousin; hierarchy
  tends to a tree, factors aren't crisp attribute sets. Read for the objective.
- **Cascading Indian Buffet Process** (Adams, Wallach, Ghahramani 2010): a prior
  over unbounded-width, unbounded-depth DAGs of binary latent units — the
  Bayesian target for the deep version. **Pachinko Allocation** is the
  DAG-structured generalization of LDA.

## Success criteria

On planted-DAG synthetic data:
1. Recover the planted concepts (match by extent overlap) — edge precision/recall.
2. Match or beat the planted model's codelength.
3. Recover the **partial order** (learned reachability agrees with planted order)
   — this is the claim that matters for the ontology interpretation.
4. Show the full FCA lattice size on the same data as the "what we saved you
   from" baseline.
