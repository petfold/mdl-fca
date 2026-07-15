# 03 — Algorithm

## The agreed starting algorithm: greedy bottom-up pair-merge

Start with the simplest model — each input attribute independent (equivalently:
no concepts; or a single trivial top with no intermediate layers) — and
**gradually add concepts with the strongest "correlations"**, where correlation
is measured over not just the raw inputs but also **previously introduced
concepts (the current dictionary)**. The structure is dynamic: an added concept
can make concepts below it less useful.

Pedigree: this is Barlow's suspicious coincidences made constructive, and it is
almost exactly Slim's move structure (pairwise unions of current dictionary
elements, evaluated by exact ΔL). Slim's empirical lesson is encouraging:
pairwise candidates from the current dictionary get very close to what expensive
global search finds, at a fraction of the cost. What we add over Slim: DAG
semantics (a new node sits *above* its two constituents and inherits their
closures) and explicit devaluation dynamics.

### Main loop

```
initialize dictionary = attribute nodes; U = X   (usage matrix, objects × items)
loop:
    update candidate-pair priority queue from co-usage counters
    pop best candidate (ranked by cheap upper bound n_ab · PMI)
    compute exact ΔL (lazy-greedy: re-evaluate only the top few;
        gains almost only decrease as the dictionary changes,
        so lazy re-evaluation is safe — same trick as submodular maximization)
    if ΔL < 0: accept
        create concept c with children {a, b}
        LOCAL RE-ENCODE: only objects with both a,b active are touched;
            rewrite their codes to use c; adjust counters incrementally
    every m accepts: PRUNE-AND-COLLAPSE sweep
    stop when the queue's best bound is unprofitable
```

### Track co-usage, not raw correlation

All candidate statistics come from the usage matrix `U` (each object's row is
its current code `S_n`), not from `X`. This gives **explaining-away
automatically**: once `c = {a,b}` exists and absorbs their joint occurrences,
the residual co-usage of `a` and `b` collapses (no re-proposing), and *new*
correlations involving `c` itself become visible — which is exactly how the
hierarchy climbs.

Pleasing FCA inversion: each accepted pair-merge is a formal concept *of the
current usage context* (extent = objects co-using a,b; intent = closure(a) ∪
closure(b)). The algorithm is "FCA restricted to pairwise-generated,
MDL-profitable concepts, computed on a self-modifying context." FCA becomes the
proposal distribution, not the answer.

### Devaluation dynamics — two explicit mechanisms (required, or greedy silts up)

1. **Local re-encoding on every accept** (above). Cheap, incremental.
2. **Periodic prune-and-collapse sweeps.** Re-evaluate each node's marginal ΔL:
   what happens to total codelength if it is deleted and its uses re-routed to
   its children. This catches the case of a later, larger concept starving an
   earlier one, and the main artifact of pairwise merging: a five-attribute
   concept gets built as a **ladder of binary merges**, and intermediate rungs
   like `c = {a,b}` often end up used *only* inside `d ⊃ c`.
   **Collapse rule:** if nearly all of `c`'s usage is as a child of a single
   parent, absorb it. (Krimp/Slim post-pruning; matters a lot in practice.)

### Known limitation (accept and document)

Pairwise statistics are blind to purely higher-order structure (parity-like
dependencies with zero pairwise signal). For ontology-shaped data this is a
non-issue — is-a structure creates massive pairwise signal. The prune/merge
sweeps plus occasional closed-itemset proposals from residuals would catch
stragglers.

## The broader search framework (the greedy constructor slots in as the "add" engine)

For later refinement beyond greedy, the framework is: **greedy-first** (run pure
hill-climbing over proposals until no move improves L — the Krimp/Slim recipe),
then a metaheuristic refinement phase. Two things matter far more than the
choice of metaheuristic:

1. **Data-driven proposals** (random structure edits almost never help; SA with
   random moves stalls):
   - **add-concept**: mine a frequent itemset / formal concept from the
     currently-unexplained 1s (residual co-occurrence context)
   - **merge** two concepts with heavily overlapping extents or intents
   - **split** a concept whose users fall into two clusters
   - **add-edge k→j** when j is active in most objects where k is active —
     this is what discovers hierarchy: attributes shared by several concepts
     get refactored upward into a parent
   - **remove-concept/edge** chosen by lowest usage
   Each move includes the acyclicity check (incremental topological order).
2. **Refinement metaheuristic**: late-acceptance hill climbing or basin-hopping
   with restarts are simpler to tune than an annealing schedule; simulated
   annealing also fine.

### The inner/outer split

Given `G`, finding the optimal `S_n` per object is a weighted set-cover-like
problem — the E-step. Greedy works well: repeatedly add the reachable-set whose
inclusion most reduces that object's codelength (covering 1s at ε⁻ price,
introducing 0-violations at ε⁺ price, plus the usage price of the item), stop
when no addition helps.

### Efficiency notes

- Cache reachability closures per concept; after a structure edit, recompute
  closures only for ancestors of touched nodes.
- Re-encode only objects whose current `S_n` intersects affected concepts.
- Incremental ΔL evaluation is what makes this tractable — naive full re-score
  per proposal is ~100× too slow.
- Keep an item → objects inverted index for cheap local re-encoding.

## Alternatives considered (baselines / initializers, not v1)

- **Asso / MDL4BMF**: batch candidates from the thresholded pairwise
  association ("implication") matrix + greedy MDL selection. Simple baseline
  and initializer; blind to latent-latent correlations, so weak on depth.
- **CorEx**: layer-wise total-correlation maximization with continuous
  optimization; principled multivariate objective, scales well; but factors
  aren't crisp attribute sets with closure semantics and the hierarchy tends to
  a tree.
- **Top-down divisive**: fights the semantics; closure-DAGs grow naturally
  bottom-up since a concept's meaning is composed from parts that must already
  exist.
