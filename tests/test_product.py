"""Product-of-DAGs regression (docs/06 §5).

Two INDEPENDENT planted hierarchies on disjoint attribute blocks, concatenated
into one matrix over the same objects. Classical FCA on this data would
materialise (much of) the direct product of the two concept lattices. The MDL
learner must instead return the two factors as *disconnected components* with
**zero cross-block concepts**, and carry the product in the object codes (nearly
every object co-activates one concept from each factor).

This is the empirical face of the coproduct->product identity written up in
``note/product-of-dags-note.tex``: an additive (disjoint-union) generator
already represents the multiplicative (direct-product) lattice, so under
statistical independence no cross-factor concept can pay its usage rent
(PMI ~ 0 => dL >= 0) and the learner declines every one of them.
"""

import numpy as np

from mdlfca.counters import CounterStore
from mdlfca.generator import concept_extents, make_planted
from mdlfca.learner import GreedyLearner


def jaccard(s1, s2):
    union = len(s1 | s2)
    return len(s1 & s2) / union if union else 0.0


def _two_factor(seed_a, seed_b, n_attrs=20, n_objects=2000):
    """Two independent planted 2-level hierarchies, side by side (X = [XA | XB])."""
    common = dict(n_attrs=n_attrs, level_sizes=(4, 2), n_objects=n_objects,
                  attrs_per_base=(3, 5), eps_plus=0.02, eps_minus=0.02)
    pA = make_planted(seed=seed_a, **common)
    pB = make_planted(seed=seed_b, **common)
    X = np.hstack([pA.X, pB.X])
    return pA, pB, X, n_attrs


def test_product_factors_are_not_multiplied():
    pA, pB, X, A = _two_factor(seed_a=1, seed_b=2)
    block_a, block_b = set(range(A)), set(range(A, 2 * A))

    res = GreedyLearner(X).fit()

    # --- global invariants (as in test_recovery) ---
    assert all(b < a + 1e-6 for a, b in zip(res.trajectory, res.trajectory[1:]))
    ref = CounterStore.from_codes(
        (set(c), f, m, g) for c, f, m, g in zip(res.codes, res.fp, res.fn, res.gen))
    assert res.counters.equals(ref)

    # attributes reachable from a learned item, restricted to the sinks
    def attrs_of(k):
        return {a for a in res.dag.closure(k) if res.dag.is_attribute(a)}

    # --- (1) the headline: no concept mixes the two independent blocks ---
    cross = [c for c in res.dag.concepts
             if (attrs_of(c) & block_a) and (attrs_of(c) & block_b)]
    assert cross == [], \
        f"cross-block concepts leaked: {[(c, sorted(attrs_of(c))) for c in cross]}"

    # every concept is confined to exactly one block (the loading graph splits
    # into two components) and both factors are actually present
    in_a = [c for c in res.dag.concepts if attrs_of(c) <= block_a]
    in_b = [c for c in res.dag.concepts if attrs_of(c) <= block_b]
    assert len(in_a) + len(in_b) == res.dag.num_concepts
    assert in_a and in_b, "a whole factor went missing"

    # --- (2) each factor's planted concepts are recovered ---
    ext_learned = concept_extents(res.dag, res.codes)
    for planted in (pA, pB):
        ext_planted = concept_extents(planted.dag, planted.codes)
        for c in planted.concept_ids:
            best = max((jaccard(ext_planted[c], le) for le in ext_learned.values()),
                       default=0.0)
            assert best >= 0.9, f"factor concept {c} unmatched (best J={best:.2f})"

    # --- (3) the product lives in the object codes: nearly every object
    #         co-activates a concept from each factor ---
    def uses(code, block):
        return any(attrs_of(k) <= block
                   for k in code if not res.dag.is_attribute(k))

    both = sum(1 for code in res.codes if uses(code, block_a) and uses(code, block_b))
    assert both >= 0.95 * len(res.codes), \
        f"only {both}/{len(res.codes)} objects co-activate both factors"
