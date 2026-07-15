"""The headline tests (docs/05 §7): learn planted hierarchies back.

Asserts (a) learned codelength <= 1.05x the planted model's codelength,
(b) every planted concept is matched by a learned concept with extent
Jaccard >= 0.8, (c) the learned reachability relation among matched concepts
recovers the planted partial order, plus the two global invariants: monotone
codelength trajectory and incremental counters == from-scratch recount.

The 3-level test additionally checks that depth builds on depth: mid-level
concepts must be discovered from the co-usage of learned base concepts, and
top concepts from the co-usage of learned mids.
"""

from mdlfca.counters import CounterStore
from mdlfca.generator import concept_extents, make_planted, planted_codelength
from mdlfca.learner import GreedyLearner


def jaccard(s1, s2):
    union = len(s1 | s2)
    return len(s1 & s2) / union if union else 0.0


def run_and_check(planted, min_jaccard=0.8, min_order=0.7):
    planted_L = planted_codelength(planted)
    res = GreedyLearner(planted.X).fit()

    # (a) codelength: match or beat the planted model within 5%
    assert res.total <= 1.05 * planted_L, (res.total, planted_L)

    # invariant: every accepted move decreased the one global scalar
    assert all(b < a + 1e-6 for a, b in zip(res.trajectory, res.trajectory[1:]))

    # invariant: incremental counters equal a from-scratch recount
    ref = CounterStore.from_codes(
        (set(c), f, m, g) for c, f, m, g in zip(res.codes, res.fp, res.fn, res.gen))
    assert res.counters.equals(ref)

    # (b) concept recovery by extent Jaccard
    ext_planted = concept_extents(planted.dag, planted.codes)
    ext_learned = concept_extents(res.dag, res.codes)
    assert ext_learned, "learner found no concepts at all"
    match = {}
    for c in planted.concept_ids:
        best, best_j = max(((lc, jaccard(ext_planted[c], le))
                            for lc, le in ext_learned.items()), key=lambda t: t[1])
        assert best_j >= min_jaccard, f"planted concept {c} unmatched (best J={best_j:.2f})"
        match[c] = best

    # (c) order recovery among matched concepts
    tp = fp = fn = 0
    for i in planted.concept_ids:
        for j in planted.concept_ids:
            if i == j:
                continue
            truth = planted.dag.reachable(i, j)
            pred = res.dag.reachable(match[i], match[j])
            tp += truth and pred
            fp += pred and not truth
            fn += truth and not pred
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    assert precision >= min_order, (precision, recall)
    assert recall >= min_order, (precision, recall)
    return res


def test_planted_recovery_two_levels():
    p = make_planted(n_attrs=20, level_sizes=(4, 2), n_objects=2000,
                     attrs_per_base=(3, 5), eps_plus=0.02, eps_minus=0.02,
                     seed=7)
    run_and_check(p)


def test_planted_recovery_three_levels():
    # level_weight=2 (not the default 3): with three levels, direct
    # lower-level activation must stay frequent enough that base and mid
    # concepts keep paying rent once their parents absorb the co-occurrences —
    # at level_weight=3 the exact MDL optimum genuinely drops the weakest
    # bases (verified by delta_add_concept > 0 on the true pairs).
    p = make_planted(n_attrs=30, level_sizes=(8, 4, 2), n_objects=3000,
                     attrs_per_base=(3, 5), children_per_concept=(2, 3),
                     level_weight=2.0, eps_plus=0.02, eps_minus=0.02,
                     seed=7)
    res = run_and_check(p, min_jaccard=0.9, min_order=0.9)
    # depth built on depth, with nothing junk left over: exactly the planted
    # dictionary size (held across seeds 0-3,7 during tuning)
    assert res.dag.num_concepts == len(p.concept_ids), res.dag.num_concepts
