"""The headline test (docs/05 §7): learn a small planted hierarchy back.

Asserts (a) learned codelength <= 1.05x the planted model's codelength,
(b) every planted concept is matched by a learned concept with extent
Jaccard >= 0.8, (c) the learned reachability relation among matched concepts
recovers the planted partial order, plus the two global invariants: monotone
codelength trajectory and incremental counters == from-scratch recount.
"""

from mdlfca.counters import CounterStore
from mdlfca.generator import concept_extents, make_planted, planted_codelength
from mdlfca.learner import GreedyLearner


def jaccard(s1, s2):
    union = len(s1 | s2)
    return len(s1 & s2) / union if union else 0.0


def test_planted_recovery():
    p = make_planted(n_attrs=20, level_sizes=(4, 2), n_objects=2000,
                     attrs_per_base=(3, 5), eps_plus=0.02, eps_minus=0.02,
                     seed=7)
    planted_L = planted_codelength(p)
    res = GreedyLearner(p.X).fit()

    # (a) codelength: match or beat the planted model within 5%
    assert res.total <= 1.05 * planted_L, (res.total, planted_L)

    # invariant: every accepted move decreased the one global scalar
    assert all(b < a + 1e-6 for a, b in zip(res.trajectory, res.trajectory[1:]))

    # invariant: incremental counters equal a from-scratch recount
    ref = CounterStore.from_codes(
        (set(c), f, m, g) for c, f, m, g in zip(res.codes, res.fp, res.fn, res.gen))
    assert res.counters.equals(ref)

    # (b) concept recovery by extent Jaccard
    ext_planted = concept_extents(p.dag, p.codes)
    ext_learned = concept_extents(res.dag, res.codes)
    assert ext_learned, "learner found no concepts at all"
    match = {}
    for c in p.concept_ids:
        best, best_j = max(((lc, jaccard(ext_planted[c], le))
                            for lc, le in ext_learned.items()), key=lambda t: t[1])
        assert best_j >= 0.8, f"planted concept {c} unmatched (best J={best_j:.2f})"
        match[c] = best

    # (c) order recovery among matched concepts
    tp = fp = fn = 0
    for i in p.concept_ids:
        for j in p.concept_ids:
            if i == j:
                continue
            truth = p.dag.reachable(i, j)
            pred = res.dag.reachable(match[i], match[j])
            tp += truth and pred
            fp += pred and not truth
            fn += truth and not pred
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    assert precision >= 0.7, (precision, recall)
    assert recall >= 0.7, (precision, recall)
