"""ΔL functions must equal brute-force before/after differences (docs/05 §7)."""

import math

from mdlfca.codelength import Scorer, kt_codelength, universal_int_codelength
from mdlfca.counters import CounterStore
from mdlfca.dag import DAG

# A=4 attributes; residual-free toy codes over bare attributes.
TOY_CODES = [{0, 1}, {0, 1}, {0, 1, 2}, {1}, {0, 1, 3}, {2}, {0, 1}, {0, 1, 2, 3}]


def build_state():
    dag = DAG(4)
    codes = [set(c) for c in TOY_CODES]
    stats = [(0, 0, len(c)) for c in codes]  # (fp, fn, gen)
    cs = CounterStore.from_codes((c, *s) for c, s in zip(codes, stats))
    return dag, codes, stats, cs


def recompute(dag, codes, stats):
    cs = CounterStore.from_codes((c, *s) for c, s in zip(codes, stats))
    return Scorer(dag, cs).total_codelength()


def merge_rewrite(codes, a, b, c):
    return [(code - {a, b}) | {c} if a in code and b in code else set(code) for code in codes]


def remove_rewrite(codes, cpt, kids):
    return [(code - {cpt}) | set(kids) if cpt in code else set(code) for code in codes]


def test_kt_hand_values():
    assert kt_codelength(0, 0) == 0.0
    assert math.isclose(kt_codelength(1, 0), 1.0)          # first symbol: p=1/2
    assert math.isclose(kt_codelength(1, 1), 3.0)          # 1/2 * 1/4
    assert math.isclose(kt_codelength(2, 0), -math.log2(0.375))  # 1/2 * 3/4
    # KT is symmetric in the two symbols
    assert math.isclose(kt_codelength(5, 2), kt_codelength(2, 5))


def test_universal_int_code_positive_increasing():
    vals = [universal_int_codelength(k) for k in range(12)]
    assert all(v > 0 for v in vals)
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_delta_add_concept_matches_bruteforce():
    dag, codes, stats, cs = build_state()
    sc = Scorer(dag, cs)
    before = sc.total_codelength()
    d = sc.delta_add_concept(0, 1)
    c = dag.add_concept({0, 1})
    after = recompute(dag, merge_rewrite(codes, 0, 1, c), stats)
    assert math.isclose(after - before, d, abs_tol=1e-9)


def test_delta_remove_concept_matches_bruteforce():
    dag, codes, stats, cs = build_state()
    c = dag.add_concept({0, 1})
    codes = merge_rewrite(codes, 0, 1, c)
    cs = CounterStore.from_codes((code, *s) for code, s in zip(codes, stats))
    sc = Scorer(dag, cs)
    before = sc.total_codelength()
    d = sc.delta_remove_concept(c)
    dag.remove_concept(c, rewire=True)
    after = recompute(dag, remove_rewrite(codes, c, {0, 1}), stats)
    assert math.isclose(after - before, d, abs_tol=1e-9)


def test_delta_remove_with_parent_rewire_matches_bruteforce():
    dag, codes, stats, cs = build_state()
    c1 = dag.add_concept({0, 1})
    codes = merge_rewrite(codes, 0, 1, c1)
    c2 = dag.add_concept({c1, 2})
    codes = merge_rewrite(codes, c1, 2, c2)
    cs = CounterStore.from_codes((code, *s) for code, s in zip(codes, stats))
    sc = Scorer(dag, cs)
    before = sc.total_codelength()
    d = sc.delta_remove_concept(c1)
    kids = set(dag.children[c1])
    dag.remove_concept(c1, rewire=True)
    assert dag.children[c2] == {0, 1, 2}  # parent inherited the children
    after = recompute(dag, remove_rewrite(codes, c1, kids), stats)
    assert math.isclose(after - before, d, abs_tol=1e-9)


def test_delta_swap_code_matches_bruteforce():
    dag, codes, stats, cs = build_state()
    sc = Scorer(dag, cs)
    before = sc.total_codelength()
    old_code, old_stats = codes[0], stats[0]
    new_code, new_stats = {2, 3}, (1, 2, 3)
    d = sc.delta_swap_code(old_code, old_stats, new_code, new_stats)
    codes2 = [set(new_code)] + [set(c) for c in codes[1:]]
    stats2 = [new_stats] + stats[1:]
    after = recompute(dag, codes2, stats2)
    assert math.isclose(after - before, d, abs_tol=1e-9)
