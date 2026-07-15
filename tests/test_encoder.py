"""Encoder invariants (docs/05 §7): antichain output; noise-free closure of a
single concept is encoded as exactly that concept."""

from mdlfca.encoder import encode_object, row_to_mask
from mdlfca.generator import make_planted, planted_counters


def _setup():
    p = make_planted(n_attrs=20, level_sizes=(4, 2), n_objects=600,
                     attrs_per_base=(3, 5), seed=3)
    return p, planted_counters(p)


def test_noise_free_closure_recovers_the_concept():
    p, cs = _setup()
    for c in p.concept_ids:
        x = p.dag.closure_mask(c)
        code, n_fp, n_fn, n_gen = encode_object(x, p.dag, cs)
        assert code == frozenset({c})
        assert n_fp == 0 and n_fn == 0
        assert n_gen == x.bit_count()


def test_encoded_codes_are_antichains():
    p, cs = _setup()
    for n in range(200):
        code, *_ = encode_object(row_to_mask(p.X[n]), p.dag, cs)
        for u in code:
            for v in code:
                assert u == v or not p.dag.reachable(u, v), (u, v, code)
