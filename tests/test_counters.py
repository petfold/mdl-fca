"""Incremental counters must equal a from-scratch recount (docs/05 §7)."""

import random

from mdlfca.counters import CounterStore


def random_coded_object(rng):
    items = set(rng.sample(range(12), rng.randint(0, 5)))
    return (items, rng.randint(0, 3), rng.randint(0, 3), rng.randint(0, 8))


def test_incremental_equals_recount_after_random_add_remove():
    rng = random.Random(0)
    cs = CounterStore()
    live = {}
    for i in range(400):
        obj = random_coded_object(rng)
        cs.add_object_code(*obj)
        live[i] = obj
        if live and rng.random() < 0.45:
            j = rng.choice(list(live))
            cs.remove_object_code(*live.pop(j))
    ref = CounterStore.from_codes(live.values())
    assert cs.equals(ref)
    assert cs.n_objects == len(live)


def test_remove_below_zero_raises():
    cs = CounterStore()
    cs.add_object_code({1, 2}, 0, 0, 2)
    cs.remove_object_code({1, 2}, 0, 0, 2)
    try:
        cs.remove_object_code({1}, 0, 0, 1)
    except (ValueError, KeyError):
        return
    raise AssertionError("expected an error on negative counts")
