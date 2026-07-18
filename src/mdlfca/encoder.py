"""Per-object greedy encoder — the E-step (docs/03, inner/outer split).

Given the DAG and one object row (as an attribute bitmask), greedily build an
antichain code S: repeatedly add the dictionary item whose closure most
reduces the object's codelength — covering 1s at the ε⁻ price, introducing
0-violations at the ε⁺ price, plus the item's usage price — and stop when no
addition helps. Bare attributes are candidate items like any concept.

Prices are plug-in estimates from the counter store (KT posterior means);
acceptance of the resulting code is decided by the exact
``Scorer.delta_swap_code``, so the plug-in approximation can only cost us a
proposal, never correctness of the descent.
"""

from __future__ import annotations

from math import log2

import numpy as np


def row_to_mask(row) -> int:
    m = 0
    for i in np.flatnonzero(row):
        m |= 1 << int(i)
    return m


def noise_rates(counters, n_attrs: int) -> tuple[float, float]:
    """Plug-in (add-1/2) estimates of (eps_plus, eps_minus)."""
    cells = counters.n_objects * n_attrs
    g = counters.n_gen
    eps_minus = (counters.n_fn + 0.5) / (g + 1)
    eps_plus = (counters.n_fp + 0.5) / ((cells - g) + 1)
    clamp = lambda x: min(max(x, 1e-9), 0.5)
    return clamp(eps_plus), clamp(eps_minus)


def activation_cost_bits(counters, k: int) -> float:
    """Marginal usage-column cost of switching item k's cell from 0 to 1."""
    u = counters.usage_of(k)
    n = counters.n_objects
    return log2((n - u + 0.5) / (u + 0.5))


def encode_object(x_mask: int, dag, counters, tol: float = 1e-9,
                  eps_override: tuple[float, float] | None = None):
    """Greedy antichain code for one object.

    Returns (code, n_fp, n_fn, n_gen): the code as a frozenset of item ids and
    the residual stats of the implied prediction against x_mask.

    ``eps_override`` replaces the plug-in (eps_plus, eps_minus) prices for
    proposal purposes only — used by the learner's batch E-step to escape the
    zero-residual local minimum (the first false negative is near-infinitely
    expensive under the KT plug-in when n_fn == 0).
    """
    eps_plus, eps_minus = eps_override or noise_rates(counters, dag.n_attrs)
    gain_one = log2((1 - eps_minus) / eps_plus)   # newly covered observed-1
    gain_zero = log2(eps_minus / (1 - eps_plus))  # newly covered observed-0 (negative)

    items = dag.items()
    # The counters do not change while a single object is being encoded, so the
    # per-item closure mask and activation price are constant across the greedy
    # rounds below — compute them once instead of inside the inner loop.
    clm = {k: dag.closure_mask(k) for k in items}
    act = {k: activation_cost_bits(counters, k) for k in items}
    reachable = dag.reachable

    code: set[int] = set()
    covered = 0
    while True:
        best_gain = tol
        best = None
        for k in items:
            if k in code:
                continue
            new = clm[k] & ~covered
            if new == 0:
                continue
            ones = (new & x_mask).bit_count()
            zeros = new.bit_count() - ones
            g = gain_one * ones + gain_zero * zeros - act[k]
            # adding k retires any current item below it (antichain): refund it
            for s in code:
                if reachable(k, s):
                    g += act[s]
            if g > best_gain:
                best_gain = g
                best = k
        if best is None:
            break
        covered |= clm[best]
        code = {s for s in code if not reachable(best, s)}
        code.add(best)

    n_fp = (x_mask & ~covered).bit_count()
    n_fn = (covered & ~x_mask).bit_count()
    n_gen = covered.bit_count()
    return frozenset(code), n_fp, n_fn, n_gen
