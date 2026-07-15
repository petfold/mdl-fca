"""Counter store: the self-contained sufficient statistics (docs/04).

Everything the codelength scorer needs lives here as counts:

- per-item usage (how many objects explicitly activate item k),
- pairwise co-usage (sparse dict keyed by sorted tuple — never a dense matrix),
- residual totals (false positives, false negatives, generated-cell total).

The scorer talks only to this object, never to raw data. Batch and online are
two drivers over this one core: batch uses exact add/remove on re-encoding,
online adds ``decay``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable


class CounterStore:
    def __init__(self):
        self.n_objects = 0
        self.usage: dict[int, float] = {}
        self.pair: dict[tuple[int, int], float] = {}
        self.n_fp = 0.0  # observed 1, not generated
        self.n_fn = 0.0  # generated, observed 0
        self.n_gen = 0.0  # total generated cells: sum over objects of |Yhat_n|

    # ------------------------------------------------------------- queries
    def usage_of(self, k: int) -> float:
        return self.usage.get(k, 0)

    def pair_count(self, a: int, b: int) -> float:
        return self.pair.get((a, b) if a < b else (b, a), 0)

    # ------------------------------------------------------------- updates
    def add_object_code(self, items: Iterable[int], n_fp=0, n_fn=0, n_gen=0):
        self.n_objects += 1
        it = sorted(items)
        for k in it:
            self.usage[k] = self.usage.get(k, 0) + 1
        for p in combinations(it, 2):
            self.pair[p] = self.pair.get(p, 0) + 1
        self.n_fp += n_fp
        self.n_fn += n_fn
        self.n_gen += n_gen

    def remove_object_code(self, items: Iterable[int], n_fp=0, n_fn=0, n_gen=0):
        self.n_objects -= 1
        it = sorted(items)
        for k in it:
            u = self.usage[k] - 1
            if u < 0:
                raise ValueError(f"usage of {k} went negative")
            if u == 0:
                del self.usage[k]
            else:
                self.usage[k] = u
        for p in combinations(it, 2):
            n = self.pair[p] - 1
            if n < 0:
                raise ValueError(f"pair count of {p} went negative")
            if n == 0:
                del self.pair[p]
            else:
                self.pair[p] = n
        self.n_fp -= n_fp
        self.n_fn -= n_fn
        self.n_gen -= n_gen

    def decay(self, gamma: float) -> None:
        """Fading factor for the online variant (docs/04). Unused in batch."""
        self.n_objects *= gamma
        self.n_fp *= gamma
        self.n_fn *= gamma
        self.n_gen *= gamma
        for d in (self.usage, self.pair):
            for k in list(d):
                v = d[k] * gamma
                if v < 1e-12:
                    del d[k]
                else:
                    d[k] = v

    # --------------------------------------------------------- test hooks
    @classmethod
    def from_codes(cls, coded: Iterable[tuple]) -> "CounterStore":
        """From-scratch recount. Each element: (items, n_fp, n_fn, n_gen)."""
        cs = cls()
        for items, n_fp, n_fn, n_gen in coded:
            cs.add_object_code(items, n_fp, n_fn, n_gen)
        return cs

    def equals(self, other: "CounterStore", tol: float = 1e-9) -> bool:
        def clean(d):
            return {k: v for k, v in d.items() if abs(v) > tol}

        return (
            abs(self.n_objects - other.n_objects) <= tol
            and abs(self.n_fp - other.n_fp) <= tol
            and abs(self.n_fn - other.n_fn) <= tol
            and abs(self.n_gen - other.n_gen) <= tol
            and clean(self.usage) == clean(other.usage)
            and clean(self.pair) == clean(other.pair)
        )
