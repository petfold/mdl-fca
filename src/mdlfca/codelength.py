"""The codelength scorer (docs/02).

L(G, X) = L_structure(G) + L_usage + L_residual, all in bits, all built from
KT/prequential (add-1/2) Bernoulli codes so nothing is hand-tuned.

The scorer talks ONLY to the counter store and the structure object — never to
raw data. Every ΔL function is exact: it must equal
``total_codelength(after) - total_codelength(before)`` for the canonical
rewrite the learner performs (asserted in tests/test_codelength.py).
"""

from __future__ import annotations

from math import lgamma, log2

LN2 = 0.6931471805599453
_LG_HALF = lgamma(0.5)


def kt_codelength(n1: float, n0: float) -> float:
    """Bits to code a binary sequence with n1 ones and n0 zeros under the
    Krichevsky–Trofimov (Jeffreys-mixture) estimator. Exchangeable, so only
    the counts matter."""
    if n1 < 0 or n0 < 0:
        raise ValueError(f"negative counts ({n1}, {n0})")
    if n1 == 0 and n0 == 0:
        return 0.0
    return -(lgamma(n1 + 0.5) + lgamma(n0 + 0.5) - 2 * _LG_HALF - lgamma(n1 + n0 + 1)) / LN2


def universal_int_codelength(k: int) -> float:
    """Rissanen's universal code for the nonneg integer k (coded as k+1):
    log2(c0) + log2*(k+1) bits."""
    if k < 0:
        raise ValueError("k must be >= 0")
    total = log2(2.865064)
    x = log2(k + 1)
    while x > 0:
        total += x
        x = log2(x) if x > 1 else 0.0
    return total


class Scorer:
    """Stateless view over (dag, counters); all methods read current state."""

    def __init__(self, dag, counters):
        self.dag = dag
        self.counters = counters

    # ------------------------------------------------------------- totals
    def _possible_edges(self, n_concepts: int) -> int:
        # each concept may point at any attribute, plus one slot per concept
        # pair (orientation fixed by the topological order)
        return n_concepts * self.dag.n_attrs + n_concepts * (n_concepts - 1) // 2

    def structure_codelength(self) -> float:
        k = self.dag.num_concepts
        e = self.dag.num_edges
        return universal_int_codelength(k) + kt_codelength(e, self._possible_edges(k) - e)

    def usage_codelength(self) -> float:
        n = self.counters.n_objects
        total = 0.0
        for k in self.dag.items():
            u = self.counters.usage_of(k)
            total += kt_codelength(u, n - u)
        return total

    def residual_codelength(self) -> float:
        c = self.counters
        cells = c.n_objects * self.dag.n_attrs
        return kt_codelength(c.n_fn, c.n_gen - c.n_fn) + kt_codelength(
            c.n_fp, (cells - c.n_gen) - c.n_fp
        )

    def total_codelength(self) -> float:
        return self.structure_codelength() + self.usage_codelength() + self.residual_codelength()

    # ------------------------------------------------------------- deltas
    def delta_add_concept(self, a: int, b: int) -> float:
        """Exact ΔL of creating concept c with children {a, b} and rewriting
        every object that explicitly co-uses a and b to use c instead.

        Prediction (hence residual and n_gen) is unchanged: closure(c) =
        closure(a) ∪ closure(b). Usage moves n_ab from a's and b's columns
        into c's new column — the re-pricing of a and b that naive
        correlation scores miss."""
        c = self.counters
        n = c.n_objects
        n_ab = c.pair_count(a, b)
        d = kt_codelength(n_ab, n - n_ab)  # the new usage column
        for k in (a, b):
            u = c.usage_of(k)
            d += kt_codelength(u - n_ab, n - u + n_ab) - kt_codelength(u, n - u)
        k_now = self.dag.num_concepts
        e = self.dag.num_edges
        d += universal_int_codelength(k_now + 1) - universal_int_codelength(k_now)
        d += kt_codelength(e + 2, self._possible_edges(k_now + 1) - (e + 2)) - kt_codelength(
            e, self._possible_edges(k_now) - e
        )
        return d

    def delta_remove_concept(self, cpt: int) -> float:
        """Exact ΔL of deleting concept cpt: its explicit uses are re-routed
        to its children (set-union rewrite), and every parent inherits its
        children so all surviving closures are unchanged (rewire)."""
        c = self.counters
        n = c.n_objects
        u_c = c.usage_of(cpt)
        kids = self.dag.children[cpt]
        pars = self.dag.parents.get(cpt, set())
        d = -kt_codelength(u_c, n - u_c)
        for ch in kids:
            u = c.usage_of(ch)
            u2 = u + u_c - c.pair_count(cpt, ch)  # objects already using ch don't double-count
            d += kt_codelength(u2, n - u2) - kt_codelength(u, n - u)
        k_now = self.dag.num_concepts
        e = self.dag.num_edges
        e2 = e - len(kids) - len(pars) + sum(len(kids - self.dag.children[p]) for p in pars)
        d += universal_int_codelength(k_now - 1) - universal_int_codelength(k_now)
        d += kt_codelength(e2, self._possible_edges(k_now - 1) - e2) - kt_codelength(
            e, self._possible_edges(k_now) - e
        )
        return d

    def delta_swap_code(self, old_items, old_res, new_items, new_res) -> float:
        """Exact ΔL of re-encoding one object: replace code old_items (with
        residual stats old_res = (fp, fn, gen)) by new_items/new_res."""
        c = self.counters
        n = c.n_objects
        old_items = set(old_items)
        new_items = set(new_items)
        d = 0.0
        for k in old_items - new_items:
            u = c.usage_of(k)
            d += kt_codelength(u - 1, n - u + 1) - kt_codelength(u, n - u)
        for k in new_items - old_items:
            u = c.usage_of(k)
            d += kt_codelength(u + 1, n - u - 1) - kt_codelength(u, n - u)
        ofp, ofn, og = old_res
        nfp, nfn, ng = new_res
        fp2 = c.n_fp - ofp + nfp
        fn2 = c.n_fn - ofn + nfn
        g2 = c.n_gen - og + ng
        cells = n * self.dag.n_attrs
        d += kt_codelength(fn2, g2 - fn2) - kt_codelength(c.n_fn, c.n_gen - c.n_fn)
        d += kt_codelength(fp2, (cells - g2) - fp2) - kt_codelength(
            c.n_fp, (cells - c.n_gen) - c.n_fp
        )
        return d
