"""Greedy pair-merge constructor (docs/03).

Start from the attribute-only model (each object's code is its raw attribute
set — zero residual). Repeat: rank candidate pairs from the co-usage counters
by the cheap upper bound n_ab·PMI(a,b); evaluate exact ΔL lazily in bound
order; accept the best strictly-improving merge; locally re-encode only the
objects that co-use the pair (via the item→objects inverted index). Every m
accepts — and after the merge loop drains — run a prune-and-collapse sweep
(marginal ΔL per concept, uses re-routed to children, parents rewired) and a
greedy re-encoding pass (the E-step), each move gated by its exact ΔL.

Every accepted move strictly decreases the one global scalar (total
codelength, asserted after each move against a from-counters recompute), and
the counter store is updated only through add/remove of whole object codes so
the incremental-equals-recount invariant is testable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log2

import numpy as np

from .codelength import Scorer
from .counters import CounterStore
from .dag import DAG
from .encoder import encode_object, row_to_mask


@dataclass
class LearnResult:
    dag: DAG
    codes: list[set[int]]
    fp: list[int]
    fn: list[int]
    gen: list[int]
    counters: CounterStore
    total: float
    trajectory: list[float]
    events: list[str]


class GreedyLearner:
    def __init__(self, X, sweep_every: int = 4, tol: float = 1e-9,
                 max_outer: int = 200, verbose: bool = False, on_commit=None):
        self.X = np.asarray(X, dtype=np.uint8)
        self.sweep_every = sweep_every
        self.tol = tol
        self.max_outer = max_outer
        self.verbose = verbose
        # optional callback(learner, event) fired after every accepted move,
        # used by the animation demo to snapshot each step; no effect if None.
        self.on_commit = on_commit

    # ------------------------------------------------------------------ fit
    def fit(self) -> LearnResult:
        n_objects, n_attrs = self.X.shape
        self.dag = DAG(n_attrs)
        self.xmask = [row_to_mask(r) for r in self.X]
        self.codes = [{int(a) for a in np.flatnonzero(r)} for r in self.X]
        self.fp = [0] * n_objects
        self.fn = [0] * n_objects
        self.gen = [len(c) for c in self.codes]
        self.counters = CounterStore()
        self.index: dict[int, set[int]] = defaultdict(set)
        for n, code in enumerate(self.codes):
            self.counters.add_object_code(code, 0, 0, self.gen[n])
            for k in code:
                self.index[k].add(n)
        self.scorer = Scorer(self.dag, self.counters)
        self.total = self.scorer.total_codelength()
        self.trajectory = [self.total]
        self.events: list[str] = []

        accepts = 0
        for _outer in range(self.max_outer):
            changed = 0
            while True:
                cand = self._best_merge()
                if cand is None:
                    break
                self._apply_merge(*cand)
                changed += 1
                accepts += 1
                if accepts % self.sweep_every == 0:
                    changed += self._sweep()
            changed += self._sweep()
            changed += self._estep()
            if changed == 0:
                # per-object moves are stuck; try whole-pass re-encodings with
                # probed noise prices to escape the zero-residual minimum
                for eps in (0.25, 1 / 16, 1 / 64, 1 / 256):
                    if self._estep_batch(eps):
                        changed = 1
                        break
            if changed == 0:
                break
        else:
            raise RuntimeError("learner did not converge within max_outer rounds")

        self._check_total()
        return LearnResult(self.dag, self.codes, self.fp, self.fn, self.gen,
                           self.counters, self.total, self.trajectory, self.events)

    # ---------------------------------------------------------------- moves
    def _best_merge(self):
        """Lazy-greedy: sort candidates by the n_ab·PMI upper bound, evaluate
        exact ΔL down the list, stop once the next bound can't beat the best
        exact gain found so far."""
        c = self.counters
        n = c.n_objects
        cands = []
        for (a, b), n_ab in c.pair.items():
            if n_ab < 2:
                continue
            pmi = log2(n_ab * n / (c.usage_of(a) * c.usage_of(b)))
            if pmi <= 0:
                continue
            cands.append((n_ab * pmi, a, b))
        cands.sort(reverse=True)
        best = None
        best_gain = self.tol
        for bound, a, b in cands:
            if bound <= best_gain:
                break
            d = self.scorer.delta_add_concept(a, b)
            if -d > best_gain:
                best_gain = -d
                best = (a, b, d)
        return best

    def _apply_merge(self, a: int, b: int, delta: float) -> None:
        c_new = self.dag.add_concept({a, b})
        objs = self.index[a] & self.index[b]
        for n in objs:
            self.counters.remove_object_code(self.codes[n], self.fp[n], self.fn[n], self.gen[n])
            code = self.codes[n]
            code.discard(a)
            code.discard(b)
            code.add(c_new)
            self.counters.add_object_code(code, self.fp[n], self.fn[n], self.gen[n])
            self.index[c_new].add(n)
        self.index[a] -= objs
        self.index[b] -= objs
        self._commit(delta, f"merge {a}+{b} -> C{c_new} (n={len(objs)}, dL={delta:.2f})")

    def _sweep(self) -> int:
        """Prune-and-collapse: remove any concept whose marginal ΔL is
        negative; its uses re-route to its children and parents inherit them
        (which is exactly the ladder-rung collapse of docs/03)."""
        removed = 0
        again = True
        while again:
            again = False
            for cpt in list(self.dag.children):
                d = self.scorer.delta_remove_concept(cpt)
                if d < -self.tol:
                    self._apply_remove(cpt, d)
                    removed += 1
                    again = True
        return removed

    def _apply_remove(self, cpt: int, delta: float) -> None:
        kids = set(self.dag.children[cpt])
        objs = set(self.index.get(cpt, ()))
        for n in objs:
            self.counters.remove_object_code(self.codes[n], self.fp[n], self.fn[n], self.gen[n])
            code = self.codes[n]
            code.discard(cpt)
            for ch in kids:
                if ch not in code:
                    code.add(ch)
                    self.index[ch].add(n)
            self.counters.add_object_code(code, self.fp[n], self.fn[n], self.gen[n])
        self.index.pop(cpt, None)
        self.dag.remove_concept(cpt, rewire=True)
        self._commit(delta, f"prune C{cpt} (n={len(objs)}, dL={delta:.2f})")

    def _estep(self) -> int:
        """Re-encode each object with the greedy encoder; keep a new code only
        if its exact ΔL improves the global total. This is where noisy objects
        adopt a concept and pay the ε⁻ price instead of coding raw leftovers."""
        changed = 0
        pass_delta = 0.0
        for n in range(len(self.codes)):
            new_code, nfp, nfn, ngen = encode_object(self.xmask[n], self.dag, self.counters)
            if new_code == self.codes[n]:
                continue
            d = self.scorer.delta_swap_code(
                self.codes[n], (self.fp[n], self.fn[n], self.gen[n]),
                new_code, (nfp, nfn, ngen))
            if d < -self.tol:
                self.counters.remove_object_code(self.codes[n], self.fp[n], self.fn[n], self.gen[n])
                for k in self.codes[n] - new_code:
                    self.index[k].discard(n)
                for k in new_code - self.codes[n]:
                    self.index[k].add(n)
                self.codes[n] = set(new_code)
                self.fp[n], self.fn[n], self.gen[n] = nfp, nfn, ngen
                self.counters.add_object_code(self.codes[n], nfp, nfn, ngen)
                self.total += d
                pass_delta += d
                changed += 1
        if changed:
            self.trajectory.append(self.total)
            self.events.append(f"e-step: re-encoded {changed} objects (dL={pass_delta:.2f})")
            self._check_total()
            if self.verbose:
                print(self.events[-1])
        return changed

    def _estep_batch(self, eps: float) -> bool:
        """Trial re-encoding of ALL objects with (eps_plus, eps_minus) = (eps,
        eps) as proposal prices; committed only if the exact global codelength
        improves, otherwise discarded.

        This breaks the chicken-and-egg at n_fn == 0: no single object can
        afford the first false negative (its KT plug-in price is ~-log2 of a
        near-zero rate), but a whole cohort adopting concepts+noise together
        is cheap. Proposal prices are probed; acceptance is still the one
        global scalar — no significance threshold enters the model."""
        trial = [encode_object(self.xmask[n], self.dag, self.counters,
                               eps_override=(eps, eps))
                 for n in range(len(self.codes))]
        counters = CounterStore.from_codes(trial)
        total = Scorer(self.dag, counters).total_codelength()
        if total >= self.total - self.tol:
            return False
        n_changed = sum(code != old for (code, *_), old in zip(trial, self.codes))
        self.codes = [set(code) for code, *_ in trial]
        self.fp = [t[1] for t in trial]
        self.fn = [t[2] for t in trial]
        self.gen = [t[3] for t in trial]
        self.counters = counters
        self.scorer = Scorer(self.dag, counters)
        self.index = defaultdict(set)
        for n, code in enumerate(self.codes):
            for k in code:
                self.index[k].add(n)
        delta = total - self.total
        self.total = total
        self.trajectory.append(total)
        self.events.append(
            f"batch e-step (probe eps={eps:g}): re-encoded {n_changed} objects (dL={delta:.2f})")
        if self.verbose:
            print(self.events[-1])
        return True

    # ------------------------------------------------------------- plumbing
    def _commit(self, delta: float, event: str) -> None:
        assert delta < 0, f"non-improving move committed: {event}"
        self.total += delta
        self.trajectory.append(self.total)
        self.events.append(event)
        self._check_total()
        if self.verbose:
            print(f"{event}  L={self.total:.1f}")
        if self.on_commit is not None:
            self.on_commit(self, event)

    def _check_total(self) -> None:
        recomputed = self.scorer.total_codelength()
        assert abs(recomputed - self.total) < 1e-6 * max(1.0, abs(recomputed)), (
            f"incremental total {self.total} != recomputed {recomputed}")
        self.total = recomputed  # kill float drift
