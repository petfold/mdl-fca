"""Planted-DAG generator: ground truth for validation (docs/05 §1).

Builds a layered random concept DAG (base concepts own disjoint attribute
blocks; upper concepts pick children from the level below), samples objects by
activating a sparse random antichain of concepts (plus occasional bare
attributes), applies downward closure, and flips cells with noise rates
ε⁺/ε⁻. Also prices the *planted* model on its own data — the codelength
target the learner has to match or beat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codelength import Scorer
from .counters import CounterStore
from .dag import DAG


@dataclass
class PlantedData:
    dag: DAG
    levels: list[list[int]]        # concept ids per level, bottom-up
    codes: list[frozenset[int]]    # true activation antichains per object
    Y: np.ndarray                  # N×A generated (pre-noise) matrix
    X: np.ndarray                  # N×A observed matrix
    eps_plus: float
    eps_minus: float

    @property
    def concept_ids(self) -> list[int]:
        return [c for lvl in self.levels for c in lvl]


def make_planted(
    n_attrs: int = 30,
    level_sizes: tuple[int, ...] = (8, 4, 2),
    n_objects: int = 1000,
    attrs_per_base: tuple[int, int] = (2, 4),
    children_per_concept: tuple[int, int] = (2, 3),
    eps_plus: float = 0.02,
    eps_minus: float = 0.02,
    bare_attr_rate: float = 0.02,
    p_second_pick: float = 0.35,
    level_weight: float = 3.0,
    seed: int = 0,
) -> PlantedData:
    """Plant a hierarchy that is actually the MDL optimum of its own data.

    Two conditions are needed for a parent concept to "pay rent" under
    per-item usage pricing, and both are enforced here:

    - **Disjoint child sets per level** (each level partitions the one below,
      like the base level partitions attributes). If two siblings share a
      child, the shared child's usage column stays dense after either parent
      is introduced, and the parent cannot pay for itself — an exact MDL
      learner then correctly refuses to create it. Recovering genuinely
      overlapping DAGs needs richer usage coding (docs/06).
    - ``level_weight``: an object picks a concept from level l with
      probability proportional to level_weight**l. Upper levels must be
      activated often relative to direct lower-level activation, otherwise
      child columns stay near-independent (PMI → 0) and again the hierarchy
      is not what an exact learner should return."""
    rng = np.random.default_rng(seed)
    dag = DAG(n_attrs)

    # base level: disjoint attribute blocks (leftover attributes stay bare)
    sizes = [int(rng.integers(attrs_per_base[0], attrs_per_base[1] + 1)) for _ in range(level_sizes[0])]
    while sum(sizes) > n_attrs:
        i = max(range(len(sizes)), key=sizes.__getitem__)
        if sizes[i] <= 2:
            raise ValueError("n_attrs too small for the requested base level")
        sizes[i] -= 1
    perm = [int(a) for a in rng.permutation(n_attrs)]
    levels: list[list[int]] = [[]]
    pos = 0
    for s in sizes:
        levels[0].append(dag.add_concept(set(perm[pos : pos + s])))
        pos += s

    for width in level_sizes[1:]:
        prev = levels[-1]
        lo, hi = children_per_concept
        if width * lo > len(prev):
            raise ValueError(f"level of width {width} needs >= {width * lo} children below")
        child_counts = [lo] * width
        spare = min(len(prev), width * hi) - width * lo
        while spare > 0:
            i = int(rng.integers(width))
            if child_counts[i] < hi:
                child_counts[i] += 1
                spare -= 1
        shuffled = [int(x) for x in rng.permutation(prev)]
        lvl = []
        pos = 0
        for s in child_counts:
            lvl.append(dag.add_concept(set(shuffled[pos : pos + s])))
            pos += s
        levels.append(lvl)

    all_concepts = [c for lvl in levels for c in lvl]
    weights = np.array([level_weight**l for l, lvl in enumerate(levels) for _ in lvl])
    weights /= weights.sum()
    codes: list[frozenset[int]] = []
    Y = np.zeros((n_objects, n_attrs), dtype=np.uint8)
    for n in range(n_objects):
        m = 1 + (rng.random() < p_second_pick)
        picks = {int(x) for x in rng.choice(all_concepts, size=m, replace=False, p=weights)}
        # reduce to the maximal antichain
        picks = {c for c in picks if not any(dag.reachable(o, c) for o in picks if o != c)}
        mask = 0
        for c in picks:
            mask |= dag.closure_mask(c)
        bare = [a for a in range(n_attrs) if not (mask >> a) & 1 and rng.random() < bare_attr_rate]
        for a in bare:
            mask |= 1 << a
        codes.append(frozenset(picks | set(bare)))
        for a in range(n_attrs):
            if (mask >> a) & 1:
                Y[n, a] = 1

    flips = rng.random(Y.shape)
    X = np.where(Y == 1, flips >= eps_minus, flips < eps_plus).astype(np.uint8)
    return PlantedData(dag, levels, codes, Y, X, eps_plus, eps_minus)


def planted_counters(p: PlantedData) -> CounterStore:
    """Counter store of the planted model's own encoding of its data."""
    cs = CounterStore()
    for n, code in enumerate(p.codes):
        y, x = p.Y[n], p.X[n]
        n_fp = int(((x == 1) & (y == 0)).sum())
        n_fn = int(((x == 0) & (y == 1)).sum())
        cs.add_object_code(code, n_fp, n_fn, int(y.sum()))
    return cs


def planted_codelength(p: PlantedData) -> float:
    """Total codelength of the planted model on its own data: the target."""
    return Scorer(p.dag, planted_counters(p)).total_codelength()


def concept_extents(dag: DAG, codes) -> dict[int, set[int]]:
    """FCA extent of each concept under the model: the objects whose generated
    attribute set contains the concept's closure (intent). Containment — not
    explicit activation — so that two representations of the same geometry
    (e.g. a parent activated instead of the child it subsumes) get the same
    extents and can be matched across DAGs."""
    y_masks = []
    for code in codes:
        m = 0
        for s in code:
            m |= dag.closure_mask(s)
        y_masks.append(m)
    extents: dict[int, set[int]] = {}
    for c in dag.concepts:
        cm = dag.closure_mask(c)
        extents[c] = {n for n, ym in enumerate(y_masks) if cm & ~ym == 0}
    return extents
