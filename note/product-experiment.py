"""Product-lattice experiment: two independent planted hierarchies on
disjoint attribute blocks, concatenated. Classic FCA on this data yields
(approximately) the PRODUCT of the two concept lattices. Question: does the
MDL learner materialize cross-block 'combination' concepts, or does it
return the disjoint union of the two factors (with the product living in
the object codes)?"""
import os
import sys

import numpy as np

# Resolve src/ relative to this file so the script runs from anywhere.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mdlfca.generator import concept_extents, make_planted
from mdlfca.learner import GreedyLearner


def jaccard(s1, s2):
    u = len(s1 | s2)
    return len(s1 & s2) / u if u else 0.0


N = 1000
pA = make_planted(n_attrs=15, level_sizes=(4, 2), n_objects=N, seed=1)
pB = make_planted(n_attrs=15, level_sizes=(4, 2), n_objects=N, seed=2)
X = np.hstack([pA.X, pB.X])
n_attrs = X.shape[1]
blockA = set(range(15))
blockB = set(range(15, 30))

# how many combinations actually occur (product-lattice pressure is real):
extA = concept_extents(pA.dag, pA.codes)
extB = concept_extents(pB.dag, pB.codes)
combos = sum(1 for ca in extA.values() for cb in extB.values() if ca & cb)
print(f"co-occurring (A-concept, B-concept) pairs: {combos} / {len(extA)*len(extB)}")

res = GreedyLearner(X).fit()
print(f"learned concepts: {res.dag.num_concepts}, total L = {res.total:.0f}")

cross = []
for c in res.dag.concepts:
    attrs = {a for a in res.dag.closure(c) if res.dag.is_attribute(a)}
    inA, inB = bool(attrs & blockA), bool(attrs & blockB)
    tag = "CROSS" if (inA and inB) else ("A" if inA else "B")
    if tag == "CROSS":
        cross.append((c, sorted(attrs)))
    print(f"  concept {c}: block={tag}, attrs={sorted(attrs)}")

print(f"\ncross-block concepts: {len(cross)}")

# factor recovery: every planted concept in each factor matched by a learned one
ext_learned = concept_extents(res.dag, res.codes)
for name, planted, ext_true, offset in (("A", pA, extA, 0), ("B", pB, extB, 15)):
    worst = 1.0
    for c in planted.concept_ids:
        best = max(jaccard(ext_true[c], le) for le in ext_learned.values())
        worst = min(worst, best)
    print(f"factor {name}: worst planted-concept extent Jaccard = {worst:.2f}")

# and the code IS conjunctive: objects use one concept from each block
both = sum(
    1 for code in res.codes
    if any({a for a in res.dag.closure(k) if res.dag.is_attribute(a)} <= blockA for k in code if not res.dag.is_attribute(k))
    and any({a for a in res.dag.closure(k) if res.dag.is_attribute(a)} <= blockB for k in code if not res.dag.is_attribute(k))
)
print(f"objects whose code activates concepts from BOTH factors: {both}/{N}")
