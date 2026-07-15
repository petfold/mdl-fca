"""Planted-DAG demo (docs/05, definition of done).

Generates a planted hierarchy, runs the greedy MDL learner, and prints the
codelength trajectory, planted vs learned codelength, the size of the full
FCA concept lattice on the same data (the "what we saved you from" number),
and the recovered DAG as an indented text rendering.

Usage: python examples/run_planted.py [--seed 7] [--objects 2000]
"""

import argparse

from mdlfca import (GreedyLearner, concept_extents, make_planted,
                    planted_codelength, row_to_mask)


def count_closed_itemsets(X, cap=50_000):
    """Number of intents of the full FCA lattice: the closure of the row set
    under intersection (rows as bitmasks). Capped — with noise this explodes,
    which is the point."""
    closed: set[int] = set()
    for r in {row_to_mask(row) for row in X}:
        closed |= {r} | {r & c for c in closed}
        if len(closed) > cap:
            return len(closed), True
    return len(closed), False


def render_dag(dag, counters, indent="  "):
    """Indented rendering of the concept DAG; shared nodes reappear under
    each parent."""
    lines = []

    def name(a):
        return f"a{a:02d}"

    def walk(k, depth):
        pad = indent * depth
        if dag.is_attribute(k):
            lines.append(f"{pad}{name(k)}")
            return
        attrs = ", ".join(name(a) for a in sorted(dag.closure(k)))
        lines.append(f"{pad}C{k}  (used {int(counters.usage_of(k))}x)  {{{attrs}}}")
        for ch in sorted(dag.children[k]):
            walk(ch, depth + 1)

    roots = [c for c in sorted(dag.concepts) if not dag.parents.get(c)]
    for r in roots:
        walk(r, 0)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--objects", type=int, default=2000)
    args = ap.parse_args()

    p = make_planted(n_attrs=20, level_sizes=(4, 2), n_objects=args.objects,
                     attrs_per_base=(3, 5), eps_plus=0.02, eps_minus=0.02,
                     seed=args.seed)
    planted_L = planted_codelength(p)
    print(f"planted: {len(p.concept_ids)} concepts over {p.dag.n_attrs} attributes, "
          f"{args.objects} objects, eps={p.eps_plus:g}/{p.eps_minus:g}")

    res = GreedyLearner(p.X, verbose=True).fit()

    t = res.trajectory
    print(f"\ncodelength trajectory: {t[0]:.0f} -> {t[len(t)//2]:.0f} -> {t[-1]:.0f} bits "
          f"({len(t) - 1} accepted moves, monotone)")
    print(f"planted codelength: {planted_L:.0f} bits")
    print(f"learned codelength: {res.total:.0f} bits "
          f"(ratio {res.total / planted_L:.3f}; <1 means the learner encodes "
          f"the noise better than the planted codes do)")
    print(f"learned dictionary: {res.dag.num_concepts} concepts "
          f"(residual: {int(res.counters.n_fp)} FP, {int(res.counters.n_fn)} FN cells)")

    n_closed, truncated = count_closed_itemsets(p.X)
    more = "+" if truncated else ""
    print(f"\nfull FCA lattice on the same data: {n_closed}{more} concepts "
          f"(vs {res.dag.num_concepts} that pay rent)")

    print("\nrecovered DAG:")
    print(render_dag(res.dag, res.counters))

    ext_p = concept_extents(p.dag, p.codes)
    ext_l = concept_extents(res.dag, res.codes)
    print("\nplanted -> learned concept match (extent Jaccard):")
    for c in p.concept_ids:
        best, bj = max(((lc, len(ext_p[c] & ext_l[lc]) / max(1, len(ext_p[c] | ext_l[lc])))
                        for lc in ext_l), key=lambda x: x[1])
        print(f"  C{c} {{{', '.join(f'a{a:02d}' for a in sorted(p.dag.closure(c)))}}} "
              f"-> C{best}  J={bj:.2f}")


if __name__ == "__main__":
    main()
