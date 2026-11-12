"""A context from elsewhere (docs/06-roadmap.md §11).

Runs the greedy MDL learner on a table written by OntoDAG's
`experiments/context_export.py` and says how much of the store's own graph
it found. The reference is in the table's `.json` file: the store's edges
among the names, from which each category's closed code follows (itself
and every attribute above it). A learned concept counts as a category
found when its attributes are exactly such a code.

Usage: python examples/run_context.py PREFIX      (PREFIX.npy, PREFIX.json)
"""

import argparse
import json
import time

import numpy as np

from mdlfca import GreedyLearner


def closed_codes(attributes, edges):
    """attribute -> the columns of itself and of every attribute above it."""
    column = {name: i for i, name in enumerate(attributes)}
    above = {}
    for low, high in edges:
        above.setdefault(low, set()).add(high)
    codes = {}
    for name in attributes:
        seen, stack = {name}, [name]
        while stack:
            for high in above.get(stack.pop(), ()):
                if high in column and high not in seen:
                    seen.add(high)
                    stack.append(high)
        codes[name] = frozenset(column[n] for n in seen)
    return codes


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prefix", help="the path PREFIX.npy and PREFIX.json share")
    ap.add_argument("--show", type=int, default=8,
                    help="how many of the most used concepts to print")
    args = ap.parse_args()

    X = np.load(args.prefix + ".npy")
    with open(args.prefix + ".json", encoding="utf-8") as handle:
        meta = json.load(handle)
    attributes = meta["attributes"]
    codes = closed_codes(attributes, meta["edges"])
    named = {}
    for name, code in codes.items():
        named.setdefault(code, []).append(name)
    several = {name for name, code in codes.items() if len(code) > 1}
    print(f"{X.shape[0]} objects, {X.shape[1]} attributes, {int(X.sum())} "
          f"crosses; {len(several)} categories whose code has more than one "
          f"attribute")

    start = time.perf_counter()
    result = GreedyLearner(X).fit()
    spent = time.perf_counter() - start
    dag = result.dag
    closures = {c: frozenset(dag.closure(c)) for c in dag.concepts}
    found = {name for closure in closures.values()
             for name in named.get(closure, ())}
    empty = result.trajectory[0]
    print(f"learned in {spent:.1f} s: {dag.num_concepts} concepts, "
          f"{dag.num_edges} edges; {empty:.0f} -> {result.total:.0f} bits "
          f"({100 * result.total / empty:.0f}% of the empty model)")
    print(f"concepts that are exactly a category's code: "
          f"{sum(1 for c in closures.values() if c in named)} of "
          f"{len(closures)}; categories found: {len(found & several)} of "
          f"{len(several)}")
    by_use = sorted(closures, key=lambda c: -result.counters.usage_of(c))
    for c in by_use[:args.show]:
        names = sorted(attributes[i] for i in closures[c])
        label = named.get(closures[c], ["no category"])[0]
        print(f"  used {int(result.counters.usage_of(c))}x  {label}: "
              f"{{{', '.join(names[:6])}{', …' if len(names) > 6 else ''}}}")


if __name__ == "__main__":
    main()
