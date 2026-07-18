"""Visual demos for mdl-fca. Renders, for a small planted example:

  1. the planted DAG and the learned DAG           -> dag_planted.png, dag_learned.png
  2. the object codes as a heatmap (which items    -> codes.png
     each object activates, concepts + attributes)
  3. the total codelength trajectory and the       -> codelength.png
     per-move delta L (how many bits each move saved)

Deps beyond the project: matplotlib (plots) and Graphviz `dot` on PATH (DAGs).
Run:  PYTHONPATH=src python3 demos/visualize.py
Output PNGs land in demos/out/.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mdlfca.generator import make_planted
from mdlfca.learner import GreedyLearner

OUT = os.path.join(os.path.dirname(__file__), "out")


def dag_to_dot(dag, title, active=None):
    """Graphviz source. Concepts are boxes, attributes are ellipses. If `active`
    (a set of item ids) is given, activated nodes are highlighted."""
    active = active or set()
    lines = [f'digraph G {{ rankdir=TB; label="{title}"; labelloc=t;',
             '  node [fontname="Helvetica"];']
    for a in range(dag.n_attrs):
        fill = "#ffd27f" if a in active else "#cfe0ee"
        lines.append(f'  a{a} [label="a{a}", shape=ellipse, style=filled, fillcolor="{fill}"];')
    for c in dag.concepts:
        fill = "#e8873a" if c in active else "#f2c48a"
        lines.append(f'  c{c} [label="C{c}", shape=box, style="rounded,filled", fillcolor="{fill}"];')
    for c in dag.concepts:
        for ch in dag.children[c]:
            tgt = f"a{ch}" if dag.is_attribute(ch) else f"c{ch}"
            lines.append(f"  c{c} -> {tgt};")
    lines.append("}")
    return "\n".join(lines)


def render_dot(dot_src, png_path):
    if not shutil.which("dot"):
        print("  (skipping DAG render: Graphviz `dot` not on PATH)")
        return
    subprocess.run(["dot", "-Tpng", "-o", png_path], input=dot_src.encode(), check=True)
    print(f"  wrote {png_path}")


def plot_codes(res, dag, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    concepts = sorted(dag.concepts)
    items = list(range(dag.n_attrs)) + concepts
    labels = [f"a{a}" for a in range(dag.n_attrs)] + [f"C{c}" for c in concepts]
    idx = {it: j for j, it in enumerate(items)}

    # show a sample of objects to keep it readable
    n_show = min(60, len(res.codes))
    M = np.zeros((n_show, len(items)))
    for i in range(n_show):
        for it in res.codes[i]:
            M[i, idx[it]] = 1

    fig, ax = plt.subplots(figsize=(max(6, len(items) * 0.28), 8))
    ax.imshow(M, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.axvline(dag.n_attrs - 0.5, color="crimson", lw=1.2)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel(f"objects (first {n_show})")
    ax.set_title("object codes: activated items (attributes | concepts)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {png_path}")


def plot_codelength(res, planted_L, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj = res.trajectory
    deltas = [b - a for a, b in zip(traj, traj[1:])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    ax1.plot(range(len(traj)), traj, "-o", ms=3, color="#c8782a", label="total L")
    ax1.axhline(planted_L, ls="--", color="#2a8c50", label=f"planted L = {planted_L:.0f}")
    ax1.set_ylabel("total codelength (bits)")
    ax1.set_title("compression trajectory")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar(range(len(deltas)), deltas, color=["#2a8c50" if d < 0 else "#be3c3c" for d in deltas])
    ax2.set_xlabel("accepted move")
    ax2.set_ylabel("delta L (bits)")
    ax2.set_title("per-move delta L (negative = compresses)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {png_path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    from mdlfca.generator import planted_codelength

    p = make_planted(n_attrs=12, level_sizes=(4, 2), n_objects=800,
                     attrs_per_base=(2, 3), eps_plus=0.02, eps_minus=0.02, seed=3)
    print(f"planted: {p.dag.num_concepts} concepts, {p.dag.n_attrs} attributes, "
          f"{len(p.codes)} objects")

    res = GreedyLearner(p.X).fit()
    planted_L = planted_codelength(p)
    print(f"learned: {res.dag.num_concepts} concepts, total L = {res.total:.0f} "
          f"(planted {planted_L:.0f})")

    render_dot(dag_to_dot(p.dag, "planted DAG"), os.path.join(OUT, "dag_planted.png"))
    # highlight one object's code on the learned DAG
    example = res.codes[0]
    render_dot(dag_to_dot(res.dag, f"learned DAG (object 0 code highlighted: {sorted(example)})",
                          active=example),
               os.path.join(OUT, "dag_learned.png"))
    plot_codes(res, res.dag, os.path.join(OUT, "codes.png"))
    plot_codelength(res, planted_L, os.path.join(OUT, "codelength.png"))
    print(f"\nDone. See {OUT}/")


if __name__ == "__main__":
    main()
