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


def _node_layout(dag):
    """x = mean attribute-column index the node covers; y = depth above attrs."""
    attrs_of = {a: {a} for a in range(dag.n_attrs)}
    for c in dag.concepts:
        attrs_of[c] = {a for a in dag.closure(c) if dag.is_attribute(a)}

    def depth(node, seen=None):
        if dag.is_attribute(node):
            return 0
        seen = seen or set()
        return 1 + max((depth(ch, seen | {node}) for ch in dag.children[node]),
                       default=0)

    x = {n: (n if dag.is_attribute(n) else float(np.mean(sorted(attrs_of[n]))))
         for n in list(range(dag.n_attrs)) + list(dag.concepts)}
    y = {n: depth(n) for n in x}

    # spread out concepts that share a level and sit too close (min gap 0.9)
    by_level = {}
    for c in dag.concepts:
        by_level.setdefault(y[c], []).append(c)
    for level, nodes in by_level.items():
        nodes.sort(key=lambda c: x[c])
        for prev, cur in zip(nodes, nodes[1:]):
            if x[cur] - x[prev] < 0.9:
                x[cur] = x[prev] + 0.9
    return x, y


def plot_dag_over_matrix(res, dag, png_path, active=None, n_show=40):
    """Data array with the learned DAG drawn on top, each attribute node aligned
    to its column. Objects are sorted by their predicted attribute pattern so the
    blocks a concept generates line up under it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A = dag.n_attrs
    # predicted attributes per object (closure of the code) drive the row order
    pred = []
    for code in res.codes:
        m = 0
        for k in code:
            m |= dag.closure_mask(k) if not dag.is_attribute(k) else (1 << k)
        pred.append(m)
    # sample evenly across the sorted range so sparse-to-dense variety shows
    ranked = sorted(range(len(res.codes)), key=lambda i: pred[i])
    if len(ranked) > n_show:
        picks = np.linspace(0, len(ranked) - 1, n_show).round().astype(int)
        order = [ranked[j] for j in picks]
    else:
        order = ranked
    M = np.array([[(pred[i] >> a) & 1 for a in range(A)] for i in order], dtype=float)

    x, y = _node_layout(dag)
    ymax = max(y.values())

    fig, (axd, axm) = plt.subplots(
        2, 1, figsize=(max(7, A * 0.55), 9),
        gridspec_kw=dict(height_ratios=[ymax + 1.2, 5], hspace=0.05))

    # ---- DAG on top ----
    for c in dag.concepts:
        for ch in dag.children[c]:
            axd.plot([x[c], x[ch]], [y[c], y[ch]], "-", color="#b9b9b9", lw=1, zorder=1)
    for c in dag.concepts:
        on = active is not None and c in active
        axd.scatter([x[c]], [y[c]], s=520, marker="s", zorder=2,
                    color="#e8873a" if on else "#f2c48a", edgecolors="#7a4a12")
        axd.text(x[c], y[c], f"C{c}", ha="center", va="center", fontsize=8, zorder=3)
    for a in range(A):
        on = active is not None and a in active
        axd.scatter([a], [0], s=300, zorder=2,
                    color="#ffd27f" if on else "#cfe0ee", edgecolors="#31597a")
    axd.set_xlim(-0.7, A - 0.3)
    axd.set_ylim(-0.6, ymax + 0.6)
    axd.axis("off")
    axd.set_title("learned concept DAG (sinks aligned to the columns below)")

    # ---- data array below, columns aligned ----
    axm.imshow(M, aspect="auto", cmap="Blues", interpolation="nearest",
               extent=[-0.5, A - 0.5, n_show - 0.5, -0.5])
    axm.set_xlim(-0.7, A - 0.3)
    axm.set_xticks(range(A))
    axm.set_xticklabels([f"a{a}" for a in range(A)], fontsize=8)
    axm.set_ylabel(f"objects (sorted, first {n_show})")
    axm.set_title("predicted attributes per object", fontsize=10)

    fig.savefig(png_path, dpi=120, bbox_inches="tight")
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
    plot_dag_over_matrix(res, res.dag, os.path.join(OUT, "codes.png"), active=example)
    plot_codelength(res, planted_L, os.path.join(OUT, "codelength.png"))
    print(f"\nDone. See {OUT}/")


if __name__ == "__main__":
    main()
