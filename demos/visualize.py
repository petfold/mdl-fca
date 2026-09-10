"""Visual demos for mdl-fca. For each scenario it renders into demos/out/:

  1. <name>_codes.png      -- planted DAG (if known) stacked above the learned
                              DAG stacked above the data array, all aligned so
                              each attribute node sits over its data column; a
                              labelled legend explains every colour.
  2. <name>_codelength.png -- the total codelength trajectory and the per-move
                              delta L (bits each accepted move saved).

Dep beyond the project: matplotlib. Run: PYTHONPATH=src python3 demos/visualize.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mdlfca.generator import make_planted
from mdlfca.learner import GreedyLearner

OUT = os.path.join(os.path.dirname(__file__), "out")


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


# colour scheme, shared by every panel and the legend
COL = dict(attr="#cfe0ee", attr_on="#ffd27f", concept="#f2c48a", concept_on="#e8873a",
           attr_edge="#31597a", concept_edge="#7a4a12")


def _draw_dag(ax, dag, A, active, panel_label):
    """Draw one DAG into `ax` with attribute sinks aligned to columns 0..A-1."""
    x, y = _node_layout(dag)
    ymax = max(y.values()) if y else 0
    for c in dag.concepts:
        for ch in dag.children[c]:
            ax.plot([x[c], x[ch]], [y[c], y[ch]], "-", color="#b9b9b9", lw=1, zorder=1)
    for c in dag.concepts:
        on = active is not None and c in active
        ax.scatter([x[c]], [y[c]], s=460, marker="s", zorder=2,
                   color=COL["concept_on"] if on else COL["concept"],
                   edgecolors=COL["concept_edge"])
        ax.text(x[c], y[c], f"C{c}", ha="center", va="center", fontsize=7.5, zorder=3)
    for a in range(A):
        on = active is not None and a in active
        ax.scatter([a], [0], s=240, zorder=2,
                   color=COL["attr_on"] if on else COL["attr"], edgecolors=COL["attr_edge"])
    ax.set_xlim(-0.7, A - 0.3)
    ax.set_ylim(-0.6, ymax + 0.6)
    ax.axis("off")
    ax.text(-0.02, 0.5, panel_label, transform=ax.transAxes, ha="right", va="center",
            fontsize=11, fontweight="bold", rotation=90)
    return ymax


def _legend_handles():
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    m = lambda c, e: Line2D([], [], marker="o", ls="", mfc=c, mec=e, ms=11)
    s = lambda c, e: Line2D([], [], marker="s", ls="", mfc=c, mec=e, ms=11)
    return [
        (m(COL["attr"], COL["attr_edge"]), "attribute (sink)"),
        (m(COL["attr_on"], COL["attr_edge"]), "attribute in example object's code"),
        (s(COL["concept"], COL["concept_edge"]), "concept"),
        (s(COL["concept_on"], COL["concept_edge"]), "concept in example object's code"),
        (mpatches.Patch(fc="#08306b"), "attribute predicted (matrix cell = 1)"),
        (mpatches.Patch(fc="#f7fbff", ec="#c8c8c8"), "not predicted (cell = 0)"),
    ]


def plot_overview(res, dag, planted, png_path, active=None, n_show=40):
    """Stacked, column-aligned figure: planted DAG (if known) above the learned
    DAG above the data array, so matching concepts line up over the same columns.
    A labelled legend explains every colour."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A = dag.n_attrs
    pred = []
    for code in res.codes:
        m = 0
        for k in code:
            m |= dag.closure_mask(k) if not dag.is_attribute(k) else (1 << k)
        pred.append(m)
    ranked = sorted(range(len(res.codes)), key=lambda i: pred[i])
    if len(ranked) > n_show:
        picks = np.linspace(0, len(ranked) - 1, n_show).round().astype(int)
        order = [ranked[j] for j in picks]
    else:
        order = ranked
    M = np.array([[(pred[i] >> a) & 1 for a in range(A)] for i in order], dtype=float)

    planted_dag = planted.dag if planted is not None else None
    pl_ymax = (max(_node_layout(planted_dag)[1].values()) if planted_dag else 0)
    ln_ymax = max(_node_layout(dag)[1].values()) if list(dag.concepts) else 0

    if planted_dag is not None:
        heights = [pl_ymax + 1.4, ln_ymax + 1.4, 6]
        fig, axes = plt.subplots(3, 1, figsize=(max(8, A * 0.55), 11),
                                 gridspec_kw=dict(height_ratios=heights, hspace=0.12))
        ax_pl, ax_ln, axm = axes
        _draw_dag(ax_pl, planted_dag, A, None, "planted")
    else:
        heights = [ln_ymax + 1.4, 6]
        fig, axes = plt.subplots(2, 1, figsize=(max(8, A * 0.55), 9),
                                 gridspec_kw=dict(height_ratios=heights, hspace=0.12))
        ax_ln, axm = axes

    _draw_dag(ax_ln, dag, A, active, "learned")

    axm.imshow(M, aspect="auto", cmap="Blues", interpolation="nearest",
               vmin=0, vmax=1, extent=[-0.5, A - 0.5, len(order) - 0.5, -0.5])
    axm.set_xlim(-0.7, A - 0.3)
    axm.set_xticks(range(A))
    axm.set_xticklabels([f"a{a}" for a in range(A)], fontsize=8)
    axm.set_ylabel(f"data — objects (sampled, {len(order)})", fontweight="bold")

    handles = _legend_handles()
    fig.legend([h for h, _ in handles], [t for _, t in handles],
               loc="upper center", ncol=3, fontsize=8, frameon=True,
               bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("planted vs. learned concept DAG, aligned to the data columns", y=1.03)
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png_path}")


def plot_rent(res, png_path, title=""):
    """Colour each node by its 'rent': the bits total L would RISE if the node
    were removed now and its uses rerouted to its children
    (Scorer.delta_remove_concept, the exact leave-one-out value the pruning
    sweep uses). Positive => the node earns its keep by that many bits.

    Encoding: node COLOUR = rent (colourbar); node SHAPE = type (square =
    concept, circle = attribute sink). Attributes are sinks with no rent, drawn
    neutral. This is a marginal / leave-one-out attribution with every other
    node present, so per-node rents do NOT sum to the total saving — a mid
    concept is cheap only because its base children exist.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    from mdlfca.codelength import Scorer

    dag = res.dag
    scorer = Scorer(dag, res.counters)
    concepts = list(dag.concepts)
    if not concepts:
        return
    rent = {c: scorer.delta_remove_concept(c) for c in concepts}
    A = dag.n_attrs
    x, y = _node_layout(dag)
    ymax = max(y.values())

    cmap = plt.cm.YlGn
    norm = Normalize(vmin=min(0.0, min(rent.values())), vmax=max(rent.values()))

    fig, ax = plt.subplots(figsize=(max(8, A * 0.55), max(4.5, ymax * 1.7 + 2.5)))
    for c in concepts:
        for ch in dag.children[c]:
            ax.plot([x[c], x[ch]], [y[c], y[ch]], "-", color="#b9b9b9", lw=1, zorder=1)
    # attributes: circles, neutral (sinks, no rent)
    for a in range(A):
        ax.scatter([a], [0], s=240, marker="o", zorder=2,
                   color="#e3e9ee", edgecolors=COL["attr_edge"])
    # concepts: squares, colour = rent
    for c in concepts:
        ax.scatter([x[c]], [y[c]], s=560, marker="s", zorder=2,
                   color=cmap(norm(rent[c])), edgecolors="#333")
        ax.text(x[c], y[c], f"C{c}", ha="center", va="center", fontsize=7.5,
                zorder=3, color="#111")
        ax.annotate(f"{rent[c]:.0f} b", (x[c], y[c]), (0, -14),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=6.5, color="#333", zorder=4)
    ax.set_xlim(-0.7, A - 0.3)
    ax.set_ylim(-0.9, ymax + 0.6)
    ax.axis("off")
    ax.set_title(f"earning its keep: node colour = leave-one-out rent  ({title})")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("rent = bits lost if the node were removed")

    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", mfc="#cfe0ee", mec="#333", ms=11,
               label="concept (colour = rent)"),
        Line2D([], [], marker="o", ls="", mfc="#e3e9ee", mec=COL["attr_edge"], ms=11,
               label="attribute (sink; no rent)"),
    ], loc="upper left", fontsize=8, frameon=True)

    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png_path}")


def plot_codelength(res, planted_L, png_path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj = res.trajectory
    deltas = [b - a for a, b in zip(traj, traj[1:])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    ax1.plot(range(len(traj)), traj, "-o", ms=3, color="#c8782a", label="total L")
    if planted_L is not None:
        ax1.axhline(planted_L, ls="--", color="#2a8c50", label=f"planted L = {planted_L:.0f}")
    ax1.set_ylabel("total codelength (bits)")
    ax1.set_title(f"compression trajectory{' — ' + title if title else ''}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    if deltas:
        ax2.bar(range(len(deltas)), deltas,
                color=["#2a8c50" if d < 0 else "#be3c3c" for d in deltas])
    else:
        ax2.text(0.5, 0.5, "no compressive move found\n(model stays at the empty DAG)",
                 ha="center", va="center", transform=ax2.transAxes, color="#be3c3c")
    ax2.set_xlabel("accepted move")
    ax2.set_ylabel("delta L (bits)")
    ax2.set_title("per-move delta L (negative = compresses)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {png_path}")


def build_scenarios():
    """Return a list of (name, title, X, planted). planted is a PlantedData or
    None (noise). These mirror the recovery / product tests plus a noise control."""
    import numpy as np
    from mdlfca.generator import make_planted

    scenarios = []

    # 1. pure uncorrelated noise: independent Bernoulli(0.3), no structure.
    #    Expectation: the learner posits ~no concepts (nothing pays rent).
    rng = np.random.default_rng(0)
    noise = (rng.random((600, 15)) < 0.3).astype(np.uint8)
    scenarios.append(("noise", "uncorrelated noise (no structure)", noise, None))

    # 2. two-level hierarchy (as in tests/test_recovery.py, smaller for legibility)
    p2 = make_planted(n_attrs=16, level_sizes=(4, 2), n_objects=1500,
                      attrs_per_base=(2, 4), eps_plus=0.02, eps_minus=0.02, seed=7)
    scenarios.append(("two_level", "planted 2-level hierarchy", p2.X, p2))

    # 3. three-level hierarchy (as in tests/test_recovery.py): deeper structure
    p3 = make_planted(n_attrs=30, level_sizes=(8, 4, 2), n_objects=2000,
                      attrs_per_base=(3, 5), children_per_concept=(2, 3),
                      level_weight=2.0, eps_plus=0.02, eps_minus=0.02, seed=7)
    scenarios.append(("three_level", "planted 3-level hierarchy", p3.X, p3))

    # 4. product of two independent factors (as in tests/test_product.py)
    common = dict(n_attrs=12, level_sizes=(4, 2), n_objects=1500,
                  attrs_per_base=(2, 3), eps_plus=0.02, eps_minus=0.02)
    pA = make_planted(seed=1, **common)
    pB = make_planted(seed=2, **common)
    scenarios.append(("product", "two independent factors (product)",
                      np.hstack([pA.X, pB.X]), None))

    return scenarios


def main():
    os.makedirs(OUT, exist_ok=True)
    from mdlfca.generator import planted_codelength

    for name, title, X, planted in build_scenarios():
        print(f"\n=== {name}: {title} ===")
        res = GreedyLearner(X).fit()
        planted_L = planted_codelength(planted) if planted is not None else None
        extra = f" (planted {planted_L:.0f})" if planted_L is not None else ""
        print(f"  X: {X.shape[0]} objects x {X.shape[1]} attributes | "
              f"learned {res.dag.num_concepts} concepts, L = {res.total:.0f}{extra}")

        # highlight the code of a well-populated object on the learned DAG
        example = max(res.codes, key=len) if res.codes else set()
        plot_overview(res, res.dag, planted, os.path.join(OUT, f"{name}_codes.png"),
                      active=example)
        plot_codelength(res, planted_L, os.path.join(OUT, f"{name}_codelength.png"),
                        title=title)
        plot_rent(res, os.path.join(OUT, f"{name}_rent.png"), title=title)

    print(f"\nDone. See {OUT}/")


if __name__ == "__main__":
    main()
