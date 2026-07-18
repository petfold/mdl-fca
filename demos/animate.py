"""Animate the learning process: every accepted move is a frame, assembled into
a GIF. Two animations per scenario, both with FIXED axes so nothing jumps:

  <name>_learn.gif       the learned DAG growing over the (fixed) data array
  <name>_codelength.gif  the codelength trajectory + per-move delta revealed

Snapshots are captured through GreedyLearner's on_commit hook. Needs matplotlib
(+ its Pillow writer). Run: PYTHONPATH=src python3 demos/animate.py
GIFs land in demos/out/anim/.
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mdlfca.dag import DAG
from mdlfca.generator import planted_codelength
from mdlfca.learner import GreedyLearner

from visualize import COL, _node_layout, build_scenarios  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "out", "anim")
N_SHOW = 20
FPS = 2.5


def _predicted_mask(dag, code):
    m = 0
    for k in code:
        m |= dag.closure_mask(k) if not dag.is_attribute(k) else (1 << k)
    return m


def _run_with_snapshots(X):
    """Fit, capturing (dag, codes, total) after every accepted move."""
    snaps = []

    def hook(learner, event):
        snaps.append((copy.deepcopy(learner.dag),
                      [set(c) for c in learner.codes], learner.total))

    res = GreedyLearner(X, on_commit=hook).fit()
    # prepend the initial state (empty DAG, raw attribute codes)
    A = X.shape[1]
    dag0 = DAG(A)
    codes0 = [{int(a) for a in np.flatnonzero(r)} for r in X]
    snaps.insert(0, (dag0, codes0, res.trajectory[0]))
    return res, snaps


def _sample_rows(res, dag):
    pred = [_predicted_mask(dag, c) for c in res.codes]
    ranked = sorted(range(len(res.codes)), key=lambda i: pred[i])
    if len(ranked) > N_SHOW:
        picks = np.linspace(0, len(ranked) - 1, N_SHOW).round().astype(int)
        return [ranked[j] for j in picks]
    return ranked


def animate_learn(res, snaps, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    A = res.dag.n_attrs
    rows = _sample_rows(res, res.dag)                      # fixed object sample & order
    # y-range must cover the tallest transient DAG (the learner over-builds
    # before the sweep prunes), or mid-animation nodes get clipped.
    ymax = max((max(_node_layout(d)[1].values()) for d, _, _ in snaps
                if list(d.concepts)), default=1)

    fig, (axd, axm) = plt.subplots(
        2, 1, figsize=(max(8, A * 0.5), 8),
        gridspec_kw=dict(height_ratios=[ymax + 1.6, 5], hspace=0.08))

    def draw(f):
        dag, codes, total = snaps[f]
        prev = set(snaps[f - 1][0].concepts) if f > 0 else set()
        new = set(dag.concepts) - prev
        axd.clear(); axm.clear()
        x, y = _node_layout(dag)
        for c in dag.concepts:
            for ch in dag.children[c]:
                axd.plot([x[c], x[ch]], [y[c], y[ch]], "-", color="#b9b9b9", lw=1, zorder=1)
        for a in range(A):
            axd.scatter([a], [0], s=200, zorder=2, color=COL["attr"], edgecolors=COL["attr_edge"])
        for c in dag.concepts:
            axd.scatter([x[c]], [y[c]], s=430, marker="s", zorder=2,
                        color=COL["concept_on"] if c in new else COL["concept"],
                        edgecolors=COL["concept_edge"])
            axd.text(x[c], y[c], f"C{c}", ha="center", va="center", fontsize=7, zorder=3)
        axd.set_xlim(-0.7, A - 0.3)
        axd.set_ylim(-0.6, ymax + 0.6)
        axd.axis("off")
        axd.set_title(f"{title}\nstep {f}/{len(snaps)-1}   "
                      f"concepts: {dag.num_concepts}   L = {total:.0f} bits", fontsize=10)

        M = np.array([[(_predicted_mask(dag, codes[i]) >> a) & 1 for a in range(A)]
                      for i in rows], dtype=float)
        axm.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                   interpolation="nearest", extent=[-0.5, A - 0.5, len(rows) - 0.5, -0.5])
        axm.set_xlim(-0.7, A - 0.3)
        axm.set_xticks(range(A))
        axm.set_xticklabels([f"a{a}" for a in range(A)], fontsize=7)
        axm.set_ylabel("objects (fixed sample)")

    frames = list(range(len(snaps))) + [len(snaps) - 1] * 4   # hold final frame
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / FPS)
    anim.save(path, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"  wrote {path}")


def animate_codelength(res, path, planted_L, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    traj = res.trajectory
    deltas = [b - a for a, b in zip(traj, traj[1:])]
    K = len(traj) - 1
    ymin, ymax = min(traj), max(traj)
    pad = 0.05 * (ymax - ymin or 1)
    dmin = min(deltas + [0]); dpad = 0.1 * (abs(dmin) or 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

    def draw(k):
        ax1.clear(); ax2.clear()
        ax1.plot(range(k + 1), traj[:k + 1], "-o", ms=3, color="#c8782a", label="learned L")
        if planted_L is not None:
            ax1.axhline(planted_L, ls="--", color="#2a8c50", label=f"planted L = {planted_L:.0f}")
        ax1.set_xlim(-0.5, K + 0.5); ax1.set_ylim(ymin - pad, ymax + pad)
        ax1.set_ylabel("total codelength (bits)")
        ax1.set_title(f"{title}\ncompression trajectory — move {k}/{K}", fontsize=10)
        ax1.legend(loc="upper right"); ax1.grid(alpha=0.3)

        d = deltas[:k]
        ax2.bar(range(len(d)), d, color=["#2a8c50" if v < 0 else "#be3c3c" for v in d])
        ax2.set_xlim(-0.5, K + 0.5); ax2.set_ylim(dmin - dpad, dpad)
        ax2.set_xlabel("accepted move"); ax2.set_ylabel("delta L (bits)")
        ax2.set_title("per-move delta L (negative = compresses)")
        ax2.grid(alpha=0.3)
        fig.tight_layout()

    frames = list(range(K + 1)) + [K] * 4
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / FPS)
    anim.save(path, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, title, X, planted in build_scenarios():
        res, snaps = _run_with_snapshots(X)
        planted_L = planted_codelength(planted) if planted is not None else None
        print(f"=== {name}: {len(snaps)-1} accepted moves ===")
        if len(snaps) < 3:
            print("  (no structure learned — skipping animations)")
            continue
        animate_learn(res, snaps, os.path.join(OUT, f"{name}_learn.gif"), title)
        animate_codelength(res, os.path.join(OUT, f"{name}_codelength.gif"), planted_L, title)
    print(f"\nDone. See {OUT}/")


if __name__ == "__main__":
    main()
