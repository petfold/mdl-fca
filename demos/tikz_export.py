"""Generate a LaTeX document of the demo figures as native TikZ/pgfplots
(drawn from scratch out of the learner's output -- not traced from the PNGs),
so the same figures drop straight into a Beamer deck or paper.

Run:  PYTHONPATH=src python3 demos/tikz_export.py
Writes demos/tikz/mdl-fca-demos.tex; compile with pdflatex (see demos/tikz/README).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mdlfca.codelength import Scorer
from mdlfca.generator import planted_codelength
from mdlfca.learner import GreedyLearner

from visualize import _node_layout, build_scenarios  # noqa: E402  (same dir)

OUTDIR = os.path.join(os.path.dirname(__file__), "tikz")
FIG_W = 13.0          # target figure width (cm)
LAYER_H = 1.05        # vertical gap between DAG levels (cm)
ROW_H = 0.24          # matrix row height (cm)
N_SHOW = 22           # objects drawn in the matrix


def _ylgn(t: float) -> str:
    """YlGn colormap sample as 'r,g,b' in 0..1 (matches the PNG rent colouring)."""
    import matplotlib.cm as cm
    r, g, b = cm.YlGn(max(0.0, min(1.0, t)))[:3]
    return f"{r:.3f},{g:.3f},{b:.3f}"


def _predicted(res, dag):
    pred = []
    for code in res.codes:
        m = 0
        for k in code:
            m |= dag.closure_mask(k) if not dag.is_attribute(k) else (1 << k)
        pred.append(m)
    return pred


def _dag_tikz(L, dag, dx, y0, prefix, active, rent=None, rnorm=None):
    """Emit one DAG: attribute sinks on the baseline y0, concepts stacked above."""
    x, y = _node_layout(dag)
    for a in range(dag.n_attrs):
        L.append(f"  \\node[attr] ({prefix}a{a}) at ({a*dx:.3f},{y0:.3f}) {{}};")
    for c in dag.concepts:
        cy = y0 + y[c] * LAYER_H
        cx = x[c] * dx
        if rent is not None:
            t = rnorm(rent[c])
            L.append(f"  \\definecolor{{rc{prefix}{c}}}{{rgb}}{{{_ylgn(t)}}}")
            style = f"concept, fill=rc{prefix}{c}, text={'white' if t > 0.55 else 'black'}"
        elif active is not None and c in active:
            style = "concept, fill=cConcOn"
        else:
            style = "concept"
        L.append(f"  \\node[{style}] ({prefix}c{c}) at ({cx:.3f},{cy:.3f}) {{\\tiny C{c}}};")
    for c in dag.concepts:
        for ch in dag.children[c]:
            tgt = f"{prefix}a{ch}" if dag.is_attribute(ch) else f"{prefix}c{ch}"
            L.append(f"  \\draw[edge] ({prefix}c{c}) -- ({tgt});")


def _matrix_tikz(L, res, dag, dx, top):
    """Filled cells for the predicted-attribute array; rows go downward from `top`."""
    A = dag.n_attrs
    pred = _predicted(res, dag)
    ranked = sorted(range(len(res.codes)), key=lambda i: pred[i])
    if len(ranked) > N_SHOW:
        picks = np.linspace(0, len(ranked) - 1, N_SHOW).round().astype(int)
        order = [ranked[j] for j in picks]
    else:
        order = ranked
    for r, i in enumerate(order):
        ytop = top - r * ROW_H
        for a in range(A):
            if (pred[i] >> a) & 1:
                L.append(f"  \\fill[cCell] ({a*dx - dx/2:.3f},{ytop:.3f}) "
                         f"rectangle ({a*dx + dx/2:.3f},{ytop - ROW_H:.3f});")
    bottom = top - len(order) * ROW_H
    L.append(f"  \\draw[cellframe] ({-dx/2:.3f},{top:.3f}) "
             f"rectangle ({(A-1)*dx + dx/2:.3f},{bottom:.3f});")
    for a in range(A):
        L.append(f"  \\node[collab] at ({a*dx:.3f},{bottom-0.05:.3f}) {{a{a}}};")
    return bottom, len(order)


def _panel_label(L, text, x, y):
    L.append(f"  \\node[panellab] at ({x:.3f},{y:.3f}) {{{text}}};")


def codes_figure(res, dag, planted, dx):
    A = dag.n_attrs
    L = ["\\begin{tikzpicture}"]
    # matrix at the bottom (top edge y=0, growing downward)
    mtop = 0.0
    mbot, nrow = _matrix_tikz(L, res, dag, dx, mtop)
    _panel_label(L, "data", -1.15, (mtop + mbot) / 2)
    # learned DAG above the matrix
    _, yl = _node_layout(dag)
    Dl = max(yl.values()) if list(dag.concepts) else 0
    yl0 = mtop + 0.95
    ex = max(res.codes, key=len) if res.codes else set()
    _dag_tikz(L, dag, dx, yl0, "ln", active=ex)
    _panel_label(L, "learned", -1.15, yl0 + Dl * LAYER_H / 2)
    top = yl0 + Dl * LAYER_H
    # planted DAG on top (when known)
    if planted is not None:
        _, yp = _node_layout(planted.dag)
        Dp = max(yp.values())
        yp0 = top + 1.5
        _dag_tikz(L, planted.dag, dx, yp0, "pl", active=None)
        _panel_label(L, "planted", -1.15, yp0 + Dp * LAYER_H / 2)
    L.append("\\end{tikzpicture}")
    return "\n".join(L)


def rent_figure(res, dag, dx):
    concepts = list(dag.concepts)
    if not concepts:
        return None
    scorer = Scorer(dag, res.counters)
    rent = {c: scorer.delta_remove_concept(c) for c in concepts}
    lo, hi = min(0.0, min(rent.values())), max(rent.values())
    rng = hi - lo or 1.0
    rnorm = lambda v: (v - lo) / rng
    L = ["\\begin{tikzpicture}"]
    _dag_tikz(L, dag, dx, 0.0, "rt", active=None, rent=rent, rnorm=rnorm)
    # rent value labels under each concept
    x, y = _node_layout(dag)
    for c in concepts:
        L.append(f"  \\node[rentval] at ({x[c]*dx:.3f},{y[c]*LAYER_H - 0.34:.3f}) "
                 f"{{{rent[c]:.0f}\\,b}};")
    # colourbar
    A = dag.n_attrs
    cbx, cby, cbw = (A - 1) * dx + 0.6, 0.0, 0.45
    cbh = (max(y.values())) * LAYER_H
    steps = 24
    for s in range(steps):
        t0, t1 = s / steps, (s + 1) / steps
        L.append(f"  \\definecolor{{cb{s}}}{{rgb}}{{{_ylgn(t0)}}}")
        L.append(f"  \\fill[cb{s}] ({cbx:.3f},{cby + t0*cbh:.3f}) "
                 f"rectangle ({cbx+cbw:.3f},{cby + t1*cbh:.3f});")
    L.append(f"  \\draw[cellframe] ({cbx:.3f},{cby:.3f}) rectangle ({cbx+cbw:.3f},{cby+cbh:.3f});")
    L.append(f"  \\node[cblab] at ({cbx+cbw+0.1:.3f},{cby:.3f}) {{{lo:.0f}}};")
    L.append(f"  \\node[cblab] at ({cbx+cbw+0.1:.3f},{cby+cbh:.3f}) {{{hi:.0f}\\,b}};")
    L.append("\\end{tikzpicture}")
    return "\n".join(L)


def codelength_figure(res, planted_L):
    traj = res.trajectory
    deltas = [b - a for a, b in zip(traj, traj[1:])]
    tcoords = " ".join(f"({i},{v:.0f})" for i, v in enumerate(traj))
    L = ["\\begin{tikzpicture}",
         "\\begin{axis}[cltraj, xlabel={accepted move}, ylabel={total $L$ (bits)}]"]
    L.append(f"  \\addplot[cLine] coordinates {{{tcoords}}};")
    if planted_L is not None:
        L.append(f"  \\addplot[cPlanted, domain=0:{len(traj)-1}] {{{planted_L:.0f}}};")
        L.append(f"  \\addlegendentry{{learned $L$}} \\addlegendentry{{planted $L$}}")
    L.append("\\end{axis}")
    if deltas:
        dcoords = " ".join(f"({i},{d:.0f})" for i, d in enumerate(deltas))
        L.append("\\begin{axis}[cldelta, xlabel={accepted move}, ylabel={$\\Delta L$ (bits)}]")
        L.append(f"  \\addplot[ybar, bar width=3pt, draw=cAmberE, fill=cGood] coordinates {{{dcoords}}};")
        L.append("\\end{axis}")
    L.append("\\end{tikzpicture}")
    return "\n".join(L)


PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1.8cm]{geometry}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usetikzlibrary{arrows.meta}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}

\definecolor{cAttr}{RGB}{207,224,238}
\definecolor{cAttrE}{RGB}{49,89,122}
\definecolor{cConc}{RGB}{242,196,138}
\definecolor{cConcOn}{RGB}{232,135,58}
\definecolor{cAmberE}{RGB}{122,74,18}
\definecolor{cCell}{RGB}{8,48,107}
\definecolor{cGood}{RGB}{42,140,80}
\definecolor{cPlantedC}{RGB}{42,140,80}

\tikzset{
  attr/.style={draw=cAttrE, circle, fill=cAttr, minimum size=3.4mm, inner sep=0pt},
  concept/.style={draw=cAmberE, rounded corners=1pt, fill=cConc, minimum size=4.8mm, inner sep=1pt},
  edge/.style={-{Latex[length=1.3mm]}, draw=black!30, line width=.5pt},
  panellab/.style={font=\bfseries\small, rotate=90, anchor=center},
  collab/.style={font=\fontsize{4.4}{5}\selectfont, anchor=north},
  rentval/.style={font=\fontsize{5}{6}\selectfont, text=black!55, anchor=north},
  cblab/.style={font=\scriptsize, anchor=west},
}
\pgfplotsset{
  cltraj/.style={width=13cm, height=5cm, grid=both, grid style={black!12},
     legend style={font=\scriptsize, at={(0.98,0.95)}, anchor=north east}, tick label style={font=\scriptsize}},
  cldelta/.style={width=13cm, height=4cm, grid=both, grid style={black!12},
     tick label style={font=\scriptsize}, yshift=-5.4cm},
  cLine/.style={color=cConcOn, very thick, mark=*, mark size=1pt},
  cPlanted/.style={color=cPlantedC, dashed, thick},
}

\newcommand{\keyfig}{%
\begin{tikzpicture}[baseline]
  \node[attr] (a) at (0,0) {}; \node[anchor=west, font=\footnotesize] at (0.35,0) {attribute (sink)};
  \node[concept] (c) at (4.6,0) {}; \node[anchor=west, font=\footnotesize] at (5.0,0) {concept};
  \node[concept, fill=cConcOn] (o) at (8.0,0) {}; \node[anchor=west, font=\footnotesize] at (8.4,0) {in example code};
  \fill[cCell] (0,-0.6) rectangle (0.3,-0.9); \node[anchor=west, font=\footnotesize] at (0.35,-0.75) {matrix cell = 1 (predicted)};
  \draw[cellframe] (4.6,-0.6) rectangle (4.9,-0.9); \node[anchor=west, font=\footnotesize] at (5.0,-0.75) {cell = 0 (not predicted)};
\end{tikzpicture}}
\tikzset{cellframe/.style={draw=black!35, line width=.4pt}}
"""


def build():
    os.makedirs(OUTDIR, exist_ok=True)
    meta = {
        "noise": ("Control: uncorrelated noise",
                  "Independent Bernoulli(0.3) with no hidden structure. Nothing is a "
                  "suspicious coincidence, so no concept pays for itself: the learner returns "
                  "\\textbf{0 concepts}. The control for over-generation."),
        "two_level": ("Planted two-level hierarchy",
                  "Four base concepts under two parents. Recovered, and compressed below the "
                  "codelength of the generating model."),
        "three_level": ("Planted three-level hierarchy",
                  "Base~$\\to$~mid~$\\to$~top. All concepts recovered in three tiers that align "
                  "with the planted ones column-for-column; base concepts carry most of the rent."),
        "product": ("Two independent factors (a product)",
                  "Two independent hierarchies on disjoint attribute blocks, concatenated. The "
                  "learner returns two disconnected components with no cross-block concept; the "
                  "product lives in the object codes."),
    }
    parts = [PREAMBLE, r"\begin{document}",
             r"\begin{center}\Large\textbf{mdl-fca demo figures}\end{center}",
             r"\noindent The learner builds a concept DAG that compresses a binary "
             r"object$\times$attribute matrix, positing a concept only when it shortens the "
             r"total description length. Figures are generated as native TikZ from the "
             r"learner's output.\par\medskip",
             r"\begin{center}\keyfig\end{center}"]

    for name, title, X, planted in build_scenarios():
        res = GreedyLearner(X).fit()
        planted_L = planted_codelength(planted) if planted is not None else None
        A = X.shape[1]
        dx = FIG_W / max(1, A - 1)
        head, desc = meta[name]
        parts.append(f"\\section*{{{head}}}")
        parts.append(desc + r"\par\medskip")
        parts.append(r"\begin{center}" + codes_figure(res, res.dag, planted, dx) + r"\end{center}")
        rf = rent_figure(res, res.dag, dx)
        if rf:
            parts.append(r"\medskip\noindent\textit{Rent} (bits lost if a node were removed; "
                         r"node colour):\par\begin{center}" + rf + r"\end{center}")
        parts.append(r"\begin{center}" + codelength_figure(res, planted_L) + r"\end{center}")
        parts.append(r"\clearpage")

    parts.append(r"\end{document}")
    path = os.path.join(OUTDIR, "mdl-fca-demos.tex")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    print("wrote", path)


if __name__ == "__main__":
    build()
