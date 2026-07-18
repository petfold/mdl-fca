# TikZ demo figures (for Beamer / paper)

Native TikZ/pgfplots versions of the demo figures, generated **from scratch out
of the learner's output** (DAG structure, predicted-attribute matrix, codelength
trajectory) — not traced from the PNGs. Vector, restyleable, and matching the
note's fonts, so they drop straight into a slide deck or paper.

## Generate + build

```sh
PYTHONPATH=src python3 demos/tikz_export.py     # writes mdl-fca-demos.tex
cd demos/tikz
pdflatex mdl-fca-demos
pdflatex mdl-fca-demos                           # 2nd pass for pgfplots
```

Needs `matplotlib` (only to sample the YlGn rent colourmap) and a TeX Live with
`tikz`, `pgfplots`, and `lmodern`.

## Reusing single figures

Each scenario emits three `tikzpicture`s (codes / rent / codelength) via the
functions in `demos/tikz_export.py` (`codes_figure`, `rent_figure`,
`codelength_figure`). To pull one figure into another document, copy the colour
`\definecolor`s and the `\tikzset` / `\pgfplotsset` styles from the preamble of
`mdl-fca-demos.tex`, then paste the `tikzpicture` body. Adjust `FIG_W` in the
exporter to change the target width.

The raster PNG versions remain in `demos/out/` (produced by `demos/visualize.py`)
for side-by-side comparison.
