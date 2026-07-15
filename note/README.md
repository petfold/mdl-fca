# Research note

`concept-dag-note.tex` — a tutorial-level research note explaining the whole
project argument: from Formal Concept Analysis, through Barlow's sparse/factorial
codes and MDL, to the greedy pair-merge learner, the planted-recovery results,
and the overlapping-concepts limitation uncovered so far. Written for a reader
with little specific background; all figures are TikZ/pgfplots, all references
in `refs.bib`.

## Build

Overleaf: upload `concept-dag-note.tex` and `refs.bib`, compile with pdfLaTeX.

Locally:

```sh
pdflatex concept-dag-note
bibtex   concept-dag-note
pdflatex concept-dag-note
pdflatex concept-dag-note
```

Compiles to a 13-page PDF with TeX Live 2023 (pdfLaTeX + BibTeX), no external
assets.
