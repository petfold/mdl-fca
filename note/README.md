# Research notes

## `concept-dag-note.tex`

A tutorial-level research note explaining the whole project argument: from Formal
Concept Analysis, through Barlow's sparse/factorial codes and MDL, to the greedy
pair-merge learner, the planted-recovery results, and the overlapping-concepts
limitation uncovered so far. Written for a reader with little specific
background; all figures are TikZ/pgfplots, all references in `refs.bib`.

## `product-of-dags-note.tex`

A standalone note on one clean result: over **disjoint** attribute alphabets, the
**coproduct** (disjoint union) of two closure DAGs generates the **direct
product** of the lattices they generate individually. The generator combines
*additively* while the represented concept lattice combines *multiplicatively* —
i.e. the exponential compression of a factorial code, written as a lattice
identity. Stated decoder-agnostically (Lemma: separable decoders factor), it
stands independently of `mdl-fca`; it also explains why the learner returns a
*product of factor DAGs* rather than one tangled graph, and why "learn the neural
code" vs. "learn the DAG" is not a dichotomy. Boundary conditions (disjoint
alphabets, statistical independence, realisability) are spelled out.

- `product-experiment.py` — the confirming experiment (two independent planted
  hierarchies, concatenated): the MDL learner returns the two factors with **zero
  cross-block concepts** and carries the product in the object codes. Run with
  `python3 note/product-experiment.py` (resolves `src/` relatively).
- `figures/factor-dags.dot` — Graphviz source for the coproduct figure (the note
  ships a self-contained TikZ copy; this is kept so the graph can be regenerated
  or restyled). Render: `dot -Tpdf figures/factor-dags.dot -o factor-dags.pdf`
  (or `dot2tex --format tikz` if installed).

## Build (both notes)

Overleaf: upload the `.tex` and `refs.bib`, compile with pdfLaTeX.

Locally (TeX Live; `product-of-dags-note` needs the `lmodern` package):

```sh
pdflatex <note>
bibtex   <note>
pdflatex <note>
pdflatex <note>
```

Both compile with pdfLaTeX + BibTeX, no external assets (all figures TikZ).
