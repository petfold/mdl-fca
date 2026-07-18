# Demos

## `visualize.py`

Runs a small planted example through the learner and renders four PNGs into
`demos/out/`:

- `dag_planted.png` / `dag_learned.png` — the planted vs. learned concept DAG
  (concepts = boxes, attributes = ellipses). The learned DAG highlights one
  object's activated code.
- `codes.png` — object codes as a heatmap: which items (attributes | concepts,
  split by the red line) each object switches on. Sparse, concept-only codes
  show up as activity only to the right of the line.
- `codelength.png` — the total codelength trajectory (vs. the planted model's L)
  and the per-move delta L (bits saved by each accepted move; negative = it
  compressed).

Run:

```sh
PYTHONPATH=src python3 demos/visualize.py
```

Needs `matplotlib` (plots) and Graphviz `dot` on PATH (DAG images). Edit the
`make_planted(...)` call in `main()` to try other sizes; keep it smallish so the
DAG stays legible.
