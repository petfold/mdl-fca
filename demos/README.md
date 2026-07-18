# Demos

## `visualize.py`

Runs a small planted example through the learner and renders four PNGs into
`demos/out/`:

- `dag_planted.png` / `dag_learned.png` — the planted vs. learned concept DAG
  (concepts = boxes, attributes = ellipses). The learned DAG highlights one
  object's activated code.
- `codes.png` — the learned DAG drawn on top of the data array, with each
  attribute (sink) node aligned exactly above its column. Rows are objects
  (sampled sparse-to-dense) and cells are the attributes predicted by each
  object's code; the column-blocks that light up sit directly under the concept
  that generates them.
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
