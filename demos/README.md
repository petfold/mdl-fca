# Demos

## `visualize.py`

Runs several scenarios through the learner and renders figures into `demos/out/`,
one set per scenario (files prefixed with the scenario name):

- `noise` — uncorrelated Bernoulli noise, no structure. The learner posits
  **0 concepts** (nothing pays rent); the codelength panel shows no compressive
  move. A control: this is what "no suspicious coincidence" looks like.
- `two_level` — a planted 2-level hierarchy (as in `tests/test_recovery.py`).
- `three_level` — a planted 3-level hierarchy: deeper structure, concepts stack
  base → mid → top in the DAG panel.
- `product` — two independent factors concatenated (as in `tests/test_product.py`):
  the learned DAG splits into two disconnected components, no cross-block concept.

For each scenario the outputs are:

- `<name>_dag_planted.png` (planted scenarios only) / `<name>_dag_learned.png` —
  the planted vs. learned concept DAG (concepts = boxes, attributes = ellipses).
  The learned DAG highlights one object's activated code.
- `<name>_codes.png` — the learned DAG drawn on top of the data array, with each
  attribute (sink) node aligned exactly above its column. Rows are objects
  (sampled sparse-to-dense) and cells are the attributes predicted by each
  object's code; the column-blocks that light up sit directly under the concept
  that generates them.
- `<name>_codelength.png` — the total codelength trajectory (vs. the planted
  model's L where known) and the per-move delta L (bits saved by each accepted
  move; negative = it compressed).

Run:

```sh
PYTHONPATH=src python3 demos/visualize.py
```

Needs `matplotlib` (plots) and Graphviz `dot` on PATH (DAG images). Edit
`build_scenarios()` to add cases or resize existing ones; keep them smallish so
the DAG stays legible.
