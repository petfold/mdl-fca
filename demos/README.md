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

- `<name>_codes.png` — the planted DAG (for planted scenarios) stacked directly
  above the learned DAG, stacked above the data array, all column-aligned so each
  attribute (sink) node sits over its data column. Matching concepts appear over
  the same columns in the planted and learned panels, so recovery is read off by
  eye. Rows of the data array are objects (sampled sparse-to-dense); cells are
  the attributes predicted by each object's code. A labelled legend explains
  every colour (attribute vs. concept, highlighted = in the example object's
  code, matrix cell = predicted/not).
- `<name>_codelength.png` — the total codelength trajectory (vs. the planted
  model's L where known) and the per-move delta L (bits saved by each accepted
  move; negative = it compressed).
- `<name>_rent.png` — "earning its keep" per concept. Each node is priced by its
  **rent**: the bits total L would *rise* if that node were removed now and its
  uses rerouted to its children (`Scorer.delta_remove_concept`, the exact
  leave-one-out value the pruning sweep uses). Encoding: node **colour** = rent
  (colourbar), node **shape** = type (square = concept, circle = attribute sink;
  sinks have no rent and are drawn neutral). Positive rent = the node pays for
  itself. Note this is a *marginal / leave-one-out* attribution with all other
  nodes present, so per-node rents do **not** sum to the total saving — the
  hierarchy's value is partly joint (a mid concept is cheap only because its base
  children exist). Empirically the base concepts carry most of the rent (dark)
  while the top concepts earn less at the margin (pale) even when heavily used.

Run:

```sh
PYTHONPATH=src python3 demos/visualize.py
```

Needs `matplotlib`. Edit `build_scenarios()` to add cases or resize existing
ones; keep them smallish so the DAG stays legible.
