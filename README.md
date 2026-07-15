# mdl-fca

Learning "good" concept DAGs from binary data by minimum description length —
a probabilistic/information-theoretic reworking of Formal Concept Analysis.

Plain FCA maps a binary object×attribute context to a concept lattice, but it
is not a probabilistic model: it creates a concept for every random coincidence,
so the lattice explodes. This project builds instead the small DAG of concepts
that best **compresses** the data: a concept exists only if it pays for its own
description. Concepts can have multiple parents (unlike hierarchical
clustering's trees) and are organized hierarchically (unlike flat topic models).

Status: design phase complete, prototype under construction.

- Read `CLAUDE.md` for orientation and design commitments.
- Read `docs/` in numeric order for the full design:
  1. background and goal
  2. model and codelength (the core)
  3. algorithm (greedy pair-merge constructor + search framework)
  4. batch vs online
  5. prototype spec (current build target)
  6. roadmap

## Quick start (once the prototype exists)

```bash
pip install -e ".[dev]"
pytest
python examples/run_planted.py
```
