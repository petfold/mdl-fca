"""DAG of concept and attribute nodes with cached closures.

Node ids: attributes are ``0..n_attrs-1`` (observed sinks); concepts are
assigned ids ``n_attrs, n_attrs+1, ...`` in creation order. Edges point from a
concept to its children (concepts or attributes).

Acyclicity is structural in this prototype: a concept's children must already
exist when the concept is created, and rewiring on ``remove_concept`` only adds
edges from a parent (created later) to the removed concept's children (created
earlier). Every edge therefore points from a later-created node to an
earlier-created one, so creation order is an incremental topological order.
``add_edge`` performs an explicit reachability check anyway so that future
proposal types (docs/03) stay safe.

Closures are cached as attribute bitmasks; on a structure edit only the
ancestors of touched nodes are invalidated.
"""

from __future__ import annotations


class DAG:
    def __init__(self, n_attrs: int):
        self.n_attrs = n_attrs
        self.children: dict[int, set[int]] = {}  # concept -> child node ids
        self.parents: dict[int, set[int]] = {}   # node -> concepts pointing at it
        self._closure_mask: dict[int, int] = {}  # concept -> attr bitmask
        self._next_id = n_attrs

    # ------------------------------------------------------------- queries
    def is_attribute(self, k: int) -> bool:
        return 0 <= k < self.n_attrs

    @property
    def concepts(self):
        return self.children.keys()

    @property
    def num_concepts(self) -> int:
        return len(self.children)

    @property
    def num_edges(self) -> int:
        return sum(len(kids) for kids in self.children.values())

    def items(self) -> list[int]:
        """All dictionary items: every attribute (bare activation is always
        allowed) plus every concept."""
        return list(range(self.n_attrs)) + list(self.children)

    def closure_mask(self, k: int) -> int:
        """Bitmask of attributes reachable from node k (k itself if attribute)."""
        if self.is_attribute(k):
            return 1 << k
        m = self._closure_mask.get(k)
        if m is None:
            m = 0
            for ch in self.children[k]:
                m |= self.closure_mask(ch)
            self._closure_mask[k] = m
        return m

    def closure(self, k: int) -> frozenset[int]:
        m = self.closure_mask(k)
        out = set()
        while m:
            low = m & -m
            out.add(low.bit_length() - 1)
            m ^= low
        return frozenset(out)

    def reachable(self, u: int, v: int) -> bool:
        """True iff v is reachable from u by one or more edges (strict)."""
        if u == v or self.is_attribute(u):
            return False
        if self.is_attribute(v):
            return bool((self.closure_mask(u) >> v) & 1)
        stack, seen = [u], set()
        while stack:
            k = stack.pop()
            for ch in self.children[k]:
                if ch == v:
                    return True
                if not self.is_attribute(ch) and ch not in seen:
                    seen.add(ch)
                    stack.append(ch)
        return False

    def ancestors(self, k: int) -> set[int]:
        out: set[int] = set()
        stack = list(self.parents.get(k, ()))
        while stack:
            p = stack.pop()
            if p not in out:
                out.add(p)
                stack.extend(self.parents.get(p, ()))
        return out

    # --------------------------------------------------------------- edits
    def add_concept(self, children: set[int]) -> int:
        kids = set(children)
        if not kids:
            raise ValueError("a concept needs at least one child")
        for ch in kids:
            if not (self.is_attribute(ch) or ch in self.children):
                raise ValueError(f"unknown child node {ch}")
        c = self._next_id
        self._next_id += 1
        self.children[c] = kids
        for ch in kids:
            self.parents.setdefault(ch, set()).add(c)
        return c

    def add_edge(self, parent: int, child: int) -> None:
        if parent == child or self.reachable(child, parent):
            raise ValueError(f"edge {parent}->{child} would create a cycle")
        self.children[parent].add(child)
        self.parents.setdefault(child, set()).add(parent)
        self._invalidate(parent)

    def remove_concept(self, c: int, rewire: bool = True):
        """Delete concept c. With rewire=True every parent of c inherits c's
        children, so the closure of every surviving node is unchanged."""
        kids = self.children.pop(c)
        for ch in kids:
            self.parents[ch].discard(c)
        pars = self.parents.pop(c, set())
        for p in pars:
            self.children[p].discard(c)
            if rewire:
                for ch in kids:
                    if ch not in self.children[p]:
                        self.children[p].add(ch)
                        self.parents.setdefault(ch, set()).add(p)
        self._closure_mask.pop(c, None)
        if not rewire:
            for p in pars:
                self._invalidate(p)
        return kids, pars

    def _invalidate(self, k: int) -> None:
        self._closure_mask.pop(k, None)
        for a in self.ancestors(k):
            self._closure_mask.pop(a, None)
