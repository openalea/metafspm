"""
The junctions between the anatomies of linked vertices (MPG.wire_junctions), wired for every pair at once.

A wiring rule is a declared dict, evaluated once over all the (linked parent, child) pairs being wired
(SectionPairs), not once per pair:

  {"node_label": L, "edge_label": E, "ordering": prop, "match": "nearest" | "equal" | "all"}
      the platform's rule: pairs the Compartments labelled L of both sides ("all": every one with every one, the
      default without ordering; "nearest": the closest values of *ordering*, greedily, the default with it;
      "equal": equal values of *ordering*);
  {"pairs": f, "edge_label": E}
      f(g, section_pairs) -> (pair, tails, heads): integer arrays, one entry per junction: the index of its pair in
      section_pairs, its Compartment on the parent's side (n_id_a) and on the child's (n_id_b).

Either may give "properties": p(g, section_pairs, pair, tails, heads) -> {name: one value per junction}, written on
the junctions it creates (their geometry, say).
"""
import numpy as np


class SectionPairs:
    """
    The (linked parent, child) pairs of vertices a wiring joins, and the Compartments of each one:

      parents, children   one entry per pair (children ascending);
      sections            the vertices of the pairs owning Compartments, ascending;
      indptr, compartments  their Compartments, compartments[indptr[k]:indptr[k + 1]] for sections[k]: by label in
                          order of first appearance, ascending vids within each label;
      parent_at, child_at   each pair's sides as positions in sections (-1: the vertex owns no Compartment);
      labels              each Compartment's label (None: none).
    """

    def __init__(self, parents, children, sections, indptr, compartments, labels):
        self.parents = np.asarray(parents, dtype=np.int64)
        self.children = np.asarray(children, dtype=np.int64)
        self.sections = np.asarray(sections, dtype=np.int64)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.compartments = np.asarray(compartments, dtype=np.int64)
        self.labels = labels
        self.parent_at = self._positions(self.parents)
        self.child_at = self._positions(self.children)

    @classmethod
    def build(cls, g, from_scale, parents, children) -> "SectionPairs":
        """The pairs (*parents*, *children*: vertices at *from_scale*) of MPG *g*, with the Compartments the vertices
        own (MPG.compartment_owners) read in bulk: one label read, no dict per owner."""
        parents, children = np.asarray(parents, dtype=np.int64), np.asarray(children, dtype=np.int64)
        vids, owners = g.compartment_owners()
        owners = g.owners_at_scale(owners, from_scale)
        involved = np.union1d(parents, children)
        kept = (owners >= 0) & np.isin(owners, involved)
        vids, owners = vids[kept], owners[kept]
        order = np.argsort(owners, kind="stable")             # by owner, ascending vids within each
        vids, owners = vids[order], owners[order]
        found, values = g._property_at("label", np.sort(vids))
        labels = np.full(vids.size, None, dtype=object)
        if found.size:
            labels[_positions_in(vids, found)] = values.tolist()
        codes = _codes(labels)
        position = np.arange(vids.size, dtype=np.int64)
        # each label's first position within its owner: the groups' order (first appearance)
        grouped = np.lexsort((position, codes, owners))
        start = np.ones(vids.size, dtype=bool)
        if vids.size:
            start[1:] = (owners[grouped][1:] != owners[grouped][:-1]) | (codes[grouped][1:] != codes[grouped][:-1])
        first = np.empty(vids.size, dtype=np.int64)
        first[grouped] = position[grouped][np.flatnonzero(start)[np.cumsum(start) - 1]]
        order = np.lexsort((position, first, owners))
        vids, owners, labels = vids[order], owners[order], labels[order]
        sections, counts = np.unique(owners, return_counts=True)
        indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        return cls(parents, children, sections, indptr, vids, labels)

    def __len__(self) -> int:
        return self.parents.size

    def _positions(self, vertices) -> np.ndarray:
        at = np.searchsorted(self.sections, vertices).clip(0, max(self.sections.size - 1, 0))
        held = (self.sections[at] == vertices) if self.sections.size else np.zeros(vertices.size, dtype=bool)
        return np.where(held, at, -1)

    def compartments_at(self, position: int) -> np.ndarray:
        """The Compartments of sections[*position*] (none for -1)."""
        if position < 0:
            return self.compartments[:0]
        return self.compartments[self.indptr[position]:self.indptr[position + 1]]

    def parent_compartments(self, pair: int) -> np.ndarray:
        """The Compartments of pair *pair*'s parent."""
        return self.compartments_at(int(self.parent_at[pair]))

    def child_compartments(self, pair: int) -> np.ndarray:
        """The Compartments of pair *pair*'s child."""
        return self.compartments_at(int(self.child_at[pair]))

    def compartments_of(self, vertex: int) -> np.ndarray:
        """The Compartments of *vertex* (one of the pairs' vertices; none when it owns none)."""
        return self.compartments_at(int(self._positions(np.array([vertex], dtype=np.int64))[0]))


def _positions_in(vids, found) -> np.ndarray:
    """The positions in *vids* (unsorted, unique) of *found* (sorted, among them)."""
    order = np.argsort(vids, kind="stable")
    return order[np.searchsorted(vids[order], found)]


def _codes(values) -> np.ndarray:
    """An integer code per value (an object array; None included), equal values sharing one."""
    table = {}
    return np.fromiter((table.setdefault(value, len(table)) for value in values.tolist()), dtype=np.int64,
                       count=values.size)


def junctions(g, rules, section_pairs) -> tuple:
    """
    Every rule evaluated over *section_pairs*: (tails, heads, edge labels, {name: (positions, values)}), the
    junctions in the order of their pairs, then of the rules, then as each rule gives them; each property's values
    at the positions of the junctions giving it.
    """
    tails, heads, pair_of, rule_of, within, given = [], [], [], [], [], []
    for index, rule in enumerate(rules):
        if "rule" in rule:
            raise ValueError("a wiring rule's callable is declared as {'pairs': f, ...}, f(g, section_pairs) -> "
                             "(pair, tails, heads), evaluated over every pair at once (MPG.wire_junctions)")
        if "pairs" in rule:
            pair, tail, head = rule["pairs"](g, section_pairs)
        else:
            pair, tail, head = label_pairs(g, rule, section_pairs)
        pair, tail, head = (np.asarray(a, dtype=np.int64).reshape(-1) for a in (pair, tail, head))
        if not pair.size == tail.size == head.size:
            raise ValueError(f"wiring rule {index}: pair, tails and heads differ in length")
        tails.append(tail)
        heads.append(head)
        pair_of.append(pair)
        rule_of.append(np.full(pair.size, index, dtype=np.int64))
        within.append(np.arange(pair.size, dtype=np.int64))
        given.append((rule, pair, tail, head))
    if not rules:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, [], {}
    pair_of, rule_of, within = np.concatenate(pair_of), np.concatenate(rule_of), np.concatenate(within)
    order = np.lexsort((within, rule_of, pair_of))
    position = np.empty(order.size, dtype=np.int64)
    position[order] = np.arange(order.size, dtype=np.int64)
    edge_labels = [rule["edge_label"] for rule in rules]
    labels = [edge_labels[k] for k in rule_of[order].tolist()]
    properties = {}
    offset = 0
    for rule, pair, tail, head in given:
        at = position[offset:offset + pair.size]
        offset += pair.size
        if "properties" not in rule or not pair.size:
            continue
        for name, values in rule["properties"](g, section_pairs, pair, tail, head).items():
            values = np.asarray(values)
            if values.shape[:1] != (pair.size,):
                raise ValueError(f"wiring rule properties '{name}': {len(values)} values for {pair.size} junctions")
            properties.setdefault(name, []).append((at, values))
    merged = {}
    for name, chunks in properties.items():
        at = np.concatenate([chunk for chunk, _ in chunks])
        values = np.concatenate([chunk for _, chunk in chunks])
        rank = np.argsort(at, kind="stable")
        merged[name] = (at[rank], values[rank])
    return np.concatenate(tails)[order], np.concatenate(heads)[order], labels, merged


def label_pairs(g, rule, section_pairs) -> tuple:
    """The platform's rule {"node_label", "ordering", "match"} over every pair: (pair, tails, heads)."""
    label = rule["node_label"]
    ordering = rule.get("ordering")
    match = rule.get("match", "nearest" if ordering else "all")
    if match not in ("all", "nearest", "equal"):
        raise ValueError(f"junction rule {rule}: match must be 'all', 'nearest' or 'equal'")
    if match != "all" and ordering is None:
        raise ValueError(f"junction rule {rule}: match '{match}' needs an ordering property")
    labelled = np.array([value == label for value in section_pairs.labels.tolist()], dtype=bool) \
        if section_pairs.labels.size else np.zeros(0, dtype=bool)
    value_of = None
    if match != "all":
        compartments = section_pairs.compartments
        unique = np.unique(compartments[labelled])
        found, values = g._property_at(ordering, unique)
        value_of = dict(zip(found.tolist(), np.asarray(values, dtype=np.float64).tolist()))
    pairs, tails, heads = [], [], []
    for index in range(len(section_pairs)):
        a_at, b_at = int(section_pairs.parent_at[index]), int(section_pairs.child_at[index])
        if a_at < 0 or b_at < 0:
            continue
        n_as = _labelled(section_pairs, a_at, labelled)
        n_bs = _labelled(section_pairs, b_at, labelled)
        if not n_as or not n_bs:
            continue
        if match == "all":
            found = [(n_a, n_b) for n_a in n_as for n_b in n_bs]
        else:
            found = _matched(n_as, n_bs, lambda nv: value_of.get(nv, 0.0), match)
        for n_a, n_b in found:
            pairs.append(index)
            tails.append(n_a)
            heads.append(n_b)
    return np.asarray(pairs, dtype=np.int64), np.asarray(tails, dtype=np.int64), np.asarray(heads, dtype=np.int64)


def _labelled(section_pairs, position, labelled) -> list:
    start, stop = section_pairs.indptr[position], section_pairs.indptr[position + 1]
    return section_pairs.compartments[start:stop][labelled[start:stop]].tolist()


def _matched(n_as, n_bs, value, match) -> list:
    """The greedy matching of "nearest" and "equal": each parent-side Compartment, by increasing value, takes the
    unused child-side one nearest in value (of equal value for "equal")."""
    sorted_a, sorted_b = sorted(n_as, key=value), sorted(n_bs, key=value)
    used_b, pairs = set(), []
    for n_a in sorted_a:
        candidates = [n_b for n_b in sorted_b if n_b not in used_b]
        if match == "equal":
            candidates = [n_b for n_b in candidates if value(n_b) == value(n_a)]
        if candidates:
            best_b = min(candidates, key=lambda n_b: abs(value(n_a) - value(n_b)))
            used_b.add(best_b)
            pairs.append((n_a, best_b))
    return pairs
