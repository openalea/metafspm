# Design note: structure, growth, active sets and boundaries (step 2)

Status: **agreed** (2026-10-02, P1–P7 answered in §8). 2a done (`4fbe3dc`), 2b done (`e659bd0`), 2c done (`6430b08`), 2d done (`9e0b00b`), 2e done. It covers step 2 of `devplan_datastructures.md` §7:
- DS2: traversal orders;
- DS19: the StructuralComponent contract;
- DS20: repartition and the active mask;
- DS21: the active subgraph;
- DS6: boundary sets, with UC5;
- DS8: multiscale graph assembly.

Branch `data_structure_api`, written against `a987cba` (step 1 complete). Code starts only after this note is agreed. The points to agree are in §8.

It builds on decisions D11 (the MPG is the source of truth), D12 (conditional rewiring), D13 (sync around structural steps), D14 (repartition), D15 (active mask) and D16 (active subgraph opt-in), and on the step 1 contract (`datastructure_contract.md`, `docs/conventions.md`).

## 1. Where we start from (code facts)

- **Scheduling is per component.** The Choregrapher registers steps by class name, and `Component.__call__` runs **all** of that component's steps, in the order of its rows. For growth these are `potential, deficit, allocation, actual, segmentation, postsegmentation` (`choregrapher.py:30`). Components are interleaved only by the order in which the composite calls them.
  - **Consequence:** the reference composite (`provide_usage_examples/composite_wrapper_example.py:171`) calls `root_growth(...)` first. Its potential, actual and segmentation steps all run in that call, reading the carbon state of the previous step, and the carbon and water models run after.
  - So **"structural potential → functional allocation → structural actual" across components does not exist today, and is not wanted (P2).** The potential / allocation / actual rows order the processes **within** one component. Each component is called as a whole, in the composite's order.
- **`StructuralComponent`** is an empty class (`component.py:213`): no DataStructure, no registration, no sync.
- **Steps without arguments** are already called with the instance only (`Functor`: no inputs → `iterating`). This is the "MPG-style" step of Q17.
- **Topology change.**
  - `MPGDataStructure.update_topology()` calls `MPG.repopulate_graph(from_scale)`, which deletes every Compartment and Connection and recreates them. It then rebuilds the index maps and carries registered variables over by vid, with `on_grow` default or inherit from the nearest known ancestor (`_carry_over`).
  - The MPG has **no modification counter**. A topology change is known only because the growth model calls `update_topology()` by hand.
- **Traversal.**
  - The node order is the Compartment post-order produced by population.
  - `MPG.pre_order_mpg` / `post_order_mpg` exist, but act on vertices, not on DataStructure indices.
  - Parents and children in local indices are not exposed. Tests use the private `_idx_to_vid` / `_vid_to_idx`, and `index_of` (D8) does not exist.
- **Boundaries.**
  - `@boundary_condition(location="node", kind="dirichlet" | "neumann", filters=)` overrides or adds node residual rows on the filtered nodes (`decorator.py:~520`).
  - Robin conditions use `BoundaryPort(name, node_id, kind, value, weight, orientation)`. These are frozen dataclasses set by hand in `component._boundary_ports`, turned into `GraphView.boundary_incidence`. The model then assembles `B_b·diag(w)·(B_bᵀx − v)` itself, in its balance **and** its Jacobian (`test_uc3_uc4_anatomy_bc.py:134`).
  - Port values are frozen, keyed by vid, and do not follow growth.
- **Residual convention** (UC1, UC3): `storage + B·outflux − sources = 0`. A Robin outflow `w·(x − v)` therefore enters with a `+` sign.
- **Anatomies.**
  - `MPG.populate_graph_custom_connections` (`mpg.py:273`) wires Connection vertices between same-label Compartments of adjacent SubOrgans: all-to-all, or nearest-neighbour on an `ordering` property.
  - `MPGDataStructure` cannot wrap the result: nodes are keyed on `vertex_id` (one Compartment per SubOrgan), and edges join SubOrgan vids.

## 2. Traversal on the DataStructure (DS2), sub-step 2a

All in **local indices**, derived from the Connections (not from the population order), and cached per `topology_version`:

| Method | Returns |
|---|---|
| `ds.parents()` | int array (n,), the parent's index, −1 at a root |
| `ds.children()` | CSR `(indptr, indices)` |
| `ds.roots()`, `ds.tips()` | int arrays |
| `ds.order("pre" \| "post")` | a permutation of `range(n)` (parents before children, or the reverse) |
| `ds.index_of(ids, location="node")` | local indices of entity ids, raising on unknown ids (closes the D8 gap) |
| `ds.owner(location)` | int array (n,): the index of each node's entity at a coarse location (already computed by `_membership`, now public) |

- Tests and components stop using `_idx_to_vid` / `_vid_to_idx`.
- The grid has no parents. Its traversal methods raise `NotImplementedError`.

## 3. StructuralComponent (DS19), sub-step 2b

```python
@dataclass
class RootGrowth(StructuralComponent):
    C_hexose_root: float = input_variable(..., scale=scales.SubOrgan)
    length:        float = state_variable(..., scale=scales.SubOrgan, state_variable_type="descriptor")
    struct_mass:   float = state_variable(..., scale=scales.SubOrgan, state_variable_type="extensive")

    @potential
    def _potential_growth(self):            # MPG-style: no arguments, edits self.mtg
        ...
    @actual
    def _thickening(self, radius, C_hexose_root):   # array-style: vectorised, outputs to the DataStructure
        ...
    @segmentation
    def _segmentation(self):
        ...
```

- **Construction:** `StructuralComponent(data_structure=ds)`, with the same declaration resolution and registration as FunctionalComponent (step 1). `self.mtg` is `ds.mtg`. The DataStructure does not re-expose MPG edit methods (D11).
- **Scheduling:** its steps are registered and called by the Choregrapher like FunctionalComponent steps. They run within the component's own call, in the growth rows (§1). The `postsegmentation` row replaces `post_growth_updating` in `__call__`.
- **Sync, per step (P1):**

  | Before an MPG-style step | After it |
  |---|---|
  | write to the MPG the component's declared variables, inputs and outputs, from the DataStructure (`write_mtg` of step 1b, used for every variable type, not only state variables) | if the topology changed: `ds.update_topology()` and the repartition (DS20); then re-read the declared outputs from the MPG (`read_mtg`) |

  - Array-style steps need no sync: they read and write the DataStructure. Their outputs reach the MPG at the next MPG-style step's flush, or at the write-back that ends every call (N4).
- **Topology change detection (P3):** a cheap **signature** of the MPG, `(number of vertices at from_scale, maximum vid, number of Compartments, number of Connections)`, compared before and after each MPG-style step. It is computed from the `scale` property array, with no traversal. There is no MPG override, so it also works with the node API (`node.add_child`).
- **Validates on (`test/structure_tests/`):** a rhizodep-like growth helper in this repo, with:
  - potential elongation and actual elongation: MPG-style, reading `C_hexose_root`;
  - thickening: array-style;
  - segmentation: MPG-style, adding segments;
  - post-segmentation: distance from tip.

  A FunctionalComponent runs after it in the same step. The test checks:
  - inputs flushed before each MPG-style step;
  - outputs re-read after;
  - `update_topology` called only by segmentation;
  - the functional component's graph view rebuilt.

## 4. Repartition and the active mask (DS20, D14, D15), sub-step 2c

**Lineage.** At each `update_topology`, every new entity gets an **origin**: its nearest pre-existing ancestor. A chain of new segments created by one segmentation is processed from the base to the tip (DS2 pre-order), so pairwise splits compose.

**Rules** (the `post_growth_updating` table, `devplan_datastructures.md` DS20), applied to **every registered variable** not written by the structural step, by its recorded `kind`. Here `f = w_v / (w_v + w_p)`, from the declared partition weight:

| kind | new and active (weight > 0) | new and inactive (e.g. primordium) | existing, weight grew |
|---|---|---|---|
| massic_concentration | split the amount `c_p·w_p⁰` | copy from the parent | dilute: `c·w⁰/w` |
| extensive | split: `f`, `1 − f` | 0 | unchanged |
| NonInertialIntensive, intensive | copy | copy | unchanged |
| NonInertialExtensive | `f·x_p`, the parent unchanged | 0 | unchanged |
| descriptor, or no kind | `on_grow` (default or inherit) | `on_grow` | unchanged |

- An inactive entity that becomes active, e.g. a primordium's first elongation, is split from its parent like a new active entity.
- **Declaration**, class attributes of the StructuralComponent (Q18, a), each accepting a callable `f(ds) -> array`:

  ```python
  partition_weight = "living_struct_mass"
  previous_weight  = ("initial_struct_mass", "initial_living_root_hairs_struct_mass")   # summed
  active           = {"struct_mass": ">0", "label": [1, 2], "type": [1, 7, 8, 9, 10, 11, 12]}
  ```

  Without `partition_weight`, there is no repartition: `on_grow` only, as today.
- **As implemented (2c): no `previous_weight` to declare.** The framework records the partition weight just before each MPG-style step and uses it as the concentrations' reference weight:
  - dilution after a step is `c · w_before / w_after`, and successive steps compose: `w0/w1 · w1/w2 = w0/w2`;
  - a split uses the parent's amount `c_p · w_before`.

  This gives the same result as rhizodep's `initial_struct_mass` bookkeeping, without the component maintaining it.
- **The active mask is a DataStructure object.**
  - `ds.define_mask("active", rule, location="node")` stores the rule. `ds.mask("active")` returns a boolean array, recomputed when one of the rule's variables was written (the DS17 stamps), and `ds.mask_version("active")` changes when its values change.
  - The structural component's `active` defines it.
  - **Vectorised steps** (functional and array-style structural) compute on the active entities by default: their node-located arguments are sliced, and their node outputs are scattered back, with inactive values unchanged. `@rate(where=None)` opts out, and `where="name"` chooses another mask. Scalar and coarse arguments are passed whole.
  - MPG-style steps always see everything.
- **Validates on** the 2b helper, on a small tree. The test checks:
  - the extensive amounts and `c·w` amounts conserved over the origin and its new segments;
  - a primordium copying concentrations, with zero amounts;
  - its first elongation splitting it from its parent;
  - elongation diluting concentrations;
  - the same numbers as rhizodep's `post_growth_updating` on that tree, computed by hand in the test;
  - a masked `@rate` leaving inactive nodes unchanged.

## 5. Graph systems on the active subgraph (DS21, D16), sub-step 2d

`@graph_system(where="active")`, opt-in. The views are cached by `(topology_version, mask_version)`.

- **Subgraph:**
  - nodes: the active ones;
  - edges: those with both ends active;
  - boundary conditions and boundary sets: restricted to active nodes.

  Gather and scatter are vectorised index arrays, and writes go in place into the full arrays.
- **Inactive entities:** node unknowns are frozen; dropped edges have flux 0.
- **Activation:** a newly active node enters with its current value (repartition, or `on_grow`), and `previous()` gives that value.
- **Well-posedness, at each rebuild:** every connected piece of the subgraph must either have a time derivative in its balance, or be anchored by a Dirichlet condition or a Robin boundary with a positive weight. Otherwise the solve raises, naming its entity ids. This uses `scipy.sparse.csgraph.connected_components` on the subgraph, and is explained in §8, P4.
- **Empty subgraph:** skipped.
- **Validates on (UC1):**
  - all active → bit-for-bit equal to the whole-graph solve;
  - a node activated between two steps, with mass conserved;
  - a dead segment with its edges at zero flux;
  - an isolated algebraic node raising with its id;
  - an empty mask skipped;
  - a mask changed through its source variable rebuilding the view.

## 6. Boundary sets (DS6) and UC5, sub-step 2e

```python
@graph_system(node_unknowns=["water_potential"], ...)
class _water:
    leaves = boundary_set(select={"label": "LeafElement"}, kind="robin",
                          value="air_water_potential", weight="leaf_conductance")
    roots  = boundary_set(select="root_surface", kind="robin",
                          value="soil_water_potential", weight="radial_conductance")
    collar = boundary_set(select=lambda ds: ds.roots(), kind="dirichlet", value=-0.1)
```

- **`select`** can be:
  - a `{variable: value or values}` filter;
  - a variable name (selects where it is > 0);
  - a callable `ds -> mask or indices`.

  Membership is re-evaluated when its variables were written, and after topology changes (P5).
- **`value` and `weight`** are DataStructure variable names (node-located) or constants, read at each solve.
- **Assembly by the framework,** for the declared field's balance:
  - **robin:** `+ B_b·diag(w)·(B_bᵀx − v)` in the residual, and `+ B_b·diag(w)·B_bᵀ` added to a user `@graph_jacobian` if there is one. With finite differences, nothing is needed.
  - **dirichlet / neumann:** the existing row replacement / addition of `@boundary_condition`.
- **`BoundaryPort`** stays the internal representation, rebuilt from the sets at each view rebuild. Setting `_boundary_ports` by hand is deprecated with a warning. `@boundary_condition` stays for conditions computed by a method.
- **UC5, leaf transpiration** (Q4, a tool use case, not a validated model): steady water potential on the seedling's shoot + root graph.
  - axial conductance on edges;
  - leaves: Robin to a per-leaf air water potential. The microclimate input is a node variable set per leaf.
  - `leaf_conductance = stomatal_conductance × exchange_surface`: a `@rate` of the same component, with `stomatal_conductance` an input and `exchange_surface` a structural variable;
  - roots: Robin to the soil water potential.
  - The test compares with the direct linear solve of `(B·diag(K)·Bᵀ + B_b·diag(w)·B_bᵀ)·ψ = B_b·(w·v)`. It also checks that changing one leaf's microclimate changes its boundary flux on the next solve without rebuilding anything, and that a grown leaf joins the set.

## 7. Multiscale graph assembly (DS8, D11, D12), sub-step 2f

- **Anatomy mode:** `MPGDataStructure(g, from_scale=SubOrgan, nodes="Compartment")`. The current mode (one Compartment per SubOrgan) stays the default.
  - **Nodes** are the non-anchor Compartment vertices, keyed by **their own vid**.
  - **The owner** of a node is its MTG parent (the SubOrgan), then `complex_at_scale` upwards. `"SubOrgan"` becomes a coarse location, as N5 intended, and step 1's mappings work on the owner map unchanged.
  - **Edges** are the Connection vertices with Compartment endpoints, **keyed by their own vid** (P6), since an endpoint pair is not unique across kinds.
- **Wiring rules (MPG methods):** `populate_graph_custom_connections` gains:
  - `match="all" | "nearest" | "equal"` on a property;
  - a callable rule, `rule(g, suborgan_a, suborgan_b, comps_a, comps_b) -> pairs`.

  Each generated Connection gets `edge_kind="junction"`. Anatomy Connections, created by the anatomy generator, carry `edge_kind="anatomy"`. The DataStructure stores the rules, so that it can re-apply them.
- **Growth in anatomy mode (D12):** `update_topology()` does **not** call `repopulate_graph`. It:
  1. computes per SubOrgan its parent, its children and a hash of its Compartment ids and rule-property values, and compares them with the previous extraction;
  2. deletes the junction Connections of the SubOrgans whose neighbourhood or anatomy changed, and wires new ones for them and for new SubOrgans. Every other Connection, and its vid, is kept;
  3. extracts the graph again, and carries edge variables over by Connection vid (new junctions take `on_grow`);
  4. runs the repartition (DS20), with lineage on the owner SubOrgans.
- **Test helper:** `test/structure_tests/anatomy.py` builds a small synthetic cross-section per SubOrgan: Compartments labelled epidermis, cortex, xylem (with `vessel_index`), and anatomy Connections.
- **Validates on (UC3 on that anatomy):**
  - xylem junctions matched by vessel index, equal to `populate_graph_custom_connections`'s pairs;
  - a growth step adding a segment, keeping every old Connection vid and its edge values;
  - a differentiation (a label change) re-wiring only that SubOrgan's junctions;
  - a radial uptake summed to SubOrgan;
  - a SubOrgan concentration broadcast to the symplastic Compartments through a target filter (DS18; the filter is the one new piece needed here).

## 8. Points to agree

- **P1, sync granularity.** The flush before and the re-read after each **MPG-style** structural step, as in §3. The alternative is once around the whole component call. Per step is needed because array-style and MPG-style steps of one component alternate. **Recommendation:** per MPG-style step.
  → answer: yes to recommandation
- **P2, interleaving across components.** Keep today's behaviour: the growth component's call runs all its rows, using the other components' states of the previous step. The "potential → allocation → actual" interleaving across components needs a scene-level schedule, which belongs with per-instance scheduling (DS13, step 5). **Recommendation:** defer it to DS13, and say so in DS19.
  → answer: No keep as today, decorators with potential allocation and actual are for within component scheduling
- **P3, topology change detection** by an MPG signature (counts and maximum vid), not by an MPG modification counter, which would need overriding every MTG edit method. **Recommendation:** the signature.
  → answer: yes follow a signature
- **P4, well-posedness** needs to know whether a graph system has a time derivative: a `transient=` flag on `@graph_system`, defaulting from the method (Euler methods transient, steady Newton not). **Recommendation:** yes.
  → answer: reexplain more clearly
  - **The problem.** With `where="active"`, the solve runs only on the active nodes, and these can fall apart into **disconnected pieces**. For example:
    - a dead segment in the middle of an axis cuts it in two;
    - an active primordium whose carrying segment is inactive is a piece of one node.

    Each piece is solved as part of one system. If one piece has no unique solution, the whole matrix is singular, and the solver fails with an obscure error (singular Jacobian, or no convergence), far from the cause.
  - **When does a piece have no unique solution?** It depends on the balance equation:
    - **With a time derivative** (a storage term, e.g. `(c − c_old)/dt` in UC1): every node's equation contains its own value with a non-zero coefficient. Any piece, even an isolated node, has a unique solution, because it simply evolves from its previous value.
    - **Without a time derivative** (a steady balance, e.g. UC5's water potential, where the fluxes in and out of each node sum to zero): the equations only involve **differences** between neighbours, e.g. `K·(ψ_parent − ψ_child)`. Adding the same constant to every ψ of a piece leaves its equations satisfied. So the piece needs an **anchor**: a Dirichlet node (ψ fixed), or a Robin boundary with a positive weight (`w·(ψ − ψ_air)`). An unanchored piece, e.g. a root branch cut off from the collar with no soil contact, is singular.
  - **What the framework cannot see.** The balance is a Python function, so the framework cannot tell whether it contains a time derivative. It can see the anchors: the Dirichlet conditions and Robin boundary sets are declared (§6).
  - **The proposal.** The graph system says whether its balance is transient, `@graph_system(..., transient=True | False)`. At each rebuild of the subgraph, the framework checks every connected piece:
    - transient: always fine;
    - steady: fine only if the piece holds at least one anchor, otherwise it raises, e.g. "piece of 3 nodes [vids 31, 32, 33] has no Dirichlet or Robin anchor in steady system `_water`".
  - **Default.** `transient` defaults from the solver method:
    - `True` for the time-stepping methods (implicit and explicit Euler, IVP);
    - `False` for the steady solvers (Newton, scipy root), where it may be declared `True` when the balance has its own `(x − previous)/dt` term, as UC1's Newton solves do.

    With a wrong `True`, the check is skipped and the old obscure failure can come back. With a wrong `False`, a valid isolated piece is rejected, and the error says to declare `transient=True`.
  - **Scope.** The check runs only for `where="active"` subgraphs, where pieces appear when the mask changes. A whole graph is connected and is the modeller's responsibility, as today.

  → answer (after the re-explanation): Ok, so most of the time models won't have graph discontinuity when selecting the active set, but yes the proposition seems sound to have a check for this, I accept the recommandation.
- **P5, boundary-set membership** re-evaluated when its select variables were written, using the write counters of step 1c, and after topology changes. **Recommendation:** yes.
  → answer: yes
- **P6, edge ids:**
  - in anatomy mode, edges are keyed by their Connection vid, which is stable because junctions are rewired incrementally;
  - in segment mode they stay keyed by the child vid, because `repopulate_graph` recreates the Connection vids at every growth.

  **Recommendation:** yes. `to_dataframe(location="edge")` documents which key it uses.
  → answer: yes
- **P7, order of sub-steps:** 2a DS2 → 2b DS19 → 2c DS20 → 2d DS21 → 2e DS6 + UC5 → 2f DS8. Each sub-step lands with its tests and a commit, and UC1–UC4 and the contract tests stay unchanged. **Recommendation:** yes. DS8 comes last, because it needs DS2, DS19 and DS20.
  → answer: yes
