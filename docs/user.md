# User guide

MetaFSPM gives FSPM components one way to declare their variables, describe their processes, and couple with other
components, whatever they run on (one plant, a population, a soil grid). This guide follows the order in which a
model is usually written: the data, a component, its equations, its structure, its couplings, and the scene that runs
it. The declaration rules are summarised in [Conventions](conventions.md); porting an existing model is described in
[Migrating a model](migration.md).

## 1. DataStructures

A component never owns its data: it reads and writes the variables of a **DataStructure**, by name. Each variable has
a **location**, the entities it has one value per:

| DataStructure | Locations | Used for |
|---|---|---|
| `MPGDataStructure` | `"node"`, `"edge"`, coarse scales (`"Plant"`, `"Axis"`, `"Organ"`, …), `"scalar"` | plants, or a population of plants in one MPG |
| `ArrayDataStructure` | `"cell"`, `"edge"` (faces), `"scalar"` | regular grids (soil, atmosphere) |
| `AdaptiveGridDataStructure` | `"cell"`, `"edge"`, `"scalar"` | octree grids refined where processes are intense |
| `UnionDataStructure` | `"node"`, `"scalar"` | the nodes of several populations, for one component that must see them all (light) |

### Plants: `MPGDataStructure`

An MPG is an MTG with a solver graph: `populate_graph(scale)` creates one Compartment per vertex of that scale (the
graph's nodes) and one Connection per link (its edges).

```python
from openalea.metafspm.data_structure.data_api import MPGDataStructure

g.populate_graph(g.scales.SubOrgan)
g.convert_properties_to_arraydict()
ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)          # segments as nodes
ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=rules)   # anatomies as nodes
```

- **Segment mode** (default): one node per segment. **Anatomy mode**: the Compartments inside each segment are the
  nodes, wired to their neighbours' by junction rules.
- After the MTG's structure changes (growth, pruning), `ds.update_topology()` extends the graph incrementally and
  carries every variable over (by id; new entities get the default or their parent's value, `on_grow="inherit"`).
- Each node belongs to coarser entities: `ds.owner("Plant")`, `ds.entity_ids("Plant")`, so the variables of a whole
  population are arrays, and per-plant values are coarse-scale variables.

### Grids: `ArrayDataStructure` and `AdaptiveGridDataStructure`

```python
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.data_structure.adaptive_grid import AdaptiveGridDataStructure

soil = ArrayDataStructure(shape=(20, 20, 50), dx=(0.01, 0.01, 0.02), periodic=(True, True, False))
adaptive = AdaptiveGridDataStructure(shape=(10, 10, 25), dx=0.02, max_level=2, periodic=(True, True, False))
adaptive.refine(lambda grid: grid.get("root_length_density") > 1e3)      # between steps
adaptive.coarsen(lambda grid: grid.get("root_length_density") < 1e2)
```

Both expose the same contract: cells are the graph's nodes and the faces between them its edges, with the edge
variables `face_area` and `face_distance`, `cell_volume()`, `cell_centers()`, `locate(points)` and
`layer_mask(z=-1)` for boundary layers. An adaptive grid keeps neighbours within one level of each other and carries
its variables over refinement conservatively (extensive values split or summed, the others copied or averaged).

### Working with variables

```python
ds.register("hexose", 0., location="node", on_grow="inherit")
ds.get("hexose")                       # a live view (writes through set())
ds.set("hexose", values)               # in place, shape checked, write counted
ds.alias("sugar", "hexose")            # a second name for the same values
ds.derive("total", {"a": 1., "b": 2.}) # recomputed when read after a source changed
ds.define_mask("apices", {"is_apex": ">0"})
ds.register("pools", 0., location="cell", shape=(15,))   # a vector-valued variable, (n_cells, 15)
```

Labels and types are declared with `dtype="int"` (label names are resolved through the MTG's `LabelsConfig`); lists
and records with `dtype="object"`. `ds.validate_variables()` reports inconsistencies; `ds.to_dataframe(location=)`
and `ds.summarize(...)` export values; `ds.checkpoint(path)` and `Class.restore(path)` (or `ds.load_checkpoint(path)`
in place) save and restore everything, bit for bit.

## 2. Components

A component is a dataclass deriving from `FunctionalComponent` (processes) or `StructuralComponent` (processes that
edit the structure), built on a DataStructure. Its fields declare its variables:

```python
from dataclasses import dataclass
from openalea.metafspm import FunctionalComponent
from openalea.metafspm.coupling.component import input_variable, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.solve.decorator import rate, state

DOC = dict(unit="mol.g-1", unit_comment="", description="", min_value=0., max_value=1., value_comment="",
           references="", DOI=[])

@dataclass
class RootCarbon(FunctionalComponent):
    hexose: float = state_variable(**DOC, initialize=1e-3, scale=scales.SubOrgan,
                                   state_variable_type="massic_concentration", on_grow="inherit")
    exudation: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="extensive")
    struct_mass: float = input_variable(**DOC, by="RootGrowth", initialize=0., scale=scales.SubOrgan)
    exudation_rate: float = parameter(**DOC, by="RootCarbon", default=1e-6)

    @rate
    def _exudation(self, hexose, struct_mass, exudation_rate):
        return exudation_rate * hexose * struct_mass

    @state
    def _hexose(self, hexose, exudation, struct_mass):
        return hexose - self.dt * exudation / struct_mass

carbon = RootCarbon(data_structure=ds, time_step=3600.)
carbon()                                  # one time step: every step, in the scheduled order
```

- **Time step.** `time_step=` (s) at construction is the `dt` of the component's equations. Its default is the
  class's `time_step` attribute when set, else the simulation time step (the scene's). Pass it explicitly, even
  for a component that solves a steady state.
- **Declarations.** `scale=` places a variable at an MTG scale (its location follows from the graph), or
  `location=` gives it directly (`"cell"` on grids). `state_variable_type` (its kind: extensive, intensive,
  massic_concentration, descriptor) decides the default mappings and the repartition after growth. Variables are
  registered at construction, from the MTG's property when it exists, else from the default.
- **Parameters** without a place are stored per plant (`"Plant"`) on plant DataStructures, as a scalar elsewhere.
  Plants of a population may then differ. Steps and equations take parameters **as arguments**; reading
  `self.<parameter>` inside a step raises, while outside steps `model.k = value` sets every plant.
- **Steps** are methods named after their output (`_exudation` writes `exudation`), decorated by their place in the
  schedule: `@stepinit`, `@rate`, `@totalrate`, `@state`, `@totalstate`, `@deficit`, `@axial`, then the growth rows
  `@potential`, `@allocation`, `@actual`, `@segmentation`, `@postsegmentation`. They receive whole arrays; a step
  written with scalar logic opts out with `@rate(vectorized=False)`. A step returning `-> tuple[...]` gives several
  outputs, `(value, "other_name", other_value)`; a step returning `None` writes nothing.
- **Selections.** Every decorator takes `filters=`, the entities it operates on, written as explicit key / values
  that must all hold: `@rate(filters={"label": "RootSegment", "length": ">0.01"})`. A value is matched, a list is
  one of several values, a comparison string (`">0"`, `"<=0.03"`) is tested, and label names are resolved. A key
  may name a variable at a coarser scale (e.g. a segment's label for its anatomy Compartments). A mask name
  (`ds.define_mask`) or a callable `ds -> boolean array` select as well, for geometry or masks shared with mappings.
  When the DataStructure defines an `"active"` mask (emergence, dead tissues), steps also compute on its entities
  only; `include_inactive=True` lifts it.
- **Inheritance.** A subclass runs its bases' steps, a redefined step replacing its base's; `steps_removed =
  ("name",)` removes inherited ones.
- **Compiled steps.** A step may call a `numba.njit` function on its arrays; parameters reach it as arrays (zero-stride
  views when uniform).
- **Random draws.** `self.random(stream, distribution="uniform", **parameters)` gives one reproducible draw per entity
  per call, independent of the visiting order.
- **The MTG view.** By default (`mtg_sync = "lazy"`) the DataStructure is the reference and state variables reach the
  MTG when it is read (`ds.mtg`, `ds.flush_mtg()`), before an MPG-style step. `mtg_sync = "after_call"` writes them
  after every call, `"never"` never.

## 3. Graph systems

Coupled equations over a graph (transport along a root system, diffusion in a soil grid) are declared as a graph system:
an inner class whose methods give the residuals of the node and edge unknowns.

```python
from openalea.metafspm.solve.decorator import boundary_set, edge_law, graph_system, node_balance, node_rate

@dataclass
class SoilDiffusion(FunctionalComponent):
    solute: float = state_variable(**DOC, initialize=0., location="cell")
    solute_flux: float = state_variable(**DOC, initialize=0., location="edge")
    diffusivity: float = parameter(**DOC, by="", default=1e-9, location="edge")

    @graph_system(node_unknowns=["solute"], edge_unknowns=["solute_flux"], transient=True, integrate="adaptive")
    class _diffusion:
        bottom = boundary_set(filters=lambda ds: ds.layer_mask(z=-1), kind="dirichlet", value=0.)

        @node_balance(field="solute")
        def _balance(self, solute, solute_flux):
            B = self._graph_view.incidence
            return (solute - self.previous("solute")) / self.dt \
                + (B @ solute_flux) / self.data_structure.cell_volume()

        @edge_law(field="solute_flux")
        def _fick(self, solute, solute_flux, diffusivity, face_area, face_distance):
            B = self._graph_view.incidence
            return solute_flux - diffusivity * face_area / face_distance * (B.T @ solute)
```

- **Two forms of a node balance.** The *residual form*, `@node_balance`, returns the residual at the end of the
  (sub-)step, time term included (`(u - self.previous("u")) / self.dt + ...`): it is solved by the Newton family,
  `solver="newton"` (default), `"newton_fd"`, `"scipy_krylov"`, `"scipy_hybr"`, … The *rate form*, `@node_rate`,
  returns du/dt without time term (`J - B q`); the framework writes the time term for the solver chosen: backward
  Euler with the Newton family, `u += dt * rate` with `"explicit_euler"`, an IVP integration with `"scipy_ivp_bdf"`
  or `"scipy_ivp_radau"`. One rate-form model can thus be solved by any solver; a residual can only be solved by
  Newton. The node unknowns of one system use one form. `integrate="substeps"` (with `n_substeps`) or `"adaptive"`
  (step doubling, `rtol` / `atol`) integrate over the component's time step. Edge unknowns are algebraic
  (`@edge_law`), whatever the solver.
- **Boundary sets.** `boundary_set(filters=..., kind="robin" | "dirichlet" | "neumann", value=..., weight=...)` on a
  set of nodes, with values and weights read at each solve. `kinds="variable"`
  reads each node's kind from a node variable (`boundary_set.CODES`: 1 Dirichlet, 2 Neumann, 3 Robin, otherwise
  none), e.g. a collar switching between a pressure and a flux. `kind=None` is a selection only.
- **Boundary conditions as equations.** When the condition is an expression rather than a variable,
  `@boundary_condition("node", "dirichlet" | "neumann", field=..., filters=...)` tags a method that takes its
  arguments by name like the balances: unknowns and DataStructure variables (e.g. coupled ones), sliced to the
  selected nodes. A Dirichlet method returns the residual (`p - collar_pressure`), a Neumann one the inflow
  (`uptake_rate * (soil_concentration - concentration)`), with the same sign as a `boundary_set`'s value.
- **Selections** (`filters=`, as for steps). Outside them:
  - a balance or an edge law contributes 0;
  - a boundary has no term;
  - `@graph_output(..., filters=)` gives 0.

  The method's arguments are sliced to the selection; values read through `self` (e.g. `self.previous()`) are not.
- **Pool unknowns.** `pool_unknowns={"shoot_sugar": {"location": "Plant", "exchange": "collar"}}` adds one unknown
  per plant, solved with the graph; its residual is a `@pool_balance(field=...)`, and equations exchange with it
  through `self.pool_exchange(name)` (a sparse node × pool map). Newton solvers only.
- **Active subgraphs and pieces.** `@graph_system(filters=...)` solves on the subgraph of the selected nodes (the
  others frozen, dropped edges without flux), e.g. `filters="active"`; `split="components"`
  solves each connected piece (each plant) on its own, with its own convergence and adaptive steps.
- **Forcings.** `self.forcing(name)` interpolates a time series of `self.forcings` (or of the scene's table) at the
  end of the current (sub-)step, or at the evaluation time of an IVP solver.
- **Outputs and Jacobians.** `@graph_output(name, location=...)` derives outputs after the solve; `@graph_jacobian`
  gives an analytic Jacobian (finite differences with graph colouring otherwise).

## 4. Structure: `StructuralComponent`

A structural component edits the plant (growth, segmentation, anatomy). Its steps come in two styles:

- **array-style** steps (with arguments) are vectorised like functional steps, e.g. elongating apices;
- **MPG-style** steps (without arguments) edit the MTG through `self.mtg` (`add_child`, `insert_parent`,
  `remove_vertex(..., reparent_child=True)`, `add_components_bulk`). Before them, the variables they declare are
  written to the MTG; after them, the DataStructure follows the new topology and the other components' variables are
  repartitioned by kind (`partition_weight`, `active`). They loop over `self.active_ids()` to skip inactive plants.

`StructuralComponent.initiate_plant(g, plant, parameters)` (a class method) builds one plant's initial structure in a
population MPG, from that plant's scenario.

### Tree kernels

Whole-population computations along the topology are DataStructure methods, vectorised over every plant:

| Method | Computes |
|---|---|
| `chain_scan(values, chain, op="sum" \| "max", reverse, exclusive)` | prefix sums or maxima along axes (distance from tip) or rank chains |
| `chain_shift`, `chain_write` | the value k positions earlier on a chain; forward writes (rank n sets n + 1) |
| `chain_gather(values, rank, chain, source)` | the value at a rank of another chain (a tiller reading the main stem) |
| `chain_recurrence(update, values, chain)` | non-associative recurrences along chains |
| `accumulate(values, direction, op)` | subtree totals, or path totals from the root |
| `fold(update, values, direction)` | a level-by-level fold with your function: pipe models, death propagation, filtered maxima, frames |
| `path_window(budget, extent, values)` | sums over a node's ancestors within a budget (a supply window) |
| `path_contributions`, `scatter_contributions` | the window element by element, and amounts shared back in visiting order |
| `path_compose(transforms)` | composed 4 × 4 transforms from the root |
| `parents()`, `children()`, `order("pre" \| "post", convention=None \| "openalea")`, `depth()`, `levels()` | traversal arrays |

Chains are declared with `ds.define_chain(name, edge_type="<")` or by a group and a rank variable.

## 5. Coupling components

### Within one DataStructure: `CompositeModel` and translators

A plant model is a `CompositeModel` whose components share one DataStructure. A translator states which component
provides each input: `Translator().link(receiver, variable, provider, {source: factor}, aggregation=, weight=,
formula=)`, or a nested YAML/Python dictionary. Links become aliases (same values, another name) or derived variables
(factors, sums, formulas, mappings between scales), computed when read.

```python
self.declare_data_and_couple_components(translator_path="translator.py", components=self.components)
```

`openalea.metafspm.testing.assert_component_couplable(Component, translator)` checks a component against a
translator in a package's own tests.

### Across DataStructures: mappings and exchanges

Links between components on different DataStructures are exchanged by `coupling.cross.Exchanges` at fixed points,
through a mapping:

- `CrossMapping(plants, grid, method="barycentre" | "overlap")`: segments to cells (by middle point, or by the
  length fraction in each cell), rebuilt after growth or coordinate changes;
- environment **scalars** (stored at `"scalar"`): population values reduced over every plant of every population,
  or a scalar broadcast to every node;
- `LayerMapping(column, grid, axis="z")`: a 1-D column and a 3-D grid, by layer overlaps;
- `UnionDataStructure(populations)` with `UnionMapping`: one component (a light model) over every population.

Defaults follow the variables' kinds: extensive values are summed towards cells and split back by a `weight=`;
intensive values are averaged (with `weight=`) towards cells and gathered back. Several populations feeding one cell
variable are pooled in one write.

## 6. Scenes

A `Scene` runs plant populations and environment models together, in one process.

```python
from openalea.metafspm import Scene, planting_table

table = planting_table(xrange=0.3, yrange=0.3, sowing_density=250, row_spacing=0.15,
                       plant_models=[Wheat], plant_scenarios=[scenario], seed=1)
scene = Scene(table, environment=[RhizoSoil, LightModel], environment_scenarios=[soil_scenario, light_scenario],
              translator="scene_translator.py", time_step=3600, forcings=meteo,
              output_dirpath="outputs", log_plants=[table["plant"][0]], heavy_log_period=24)
scene.simulate(n_iterations=2500)
```

- **Plant models** are built once per model of the planting table, on an MPG holding all its plants:
  `Model(data_structure, time_step, **scenario)`, with a class attribute `initiators` (the structural components
  building each plant). In anatomy mode (`nodes = "Compartment"`) the initiators build the anatomies, and a `wiring`
  attribute gives the junction rules between them. Numeric parameters may differ per plant (one scenario per plant); an `emergence_time` column
  keeps a plant frozen until then.
- **Environment models** are built as `Model(populations, scene_xrange, scene_yrange, time_step, **scenario)` and
  create their own DataStructures (a grid, a union of the populations).
- **A step** runs each environment model after the exchanges into it, then each population after the exchanges into
  it. Mappings between populations and grids or unions are inferred from the scene translator; others are given by
  `mappings=`.
- **Services.** `forcings=` (one meteo table for every model), `run_every` / `run_when(scene)` on a model (skipped
  steps keep its outputs), `spin_up(scene)`, `events=[(time, action)]`, `stop_when=condition`.
- **Outputs and restarts.** The recorder writes per-plant summaries every step and the selected plants' segments
  every `heavy_log_period` steps (CSV, one folder per population). `scene.checkpoint(path)` and
  `Scene.restore(path, *arguments)` continue a run bit for bit; models keep non-variable state through
  `checkpoint_state()` / `restore_state(state)`.

## 7. A complete example

`examples/soil_plant_atmosphere/` puts every feature above together, on water flow from a water table through the
soil and seedlings to a dry atmosphere:
- transport components for the plants' anatomy graph and for the soil grid, on shared flow equations;
- structural components building the seedlings at every scale, with their anatomies, and computing the conductances;
- boundary sets and boundary equations;
- translators within and across DataStructures, with a masked `CrossMapping` of the root surface;
- scenes with one plant and with a planted population;
- plots of the converged water potentials.

Its README describes each file and shows the figures.

## 8. Performance notes

- Steps, graph systems, tree kernels and exchanges work on whole-population arrays: a step of 1000 plants of 2 000
  segments takes about 2 s in one process.
- Growth bookkeeping is incremental: only new vertices are read from the MTG.
- An MPG-style step sees only the MTG properties its component declares; keep MTG reads in such steps to those.
- `split="components"` is not parallel; graph systems are solved by sparse LU (SuperLU).
