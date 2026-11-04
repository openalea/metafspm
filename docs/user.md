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
  nodes, wired to their neighbours' by junction rules. A segment's anatomy may have Layers and Cells
  (SubOrgan → Layer → Cell → Compartment), with Compartments shared by several cells (walls) anchored to the
  segment; `"Layer"` and `"Cell"` are then locations, and the walls, owned by no Cell, read the defaults there.
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
export values (the SceneRecorder writes the summaries); `ds.checkpoint(path)` and `Class.restore(path)` (or `ds.load_checkpoint(path)`
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

@dataclass
class RootCarbon(FunctionalComponent):
    hexose: float = state_variable(unit="mol g-1", description="The hexose concentration", default=1e-3,
                                   scale=scales.SubOrgan, kind="massic_concentration",
                                   on_grow="inherit")
    exudation: float = state_variable(unit="mol s-1", description="The hexose exuded", default=0.,
                                      scale=scales.SubOrgan, kind="extensive")
    struct_mass: float = input_variable(unit="g", description="The structural mass", by="RootGrowth",
                                        default=0., scale=scales.SubOrgan)
    exudation_rate: float = parameter(unit="g s-1", description="The exudation's rate", by="RootCarbon",
                                      default=1e-6)

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
  class's `time_step` attribute when set, else the scene's (`Scene(time_step=)`). Alone, a component runs one step
  of its own time step per call; without any, its equations may not read `self.dt` (it raises). Pass it
  explicitly, even for a component that solves a steady state. A time step shorter than the simulation step makes the component
  run once per time step within it (900 s in a 3600 s scene: four runs per scene step, forcings read at the end
  of each). It must divide the simulation step a whole number of times.
- **Declarations.** `default=` is the initial value of a state variable, the value of an input nothing provides,
  a parameter's or a setting's value. Each declaration is documented by keywords: `unit` and `description` at
  least, and optionally `min_value`, `max_value` (the bounds `variants` reads), `references`, `DOI`,
  `unit_comment`, `value_comment`; `declarations()` lists them all, with each variable's `role` (state_variable,
  input, parameter, setting, forcing) and `kind`. `scale=` places a variable at an MTG scale (its location follows
  from the graph), or `location=` gives it directly (`"cell"` on grids). `kind=` (extensive, intensive,
  massic_concentration, descriptor: how its value depends on the size of what it describes) decides the default
  aggregations and the repartition after growth. Variables are registered at construction, from the MTG's property
  when it exists, else from the default. The former `initialize=`, `state_variable_type=` and `mapping=` are
  accepted with a warning for one release.
- **Parameters** without a place are stored per plant (`"Plant"`) on plant DataStructures, as a scalar elsewhere.
  Plants of a population may then differ. Steps and equations take parameters **as arguments**; reading
  `self.<parameter>` inside a step raises, while outside steps `model.k = value` sets every plant. Declared
  `scale=Plant`, a boolean or an object (`dtype="object"`: a dict, a record) is per plant too, set by each plant's
  scenario. A parameter whose value is `{section: {field: value}}` (an anatomy's parametrization) declares its
  fields with `sections=` (a function giving `{"section.field": {default, unit, description, min_value,
  max_value}}`): `parameter_table` and `variants` list each as `name.section.field`, and a scenario may give one
  dotted, `{"root_anatomy.cortex.n_layers": 4}`, or nested.
- **Steps** are methods decorated by their place in the schedule: `@stepinit`, `@rate`, `@totalrate`, `@state`,
  `@totalstate`, `@deficit`, `@axial`, then the growth rows `@potential`, `@allocation`, `@actual`,
  `@segmentation`, `@postsegmentation` (a structural component's steps, in that order: before, while and after it
  changes the structure). Their arguments are variables, read from the DataStructure by name as whole arrays (one
  value per entity); a step written with scalar logic opts out with `@rate(vectorized=False)`. A step returns
  - one output, named after the method (`_exudation` writes `exudation`);
  - or several, named, as a dict, annotated `-> dict`: `return {"x1": x, "y1": y}` (the method's name, e.g.
    `_placement`, then only names the step);
  - or `None`, writing nothing.
  (The older form `-> tuple[...]`, `(value, "other_name", other_value)`, still works.)
- **Selections.** Every decorator takes `filters=`, the entities it operates on, written as explicit key / values
  that must all hold: `@rate(filters={"label": "RootSegment", "length": ">0.01"})`. A value is matched, a list is
  one of several values, a comparison string (`">0"`, `"<=0.03"`) is tested, and label names are resolved. A key
  may name a variable at a coarser scale (e.g. a segment's label for its anatomy Compartments). A mask name
  (`ds.define_mask`) or a callable `ds -> boolean array` select as well, for geometry or masks shared with mappings.
  When the DataStructure defines an `"active"` mask (emergence, dead tissues), steps also compute on its entities
  only; `include_inactive=True` lifts it.
- **Where a step works.** A step's arguments come at their locations, its parameters at the step's: the location of
  its per-node arguments, or, without any, that of its declared outputs (a plant's step taking a per-plant parameter
  and returning one value per plant). Its outputs are floats, or records (`dtype="object"`: a tuple, a string per
  entity) for a variable of that type. `self.dt` is the time step in every component's steps.
- **Optional inputs.** `input_variable(..., optional=True)`: when nothing provides it (no other component, no MTG
  property), it is not registered on the DataStructure (coordinates read for a display only, which a geometry
  provider would take for its own), and a step naming it gets its default: the component runs on its own.
- **Memoized steps.** `@rate(memo=True)` (any step): its outputs are computed again only when its arguments' values,
  its component's parameters or settings, the active mask or the structure changed; otherwise its last outputs are
  written again (a network's conductances, recomputed only when a barrier or a length changed). The step's
  dependencies are its arguments: a memoized step reads nothing else.
- **Own entities.** `entities = {"label": ("RootSegment", "RootApex")}` (a class attribute, the filters' syntax,
  label names resolved; or a callable `ds -> boolean array`): the entities the component owns when several models
  share one plant (the roots' and the shoot's). It then works on `ds.entity_view(entities)`, a view of the
  DataStructure restricted to them (section 4). `None` (the default): every entity.
- **Inheritance.** A subclass runs its bases' steps, a redefined step replacing its base's; `steps_removed =
  ("name",)` removes inherited ones.
- **Compiled steps.** A step may call a `numba.njit` function on its arrays; parameters reach it as arrays (zero-stride
  views when uniform).
- **Random draws.** `self.random(stream, distribution="uniform", **parameters)` gives one reproducible draw per entity
  per call, independent of the visiting order. Draws that must not depend on the rest of the population (an
  element's angles, whatever other plants exist) are keyed by lineage instead of entity id: `random_streams.plant_keys(
  seeds)` for a plant's first element, `child_keys(parent_keys, edge, rank)` for each new one, drawn with
  `random_streams.normal(keys, ...)` (or `ds.random(..., ids=keys)`).
- **Settings** (`setting(unit=, description=, default=, choices=)`): values of the model, not of the plants (a
  latitude, a light model, a solver, a grid's voxel). One value for all, never stored on the DataStructure: read as
  `self.name` anywhere, passed as the value itself to a step naming it, set by constructor keywords and scenarios
  (the same for every plant), checked against their `choices`. Numeric `parameter`s stay per plant.
- **State that is not a variable:** `self.keep(name, value)` and `self.kept(name, default)` (the time of a last
  lighting), saved by the Scene's checkpoints; `keep(..., saved=False)` for a cache rebuilt when missing (a kept
  canopy, incremental meshes); `self.cached(name, build, key)` for a value derived from the structure.
- **Reading a component:** `Component.declarations()` lists its variables as a table (kind, scale, unit, description,
  default, provider); `Component.schedule()` its steps in the order they run, grouped by stage;
  `component.variable_specs` their resolved places on its DataStructure; `component.active_ids(location)` the active
  entities (at a coarse location: those of the plants emerged), for MPG-style steps.
- **Couplings.** `model.couplings()` lists how each input of the model's components is coupled; in a Scene, the scene
  translator's links too (`"scene"`: with the environment models, the default identity links included).
- **One variable, two places.** A component reading at the nodes a variable another holds at its scale (a segment's length
  read at its Compartments), or reading at its scale a variable another holds at the nodes as a broadcast from it
  (a segment's coordinates seen by its Compartments), reads that variable through the mapping, in its steps and with
  `component.read(name)` outside them.
- **Owners.** `ds.owner(location)`: each node's entity at a coarse location (an index in `ds.entity_ids(location)`);
  `ds.owner("Axis", of="Organ")`: each Organ's Axis, between two coarse locations.
- **Public accessors** (rather than the private names behind them): `ds.from_scale`, `ds.sections_of(vids)`,
  `ds.reduction`, `ds.structure_stamp()`, `ds.default(name)`; `MPG.values_at(name, vids)` (bulk, instances
  included), `MPG.anchored(vid)` (its anchored Compartments and Connections), `MPG.has_instances()`, `MPG.last_vid`,
  `MPG.add_root_at_scale(scale)`; `self.graph_view` in graph-system terms; `Choregrapher().set_simulation_time_step(dt)`
  for components run without a Scene.
- **Outside steps** (structural code, helpers, harnesses): `self.parameter_value(name)` is a parameter's single
  value for the population (it raises when plants differ; `parameter_values(name, location)` gives one per entity);
  `self.cached(name, build, key=None)` keeps a value until *key* changes (by default the topology version), e.g. a
  per-node array read from the MTG's labels; `component.run_stage("postsegmentation")` runs the steps of one
  schedule row only (e.g. after another model wrote the inputs they read).
- **The MTG view.** By default (`mtg_sync = "lazy"`) the DataStructure is the reference and state variables reach the
  MTG when it is read (`ds.mtg`, `ds.flush_mtg()`), before an MPG-style step. `mtg_sync = "after_call"` writes them
  after every call, `"never"` never. `ds.structure` is the MTG to read the structure only (labels, parents,
  complexes), without that synchronisation.

## 3. Graph systems

Coupled equations over a graph (transport along a root system, diffusion in a soil grid) are declared as a graph system:
an inner class whose methods give the residuals of the node and edge unknowns.

```python
from openalea.metafspm.solve.decorator import boundary_set, edge_law, graph_system, node_balance, node_rate

@dataclass
class SoilDiffusion(FunctionalComponent):
    solute: float = state_variable(**DOC, default=0., location="cell")
    solute_flux: float = state_variable(**DOC, default=0., location="edge")
    diffusivity: float = parameter(**DOC, by="", default=1e-9, location="edge")

    @graph_system(node_unknowns=["solute"], edge_unknowns=["solute_flux"], transient=True, integrate="adaptive")
    class _diffusion:
        bottom = boundary_set(filters=lambda ds: ds.layer_mask(z=-1), kind="dirichlet", value=0.)

        @node_balance(field="solute")
        def _balance(self, solute, solute_flux):
            B = self.graph_view.incidence
            return (solute - self.previous("solute")) / self.dt \
                + (B @ solute_flux) / self.data_structure.cell_volume()

        @edge_law(field="solute_flux")
        def _fick(self, solute, solute_flux, diffusivity, face_area, face_distance):
            B = self.graph_view.incidence
            return solute_flux - diffusivity * face_area / face_distance * (B.T @ solute)
```

- **Two forms of a node balance.** The *residual form*, `@node_balance`, returns the residual at the end of the
  (sub-)step, time term included (`(u - self.previous("u")) / self.dt + ...`): it is solved by the Newton family,
  `solver="newton"` (default), `"newton_fd"`, `"scipy_krylov"`, `"scipy_hybr"`, … The *rate form*, `@node_rate`,
  returns du/dt without time term (`J - B q`); the framework writes the time term for the solver chosen: backward
  Euler with the Newton family, `u += dt * rate` with `"explicit_euler"`, an IVP integration with `"scipy_ivp_bdf"`
  or `"scipy_ivp_radau"`. One rate-form model can thus be solved by any solver; a residual can only be solved by
  Newton. The node unknowns of one system use one form. `integrate="substeps"` (with `substeps=`) or `"adaptive"`
  (step doubling, `rtol` / `atol`, numbers or `{unknown: value}` per node unknown) integrate over the component's
  time step (`"once"` by default). A failed solve (Newton not converging, a residual that is not finite, a singular
  system) raises `solve.solver.SolverConvergenceError`; with `"adaptive"` it rejects the sub-step and halves it, as
  the error control does, and raises naming the component and the time only below `min_step`. The Newton
  iterations take `max_iterations`, `tolerance` (on ||R||∞), `relative_tolerance` (converged too when ||R||∞ fell
  below that fraction of the first iterate's; 0 by default), `finite_difference_step`, `linesearch`; `transient=True` declares
  that a `@node_balance` writes its own time term (it only decides the anchoring check; inferred for `@node_rate`).
  `@edge_law(integrate=True)` adds the integrated amount `<field>_amount` (Q += q dt) as an edge unknown. Edge unknowns are algebraic
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
- **Forcings.** `self.forcing(name)` interpolates a time series of the scene's forcings at the end of the current
  (sub-)step, or at the evaluation time of an IVP solver. A component declares the forcings it
  reads, `PARi: float = forcing_variable(unit="µmol m-2 s-1", description="...")`: unlike an input, written per
  entity by another model through the translator, a forcing is a time series given from outside, the same for every
  entity. It is listed by `declarations()` (role "forcing"), not stored on the DataStructure, and a `Scene` checks
  at its construction that every required one is given (`required=False`: an optional one, which raises KeyError
  when read). A forcing is read at the end of the step by default; `forcing_variable(at="start")`, or
  `self.forcing(name, at="start")`, reads it at the step's start (a meteo table whose hour begins with the step).
  Outside a Scene, `component.set_forcings(forcings, start=None)` gives a component the forcings a Scene would
  (`{name: forcing}` or a DataFrame indexed by time, the Scene's `start=` calendar), read at the component's clock
  (0 at its first call); `self.has_forcing(name)` tells whether one is given as a time series, e.g. for an input
  read from the forcing of its name at each sub-step when one is given.
- **Solvers.** `@graph_system(solver=...)` takes one of the keys `solve.solver.solver_names()` lists; a solver class
  of one's own (a subclass of `NewtonSolver`, e.g. with another linear solver) is given a key with
  `solve.solver.register_solver(name, solver_class)`. A graph system's equations class, to call or re-solve its
  equations outside the component, is `Component._system.equations`.
- **Outputs and Jacobians.** `@graph_output(name, location=...)` derives outputs after the solve; `@graph_jacobian`
  gives an analytic Jacobian (finite differences with graph colouring otherwise).

- **Reductions.** A plant model may reduce its graph systems by groups (`solve.reduction.Reduction(location=, labels=,
  ports=, keep=, explicit_edges=, kind=, inputs=, systems=, outputs=)`): the groups of a location (e.g. the SubOrgans
  of some labels) keep their ports and kept nodes, their interiors eliminated once per kind (a library population of
  the model's own components, each kind its group's template, kind values and the plant's parameters, objects
  included, digested by content; the library runs the components of the reduced systems and those giving them inputs
  at the cells (per-cell osmotic presets), not the others (a shoot's development, its tables)). It is the model's
  `reduction` class attribute, or a function of its scenario choosing one, or None: `reduction = staticmethod(lambda
  scenario: Reduction(...) if scenario.get("reduced") else None)`; the entries it reads are the model's, the same for
  every plant of a population. The entries it owns but may not read (the reduced labels while `reduced` is off) are
  its attribute `scenario_keys` (`choose.scenario_keys = ("reduced", "reduced_labels")`): accepted, inactive. A graph
  system's terms the reduction must keep explicit are declared `@explicit_term`, its inputs' derivatives
  `@input_jacobian`. A structural model takes a group's reduced template from `reduction.template_for(full)`; a
  component reading a reduced group's full template (its volumes, its geometry) gets it from
  `reduction.full_template(key)`, the key of the reduced template its instance holds.
## 4. Structure: `StructuralComponent`

A structural component edits the plant (growth, segmentation, anatomy). Its steps come in two styles:

- **array-style** steps (with arguments) are vectorised like functional steps, e.g. elongating apices;
- **MPG-style** steps (without arguments) edit the MTG through `self.mtg` (`add_child`, `insert_parent`,
  `remove_vertex(..., reparent_child=True)`, `add_components_bulk`). Before them, the variables they declare are
  written to the MTG, their inputs another component holds at another location too (an architecture's SubOrgan
  lengths read by a Compartment-level anatomy model, at the SubOrgans as their holder declares them); after them,
  the DataStructure follows the new topology, reads back the states they wrote (on the vertices they created too),
  and the other components' variables are repartitioned by kind (`partition_weight`, `active`). Until their states
  are read back the MTG is ahead of the DataStructure (`ds.mtg_ahead()`): a `ds.mtg` meanwhile (an inheritance
  rule's, during the topology's update) writes nothing over them. They loop over `self.active_ids()` to skip
  inactive plants.

`StructuralComponent.initiate_plant(g, plant, parameters)` (a class method) builds one plant's initial structure in a
population MPG, from that plant's scenario.

**Creating nodes from an array-style step.** An array-style step returns a `NewNodes` among its outputs: the new
nodes' parents (existing node ids, or earlier new nodes as `-1 - index`), edges, labels and values, and their new
complexes (`new_complex` and `complex_levels`, or explicit `NewComplexes`, e.g. a phytomer holding three organs).
The framework creates them in the MPG in bulk, extends the DataStructure with their values (the others at their
`on_grow` value) and repartitions the other components' variables:

```python
@segmentation
def _new_segments(self, length, segment_length) -> dict:
    tips = np.flatnonzero(length > segment_length)
    return {"length": np.where(length > segment_length, length - segment_length, length),
            "new_segments": NewNodes(parent=self.data_structure.entity_ids("node")[tips], edge="<",
                                     label=self.data_structure.structure.labels.SubOrgan.RootSegment,
                                     values={"length": segment_length})}
```

**Geometry for light models and display.** `component.triangles(where=None)` gives the elements' triangles
(`coupling.geometry.Triangles`: vertices in the scene's frame, each triangle's node index). Segment models
declaring `x1`, `y1`, `z1`, `x2`, `y2`, `z2` and `radius` get slim cylinders by default; a component with its own
shapes (Adel's blades) overrides it. A light model computes on the triangles and gives its results back per element
(`Triangles.per_owner`). An environment model receives DataStructures, not components: `ds.triangles()` gives the
triangles of the structural component built on it (the one with its own `triangles()` first; set
`ds.geometry_provider` when several draw), and a `UnionDataStructure`'s concatenates its parts', in its node indices.
`ds.triangles(where)` takes a node mask or a mask's name (`"emerged"`: plants not yet emerged left out; every node
when the DataStructure has no such mask). `Triangles.clip_below(z)` cuts them at the soil (a buried part receives no
light), `Triangles.mean_per_owner(values, n)` averages per element by area.

**Labels.** `ds.labels()` gives each node's MTG label name (`"LeafElement"`, `"RootSegment"`, ...), and
`ds.labels(location="Organ")` the label of its entity at a coarser scale; a `UnionDataStructure` gives its parts'. They are
read once per topology, for steps that treat leaves and stems, or root orders, differently. `openalea.metafspm.display.draw(axes, triangles, colors=...)` draws them with matplotlib.

**Several architecture models on one plant: `entities`.** When two structural components grow one plant (the roots
and the shoot), each declares the entities it owns, and works on an `EntityView` of them, as if the DataStructure held
them only:

```python
@dataclass
class Roots(StructuralComponent):
    entities = {"label": ("RootSegment", "RootApex")}
    ...
```

- Its nodes are the entities of its location matching the declaration, in the base's order (the nodes in segment
  mode; the SubOrgans of an anatomy-mode DataStructure, for a component declared at that scale or coarser, a
  Phytomer-scale one included: its `owner("Phytomer")` maps its SubOrgans to their phytomers):
  `n_nodes()`, `entity_ids("node")`, `index_of` and its steps' arrays are theirs. They are selected again, at the
  next read, when the topology or the labels change (new entities matching the declaration join it) and when a
  variable the declaration reads is written (`{"axis_row": ">=0"}`, read at each node's Axis: an axis a
  `NewComplexes` creates with its `axis_row` is seen with it, the new nodes' and complexes' values being written
  before the new nodes are checked against the declaration). A callable is evaluated again on a topology or label
  change only, unless it lists the variables it reads in its attribute `variables` (`shoot.variables =
  ("axis_row",)`); it may read MPG properties (`ds.structure`), which are not tracked. The selection, its index
  maps and the view's stores are cached on these: a step costs about what it costs on the base.
- Its topology is theirs: a node's parent is its parent when that one is in the view, else -1 (a base: a seminal root
  on the seed another model owns). `children()`, `roots()`, `tips()`, `order()`, `levels()`, the folds, chains and
  `to_graph_view()` follow from these parents.
- A name is one variable of the base for every entity of the location (the roots' and the shoot's `length`):
  `get(name)` gives the view's entities' values, **a copy** (steps write through their outputs, or `ds.set`);
  `set` writes them into the base's array; a variable it creates gets the default (or the MTG's values) on the other
  entities. Coarse locations (`"Plant"`, `"Axis"`, …) and scalars are the base's.
- `map_to` between its nodes and a coarse location, `owner()` and `labels()` are over its entities only (a plant's
  root length: `map_to(length, "node", "Plant", "sum")`).
- Its `NewNodes` may name a parent outside the view (a nodal root `'+'` on a stem element): any entity of the
  location; the new nodes must match its declaration. Its masks (and `active`) are its own; the base's triangles are
  those of every component drawing its entities, together.
- `view.base` is the DataStructure the view is of, at the view's location: the MPGDataStructure in segment mode;
  in anatomy mode, whose nodes are the Compartments, its view at the SubOrgans (`ds.at("SubOrgan")`). Its
  `get(name)` gives a variable at every entity of the location, `index_of` their indices. A component reads through it what lies outside its
  entities and its steps' arguments cannot give: the end (`x2`, `y2`, `z2`) of the stem element a nodal root is borne
  by, its bearers' values. It does not write there: another model's entities are that model's.
- `couplings()` and `declarations()` are unchanged: the components still share one DataStructure.
- **Anchors.** An initiator builds onto another's vertex (a root model on the seed a shoot model created) through
  `g.find_anchor(plant, label, scale=None, where=None, first=False, last=False)`: the plant's vertex of that label
  (at that scale: found from the label when only one scale has it) whose MPG properties equal `where`'s. Several
  match: an `AnchorError`, unless `first=True` (the topologically first: the bottom of a chain of them) or
  `last=True` (the topologically last: its top, a crown's top stem element). None: an `AnchorError` naming what the
  plant has; an anchor must be built by an earlier initiator.

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
| `parents()`, `children()`, `order("pre" \| "post", traversal=None \| "openalea")`, `depth()`, `levels()` | traversal arrays |

Chains are declared with `ds.define_chain(name, edge_type="<")` or by a group and a rank variable.

## 5. Coupling components

### Within one DataStructure: `CompositeModel` and translators

A plant model is a `CompositeModel` whose components share one DataStructure. A translator states which component
provides each input: `Translator().link(receiver, variable, provider, {source: factor}, aggregation=, weight=,
formula=)`, or a nested YAML/Python dictionary. An input named as another component's variable needs no link: the
DataStructure is the coupling, both read the same array. Links state the rest, and become aliases (a rename) or
derived variables (factors, sums, formulas, aggregations between scales), computed when read.

```python
self.couple(self.growth, self.carbon, translator="translator.py")     # Translator.load: a Translator, a dict, .py, YAML
self.couple(self.anatomy, self.hydraulics)                            # every input shared by its name: no translator
self.couplings()                                                      # how each input is coupled, one row each
```

`couplings()` lists each input's provider and coupling: "shared" (by its name), "alias", "derived", "scene" (a
link with another DataStructure, exchanged by the Scene) or "not provided" (its default, or a forcing).

**Where a derived input is held.** At the receiving variable's location when it is registered; otherwise (an optional
input nothing registered) where the link's `scale=` says, else where the receiver declares it, on the view it works
on (a gas exchange at an anatomy-mode plant's SubOrgans: MECHA's Compartment water potential averaged onto them); the
two must agree. The sources' location is used only when neither says. No registration by hand in a plant model.

**Entities without contribution.** A mean (`"mean"`, `"weighted_mean"`) gives a receiving entity no source
contributes to (none of its own, or all their weights 0: an element without a liquid node in a mean weighted by
`liquid_volume`) the receiver's default, its standalone value (`missing="default"`, the default), or NaN
(`Translator().link(..., missing="nan")`, to see where). Within one DataStructure and across DataStructures alike
(a cell without roots in a mean towards the soil).

`openalea.metafspm.testing.assert_component_couplable(Component, translator)` checks a component against a
translator in a package's own tests.

**Units.** A declaration's unit follows one grammar: factors separated by spaces, signed integer exponents, `adim`
for dimensionless, what an amount is of in `unit_comment` (`unit="mol s-1", unit_comment="of hexose"`). Units are
read as quantities (`coupling.units.parse_unit`: SI dimensions and a scale; prefixes, `min`, `h`, `d`, `L`, `Pa`,
`°C`, `°`, `%`, ...). Each link is checked: the receiver's and the source's units must be the same quantity, and the
link's factor their conversion (`{"p_mpa": 1e4}` from MPa to hPa); `unit_check=False` on a link for a deliberate
factor (a share, a calibration). Undeclared units agree with anything; unreadable ones must be spelled alike.

### Across DataStructures: mappings and exchanges

Links between components on different DataStructures are exchanged by `coupling.cross.Exchanges` at fixed points,
through a mapping:

- `CrossMapping(plants, grid, method="barycentre" | "overlap")`: segments to cells (by middle point, or by the
  length fraction in each cell), rebuilt after growth or coordinate changes. On an anatomy-mode plant (its nodes the
  Compartments) it maps its SubOrgans, where the segments' ends are (`mapping.location`): a SubOrgan variable
  through its segment's cells; a Compartment variable (MECHA's root uptake, the soil's potential it receives)
  through its SubOrgan's cells, at its masked Compartments only (`mask="root_surface"`, a node mask: the roots'
  soil nodes; the others neither give nor receive): summed (or each one in a mean) up, broadcast or split by weight
  down;
- environment **scalars** (stored at `"scalar"`): population values reduced over every plant of every population,
  or a scalar broadcast to every node;
- `LayerMapping(column, grid, axis="z")`: a 1-D column and a 3-D grid, by layer overlaps;
- `UnionDataStructure(populations)` with `UnionMapping`: one component (a light model) over every population.

An exchange into a plant's DataStructure (the Scene's, before the population runs) reaches its components on views
of it too: those at its SubOrgans on an anatomy-mode plant (a gas exchange's PARa from the light's union), and
those on their entities.

Defaults follow the variables' kinds: extensive values are summed towards cells and split back by a `weight=`;
intensive values are averaged (with `weight=`) towards cells and gathered back. Several populations feeding one cell
variable are pooled in one write.

## 6. Scenes

A `Scene` runs plant populations and environment models together, in one process.

```python
from openalea.metafspm import Scene, SceneRecorder, planting, stand

table = planting(stand(xrange=0.3, yrange=0.3, sowing_density=250, row_spacing=0.15, seed=1), Wheat, scenario)
scene = Scene(table, environment=[RhizoSoil, LightModel], environment_scenarios=[soil_scenario, light_scenario],
              translator="scene_translator.py", time_step=3600, forcings=meteo,
              recorder=SceneRecorder("outputs", segment_plants=["p0"], frame_period=24, checkpoint_period=240))
scene.simulate(n_iterations=2500)

scenario = read_scenarios("scenarios.csv")["dry"]          # a scenario table's column, by model, with its forcings
scene = Scene.from_scenario(scenario, stand(...), Wheat, environment=[RhizoSoil, LightModel],
                            translator="scene_translator.py", time_step=3600, recorder="outputs")
```

- **Planting tables** come from one function, `planting(plants, model, scenario, plant_scenarios,
  emergence_times)`. *plants* are given (`[Planting(x, y, z, rotation), ...]`, positions in m, z minus the sowing
  depth) or sown: `stand(xrange, yrange, sowing_density, row_spacing, sowing_depth, models=, frequencies=, seed=)`
  lays out rows at a density, the stand's size adjusted to the rows, and with several models draws one per position
  (`planting(sown, sown.models, {Wheat: ..., Pea: ...})`). The plants are named `p0`, `p1`, ... in the table's
  order: those names are the `segment_plants=`, the frames' `entities=` and the summaries' `plant` column. A
  component placing its plants' elements declares the planting's variables it reads, `x: float =
  planting_variable("x")` (x, y, z, rotation: Plant-scale inputs the planting provides, `couplings()` listing them)
  and takes them as step arguments.
- **Plant models** are built once per model of the planting table, on an MPG holding all its plants:
  `Model(data_structure, time_step, <its own arguments>)`, with a class attribute `initiators` (the structural components
  building each plant) and a class attribute `translator` (its coupling contract: `couple()` uses it, and its links
  between initiators apply while they build each plant, a later initiator's linked inputs written before it builds;
  a link from a later initiator raises). Its `run()` calls its components in turn, or as one with
  `self.run_together(a, b)`: their steps in one schedule, each at its schedule position and the components in the
  given order within one (e.g. a development's potential lengths, the architecture's segmentation, the development's
  postsegmentation giving the new organs their lengths, then the architecture's postsegmentation). In anatomy mode (`nodes = "Compartment"`) the initiators build the anatomies, and a `wiring`
  attribute gives the junction rules between them. Numeric parameters may differ per plant (one scenario per plant); an `emergence_time` column
  keeps a plant frozen until then. Without a Scene, `population_data_structure(table, Model, time_step=)` builds
  the same: the plants' MPG and MPGDataStructure, the model on it, every plant's scenario applied
  (`population.data_structure`, `population.instance`).
- **Environment models** are built as `Model(plant_data_structures, xrange, yrange, time_step, <its own arguments>)` and
  create their own DataStructures (a grid, a union of the populations). `scene.environment.EnvironmentModel` is
  their base: `self.plant_data_structure()` (the populations' union, of one population too; an anatomy-mode
  population's elements its sections' owners; its variables reach the plant models through the scene translator,
  whose identity links are added by default for the outputs a plant model reads as inputs of the same name, and
  for the inputs a plant model holds as states of the same name, one per population), `self.build_grid(voxel,
  depth, adaptive=False)` (a grid over the scene's pattern, z down, periodic in x and y), a cadence (`interval` in s,
  or an overridden `due(time)`; at the other steps `between()`, e.g. `self.light.run_stage("state")`),
  `after_run()` (an adaptive grid refined), the time of the last full run kept by checkpoints. A light model and a
  soil model are short subclasses of it (Caribu's CaribuLightModel; the soil–plant–atmosphere example's Soil).
  Its `compartment` class attribute, `"Soil"` or `"Atmosphere"`, is required: the folder of its outputs.
- **The scene's DataStructures.** `scene.data_structures()` gives `{id: (DataStructure, model, components)}`:
  each population's (id: its model's name), then each environment model's own; the recorder's tables, the frames
  and the checkpoints walk the scene with it.
- **Scenarios** are flat dicts, `{name: value}`, applied by the Scene, so models do not forward them. A scenario's
  entries are parameters and settings of the model's components. They are set after the model is built, with
  the same effect as constructor keywords: a numeric parameter is written to the DataStructure (per plant for
  plant models, a plant without the entry keeping the default), and a non-numeric one goes to the component's
  field. Keys the model names in its `__init__` are its own arguments (e.g. `Soil(..., topsoil_K=1.5)`), and keys
  the initiators read (`parameters.get("n_segments")`) are theirs. Any other entry raises: a typo, or a parameter
  no component declares. A parameter not stored per plant must be the same for every plant.
- **A step** runs each environment model after the exchanges into it, then each population after the exchanges into
  it. Mappings between populations and grids or unions are inferred from the scene translator; others are given by
  `mappings=`.
- **Forcings, the time-varying inputs.** `forcings={name: forcing}`, or a DataFrame indexed by time (s) with one
  column each. A forcing is a number, a callable of the scene time (s), a Series indexed by time, or a `(times,
  values)` pair; tables are interpolated linearly; a callable of the time and the entity ids, `(time, ids) -> one
  value per entity`, varies in space (a soil temperature by depth). A component declaring the name
  (`forcing_variable`) reads it with `self.forcing(name)` (any component). Otherwise the Scene writes it at the start of each step into the input or parameter of that
  name, on every component declaring it and not coupled to another (a measured soil temperature, a shoot's
  supply; an input declared without a place is a plant's value, per plant). A forcing nothing takes raises. `as_forcing(value, time_unit=86400.)` (or `time_unit="d"`) turns an input given by day (a
  number, a function of the day, a Series indexed by day, one value per day) into one. `events=[(time, action)]`
  stay for one-off actions.
- **Services.** An environment model's `interval` (s) is its cadence (`due()`, and `between()` at the other
  steps), also its scenario's `"interval"`; a component reading the plants' geometry declares it,
  `geometry: object = geometry_input(relight="topology" | "geometry")`, listed by `couplings()`, and its model runs
  fully when the elements (or their triangles) changed too; `spin_up(scene)`, `stop_when=condition`.
- **A soil for tests and examples.** `openalea.metafspm.testing.soil.Soil` (not one of metafspm's models, a
  fixture): the steady water flow in a coarse fixed grid (5 cm cells), its conductivity by layer, the water table at
  the bottom, the surface evaporating; a plant model takes up water from it through scene links.
- **Calendar.** `Scene(start=(day_of_year, hour))`: the calendar of the scene's time 0 (UTC); every component's
  `self.calendar()` gives the (day of year, hour) of its forcing time (a light model's sun), and a forcing given by
  the hour of the day (or by the day of the year) takes it with `as_forcing(value, time_unit="h", origin="calendar")`.
- **Outputs: the SceneRecorder.** `recorder=SceneRecorder(folder, ...)` (or `recorder=folder`) writes:
  - `run.json`: status (running, finished, stopped, failed with its error), times, wall time per model, versions;
  - `scenario.json`: the planting, the scenarios, the environment models, the forcings (tables in `forcings/`);
  - `Plant/<Model>/summaries.csv`: one row per plant and step, its state variables summed (extensive) or averaged;
  - `Plant/<Model>/segments.csv`: the `segment_plants`' nodes every `segment_period` steps;
  - `Soil/<Model>/summaries.csv` and `Atmosphere/<Model>/summaries.csv`: each environment model's grid, by its
    `compartment` attribute;
  - your own tables, `tables={name: function(scene) -> row}`, as `<name>.csv`;
  - `frames/` (fspm-TF, every `frame_period` steps) and `checkpoints/step_<n>` (every `checkpoint_period`, the last
    `keep_checkpoints` kept).

  Each output is switched on its own (`summaries=False`, `frame_period=None`, ...); the initial state is recorded
  at the first step and the last one at the end. Without a folder the tables stay in memory, and
  `recorder.table(name)` reads any of them back. `Scene.simulate` sets the run's status; a loop of `scene.run()`
  ends with `scene.finish()` (a recording scene dropped without it warns).
- **Restarts.** `scene.checkpoint(path)` and `Scene.restore(path, **arguments)` continue a run bit for bit; models
  keep non-variable state through `checkpoint_state()` / `restore_state(state)`. A restored scene cuts its
  recorder's tables and frames back to the checkpoint's step, so a run that went past its checkpoint leaves no
  duplicate rows.
- **Scenario tables and variants** (`scene.scenarios`). `read_scenarios("scenarios.csv")` reads a table with one row
  per entry and one column per named scenario into `{name: scenario}`. Description columns (`type`, `unit`,
  `description`, ...) are skipped, empty cells leave the defaults, `forcing` rows read their CSV, and a `model`
  column nests the entries by model. `parameter_table(*components, main=)` lists the parameters with their units,
  descriptions and declared bounds. `variants(base, ranges, design="grid" | "one_at_a_time" | "sampling",
  declared=components)` builds scenarios within the declared `min_value` / `max_value`. `docs/tutorial.md` goes
  through the whole path.
- **Exchange with other platforms.** `openalea.metafspm.io.fspm_tf` writes a scene, or chosen DataStructures, as an
  [fspm-TF](https://github.com/openalea/fspm-TF) frame: one glTF file that CPlantBox, GroIMP or VirtualPlantLab can
  read, and that glTF viewers display. It holds each MPG scale as a table of vertices (ids, complex, parent, edge
  type), the variables and MPG properties as columns with their declared units, kinds and standard names, the soil
  grids, the mappings and the geometry: the DataStructure's triangles, and the shapes a component draws for display
  only (`display_triangles(where=None)`, e.g. GRANAP's sections), which light models never receive.
  `export_scene(path, scene)` or `export_frame(path, {id: ds},
  components={id: [...]})` writes a frame. `import_frame(path)` rebuilds the DataStructures: the same MPG, the
  same variables, bit for bit, from which a simulation continues as from the original. `SceneRecorder(folder,
  frame_period=24)` records a frame at the start and every 24 steps, in `folder/frames` with its index
  `scene.fspm.json`, which ParaView's fspm-TF reader plays as time steps. A frame is full by default: every scale,
  every row, every variable (what another platform reads, and a complete log), with the triangles light models get;
  the components' display shapes (e.g. GRANAP's sections), drawings only, are a recording choice:
  `display_period=96` writes them every 96 steps (a multiple of `frame_period`) and in the last frame, `None` (the
  default) never; their data rows (the Compartments) are in every frame either way, with the DataStructure's values
  (those of an instanced or reduced section's Compartments too, which the MPG holds as their template's).
  `frame_options=` passes
  `scene_to_frame`'s options (e.g. `geometry=False`). For logging, the options also filter what each frame keeps:
  `data_structures=` (which ones), `scales=` (e.g. `["Plant", "SubOrgan"]`), `entities=` (a plant, by vid or by its
  name in the planting table, an axis, a segment with its cross-section: with everything they contain) and
  `variables=`. Each is a value for every DataStructure or a `{id: value}` dict. A light log is such an explicit
  choice, e.g. `frame_options={"scales": {"Wheat": ["Plant", "Axis", "Organ", "SubOrgan"]}}` leaves out the
  anatomies' rows (the scales below SubOrgan) of population `Wheat`; never a default. A filtered frame is a
  smaller, valid fspm-TF file, for outputs only: `import_frame` refuses to rebuild it. It uses `openalea.fspm_tf`, a
  dependency of metafspm (in development, installed from the fspm-TF clone).
- **Frames ready to look at.** `SceneRecorder(folder, frame_period=24, frame_views="default")` also writes
  `frames/views.json` at the first frame: fspm-TF's views, the layers ParaView's fspm-TF reader shows, one block each,
  instead of one block per DataStructure. The default layers (`io.fspm_tf.default_views(scene)`) are, per population,
  its shoot mesh (where its SubOrgans are shoot elements) coloured by the first Organ variable its components declare
  (a state variable that is not a descriptor, else an input), its root segments as tubes on `radius` coloured by the
  first such SubOrgan variable, and each component's display shapes (e.g. sections, in the first frame with
  `display_period`) hidden; per environment model, its grid's cells coloured by its first cell variable, at opacity
  0.3. A part missing from the first frame (a shoot that appears later) gets no default layer. `frame_views=` also
  takes your own views, a dict or the path of a views file (e.g. one saved from ParaView): each name a layer gives
  (data structure, mesh, entity set, variable) is checked against the first frame, with a warning for each one it
  lacks; a layer is empty in the frames that lack what it shows. `write_views(path, views)` writes views, checked
  against fspm-TF's schema. In ParaView, the fspm-TF macros make each layer its own pipeline item and save your changes back, and "Select at scale" grows a picked
  element to its plant, axis, organ or segment with everything it contains (a segment's section shown even when the
  sections' layer is hidden), isolated in the view or in a panel of its own (fspm-TF's `doc/using.md`, "Views").
- **Running with another platform's model.** In an fspm-TF coupling session (fspm-TF's coordinator, `fspm-tf
  couple`, runs models of several platforms step by step), `openalea.metafspm.io.coupling.RemoteComponent` stands for
  the other platform's model: a component computing nothing, whose state variables are that model's outputs and whose
  inputs are sent to it. A subclass declares them as any component does, with `rights` (per scale: `"owns"`, metafspm
  creates the entities; `"refines"`, metafspm creates some, under the others' entities; `"follows"`, the other model's
  entities are created in the MPG, as are another refiner's in a scale metafspm refines) and `links` (renames,
  factors, changes of scale between the two models' variables). With `partition_weight` (e.g. a length the other
  model provides), metafspm shares its own variables over the other model's new nodes by StructuralComponent's
  repartition rules: an extensive amount split between a new segment and its parent, the total kept. `remote.start()` joins the session; each `remote()` sends the
  step's outputs and applies the next turn's inputs, so it runs last in a step. The coupling is explicit: metafspm's
  inputs at a step are what the other model gave at its previous turn. `test/exchange/test_coupling.py` couples the
  root growth model with a carbon model written in VirtualPlantLab (`vpl_carbon.jl`).

## 7. Upscaled variables, observations and calibration

A variable at a coarse scale that is the equivalent of finer ones (a root system's conductance from its cells'
walls and membranes) is an equation of the model, not a mapping: a step computes it. Its declaration says what it
summarises and under which protocol, so that observations made at that scale know which model variable they
measure:

```python
from openalea.metafspm.coupling.component import state_variable, upscaling
from openalea.metafspm.solve.network import equivalent_conductance

root_system_conductance: float = state_variable(
    unit="cm3 hPa-1 d-1", description="Krs", location="Plant",
    upscaling=upscaling(of=("kw", "kaqp", "kmb"), law="network",
                        protocol="uniform soil potential, collar fixed: Krs = Q_collar / (psi_soil - psi_collar)"))

@totalstate
def _root_system_conductance(self, conductance, soil_conductance, collar_kind):
    krs, inflow = equivalent_conductance(self.data_structure.to_graph_view(), conductance,
                                         sources=soil_conductance > 0, sinks=collar_kind == 1,
                                         source_conductance=soil_conductance)
    ...
```

- **`upscaling(of=, law=, protocol=, references=)`** is metadata (`law`: `"sum"`, `"mean"`, `"weighted_mean"`,
  `"series"` or `"network"`). `declarations()` lists it (column `upscaling`), the fspm-TF frames carry it.
- **`solve.network.equivalent_conductance(view, edge_conductance, sources, sinks, source_conductance=)`**: the
  conductance of a network between two node sets at a unit potential difference, and each source's inflow (one
  sparse solve; Robin sources with `source_conductance`). It has no physics: the model chooses the protocol.
- **Observations** (`scene.calibration.read_observations`): one row per measured value, `variable` and `value`,
  and optionally `table`, `plant`, `time` (s), `uncertainty`, `unit` and `experiment`. `compare(scene, observations)`
  reads each one's model value from the recorder (converted to the observation's unit from the declared one) and its
  residual; with several virtual experiments, `compare({experiment: scene, ...}, observations)`.
- **`objective(run, parameters, observations, declared=, scale=)`**: a callable of the parameters' values, its
  residuals, for an optimiser (`scipy.optimize.least_squares(cost, x0, bounds=cost.bounds)`): `run(values,
  experiment)` builds or updates one virtual experiment and returns its scene; the declared `min_value` /
  `max_value` bound the parameters. Calibrate on the measurement's protocol (a virtual experiment), not on a whole
  simulation, and check first which parameters the observations constrain (elasticities, a cost map).
- **Sensitivity analysis** stays in its tools: `from_samples(base, names, matrix, declared=)` turns a SALib sample into
  scenarios within the declared bounds, `design_table(scenarios)` lays them out to join with the outputs.

MECHA's developer tutorial 10 goes through all of it on a seedling's root system.

## 8. A complete example

`examples/soil_plant_atmosphere/` puts every feature above together, on water flow from a water table through the
soil and seedlings to a dry atmosphere:
- transport components for the plants' anatomy graph and for the soil grid, on shared flow equations;
- structural components building the seedlings at every scale, with their anatomies, and computing the conductances;
- boundary sets and boundary equations;
- translators within and across DataStructures, with a masked `CrossMapping` of the root surface;
- scenes with one plant and with a planted population;
- plots of the converged water potentials.

Its README describes each file and shows the figures.

## 9. Performance notes

- Steps, graph systems, tree kernels and exchanges work on whole-population arrays: a step of 1000 plants of 2 000
  segments takes about 2 s in one process.
- Growth bookkeeping is incremental: only new vertices are read from the MTG.
- An MPG-style step sees only the MTG properties its component declares; keep MTG reads in such steps to those.
- `split="components"` is not parallel; graph systems are solved by sparse LU (SuperLU).

### Compiled kernels (numba)

Loops that numpy cannot vectorise (a z-buffer, a tree walk, a disc of form factors) are compiled with numba. What the
ports learned:

- **Array arguments cost.** Each array passed to a compiled function, and each view taken in it (`a[i]`, a slice),
  updates reference counts with atomic operations. In a function called millions of times, or in threads sharing
  the same arrays, these atomics dominate (Caribu's radiosity near field ran 2.5× faster without them). In hot
  loops pass scalars and indices, index arrays element by element (`work[row, k]`), and gather small work arrays
  into one packed array allocated once (`workspace()` in Caribu's nearfield).
- **Options.** `@njit(cache=True, nogil=True)`: compiled once per environment, and releasing the GIL so that a
  `ThreadPoolExecutor` runs the calls in parallel (per source, per chunk of receivers), each thread on its own work
  arrays; results summed afterwards in a fixed order keep runs reproducible. `parallel=True` (`prange`) suits
  independent iterations writing to separate places.
- **Division.** numba raises on division by zero (Python's rule) unless `error_model="numpy"`: then IEEE results
  (`inf`, `nan`, `−0.0`), as C. Use it when porting C or C++ code, whose degenerate cases produce them.
- **Reproducing a legacy bit for bit.** Write float32 where the original used `float` (`np.float32(a * b)` rounds
  each operation), keep its order of operations, and its C library functions (`math.sqrt`, `math.acos`); C's
  `(int)` truncates towards zero (and gives INT_MIN for a NaN). Compare the outputs as text, as the original printed
  them.
- **The same code on a GPU.** A compiled function's Python source (`function.py_func`) can be compiled again as a
  CUDA device function (`numba.cuda.jit(device=True)`), so that a kernel calls the CPU code's functions and gives
  its results (Caribu's `gpu.py`); numba's CUDA simulator (`NUMBA_ENABLE_CUDASIM=1`, set before numba is imported)
  tests it without a GPU.
- **Measure before optimising.** Time the parts (the near field, the solve, the projections) on a real case; a
  laptop that mixes performance and efficient cores, or throttles in long runs, needs a run pinned to one core
  (`taskset -c 2`) for single-core comparisons.

## 10. Porting a model

The ports (GRANAP, MECHA, rhizodep, Adel, ArchiSimple, Caribu) share one approach: the original frozen in
`<package>.legacy` (or run in its own environment), the port reproducing it first, its behaviours then corrected
one by one, each correction named.

- **Comparisons.** `<package>/comparison/legacy_run.py` runs the legacy and `port_run.py` the port on the same
  inputs; `test/comparison/` asserts their outputs equal. `openalea.metafspm.testing` holds the shared pieces:
  `run_executables(steps, files, outputs)` (a legacy's executables in a temporary folder), `run_in_env(env, script)`
  (a legacy in its own conda environment), `assert_texts_equal` and `text_differences` (files row by row),
  `assert_arrays_equal` and `array_differences` (bitwise, or within tolerances: the count, the first difference,
  the largest gap), `worktree(repo, ref)` (an earlier version checked out apart, never by stash or reset).
- **Corrections.** `openalea.metafspm.testing.corrections.Corrections({name: description})`: `resolve("recommended" |
  "legacy" | names, precision)`, presets of the port's own, refused with the legacy's precision (the bit-for-bit
  reference); `table()` and `rst()` for the documentation.
- **Documentation.** Each port documents its components from their declarations: in its Sphinx `conf.py`,
  `extensions = [..., "openalea.metafspm.sphinx"]`, then `.. metafspm-component:: package.module.Component` renders
  its docstring, its variables by kind and its schedule (`openalea.metafspm.sphinx.component_rst(cls)` the same text).
  Beside it, the port's pages: a tutorial opening with the component formalism, `equations` (the model's equations
  and the functions computing them), `legacy` (the legacy's algorithms and a registry of its behaviours, each with
  what the port does) and `performance`.
