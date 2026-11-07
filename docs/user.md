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
  The arrays derived from the structure (node and edge ids, owners, the index maps) are patched from what changed,
  not rebuilt (conventions.md, "Performance rules"); `ds.edge_endpoints()` gives the edges as an (edges, 2) array.
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
`layer_mask(z=-1)` for boundary layers. Every grid holds its geometric descriptors as variables, for its components
to read as inputs (`by="grid"`; `data_api.GRID_DESCRIPTORS`): `cell_volume`, `cell_size_<axis>`,
`cell_section_<axis>` (the product of the other sizes: `cell_section_z` a cell's horizontal area),
`boundary_low_<axis>` and `boundary_high_<axis>` (1 in the first and last layer along a non-periodic axis: a soil's
surface and bottom, z pointing down) and, on the faces, `face_normal_<axis>` (1 across that axis). An adaptive grid
keeps neighbours within one level of each other and carries its variables over refinement conservatively
(extensive values split or summed, the others copied or averaged), sets its descriptors for the new cells, then
calls its components' carry-over hooks: a method decorated `@after_regrid` (`coupling.component`) sets again what
the volume averages do not carry, e.g. a soil's pressure head from the water carried (use case 1's
`RichardsSoil.carry_over`), once per refinement or coarsening.

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
  When the DataStructure defines an `"active"` mask (planting times, dead tissues), steps also compute on its entities
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
  canopy, incremental meshes); `self.cached(name, build, key)` for a value derived from the structure. Both caches
  are the component's `caches()` (with its graph systems' views and solutions): never checkpointed, all dropped
  when a checkpoint is loaded (`clear_caches()`).
- **Objects shared by content:** `ds.objects(name, saved=)` (`g.objects` on an MPG, for the initiators), a
  registry of the DataStructure (`data_structure.objects.ObjectRegistry`: `get(key, build)`, `index(key)`,
  `by_index(index)`), its objects keyed by their content (`solve.memo.content_key` of what they are made from), saved
  with it by the checkpoints, each once (a shared part, in the form `saved` gives; `saved=False`: a cache, not
  saved). No module-level registry: two Scenes in one process have their own, freed with them (GRANAP's anatomy
  templates, `g.objects("anatomy_templates")`).
- **Reading a component:** `Component.declarations()` lists its variables as a table (kind, scale, unit, description,
  default, provider); `Component.schedule()` its steps in the order they run, grouped by stage;
  `component.variable_specs` their resolved places on its DataStructure; `component.active_ids(location)` the active
  entities (at a coarse location: those of the plants planted), for MPG-style steps.
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
  `MPG.add_root_at_scale(scale)`; `self.graph_view` in graph-system terms; `ds.simulation_time_step = dt` for components
  run without a Scene (the Scene sets its time step on each of its DataStructures: Scenes of other steps run in one
  process; none: a component runs one of its own steps per call).
- **Outside steps** (structural code, helpers, harnesses): `self.parameter_value(name)` is a parameter's single
  value for the population (it raises when plants differ; `parameter_values(name, location)` gives one per entity);
  `self.cached(name, build, key=None)` keeps a value until *key* changes (by default the topology version), e.g. a
  per-node array read from the MTG's labels; `component.run_stage("postsegmentation")` runs the steps of one
  schedule row only (e.g. after another model wrote the inputs they read).
- **The MTG view.** By default (`mtg_sync = "lazy"`) the DataStructure is the reference. A state variable held at
  its own scale is not copied: the MTG reads its column. Mapped ones reach the MTG when it is read (`ds.mtg`,
  `ds.flush_mtg()`), before an MPG-style step. `mtg_sync = "after_call"` writes them after every call, `"never"`
  never. `ds.structure` is the MTG to read the structure only (labels, parents, complexes), without that
  synchronisation.
- **MTG properties** are read and written in bulk: `g.values_at(name, vids)`, `g.set_values(name, vids, values)`,
  `g.column(name)` (values and presence over the vid index), `g.view(name)` vertex by vertex. `g.property(name)` is a
  read-only mapping built on demand (openalea.mtg's API, for tests and legacy code): `g.property(name)[v] = x` raises,
  write `g.set_values(name, [v], [x])`.

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
  below that fraction of the first iterate's; 0 by default), `step_tolerance` (converged too when a step is below
  the unknowns' round-off, ||δx||∞ ≤ step_tolerance · max(1, ||x||∞); 1e-10 by default: conventions.md, Stopping
  criteria), `finite_difference_step`, `linesearch`; `transient=True` declares
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
- **Forcings.** `self.forcing(name)` reads a time series of the scene's forcings **over the (sub-)step** being
  computed, `[t, t + dt]` (`over="substep"`, the default; `over="step"`: the call's whole step, all its sub-steps), so
  that the results do not depend on the time step through the forcings; inside an IVP solve, at its evaluation time.
  A component declares the forcings it reads, `PARi: float = forcing_variable(unit="µmol m-2 s-1",
  description="...")`: unlike an input, written per entity by another model through the translator, a forcing is a
  time series given from outside, the same for every entity. It is listed by `declarations()` (role "forcing"), not
  stored on the DataStructure, and a `Scene` checks at its construction that every required one is given
  (`required=False`: an optional one, which raises KeyError when read). What a read gives is declared once,
  `forcing_variable(aggregate=...)`: `"mean"` (the default: its mean over the step, a rate or a state of the air),
  `"total"` (its integral, a rain's depth from its rate; not readable inside an IVP solve) or `"instant"` (its value
  at the step's end). The mean is exact for numbers, tables (piecewise linear) and pulses
  (`forcings.pulse("38 h", "39 h", total)`: a rate given by its total over `(start, end]`, a rain shower, whose mean is
  exact at any step), and the 4-point Gauss-Legendre rule on panels of at most an hour for other callables (a daily
  course read by the hour or by the day); `forcing_mean`, `forcing_total` and `forcing_over` compute them outside a
  component, `self.forcing_interval(over)` gives the interval read.
  Outside a Scene, `component.set_forcings(forcings, start=None)` gives a component the forcings a Scene would
  (`{name: forcing}` or a DataFrame indexed by time, the Scene's `start=` date), read at the component's clock
  (0 at its first call); `self.has_forcing(name)` tells whether one is given as a time series, e.g. for an input
  read from the forcing of its name at each sub-step when one is given.
- **Solvers.** `@graph_system(solver=...)` takes one of the keys `solve.solver.solver_names()` lists; a solver class
  of one's own (a subclass of `NewtonSolver`) is given a key with `solve.solver.register_solver(name,
  solver_class)`. The sparse solver of the Newton steps' linear systems is the run's (`Compute(linear_solver=)`,
  below; `solve.linear.LINEAR_SOLVERS`, a solver of one's own registered there by name).
  `@graph_system(symmetric=True)` declares the system's matrix symmetric (a conductance network: its held
  Dirichlet rows aside, a row holding its diagonal alone): checked once per matrix pattern (`solve.linear.
  SymmetryError` when it is not), and what lets `linear_solver="cholmod"` factorise it by Cholesky (MECHA's standard
  problem declares it; its plant problem cannot: the leaves' vapour path makes it unsymmetric once they transpire). A graph system's equations class, to call or re-solve its
  equations outside the component, is `Component._system.equations`.
- **Outputs and Jacobians.** `@graph_output(name, location=...)` derives outputs after the solve; `@graph_jacobian`
  gives an analytic Jacobian (finite differences with graph colouring otherwise).

- **Reductions.** A plant model may reduce its graph systems by groups (`solve.reduction.Reduction(location=, where=,
  ports=, keep=, explicit_edges=, kind=, inputs=, systems=, outputs=)`): the groups of a location (the SubOrgans
  `where=` selects: a filter of the entities' syntax, `{"label": "RootSegment"}`, `"<Scale>.label"` their complex's
  label, `{"label": "StemElement", "Organ.label": "StemInternode"}` the internodes' elements and not the sheaths', or
  a published entity set's name; all without) keep their ports and kept nodes (`ports=`, `keep=`: entity set names,
  `"GranapAnatomy.conduits"`, `{variable: condition}` rules on the template's properties, callables, or tuples of
  them), their interiors eliminated once per kind (a library population of
  the model's own components, each kind its group's template, kind values and the parameters of the components
  the library runs, objects included, digested by content; the library runs the components of the reduced systems and those giving them inputs
  at the cells (per-cell osmotic presets), not the others (a shoot's development, its tables)). It is the model's
  `reduction` class attribute, or a function of its scenario choosing one, or None: `reduction = staticmethod(lambda
  scenario: Reduction(...) if scenario.get("reduced") else None)`; the entries it reads are the model's, the same for
  every plant of a population. The entries it owns but may not read (the reduced labels while `reduced` is off) are
  its attribute `scenario_keys` (`choose.scenario_keys = ("reduced", "reduced_where")`): accepted, inactive. A graph
  system's terms the reduction must keep explicit are declared `@explicit_term`, its inputs' derivatives
  `@input_jacobian`. A structural model takes a group's reduced template from `reduction.template_for(full)`; a
  component reading a reduced group's full template (its volumes, its geometry) gets it from
  `reduction.full_template(key)`, the key of the reduced template its instance holds. A kind variable is exact
  (`kind={"barrier": None}`: a kind per value) or classed: `Relative(width, interpolate=True)` gives the classes
  (1 + width)^k, and a group between two classes their two kinds' blend, weighted linearly by its value's place
  between them (its D, e, inputs' terms, outputs and recovered interior; exact at the class values and for terms
  affine in the value, within width²/8 of the curvature between them: MECHA's sections within 10⁻⁶ of their own
  length's at width 0.1); `interpolate=False` takes the nearest class as it is. Classes are evaluated once each: a
  class kind no group has any more is kept while its family (its template, exact kind values and parameters) is in
  use, so a growing plant's kinds saturate; an exact kind no group has any more is released at the next solve (its
  arrays freed, evaluated again if a group comes back to it). The plants of one population share their kinds when
  those parameters are the same (plants differing in another component's only, their root draws say, share them). The missing kinds are evaluated in batches of at most `Compute.kind_batch`
  interior cells (the Scene's Compute, `Compute.current()`; the memory freed between batches), in one library
  population per plant parameters kept on the reduction (each batch's groups replacing the former's, its
  components reset to their built state); with `Compute.workers` allowing a pool, the batch's kinds are dealt over
  its processes, each its share in its own library (the same kinds, bit for bit). The kinds are checkpointed with the population (`Kind`, a shared part
  of `scene.shared`, saved once; never a factorisation): a resumed run evaluates none again. A value derived from a kind (a section's condensed volumes)
  is a method of the reduced system's component tagged `@kind_derived(name)`: `method(self, kind, full_template,
  kron, nodes)`, run once per kind on its library while the kind's factorisation is live (`kron.shares(targets,
  values)`, the interior's values shared among kept nodes by the recovery weights; `kron.recover`; no
  refactorisation), its value in `kind.outputs[name]`, released with the kind. Each kind's interior is factorised
  once: its reduced term and its inputs' injection (−Xᵀ for a symmetric operator, computed only when the reduction
  declares an input) come from that one factorisation;
  a kind keeps its term and the sparse blocks of its full interior, not their factorisation (made again for a
  recovery, `recover`), so the memory follows the plant's size, not the kinds it went through. For outputs,
  `reduction.recover_interiors(ds)` gives every reduced group's interior values at once (batched by kind, one
  factorisation each), `reduction.full_rows(key)` the full template's rows a reduced template holds and
  `reduction.unknown()` the reduced system's node unknown; a plant model's structural component asks
  `reduction.reduces(g, vertex)` for each group it instances (where= on the vertex as it is: its label, its
  complexes', its properties), and `reduction.is_reduced(template)` whether an instance's template is a reduced one.
  **Merged kept nodes** (`Reduction(merge=)`, an approximation chosen by the model): kept nodes lumped into one, a
  dict or a tuple of disjoint dicts `{"where": rule, "by": property names | callable, "properties": {name:
  aggregation}}`: `where` the kept nodes it applies to (a rule as `keep=`'s; None: all), `by` their grouping (the
  kept nodes of equal values of those template properties one node, a NaN not merged; or a callable template ->
  an integer label per vertex, < 0 not merged), `properties` the merged node's values of template properties from
  its members' (`"sum"`, `"mean"`, `"min"`, `"max"`, `"root_sum_square"`, or a picklable callable; the others its
  representative's, its first member's). The reduced template holds the representative only; a member's explicit
  edges join it (one between two members: dropped); the kind contracts its full group's matrix and sources (the
  members' rows and columns summed, PᵀAP, Pᵀc: the members held at one potential) before reducing it, exact when the
  members' potentials are equal in the full group. The model's live terms on a merged node are its representative's
  and those of the edges joined to it, so merge nodes without boundary sets of their own. A `@kind_derived` method
  sees the contracted nodes (`nodes`, the kron's order): `kind.contract(values)` sums per-Compartment values of the
  full template into them; `recover_interiors` gives a member its node's value. MECHA's
  `plant_reduction(leaf_ports="bundles", vapour_path="air_spaces")` lumps the shoot's conduits per vascular bundle
  and a leaf's evaporating walls per air space this way. These are options only; the defaults (`"conduits"`,
  `"cells"`: every conduit and evaporating wall its own node) hold at any scene size. `leaf_ports="bundles"` gives
  up each vessel's own axial path and its own junction to its successor section: a bundle's vessels are held at
  one potential and joined bundle to bundle. Measured on use case 1 (16 plants, day 8): MECHA's time per step 34 %
  less, the transpiration and uptake unchanged, and the leaves' water potential (which the stomata read) 0.36 %
  above on average, 0.54 % at most; both options together cut more (2 plants: the plant solve 60 % less) but put
  the leaves' water potential 3 % above (the use case's README).
- **Drawing a reduction's groups (any reduction, at any location).** Nothing here is specific to anatomies: a
  section's cells, a coarse soil cell's sub-cells, a leaf's reduced vein network or any group a `Reduction`
  eliminates are drawn the same way. A group's interior is a linear function of its kept values, so a frame stores
  each group as an *instance* of its template rather than its recovered interior (`display="instances"`, the
  default of `to_frame`, `scene_to_frame` and `SceneRecorder`; see *Exchange with other platforms* below). The
  model gives one hook, on the component that draws the groups for display (the one with `display_triangles`):

  ```python
  from openalea.metafspm.coupling.geometry import TemplateShapes, Triangles

  def display_instances(self, owners):            # owners: the reduced groups' vertices (the reduction's location)
      shapes = {full.key: Triangles(local_vertices, template_rows)}   # once per template, in its own coordinates
      return TemplateShapes(owners, shapes, shape_of=[full.key, ...], transforms=placements)   # (N, 4, 4)
  ```

  - `shapes[key]` is the group's shape in the template's own coordinates, each triangle owned by the row it shows in
    the group's full template (`reduction.full_template(key)`: a node of the reduced system, e.g. a protoplast's
    Compartment); a key is any hashable stable across frames (e.g. the full template's key), one shape per key;
  - `transforms[i]` places group *i* (template coordinates → the DataStructure's: any affine map, e.g. a section's
    centre and its frame, a voxel's corner and size); `TemplateShapes.placed()` gives the triangles placed, as
    `display_triangles` draws them (a test that both agree is cheap).

  The rest is the reduction's: `reduction.recovery_terms(ds)` gives each group's kept values and its kinds' weights
  (`RecoveryTerms`: a group between two classes of a `Relative` kind variable is the weighted sum of the two class
  kinds), and `kind.recovery_operator()` each kind's recovery as one linear operator M over its full template's
  Compartments, the unknown there being M · [x_kept; 1] (the identity on the kept nodes, A_II⁻¹(−A_IK x − c_I) on
  the interior, a merged member its node's value): dense, derived while the kind's factorisation is live
  (`Kind.recovery`) when it holds at most `solve.reduction.DISPLAY_DENSE_MAX` values (32768), else its sparse system
  (S, R with M = S⁻¹R, from the blocks the kind keeps anyway: no solve when written). A model without the hook gets
  its groups drawn whole as before (`display="full"`).
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
(`coupling.geometry.Triangles`: vertices in the scene's frame, each triangle's node index). Elements with ends
(`x1`, `y1`, `z1`, `x2`, `y2`, `z2` and `radius`) get slim cylinders by default (`coupling.geometry.segment_cylinders`:
a structural component's, and the DataStructure's when no component draws them); a component with its own shapes
(Adel's blades) overrides it.

**Positions.** `x1 ... z2` are topological: each element's base at its parent's end (a plant's first element at its
planting position or its anchor's end). An element drawn elsewhere gives where in its display ends, `display_x1 ...
display_z2` (declared with `display_end("x1")` ...; NaN: at its topological place): Adel's turtle draws a hidden part
on the visible ones, a seed point is drawn as its grain. The default cylinders and the readers of the drawing (an
anatomy's 3D placement, an element's height for a gas exchange) take the display ends where given
(`coupling.geometry.drawn_ends`). `ds.position(location, at="base" | "end")` gives each entity's position from its
elements' ends: its first element's base, or the end of its last element along its axis (a Phytomer's node, where
its nodal roots start); `ds.bearer(location)` gives that last element, where a model grows the axes an entity bears
(a root model its nodal roots, on the node's element: their base at its end); `ds.check_connectivity(tolerance=)`
lists the elements whose base is not at their parent's end. An initiator places its elements with the MPG's builders: `g.add_organ(plant, "StemInternode", kind="Shoot")`
(the Axis, GrowthUnit, Phytomer between) and `g.add_element("StemElement", parent=element, direction=(0, 0, 1),
length=0.01, radius=5e-4)` (its ends from its parent's end; `direction=(0, 0, 0)` a point; `display=(start, end)`),
reading its parameters with `cls.plant_parameters(scenario)` (the scenario over the declared defaults, checked
against their bounds and dtype). A light model computes on the triangles and gives its results back per element
(`Triangles.per_owner`). An environment model receives DataStructures, not components: `ds.triangles()` gives the
triangles of the structural component built on it (the one with its own `triangles()` first; set
`ds.geometry_provider` when several draw), and a `UnionDataStructure`'s concatenates its parts', in its node indices.
On an anatomy-mode DataStructure, the frames and the light models take the triangles of its view at its from_scale
(`ds.at("SubOrgan").triangles()`, the elements): a structural component working at a coarser scale (its finest
declaration at Organ) registers its geometry on `ds.at("Organ")`, never drawn nor lit, and warns at its construction;
declare one of its variables at the SubOrgan scale (or its `entities`) for its elements to be drawn.
`ds.triangles(where)` takes a node mask or a mask's name (`"planted"`: plants not yet planted left out; every node
when the DataStructure has no such mask). `Triangles.mean_per_owner(values, n)` averages per element by area.
A triangle may carry attributes (`Triangles(..., attributes={"green": shares})`: Adel's green share of each
triangle's area, 1 by default where a part does not carry it), kept through `select`, `clip` and `concatenate`.

**The soil surface and the geometry in a domain.** `Scene(soil_surface=0.)` is the one definition of the soil's
surface (m, the plants' frame): every DataStructure's `ds.soil_surface` (a component's `self.soil_surface`), the
cross mappings' grid top (`CrossMapping(soil_surface=)`: the grid's depth is `soil_surface - z`), and the MPG's
`g.soil_surface` while the initiators build (a crown placed below it). `ds.geometry` answers what of the elements
lies in a domain, vectorised: `"air"` and `"soil"` (the half spaces of the soil surface) or a `HalfSpace(level,
axis="z", above=True)`:

```python
ds.geometry.clipped("air", where="planted")                 # the triangles above the soil (Caribu lights them)
ds.geometry.fraction_in("air", location="SubOrgan", by="area", weight="green")   # each element's green share there
ds.geometry.fraction_in("soil", by="length")                # each segment's length fraction in the soil
ds.geometry.centre_in("air")                                # (n, 3): the centre of each element's part above it
```

`Triangles.clip(domain)` cuts the triangles at the plane (one or two triangles each, exact),
`Triangles.fraction_in(domain)` gives each one's area fraction inside, without a loop. A translator link takes such
quantities as sources: `FractionIn(domain, by=, weight=, empty=)` and `CentreIn(domain, axis=, by=, weight=)`
(`coupling.geometry`) are geometric variables the DataStructure derives (`ds.derive_geometry(name, query,
location=)`) and recomputes when the geometry changes (`Geometry.stamp`: the topology, the providers'
`geometry_stamp()`, by default their calls and their segment ends' writes):

```python
Translator().link("GasExchange", "green_area", "ShootArchitecture",
                  ("green_area", FractionIn("air", weight="green", empty=0.)), formula=np.multiply,
                  names={"green_area": "green_area_in_air"})        # CNW-Grass's emerged_links
```

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

Chains are declared with `ds.define_chain(name, edge_type="<")` or by a group and a rank variable, of the nodes
or of the entities of a coarse location: `ds.define_chain("phytomers", location="Phytomer", group="Axis",
rank="phytomer_rank")` chains each axis' phytomers by their rank (the group an owner at a coarser location, or a
variable), and the kernels then take and give one value per phytomer; `chain_gather(values, rank, chain=)` without
a `source` reads along each entity's own chain (use case 1's nodal root rule: the phytomer three ranks below a
leaf's, on its axis).

A component that reads values of its own last call (a rule interpolating between its calls) names them,
`kept_from_last_call = ("thermal_time", "haun_stage")`, and reads `self.last_call(name)`: the values at the end of its
last call, aligned by id (NaN before its first call and for entities new since), kept with its state (checkpoints).
With a schedule step's cadence, the last call is the cadence's, not the last scene step's (`ds.previous`: the step's
start).

## 5. Coupling components

### Within one DataStructure: `CompositeModel` and translators

A plant model is a `CompositeModel` whose components share one DataStructure, declared as its parts and run by its
schedule (`coupling.schedule`):

```python
class WheatPlant(CompositeModel):
    initiators = (WheatCrown, ScenarioShoot, RootArchitecture, GranapAnatomy)    # a nested model: its initiators
    parts = {"crown": WheatCrown, "shoot": ScenarioShoot, "rule": NodalRootEmergence, "roots": RootArchitecture,
             "anatomy": GranapAnatomy, "hydraulics": Part(MechaHydraulics, soil_boundary="node_dirichlet")}
    schedule = ("shoot", "rule", "roots", Step("anatomy", cadence="1 d"), "hydraulics")
    translator = PLANT_TRANSLATOR

    def __init__(self, data_structure, time_step):        # only what the model adds to its parts
        super().__init__(data_structure, time_step)
        data_structure.define_mask("root_surface", entities="GranapAnatomy.soil_nodes")
```

The parts (`{name: class}`, `Part(class, **keywords)` for constructor keywords) are built in order on the model's
DataStructure, each its attribute; a part may be a model (Adel's `ScenarioShoot` within the use case's plant): its
components become the model's, its schedule one step, its translator's links merged. The schedule's entries are
part names, a model's method names (a step over all its parts: `ScenarioShoot`'s `grow` loop, a soil's `adapt`),
`Together("development", "architecture")` (run_together: their step groups interleaved by schedule position) and
`Step(name, stages=("rate",), cadence=Cadence("4 h", at="start"), name=...)`: a component's stages only (a light's
lighting at its cadence, its absorbed PAR every step, two Steps of one part), its cadence (`coupling.cadence`).
Without a schedule, the parts run in their order (those without steps, an initiator alone, left out).
`model.steps()` gives the schedule bound, each step's `reads` and `writes` ((DataStructure, variable): the inputs as
stored, through aliases, derivations and private names; "topology" written by a structural component, "geometry"
read by a geometry input); `conflicts(a, b)` what keeps two steps apart; `Scene.schedule()` the scene's table, the
exchanges included. A translator states which component
provides each input: `Translator().link(receiver, variable, provider, {source: factor}, aggregation=, weight=,
formula=)`, or a nested YAML/Python dictionary. An input named as another component's variable needs no link: the
DataStructure is the coupling, both read the same array. Links state the rest, and become aliases (a rename) or
derived variables (factors, sums, formulas, aggregations between scales), computed when read.

```python
translator = "translator.py"          # the model's class attribute (Translator.load: a Translator, a dict, .py, YAML)
model.couplings()                     # how each input is coupled, one row each
```

An environment model builds its components on its own DataStructures in its `__init__` (`self.components = [...]`,
or `self.couple(a, b, translator=...)` to link them), its schedule naming them by attribute.

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
factor (a share, a calibration). Undeclared units agree with anything; unreadable ones must be spelled alike. Where
the quantity changes, the factor is a named physical equivalence (`coupling.units.Equivalence(name, value, unit)`,
e.g. `WATER_MOLAR_VOLUME`, 18e-3 cm3 mmol-1, and `RADIUS_OF_DIAMETER`, 0.5): `{"transpiration": WATER_MOLAR_VOLUME}`
from mmol s-1 to cm3 d-1 is checked and its number computed from the units (18e-3 × 86400); a YAML translator gives
it by name (`"water_molar_volume"`; `units.register_equivalence` names a new one).

### Across DataStructures: mappings and exchanges

Links between components on different DataStructures are exchanged by `coupling.cross.Exchanges` at fixed points,
through a mapping:

- `CrossMapping(plants, grid, method="barycentre" | "overlap", mask=, soil_surface=)`: segments to cells (by middle
  point, or by the length fraction in each cell), rebuilt after growth or coordinate changes. The Scene infers one
  from the scene translator's links: a link's `mapping=` (`Link(..., mapping={"mask": "root_surface", "method":
  "overlap"})`) gives its own, shared by the links of the same options (a population with planting times: its mask
  and `"planted"`); a CrossMapping given by `mappings=` replaces the inferred ones of its pair. The entities with a
  part outside the grid along its non-periodic axes (above the soil, below its bottom) are clipped into its border
  cells, counted (`mapping.outside()`, their indices) and warned of once. On an anatomy-mode plant (its nodes the
  Compartments) it maps its SubOrgans, where the segments' ends are (`mapping.location`): a SubOrgan variable
  through its segment's cells; a Compartment variable (MECHA's root uptake, the soil's potential it receives)
  through its SubOrgan's cells, at its masked Compartments only (`mask="root_surface"`, a node mask: the roots'
  soil nodes; the others neither give nor receive): summed (or each one in a mean) up, broadcast or split by weight
  down. A plant in a stand: the grid is the stand's pattern (the scene's `xrange` and `yrange`, its sides periodic),
  and `periodic=(True, True, False)` (the default) folds the pieces of a segment outside the grid's x and y into the
  cells of their periodic images, so that a root leaving the pattern takes up water where a neighbour's would come
  in, nothing dropped nor clamped into the side cells; z is clipped (a root below the grid in its bottom cells). The
  plants are drawn at their own positions (the frames: their real architectures, out of the pattern);
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

table = planting(stand(20, rows=2, row_spacing=0.15, sowing_density=250, seed=1), Wheat, scenario)
scene = Scene(table, environment=[RhizoSoil, LightModel], environment_scenarios=[soil_scenario, light_scenario],
              translator="scene_translator.py", time_step="1 h", start="2026-05-10 00:00", forcings=meteo,
              recorder=SceneRecorder("outputs", segment_plants=["p0"], frame_period="1 d", checkpoint_period="10 d"))
scene.simulate(n_iterations=2500)

scenario = read_scenarios("scenarios.csv")["dry"]          # a scenario table's column, by model, with its forcings
scene = Scene.from_scenario(scenario, stand(...), Wheat, environment=[RhizoSoil, LightModel],
                            translator="scene_translator.py", time_step="1 h", recorder="outputs")
```

**Times and durations carry their units** (`openalea.metafspm.coupling.durations`): a duration is `"12 h"`,
`"1 d 12 h"`, `"30 min"` (units `s`, `min`, `h`, `d`) or a number of seconds, the scene's one internal unit; a scene
time (an event's, a planting time) is a duration since time 0 or a date (`"2026-05-12 06:00"`, with the Scene's
`start=`); a date is ISO 8601, UTC. `duration()`, `instant()`, `start_date()` and `day_and_hour()` read them.

- **Planting tables** come from one function, `planting(plants, model, scenario, plant_scenarios,
  planting_times)`. *plants* are given (`[Planting(x, y, depth, rotation), ...]`, positions in m, the depth below
  the Scene's soil surface: the plant's z, its planting variable, is the surface minus it) or sown by the one stand
  builder, `stand(plants, rows=1, row_spacing=0.15, plant_spacing= | sowing_density=, depth=0.025, jitter=0.,
  rotation="random" | degrees, models=, frequencies=, seed=)`: the rows along x, `row_spacing` apart in y, the plants
  `plant_spacing` apart along their row (or 1 / (density × row spacing)), plant k of row r at ((k + 1/2) spacing,
  (r + 1/2) row spacing), the stand's pattern their extent, `jitter` a normal offset (m), a numeric `rotation` each
  plant turned by it from the one before, with several models one drawn per position (`planting(sown, sown.models,
  {Wheat: ..., Pea: ...})`). `planting_times` (one scene time per plant) keep each plant out of the scene until then
  (the mask `"planted"`). The plants are named `p0`, `p1`, ... in the table's
  order: those names are the `segment_plants=`, the frames' `entities=` and the summaries' `plant` column. A
  component placing its plants' elements declares the planting's variables it reads, `x: float =
  planting_variable("x")` (x, y, z, rotation: Plant-scale inputs the planting provides, `couplings()` listing them)
  and takes them as step arguments.
- **Plant models** are built once per population of the planting table (its `population` column, by default the
  model's class name), on an MPG holding all its plants:
  `Model(data_structure, time_step, <its own arguments>)`, with a class attribute `initiators` (the structural components
  building each plant, a nested model's in its place), its `parts` and `schedule` (section 5) and a class attribute
  `translator` (its coupling contract: its parts are coupled with it, and its links between initiators apply while
  they build each plant, a later initiator's linked inputs written before it builds; a link from a later initiator
  raises). `Together(a, b)` in its schedule runs two components' steps in one schedule, each at its schedule
  position and the components in the given order within one (e.g. a development's potential lengths, the
  architecture's segmentation, the development's postsegmentation giving the new organs their lengths, then the
  architecture's postsegmentation). In anatomy mode (`nodes = "Compartment"`) the initiators build the anatomies,
  and a `wiring` attribute gives the junction rules between them. Numeric parameters may differ per plant (one
  scenario per plant); a `planting_time` column keeps a plant out until then. Without a Scene, `population_data_structure(table, Model, time_step=)` builds
  the same: the plants' MPG and MPGDataStructure, the model on it, every plant's scenario applied
  (`population.data_structure`, `population.instance`).
- **Several populations of one model (intercropping).** `planting(plants, [Roots] * 4, {"deep": {...}, "shallow":
  {...}}, population=["deep", "shallow", "deep", "shallow"])`: two populations of one model, each its own MPG,
  model instance and outputs (`Plant/deep/summaries.csv`); a scenario may be given per population name. In the
  scene a component is named `"<population>.<Component>"` (an environment model's `"<EnvironmentModel>.<Component>"`):
  a translator link naming a class reaches every population's component of that class (each exchanging with the
  soil or the light), one naming `"deep.RootGrowth"` that one only; two populations' components are linked only by
  their scene names. `scene.model(key)` gives a model instance (a population's or an environment model, by name or
  class: `scene.model(Soil).grid`), `scene.population(key)` a Population (`scene.population("deep").data_structure`;
  none: the only one), raising when none or several match. Checkpoints, timing (`"deep.RootGrowth"`), conservation
  rows and the recorder's folders use the population names. `examples/intercrop/`: wheat and pea roots
  (ArchiSimple's, two parameter sets) in one soil.
- **Environment models** are built as `Model(plant_data_structures, xrange, yrange, time_step, <its own arguments>)` and
  create their own DataStructures (a grid, a union of the populations). `scene.environment.EnvironmentModel` is
  their base: `self.plant_data_structure()` (the populations' union, of one population too; an anatomy-mode
  population's elements its sections' owners; its variables reach the plant models through the scene translator,
  whose identity links are added by default for the outputs a plant model reads as inputs of the same name, and
  for the inputs a plant model holds as states of the same name, one per population), `self.build_grid(voxel,
  depth, adaptive=False)` (a grid over the scene's pattern, z down, periodic in x and y), its schedule (its
  components by attribute and its methods: a soil's `("soil", "adapt")`, an adaptive grid refined after each solve;
  a light's `(Step("light", stages="rate", cadence=Cadence("4 h", at="start")), Step("light", stages="state",
  name="absorbed_light"))`, lit every 4 h, its PARa every step). A light model and a soil model are short
  subclasses of it (Caribu's CaribuLightModel; the soil–plant–atmosphere example's Soil).
  Its `compartment` class attribute, `"Soil"` or `"Atmosphere"`, is required: the folder of its outputs.
- **The scene's DataStructures.** `scene.data_structures()` gives `{id: (DataStructure, model, components)}`:
  each population's (id: its name), then each environment model's own; the recorder's tables, the frames
  and the checkpoints walk the scene with it.
- **Scenarios** are flat dicts, `{name: value}`, applied by the Scene, so models do not forward them. A scenario's
  entries are parameters and settings of the model's components. They are set after the model is built, with
  the same effect as constructor keywords: a numeric parameter is written to the DataStructure (per plant for
  plant models, a plant without the entry keeping the default), and a non-numeric one goes to the component's
  field. Keys the model names in its `__init__` are its own arguments (e.g. `Soil(..., topsoil_K=1.5)`), and keys
  the initiators read (`parameters.get("n_segments")`) are theirs. Any other entry raises: a typo, or a parameter
  no component declares. A parameter not stored per plant must be the same for every plant.
- **A step** runs each environment model after the exchanges into it, then each population after the exchanges into
  it. Mappings between populations and grids or unions are inferred from the scene translator's links (their
  `mapping=` options); others are given by `mappings=`. The coupling is explicit by default (`coupling="explicit"`:
  a model reads what the models after it wrote at the step before, e.g. a soil the plants' uptake). A fixed point
  iterates chosen schedule steps of several models within the step instead: `Scene(...,
  coupling=FixedPoint(("Soil.water", "Wheat.hydraulics"), tolerance=1e-8, max_iterations=20))`
  (`coupling.iteration`): after the explicit pass each iterated step runs again from its start (its components'
  kept state and balances, the variables it writes), after the exchanges into its model, until what the steps write
  moves by less than the tolerance; `Scene.coupling_iterations` (and run.json) counts the iterations. An iterated
  step runs every step (no cadence) and its DataStructures keep their topology from its first run to the iteration
  (a step that grows or regrids after it, as use case 1's adaptive soil, is refused).
- **Forcings, the time-varying inputs.** `forcings={name: forcing}`, or a DataFrame indexed by time (s) with one
  column each. A forcing is a number, a callable of the scene time (s), a Series indexed by time, or a `(times,
  values)` pair; tables are interpolated linearly; a callable of the time and the entity ids, `(time, ids) -> one
  value per entity`, varies in space (a soil temperature by depth). A component declaring the name
  (`forcing_variable`) reads it with `self.forcing(name)` (any component), as declared. Otherwise the Scene writes
  its mean over the step (each entity's for a forcing varying in space) at the start of each step into the input or
  parameter of that
  name, on every component declaring it and not coupled to another (a measured soil temperature, a shoot's
  supply; an input declared without a place is a plant's value, per plant). A forcing nothing takes raises.
  `as_forcing(value, time_unit="d")` (a duration: `"h"`, `"15 min"`) turns an input given by day (a number, a
  function of the day, a Series indexed by day, one value per day) into one. `events=[("36 h", action)]` (a scene
  time) stay for one-off actions.
- **Services.** A component reading the plants' geometry declares it, `geometry: object =
  geometry_input(relight="topology" | "geometry")`, listed by `couplings()`: a step of it at a cadence at its start
  also runs when the elements (or their triangles) changed since its last call, covering the time to its interval's
  end; `spin_up(scene)`, `stop_when=condition`.
- **A soil for tests and examples.** `openalea.metafspm.testing.soil.Soil` (not one of metafspm's models, a
  fixture): the steady water flow in a coarse fixed grid (5 cm cells), its conductivity by layer, the water table at
  the bottom, the surface evaporating; a plant model takes up water from it through scene links.
- **The time step is the Scene's.** `Scene(time_step="1 h")` sets it on every DataStructure of the scene
  (`ds.simulation_time_step`, also `population_data_structure(time_step=)`'s), from which each component runs
  `simulation_time_step / time_step` sub-steps per call (`component.simulation_step()`); a component built without a
  time step runs one of the scene's steps. Nothing is process-wide: Scenes of other time steps run in one process,
  and the Choregrapher (the steps' schedule) holds no run state.
- **A step's cadence.** `Step("anatomy", cadence="1 d")` in a model's schedule, or `Cadence("3 h", at="start")`
  (`openalea.metafspm.coupling.cadence`); a model's scenario sets its steps' (`{"cadence": {"anatomy": "1 d"}}`, a
  duration keeping the declared `at`; a cadence of the scene's step or shorter is every step): the step runs once
  per interval (a whole number of the scene's steps), its call covering it: its clock at the interval's start, its
  step the interval (one step of it when it ran in the scene's step; its own finer sub-step sub-steps it), its
  forcings read over it. `at="end"` (the default): it runs at the interval's last step, the inputs it reads through
  its DataStructure their means over the interval's steps (each entity's over the steps it existed, aligned by id;
  descriptors and non-float inputs their values at the call); `at="start"`: at the first step, its inputs as they
  are then. Between its calls nothing runs and its outputs are held; MPG-style steps read the MTG as it is. The
  means accumulated are checkpointed; a `Together` step has one cadence. `scene.calls` counts the scene steps each
  component ran in (`"<Model>.<Component>"`, in run.json next to the timing). Caribu's light sets one (its lighting
  every 4 h); the use case lights every hour.
- **Calendar.** `Scene(start="2026-05-10 00:00")`: the date of the scene's time 0 (ISO 8601, UTC); every
  component's `self.calendar()` gives the (day of year, hour) of its forcing time (a light model's sun), and a forcing given by
  the hour of the day (or by the day of the year) takes it with `as_forcing(value, time_unit="h", origin="calendar")`.
- **Outputs: the SceneRecorder.** `recorder=SceneRecorder(folder, ...)` (or `recorder=folder`) writes:
  - `run.json`: status (running, finished, stopped, failed with its error), times, wall time per model and per
    component (`scene.timing`: `"<Model>"` and `"<Model>.<Component>"`; `scene.calls`, the calls each component ran;
    a component's calls and its steps run
    together with others, `Component.wall_time()`), versions;
  - `scenario.json`: the planting, the scenarios, the environment models, the forcings (tables in `forcings/`);
  - `Plant/<Model>/summaries.csv`: one row per plant and step, the float state variables of every component on the
    plants' DataStructure **and on its views** (a structural component's SubOrgans, Adel's or ArchiSimple's
    entities: each variable over the entities of the components declaring it), at every location (nodes, SubOrgans,
    Organs, ...; a light model's union of the plants too): extensive ones summed, intensive ones their mean
    weighted by their declared weight (`state_variable(kind="intensive", weight="green_area")`, a variable at the
    same location: a volume, a mass, an area), plain without one (a warning lists them once), Plant ones as they are;
    descriptors, the relations' variables (edges) and the entities without a value (NaN) left out;
  - `Plant/<Model>/segments.csv`: the `segment_plants`' nodes every `segment_period` steps;
  - `Soil/<Model>/summaries.csv` and `Atmosphere/<Model>/summaries.csv`: each environment model's grid, by its
    `compartment` attribute (intensive variables weighted by their declared weight, else by the cells' volumes);
  - your own tables, `tables={name: function(scene) -> row}`, as `<name>.csv`, or declared:
    `Summary(location, where={...}, sum=[...] or {name: factor}, mean=[...] or {name: weight}, min=[...], max=[...],
    per="scene" | "plant")`, e.g. the leaves' gas exchange `Summary("SubOrgan", where={"label": "LeafElement",
    "green_area": ">0"}, sum={"An": "green_area", "transpiration": None}, mean={"gs": "green_area"})`: the
    entities at the location where every condition holds (a variable, at that location or a coarser one; `"label"`;
    `"<Scale>.label"`, their owner's at a scale), the sums (of name × factor: an integral), the means (by their
    declared weight with a list), the extremes (`<name>_min`, `<name>_max`), the largest change since the step
    before (`max_change=[...]`, `<name>_max_change`: over the entities there at both, from the values the Scene
    keeps before each step, `ds.previous(name)`, aligned by id; `scene.keep_previous(ds, *names)` keeps others);
    `population=` a population's model name, or an environment model's DataStructure (its `scene.data_structures()`
    id: a soil grid's cells);
  - `conservation.csv`: the balances of the quantities the graph systems conserve (`graph_system(conserved=
    {"water": Conservation(amount=("water_content", "volume"), inflow="surface_inflow", outflow=("drainage",
    "plant_uptake"), unit="m3")})`): per step (the summaries' period), each one's inflow and outflow (their rates at
    the end of each accepted sub-step times its length, over the solved entities), the change of what is held and
    their residual, change − (inflow − outflow), the solver's tolerance for a conservative system
    (`component.conservation(reset=False)` reads it);
  - `frames/` (encoding3d, every `frame_period`) and `checkpoints/step_<n>` (every `checkpoint_period`, the last
    `keep_checkpoints` kept; their unchanging parts saved once in `checkpoints/shared`, below).

  Each output is switched on its own (`summaries=False`, `frame_period=None`, ...); the initial state is recorded
  at the first step and the last one at the end. Without a folder the tables stay in memory, and
  `recorder.table(name)` reads any of them back. `Scene.simulate` sets the run's status; a loop of `scene.run()`
  ends with `scene.finish()` (a recording scene dropped without it warns).
- **Restarts.** `scene.checkpoint(path)` and `Scene.restore(path, **arguments)` continue a run bit for bit; models
  keep non-variable state through `checkpoint_state()` / `restore_state(state)` (an external solver's state; the
  objects the components share by content are the DataStructures' `ds.objects`, saved with them).
  `Scene.load_checkpoint(path, scenario="checkpoint" | "current")` (and `Scene.restore(path, scenario=...,
  **arguments)`): with `"checkpoint"` (the default) the run goes on with the checkpoint's parameters and settings,
  whatever the scene was built with; with `"current"` the scene's own scenarios (its planting table's, its
  environment scenarios) are applied again after the load: a resumed run with a changed parameter (only the entries
  the scenarios give; the others keep the checkpoint's values). What does not change from one checkpoint to the
  next is saved once (`scene.shared`): a `checkpoint_state()` returns `Shared(name, value)` for such a value (at any
  depth), and the sections' templates (reduced or instanced) are shared the same way; each part is written once in
  the shared folder (`scene.checkpoint(path, shared=folder)`, by default `path/shared`; the SceneRecorder's
  `checkpoints/shared`) under its content's key, the checkpoint's `shared.json` lists those it refers to, and the
  recorder removes the parts no kept checkpoint refers to. A checkpoint is restored with its shared folder (a
  missing part raises). A restored scene cuts its recorder's tables
  and frames back to the checkpoint's step, so a run that went past its checkpoint leaves no duplicate rows. A scene
  rebuilt with a changed translator and resumed uses the changed links: a DataStructure its models coupled keeps
  their derived variables (formulas, factors, aggregations) on `load_checkpoint`, not the checkpoint's (a changed
  one recomputed from the loaded sources at its next read, a new one registered, a dropped one's variable left as
  saved); a changed alias raises (an alias has no values of its own). `DataStructure.restore` (no model) takes the
  checkpoint's.
- **Scenario tables and variants** (`scene.scenarios`). `read_scenarios("scenarios.csv")` reads a table with one row
  per entry and one column per named scenario into `{name: scenario}`. Description columns (`type`, `unit`,
  `description`, ...) are skipped, empty cells leave the defaults, `forcing` rows read their CSV, and a `model`
  column nests the entries by model. `parameter_table(*components, main=)` lists the parameters with their units,
  descriptions and declared bounds. `variants(base, ranges, design="grid" | "one_at_a_time" | "sampling",
  declared=components)` builds scenarios within the declared `min_value` / `max_value`. `docs/tutorial.md` goes
  through the whole path.
- **Exchange with other platforms.** `openalea.metafspm.io.encoding3d` writes a scene, or chosen DataStructures, as an
  [encoding3d](https://github.com/openalea/encoding3d) frame: one glTF file that CPlantBox, GroIMP or VirtualPlantLab can
  read, and that glTF viewers display. It holds each MPG scale as a table of vertices (ids, complex, parent, edge
  type), the variables and MPG properties as columns with their declared units, kinds and standard names, the soil
  grids, the mappings and the geometry: the DataStructure's triangles, and the shapes a component draws for display
  only (`display_triangles(where=None)`, e.g. GRANAP's sections), which light models never receive.
  `export_scene(path, scene)` or `export_frame(path, {id: ds},
  components={id: [...]})` writes a frame. `import_frame(path)` rebuilds the DataStructures: the same MPG, the
  same variables, bit for bit, from which a simulation continues as from the original. `SceneRecorder(folder,
  frame_period="1 d")` records a frame at the start and every day, in `folder/frames` with its index
  `scene.fspm.json`, which ParaView's encoding3d reader plays as time steps. A frame is full by default: every scale,
  every row, every variable (what another platform reads, and a complete log), with the triangles light models get;
  the components' display shapes (e.g. GRANAP's sections), drawings only, are a recording choice:
  `display_period="4 d"` writes them every 4 days (a multiple of `frame_period`) and in the last frame, `None` (the
  default) never, and `display_plants=("p0", "p3")` draws them for those plants only (a stand's frames bounded by
  the plants drawn; the others' reduced sections' interiors not recovered; `scene_to_frame(display_plants=)` too);
  their meshes share their vertices (indices); their data rows (the Compartments) are in every frame either way, with the DataStructure's values
  (those of an instanced or reduced section's Compartments too, which the MPG holds as their template's).
  **A reduction's groups in those frames** (`display=` of `SceneRecorder`, `to_frame` and `scene_to_frame`):
  - `"instances"` (the default): each reduced group an *instance* of its template (encoding3d 0.4's instance sets,
    one per component giving `display_instances`, named after it; see *Drawing a reduction's groups* above). What a
    frame stores per group is small: its owner (the group's vertex, e.g. its SubOrgan), its transform, its template's
    index, its kept nodes' values of the reduced system's unknown (then 1) and its kinds' indices and weights. What
    groups share is stored once, as *parts* in `frames/parts/<key>.glb` (the key: the SHA-256 of the part's bytes, a
    part written only when its key is new): each template (the shape, and the full template's nodes with their
    constant properties, e.g. `tissue`, those the frame's variables filter keeps) and each kind's recovery operator.
    The unknown on every node of the template (e.g. MECHA's `psi` on every protoplast) is a *derived field*, the
    weighted sum of the kinds' operators applied to the kept values, which the reader rebuilds for the instances it
    selects: `encoding3d.evaluate(frame, "psi", instances=rows)` in Python, ParaView's "encoding3d Select at scale"
    (or the reader's "Evaluate Derived Fields" for all). The values are those of `"full"` (exact: the same linear
    recovery). The index `scene.fspm.json` lists the parts its frames use (`"parts"`); the recorder removes the parts
    no listed frame refers to when it starts writing frames and when a run resumes from a checkpoint (its later
    frames dropped). `encoding3d validate frames/scene.fspm.json` checks every reference; a missing part is an error
    naming its key. A DataStructure whose display components have no `display_instances` is drawn as with `"full"`.
  - `"full"`: each group drawn whole in the component's display mesh: its triangles may be owned by id
    (`Triangles(vertices, owner, owner_ids)`, owner -1 and `owner_ids` the display ids of
    `solve.reduction.display_ids(group vertex, full template rows)`, e.g. GRANAP's interior cells), and the frame
    writes each reduced group's interior (its full template's Compartments, Cells and Layers the reduced one does not
    hold; not its Connections) as display rows of their entity sets, marked by the column `metafspm_display_only`,
    linked to their Cells, Layers and section (a selection by entity brings a section's interior with it), with the
    reduced system's unknown recovered (`Reduction.recover_interiors(ds)`, one factorisation per kind for all its
    groups, freed after), their template's properties, NaN otherwise. Every display frame then holds every value, as
    large as the full anatomy's.
  - `None`: the reduced groups not drawn (the other groups' shapes are).

  Instances and display rows are output only: `from_frame` leaves them out (the DataStructure stays reduced). In use
  case 1 (1 plant drawn, day 3) a display frame is 0.8 MB as instances against 33 MB whole, written in 0.3 s instead
  of 3 s once its parts exist (the first frame writes them: 33 MB, then a new kind's operator at a time); see
  `examples/use_case_1/README.md` for the day-8 and 2-plant numbers.
  **One file to share:** `SceneRecorder(pack_at_finish=True)` packs the frames folder into `frames.fspm.zip` when
  the run finishes (`pack_compression="deflate"`, the default, about a third of the folder for instances' frames;
  `"lzma"` smaller and slower; `"stored"`), keeping the folder (`recorder.pack()` packs it on demand; `encoding3d
  pack FOLDER OUT.fspm.zip` from the command line, `encoding3d unpack` back). ParaView's reader and
  `encoding3d.read_index`/`read` open the archive as the folder (`frames.fspm.zip/frame_000024.glb`).
  `frame_options=` passes
  `scene_to_frame`'s options (e.g. `geometry=False`). For logging, the options also filter what each frame keeps:
  `data_structures=` (which ones), `scales=` (e.g. `["Plant", "SubOrgan"]`), `entities=` (a plant, by vid or by its
  name in the planting table, an axis, a segment with its cross-section: with everything they contain) and
  `variables=`. Each is a value for every DataStructure or a `{id: value}` dict. A light log is such an explicit
  choice, e.g. `frame_options={"scales": {"Wheat": ["Plant", "Axis", "Organ", "SubOrgan"]}}` leaves out the
  anatomies' rows (the scales below SubOrgan) of population `Wheat`; never a default. `variables=` gives, per
  DataStructure, the variables kept in every table, or per table (an entity set, a relation or the scalars, by name;
  `"*"`: every table not named, the others unfiltered), e.g.
  `frame_options={"variables": {"Wheat": {"Compartment": ["psi", "tissue"], "SubOrgan": ["length", "radius"],
  "*": []}}}`: the Compartments' potential and tissue (a reduced section's interior rows too), the elements' length
  and radius, nothing else. A filtered table always keeps its structure (ids, labels, complex and parent links, edge
  types, a relation's ends, a grid cell's level and index), its segment ends and the display rows' marker; metafspm's
  own links (`metafspm_complex` ..., for `import_frame`) only when named. The filter names the data only: the
  recorder keeps besides what the frames' views select, colour and size their tubes by (your views, or the default
  views, those of the unfiltered frame, read at the first frame: `SceneRecorder.frame_variables(scene, views)`,
  `io.encoding3d.with_view_columns`; `filtered_view_columns(views, variables)` lists what a filter alone would drop).
  `variables="recorded"` (as a whole, or as a table's value) keeps the variables the declarations record: each
  declaration's `record=` (`state_variable(..., record=False)` for an intermediate of no use to look at), by default
  the state variables and inputs, not the parameters, settings or a grid's geometric descriptors
  (`io.encoding3d.unrecorded`). A filtered frame is a smaller, valid encoding3d file, for outputs only:
  `import_frame` refuses to rebuild it.
  **Folded for display:** `frame_options={"fold": True}` (`scene_to_frame(fold=True)`, `io.encoding3d.fold_frame`)
  draws the plants folded into the scene's periodic pattern (`(0, xrange) × (0, yrange)`, where its light and its
  soil repeat): each mesh triangle, segment (the roots' tubes) and instance moved by the whole periods that bring its
  centre into the pattern, so that a leaf or a root beyond the stand is drawn where the periodic scene has it. A
  folded frame is for display only (its ends are no longer the plants'): `import_frame` refuses it. It uses `openalea.encoding3d`, a
  dependency of metafspm (in development, installed from the encoding3d clone).
- **Frames ready to look at.** `SceneRecorder(folder, frame_period="1 d", frame_views="default")` also writes
  `frames/views.json` at the first frame: encoding3d's views, the layers ParaView's encoding3d reader shows, one block each,
  instead of one block per DataStructure. The default layers (`io.encoding3d.default_views(scene)`) are, per population,
  its shoot mesh (where its SubOrgans are shoot elements) coloured by the first Organ variable its components declare
  (a state variable that is not a descriptor, else an input), its root segments as tubes on `radius` coloured by the
  first such SubOrgan variable, each component's display shapes (e.g. sections, in the first frame with
  `display_period`) hidden, and each instance set ("<population> <component> instances": a reduction's groups as
  instances) hidden, coloured by its derived field (the reduced system's unknown, `SECTIONS_COLORMAP`, "Blue Orange
  (divergent)": its colours appear once the instances are selected at a scale or evaluated); per environment model, its grid's cells coloured by its first cell variable, at opacity
  0.3. Each part has its own colour map (ParaView presets: the shoot `SHOOT_COLORMAP`, "Cool to Warm"; the roots
  `ROOTS_COLORMAP`, "Inferno (matplotlib)"; the grids `GRID_COLORMAP`, "Viridis (matplotlib)", in `io.encoding3d`). A part missing from the first frame (a shoot that appears later) gets no default layer. `frame_views=` also
  takes your own views, a dict or the path of a views file (e.g. one saved from ParaView): each name a layer gives
  (data structure, mesh, entity set, variable) is checked against the first frame, with a warning for each one it
  lacks; a layer is empty in the frames that lack what it shows. `write_views(path, views)` writes views, checked
  against encoding3d's schema. In ParaView, encoding3d's plugin (`paraview/plugin_encoding3d.py`, no macros) draws
  the default configuration at Apply, each layer with its own colour map and opacity (the soil translucent), edited
  layer by layer in View → MultiBlock Inspector and saved back with the reader's "Save Views" button; its filter
  "Select at scale" grows a picked element to its plant, axis, organ or segment with everything it contains (a
  segment's section shown even when the sections' layer is hidden), isolated in the view or in a panel of its own
  (encoding3d's `paraview/README.md`, the how-to). On a frame whose reduced groups are instances, Select at scale
  also rebuilds the derived fields of the instances it picks (e.g. a segment's section coloured by `psi` on every
  cell), from the frame's parts, without evaluating the others; `pvbatch encoding3d/paraview/check_in_paraview.py
  --instances` checks the plugin by hand.
- **Running with another platform's model.** In an encoding3d coupling session (encoding3d's coordinator, `encoding3d
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
  `"series"` or `"network"`). `declarations()` lists it (column `upscaling`), the encoding3d frames carry it.
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
- A reduction's groups are arrays (`Reduction.group_table`, `assignment`): its terms, outputs (`outputs`, a value per
  group; `kept_outputs`, a value per kept node) and recoveries batched by template and kind, the terms' sparse
  pattern cached per topology (80k groups: 0.03 s per solve's terms).
- The DataStructure's topology arrays (`entity_ids`, `owner`) are cached until the topology changes and read-only:
  copy one (`np.array(ds.owner("SubOrgan"))`) before writing into it.
- Growth bookkeeping is incremental: only new vertices are read from the MTG.
- An MPG-style step sees only the MTG properties its component declares; keep MTG reads in such steps to those.
- `split="components"` solves its pieces one after the other; a whole system made of independent blocks (the
  plants of a population) has its blocks solved apart, in `Compute.threads` threads (solve.linear, below); graph systems are solved
  by sparse LU (SuperLU) unless `Compute(linear_solver=)` says otherwise.

### The run configuration: `Compute`

`Scene(..., compute=Compute(threads=1, workers="auto", device="cpu", memory="trim"))` (`Compute()` when omitted)
says how the scene uses the machine:

- `threads`: the threads of the scene's parallel work while it builds and runs: numba's kernels
  (`numba.set_num_threads`) and the independent blocks of a linear solve (in threads, each block factorised in one;
  SuperLU's at most `solve.linear.SUPERLU_BLOCK_THREADS`, 4: its factorisations are bound by the memory bandwidth and
  get slower beyond on a laptop). BLAS and OpenMP stay at one thread within the scene, whatever `threads`
  (`compute.BLAS_THREADS`, through `threadpoolctl`; numpy outside the scene keeps its own): a threaded BLAS changes
  the results' last bits with its thread count (up to 1.5e-11 relative on use case 1's potentials) and only
  oversubscribes the step's small dense work. The results are the same, bit for bit, at any number of threads: the
  kernels' chunks and the blocks come from the work only. 1 by default (the blocks then solved in sequence).
- `workers`: the process pool of the scene's parallel work: a reduction's missing kinds. `"auto"` (default) starts none until a batch of kinds is worth it (more than
  about a second of work, estimated from their interior cells, and at least two kinds), then
  `min(processors // 2, kinds)` processes, one processor left free, kept for the run and grown to the largest
  batch's need; an integer forces it from the first batch, `1` sequential and in the process. The results are the
  same, bit for bit, whatever the number (the kinds dealt by their interior cells, each process evaluating its
  share in its own library population, the results added in the requests' order). The processes are forked on
  Linux (they inherit the scene's state at their start, its templates shared with the main process: when new
  templates appear, the pool is forked again rather than sent them, each process would hold its own copy) and spawned on
  macOS and Windows (each receives the reduction at its first task: the plant model class must be importable by
  name, a module-level class, and the reduction's `merge=` rules picklable). Each holds its own libraries: a pool
  of n adds about n library populations to the memory (use case 1, 2 plants, day 1: 1.4 → 2.1 GB for 4 processes);
  `scene.finish()` (and a scene collected) stops them, `run.json` records the pool used (`"compute": {"pool":
  {"workers": n, "start_method": "fork"}}`). A component's setting named `workers` that its scenario does not give
  takes the Compute's (Caribu's projection threads).
- `linear_solver`: the sparse solver of the graph systems' Newton steps (`solve.linear`): `"superlu"` (default,
  scipy's SuperLU with the MMD ordering); `"superlu_reuse"`, the same solution bit for bit with its column ordering
  kept per system while the matrix's pattern does not change, and its factorisation while the matrix itself does
  not (a linear system's second Newton iteration), at the memory of one factorisation kept per system (measured
  on use case 1, 2 plants to day 8: no gain, the plant solve being a small share of the step, and the process's
  memory growing 0.6 GB a day more than with "superlu" although the caches hold 66 MB: not advised there);
  `"cupy"` (the default with `device="cuda"`, not allowed on the cpu: [Running on a GPU](gpu.md));
  `"cholmod"` (scikit-sparse, optional: `pip install scikit-sparse` or `conda install -c conda-forge
  scikit-sparse`), CHOLMOD's Cholesky for the systems declared `@graph_system(symmetric=True)`, its symbolic
  analysis kept while the pattern does not change, the other systems SuperLU's; `"pardiso"` (MKL's PARDISO through
  pypardiso, optional, x86 only: `pip install "openalea.metafspm[pardiso]"` or `pip install pypardiso`) and
  `"mumps"` (MUMPS through python-mumps, optional: `conda install -c conda-forge python-mumps`), registered only when
  their package is importable, each keeping its analysis (ordering, symbolic factorisation) per block while the
  pattern does not change; PARDISO's blocks in `threads` threads, MUMPS's in sequence (it is not thread-safe). A
  missing optional package raises when the Scene is built (or at the `Compute` for an unregistered one), with its
  install line. SuperLU stays the default: on use case 1 the plants grow every hour, so every step brings a new
  pattern and an analysis is not reused, where SuperLU is the fastest (8 plants, day 13: 0.74 s at 1 thread, 0.35
  at 4; PARDISO 1.43 and 0.60 with its analysis; MUMPS 1.49); PARDISO pays when the pattern stays (0.53 s at 1
  thread, 0.16 at 8, without its analysis). The optional solvers' solutions differ from SuperLU's by up to 3e-12
  relative a solve (another pivoting, the same residual; 8e-10 on the potentials over 3 days at 2 plants), the
  same at any thread count. A matrix of independent diagonal
  blocks of at least `solve.linear.BLOCKS_MIN_NONZEROS` (200k) entries is solved block by block (the blocks found
  once per pattern, each block's matrix gathered from the whole one's values), the blocks in threads per `threads`;
  the same blocks whatever the threads, so the same results.
- `device`: `"cpu"` (default) or `"cuda"`: the array backend of the heavy linear algebra (`openalea.metafspm.backend`:
  numpy/scipy or CuPy). On `"cuda"`, the kinds' Kron reductions solve their many right-hand sides on the GPU (SuperLU's
  factors copied once), the linear solves go to `linear_solver="cupy"` (its default there: a system of at least
  100k entries solved whole on the device, cuDSS when nvmath-python is installed), and the components' setting named
  `device` defaults to it (Caribu's near field). CuPy is optional: a missing one, or no CUDA device, raises when the
  Scene is built, with the install line; nothing imports it otherwise. The transfers are counted in `run.json`
  (`"compute": {"device": ...}`). What moves, when it pays and how to check it: [Running on a GPU](gpu.md).
- `memory="trim"`: glibc's `malloc_trim(0)` after each step, the memory freed given back (use case 1 at day 8: 4.7 →
  2.5 GB); `"keep"`: nothing. The allocator's arenas are an environment variable read when the process starts:
  `MALLOC_ARENA_MAX=2` (0.6 GB less at the peak, no time cost), set in the examples' `environment.yaml`
  (`variables:`) or in the shell (`export MALLOC_ARENA_MAX=2`).
- `kind_batch=200_000`: the most interior cells a reduction evaluates at once (its missing kinds in batches, the
  memory freed between them: a blade's kind is about 10k cells, a stem's 45k).

Within the scene's build and steps, `Compute.current()` is the scene's Compute (what runs there reads it without
being given it: a reduction's batches); outside, a default `Compute()`. `scenario.json` records the Compute, `run.json` the threads in use and the pool.

### Repeated runs: `run_variants`

A sensitivity analysis or a calibration runs many scenes that do not depend on each other. `run_variants(run,
scenarios, workers="auto", compute=Compute(workers=1))` (`openalea.metafspm.scene.scenarios`) runs one per scenario
in a process pool: *run* is a module-level function `run(name, scenario, compute)` that builds its scene with
`compute=compute`, runs and finishes it, and returns what is kept (picklable: a table, a path); the results come back
in the scenarios' order. Each variant has a fresh process (its memory given back when it ends), forked on Linux and
spawned on macOS, Windows or with `device="cuda"`; `workers=1` runs them one after the other in the process. The
processes stay within the processors (`variant_processes`): `"auto"` is processors − 1, at most one per variant; with
several variant processes, a scene's `workers="auto"` becomes 1 (the variants are the parallel work); an integer is
kept when `workers × Compute.workers` and `workers × Compute.threads` fit the processors, else it raises. On a laptop:
`workers="auto"` and scenes of `Compute(workers=1)`; on a many-core workstation, k variants of a pool of w each
(k × w ≤ processors); on a GPU workstation, one or two variants on the device ([Running on a GPU](gpu.md)).

### Compiled kernels (numba)

Loops that numpy cannot vectorise (a z-buffer, a tree walk, a disc of form factors) are compiled with numba, each
`@njit(cache=True)` in its package's `kernels.py` (`openalea.metafspm.kernels`, Adel's, Caribu's; conventions.md,
"Performance rules"). They compile at their first call, about a minute and a half for metafspm and the ports together,
once per machine: `python -m openalea.metafspm warm_up` (or `compute.warm_up()`) does it at install, running each
package's `kernels.warm_up` on tiny inputs (a port registers its own under the entry points `openalea.metafspm.warm_up`
of its `pyproject.toml`: `"caribu" = "openalea.caribu.kernels:warm_up"`). What the ports learned:

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
