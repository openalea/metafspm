# API reference

## Package

```{eval-rst}
.. automodule:: openalea.metafspm
   :members: FunctionalComponent, StructuralComponent, CompositeModel, Scene, SceneRecorder, Summary, Compute,
             Planting, Stand, planting, stand, read_scenarios, variants, run_variants, parameter_table, as_forcing,
             pulse
```

## Components and declarations

```{eval-rst}
.. automodule:: openalea.metafspm.coupling.component
   :members: state_variable, input_variable, parameter, setting, forcing_variable, upscaling, Upscaling,
             DataStructureComponent, FunctionalComponent, StructuralComponent, NewNodes, NewComplexes, after_regrid,
             display_end

.. automodule:: openalea.metafspm.coupling.declaration
   :members: VariableSpec, DeclarationError, resolve_declaration, default_mapping, kinds_agree, scale_names,
             entity_set, EntitySets, rule_holds, template_selection
```

## Steps and graph systems

```{eval-rst}
.. automodule:: openalea.metafspm.solve.decorator
   :members: rate, state, totalrate, totalstate, stepinit, deficit, axial, potential, allocation, actual,
             segmentation, postsegmentation, graph_system, node_balance, node_rate, edge_law, boundary_condition,
             boundary_set, pool_balance, graph_output, graph_jacobian, explicit_term, input_jacobian, kind_derived

.. automodule:: openalea.metafspm.solve.network
   :members: equivalent_conductance, condensed_solve, kron_reduction, KronReduction

.. automodule:: openalea.metafspm.solve.reduction
   :members: Reduction, Relative, Kind, RecoveryTerms, display_ids, is_display_id

.. automodule:: openalea.metafspm.solve.linear
   :members: LinearSolver, SuperLU, SuperLUReuse, Cholmod, Cupy, Pardiso, Mumps, LINEAR_SOLVERS, OPTIONAL_SOLVERS,
             linear_solver, solve, held_rows, SymmetryError, block_threads, BlockThreads

.. automodule:: openalea.metafspm.solve.conservation
   :members: Conservation

.. automodule:: openalea.metafspm.coupling.choregrapher
   :members: Choregrapher, family_of
```

The solvers are chosen by key in `graph_system(solver=...)`: `"newton"`, `"newton_fd"`, `"explicit_euler"`,
`"scipy_krylov"`, `"scipy_anderson"`, `"scipy_hybr"`, `"scipy_ivp_bdf"`, `"scipy_ivp_radau"`.

## DataStructures

```{eval-rst}
.. automodule:: openalea.metafspm.data_structure.data_api
   :members: DataStructure, VariableStoreMixin, MTGDataStructure, MPGDataStructure, ArrayDataStructure, GraphView,
             register_grid_descriptors

.. automodule:: openalea.metafspm.data_structure.adaptive_grid
   :members: AdaptiveGridDataStructure

.. automodule:: openalea.metafspm.data_structure.mpg
   :members: MPG

.. automodule:: openalea.metafspm.data_structure.objects
   :members: ObjectRegistry

.. automodule:: openalea.metafspm.data_structure.instances
   :members: SectionTemplate, Instances, InstancedView

.. automodule:: openalea.metafspm.data_structure.tree_kernels
   :members:

.. automodule:: openalea.metafspm.kernels
   :members: warm_up, follow_parents, left_of, roll_right, down, turn_left, roll_to_vertical

.. automodule:: openalea.metafspm.compute
   :members: warm_up

.. automodule:: openalea.metafspm.backend
   :members: HostBackend, CupyBackend, HostFactor, DeviceFactor, make_backend, cupy_modules

.. automodule:: openalea.metafspm.data_structure.random_streams
   :members:

.. automodule:: openalea.metafspm.data_structure.configs
   :members: ScalesConfig, PropsConfig, LabelsConfig
```

## Coupling

```{eval-rst}
.. automodule:: openalea.metafspm.coupling.translator
   :members: Link, Translator, parse_factor

.. automodule:: openalea.metafspm.coupling.composite_wrapper
   :members: CompositeModel

.. automodule:: openalea.metafspm.coupling.cross
   :members: CrossMapping, LayerMapping, UnionDataStructure, UnionMapping, Exchanges

.. automodule:: openalea.metafspm.coupling.units
   :members: parse_unit, conversion_factor, canonical, units_agree, check_link_units, Equivalence,
             register_equivalence

.. automodule:: openalea.metafspm.coupling.schedule
   :members: Step, Together, ScheduledStep, conflicts

.. automodule:: openalea.metafspm.coupling.cadence
   :members: Cadence

.. automodule:: openalea.metafspm.coupling.durations
   :members: duration, start_date, instant, day_and_hour

.. automodule:: openalea.metafspm.coupling.iteration
   :members: FixedPoint

.. automodule:: openalea.metafspm.coupling.geometry
   :members:
```

## Scenes

```{eval-rst}
.. automodule:: openalea.metafspm.scene.population
   :members: Planting, Stand, planting, stand, Population, population_data_structure

.. automodule:: openalea.metafspm.scene.scenarios
   :members: read_scenarios, parameter_table, variants, from_samples, design_table, run_variants, variant_processes

.. automodule:: openalea.metafspm.scene.calibration
   :members: read_observations, compare, residuals, objective, Objective

.. automodule:: openalea.metafspm.coupling.forcings
   :members: as_forcing, pulse, forcing_value, forcing_mean, forcing_total, forcing_over, spatial_mean, quadrature,
             forcing_table, is_spatial

.. automodule:: openalea.metafspm.scene.scene
   :members: Scene, AllMasks

.. automodule:: openalea.metafspm.compute
   :members: Compute, WorkerPool, WorkerError, worker_state, in_worker, machine, processors, core_types

.. automodule:: openalea.metafspm.scene.environment
   :members: EnvironmentModel

.. automodule:: openalea.metafspm.scene.recorder
   :members: SceneRecorder, Summary, compartment, grouped, base_of

.. automodule:: openalea.metafspm.scene.shared
   :members: Shared, SharedStore, references, prune
```

## Exchange with other platforms

```{eval-rst}
.. automodule:: openalea.metafspm.io.encoding3d
   :members: to_frame, scene_to_frame, export_frame, export_scene, from_frame, import_frame, filter_frame, ucum_unit,
             default_views, write_views, fold_frame, unrecorded, resolved_variables, Excluding, view_columns,
             with_view_columns, filtered_view_columns

.. automodule:: openalea.metafspm.io.coupling
   :members: RemoteComponent
```

## Display

```{eval-rst}
.. automodule:: openalea.metafspm.display
   :members:
```

## Testing helpers

```{eval-rst}
.. automodule:: openalea.metafspm.testing
   :members: couplability_problems, assert_component_couplable

.. automodule:: openalea.metafspm.testing.comparison
   :members:

.. automodule:: openalea.metafspm.testing.corrections
   :members:
```
