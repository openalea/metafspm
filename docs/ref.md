# API reference

## Package

```{eval-rst}
.. automodule:: openalea.metafspm
   :members: FunctionalComponent, StructuralComponent, CompositeModel, Scene, SceneRecorder, Planting, Stand,
             planting, stand, read_scenarios, variants, parameter_table, as_forcing
```

## Components and declarations

```{eval-rst}
.. automodule:: openalea.metafspm.coupling.component
   :members: state_variable, input_variable, parameter, setting, forcing_variable, DataStructureComponent,
             FunctionalComponent, StructuralComponent, NewNodes, NewComplexes

.. automodule:: openalea.metafspm.coupling.declaration
   :members: VariableSpec, DeclarationError, resolve_declaration, default_mapping, kinds_agree, scale_names
```

## Steps and graph systems

```{eval-rst}
.. automodule:: openalea.metafspm.solve.decorator
   :members: rate, state, totalrate, totalstate, stepinit, deficit, axial, potential, allocation, actual,
             segmentation, postsegmentation, graph_system, node_balance, node_rate, edge_law, boundary_condition,
             boundary_set, pool_balance, graph_output, graph_jacobian

.. automodule:: openalea.metafspm.coupling.choregrapher
   :members: Choregrapher, family_of
```

The solvers are chosen by key in `graph_system(solver=...)`: `"newton"`, `"newton_fd"`, `"explicit_euler"`,
`"scipy_krylov"`, `"scipy_anderson"`, `"scipy_hybr"`, `"scipy_ivp_bdf"`, `"scipy_ivp_radau"`.

## DataStructures

```{eval-rst}
.. automodule:: openalea.metafspm.data_structure.data_api
   :members: DataStructure, VariableStoreMixin, MTGDataStructure, MPGDataStructure, ArrayDataStructure, GraphView

.. automodule:: openalea.metafspm.data_structure.adaptive_grid
   :members: AdaptiveGridDataStructure

.. automodule:: openalea.metafspm.data_structure.mpg
   :members: MPG

.. automodule:: openalea.metafspm.data_structure.tree_kernels
   :members:

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
   :members: parse_unit, conversion_factor, canonical, units_agree, check_link_units

.. automodule:: openalea.metafspm.coupling.geometry
   :members:
```

## Scenes

```{eval-rst}
.. automodule:: openalea.metafspm.scene.population
   :members: Planting, Stand, planting, stand, Population, population_data_structure

.. automodule:: openalea.metafspm.scene.scenarios
   :members: read_scenarios, parameter_table, variants

.. automodule:: openalea.metafspm.coupling.forcings
   :members: as_forcing, forcing_value, forcing_table, is_spatial, seconds, time_unit_label

.. automodule:: openalea.metafspm.scene.scene
   :members: Scene, AllMasks

.. automodule:: openalea.metafspm.scene.environment
   :members: EnvironmentModel

.. automodule:: openalea.metafspm.scene.recorder
   :members: SceneRecorder, compartment
```

## Exchange with other platforms

```{eval-rst}
.. automodule:: openalea.metafspm.io.fspm_tf
   :members: to_frame, scene_to_frame, export_frame, export_scene, from_frame, import_frame, filter_frame, ucum_unit
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
