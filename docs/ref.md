# API reference

## Package

```{eval-rst}
.. automodule:: openalea.metafspm
   :members: FunctionalComponent, StructuralComponent, CompositeModel, Scene, planting_table
```

## Components and declarations

```{eval-rst}
.. automodule:: openalea.metafspm.coupling.component
   :members: declare, state_variable, input_variable, parameter, DataStructureComponent, FunctionalComponent,
             StructuralComponent

.. automodule:: openalea.metafspm.coupling.declaration
   :members: VariableSpec, DeclarationError, resolve_declaration, default_mapping, kinds_agree
```

## Steps and graph systems

```{eval-rst}
.. automodule:: openalea.metafspm.solve.decorator
   :members: rate, state, totalrate, totalstate, stepinit, deficit, axial, potential, allocation, actual,
             segmentation, postsegmentation, priorbalance, selfbalance, graph_system, node_balance, edge_law,
             boundary_condition, boundary_set, pool_balance, graph_output, graph_jacobian

.. automodule:: openalea.metafspm.solve.solver
   :members: NewtonSolver, ImplicitEulerSolver, ExplicitEulerSolver, ScipyIVPSolver, ScipyRootSolver

.. automodule:: openalea.metafspm.coupling.choregrapher
   :members: Choregrapher, family_of
```

## DataStructures

```{eval-rst}
.. automodule:: openalea.metafspm.data_structure.data_api
   :members: VariableStoreMixin, MPGDataStructure, ArrayDataStructure, GraphView, BoundaryPort

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
   :members: CrossMapping, ScalarMapping, LayerMapping, UnionDataStructure, UnionMapping, Exchanges,
             cross_default_mapping
```

## Scenes

```{eval-rst}
.. automodule:: openalea.metafspm.scene.population
   :members: planting_table, stand_initialization, build_population, apply_plant_scenarios

.. automodule:: openalea.metafspm.scene.scene
   :members: Scene, SceneRecorder, Population, AllMasks, load_translator
```

## Testing helpers

```{eval-rst}
.. automodule:: openalea.metafspm.testing
   :members: couplability_problems, assert_component_couplable
```
