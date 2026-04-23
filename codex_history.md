I am trying to think how to generalize metaFSPM's component_factory module. If you look at Root-CyNAPS use case, MetaFSPM is used so that each process manages its own actualization and iteration on the data structure to apply its outputs. This works well in Root-CyNAPS with explicit Euleur since each process is managed independantly, and each node of the mtg updates its states independantly, so it is vectorizable, but it suffers from several limits: since each process orchestrates its own resolution, we never tried to gather the proper differential equation system resolved to alternate different resolution strategies between implicit and explicit time discretization. 2) point is that each time there is an axial dependancy, you will see that the methods become monolithic to build their own matrices and resolution: I am trying to generalize a scale-agnostic edge and node adjacency structure within the MTG that could be operated by different types of solvers (see root-cynaps water and nitrogen axial transports as a starting point), but I need clues of how I could generalize this in metaFSPM. To illustrate this, at most write tests do not change the code elsewhere.

Show more
5:09 PM



31 previous messages
I only added tests, in test_solver_adjacency_contract.py (line 1). They characterize the solver-facing MTG kernel that both root_water.py (line 667) and root_nitrogen.py (line 1122) currently rebuild inside their axial methods: focus-local indexing, parent/child edge extraction, grouped children access, parent reductions, and the distinction between a true root and a truncated boundary when the focus set is only a subgraph. I did not change production code elsewhere.

I also verified the new tests by executing the three test functions directly in the wheat-bridges conda env. I could not run pytest itself because it is not installed in the available environments.

The main design clue is that component_factory is currently only a scheduler/executor, not a system assembler. In component_factory.py (line 47), a Functor either loops nodewise or runs a whole-model method, and in component_factory.py (line 149) axial is just another schedule bucket. That is why each axial process becomes monolithic: the process owns topology extraction, operator assembly, boundary handling, solve, and write-back all at once.

If you want to generalize this cleanly, I would separate it into four layers:

TopologyView or SolverAdjacency.
It should be built once from MTG + scale + focus_elements and expose exactly the contract captured in the new test: focus_vids, parent_vid, has_parent, parent_idx, children, parents, grouped-children offsets, is_root, is_tip, and a detached-boundary mask.

StateLayout.
This maps unknowns to vector blocks. Water is a 2-field node system, nitrogen is a 1-field node system reused over several solutes. The generic abstraction should not assume one scalar unknown per node.

Operator contributors.
Processes should contribute node storage terms, node source/sink terms, edge flux laws, and boundary conditions. That keeps the biology in the component and moves the sparse-matrix plumbing out.

Solver policy.
Explicit Euler, backward Euler, Newton, or split explicit/implicit should consume the assembled operators, not be reimplemented inside each biological process.

The most important subtlety the tests highlight is this: “no parent in the active system” is not the same thing as “root”. For subgraphs or multiscale views, a node can have a parent in the MTG but not in the current focus set. That needs to be explicit in the shared adjacency object, otherwise boundary conditions get mixed up.

So I would not try to generalize all of component_factory first. I would first extract the shared graph kernel into a small reusable object, then add a second abstraction for system assembly on top of the existing scheduler. That gives you a path where current explicit nodewise processes keep working, while axial processes can progressively move toward interchangeable solvers.




5:14 PM
1 file changed
Undo

Review
metafspm/test/test_solver_adjacency_contract.py
MTG is used to represent growing graphs, how does it adjusts our strategy, do we need to rebuild the adjacency and its valeurs propres every time or can we do it iteratively?
6:05 AM



4 previous messages
No, not every time. With a growing MTG, the useful split is:

topology changed:
A node was added, removed, or reparented by growth/segmentation.
active set changed:
focus_elements changed, even if the MTG itself did not.
coefficients changed:
Same graph, but conductances, fluxes, volumes, temperatures, etc. changed.
In your current code, the axial kernels in root_water.py (line 667) and root_nitrogen.py (line 1122) rebuild the local adjacency every call. That is correct but more expensive than necessary.

What I would do instead is:

Rebuild adjacency only when topology or focus_elements changed.
Reuse the cached adjacency, grouped children offsets, and sparse pattern when only coefficients changed.
Reassemble only matrix values and RHS at each solve.
That fits MTG growth well, because growth is event-like, not continuous. In fact your structural update code already has natural invalidation points:
mtg_structural_init.py (line 421) refreshes vertex_index and parent_id, and mtg_structural_init.py (line 786) recomputes focus_elements. So a cached SolverAdjacency can just be invalidated there.

About valeurs propres: usually, no, you should not recompute them routinely.

Your current water and nitrogen solvers use sparse LU, not eigendecomposition, so eigenvalues are not part of the solve path.
If later you use an explicit scheme, the relevant quantity is the spectral radius of the transport operator or Jacobian, not the plain MTG adjacency.
Even then, do not do a full eigendecomposition each step. Use a warm-started power iteration, and only refresh it after a topology change or a strong coefficient change.
So the practical strategy is:

Cache topology view:
focus_vids, parent_idx, children, parents, CSR-like child grouping, root/boundary masks.
Cache pattern view:
row, col, block layout of the sparse matrix.
Rebuild those caches only after segmentation/postsegmentation or focus-set changes.
Recompute only numeric coefficients every solver step.
One subtle point matters a lot for growing MTGs: “has no parent in the active system” is not always “is the root”. A growing or truncated focus set creates detached boundaries. That is exactly why I encoded the contract in test_solver_adjacency_contract.py (line 37). It is the right boundary between MTG growth management and solver logic.

If you want, I can turn this into a concrete MetaFSPM proposal with three cache invalidation flags like topology_dirty, focus_dirty, and coefficients_dirty, and map them onto the current component_factory schedule.




6:07 AM
I have added test_mtg_to_network.py to start imagining how we coud generalize resolution strategies on an adjacency graph. What I see here is that from node and edge abstract mtg scales, it is easy to build an adjacency graph in a vectorized way. A proposition I would like to forsee now is how on this graph, I can setup a generic system resolution strategy by 1) setting node states (like potentials or concentrations), 2) setting edge states (like conductances or resistances), 3) set the particular case of boundary conditions as node state or edge-level flux, 4) define differential equations from these node and edge states (btw is it possible to define equations so that they are automatically derivable in python or should I ask the modeler to input the derivative in the definition of the differential equation system?), and finally 5) define the unknowns and the picked solver, all generically. First for these 5 steps, does such strategy already exist in the literature, and if not make a test example with a full proposition for me to prototype on it.

Show more
6:19 AM



29 previous messages
Yes, in pieces. I don’t think there is a standard MTG-specific framework that already packages your five steps exactly for growing plant graphs, but your idea sits very close to a few established families:

Port-Hamiltonian systems on graphs: node/edge variables, incidence matrices, open-boundary ports, constitutive laws, and generic interconnection structure. Best match for your “node states + edge states + boundary flux/value + solver” view. Source: van der Schaft and Maschke, 2013, https://doi.org/10.1137/110840091
Network thermodynamics and bond graphs: older and still very relevant for multi-domain effort/flow modeling with storage and dissipation. Sources: Oster et al., 1971, https://www.nature.com/articles/234393a0 and Gawthrop & Pan, 2022, https://pubmed.ncbi.nlm.nih.gov/36057321/
Equation-based acausal modeling: this is the Modelica philosophy, where the modeler writes equations, not execution order. Sources: https://modelica.org/language/ and https://specification.modelica.org/master/equations.html
For the Python side of step 4, automatic differentiation is possible, but only if the equations are written as pure array functions, not as arbitrary MTG-mutating Python methods. The practical stack is:

Residual-first API
Optional analytic Jacobian
AD backend if wanted later with CasADi or JAX
Finite-difference fallback for prototyping
Sources:

CasADi AD and sparse Jacobians: https://web.casadi.org/docs/
JAX autodiff: https://docs.jax.dev/en/latest/automatic-differentiation.html
Residual-form DAE solvers like IDA: https://sundials.readthedocs.io/en/latest/ida/Introduction_link.html
I turned your test_mtg_to_network.py into a concrete prototype you can iterate on: test_mtg_to_network.py (line 1)

What the prototype now does:

Step 1, node states:
FieldState(..., location="node", ...) for potential and capacity at test_mtg_to_network.py (line 10) and test_mtg_to_network.py (line 300)
Step 2, edge states:
FieldState(..., location="edge", ...) for conductance and flux at test_mtg_to_network.py (line 314)
Step 3, boundaries:
DirichletNodeValue and BoundaryFlux, with a separate boundary_incidence matrix in OpenGraph at test_mtg_to_network.py (line 17), test_mtg_to_network.py (line 24), and test_mtg_to_network.py (line 87)
Step 4, equations:
a residual formulation
node_balance + constitutive_edge_law = 0
in test_mtg_to_network.py (line 178)
Step 5, unknowns and solver:
UnknownLayout and SolverSpec at test_mtg_to_network.py (line 32) and test_mtg_to_network.py (line 38), solved through a generic residual/Jacobian interface at test_mtg_to_network.py (line 255)
The mathematical pattern I’d recommend from here is:

M(x_n) x_n_dot + B_int q + B_b q_b - s(x_n, z_e, p, t) = 0
r_e(q, x_n, z_e, p, t) = 0
c_dirichlet(x_n, t) = 0
Where:

x_n: node unknowns/states
z_e: edge states/parameters
q: edge flux unknowns
B_int: internal incidence
B_b: boundary incidence
q_b: prescribed or unknown boundary fluxes
My recommendation for MetaFSPM is: ask modelers to provide residual equations, not Jacobians. Then support Jacobians in this order:

analytic optional
AD if backend is JAX/CasADi-compatible
finite differences fallback
I verified the prototype by directly executing the file in the wheat-bridges environment. If you want, I can take the next step and split this test prototype into a tiny reusable network_system.py test utility so other transport models can prototype against the same API.




6:26 AM
5 files changed
+442
-32
Undo

Review
metafspm/test/test_mtg_to_network.py
metafspm/test/test_mtg_to_network.py
metafspm/test/test_mtg_to_network.py
metafspm/test/test_mtg_to_network.py
metafspm/test/test_mtg_to_network.py
Comment the test in detail I can't understand the steps I am currently reading
6:36 AM



6 previous messages
I added detailed comments throughout test_mtg_to_network.py (line 1) so it now reads as a guided prototype:

module-level explanation of the 5-step strategy
what each dataclass means mathematically
how the MTG subset is converted into an open graph
what the residual equations are doing
how the Jacobian is assembled
what each test is checking and why
I also re-ran the file in the wheat-bridges environment, and it still executes successfully.

If you want, the next useful step would be for me to add one more layer of comments directly in equation form beside the code blocks, like:
R_node = B q + B_b q_b + C (p - p_old)/dt
and
R_edge = q - K B^T p
so you can read the file almost line-by-line against the math.




6:38 AM
1 file changed
Undo

Review
metafspm/test/test_mtg_to_network.py

test_mtg_to_network.py
here I need more explanation fro this problem formulation
6:40 AM


In test_mtg_to_network.py (line 1), that formulation is a mixed node-edge balance system. The key idea is that we do not solve only for node values; we solve simultaneously for:

p: a node state, here called a potential
q: an edge state, here called a flux
This is why there are two equations.

1. Node balance
B q + B_b q_b = 0

This is the conservation law. It says:

take all internal edge fluxes q
distribute them onto nodes using the incidence matrix B
add external boundary fluxes q_b through B_b
the net balance must be zero
With your sign convention, each column of B has:

+1 at the tail node of an edge
-1 at the head node
So B @ q gives, at each node, the net outgoing minus incoming internal flux.

For the 3-node chain in the test, the internal incidence is:

B = [[ 1,  0],
     [-1,  1],
     [ 0, -1]]
If q = [q1, q2], then:

B q = [ q1,
       -q1 + q2,
       -q2 ]
Meaning:

node 1 loses q1
node 2 receives q1 and loses q2
node 3 receives q2
Then B_b q_b adds exchanges with the outside world, like a sink at the collar or a source from soil/shoot.

2. Edge constitutive law
q - K (B^T p) = 0

This says flux is driven by a node-to-node difference.

B^T p computes the potential drop across each edge.
With the same sign convention, on one edge from node a to node b:

(B^T p)_e = p_a - p_b
So the equation becomes:

q_e = K_e (p_a - p_b)
This is the discrete version of a transport law like:

Ohm: current = conductance × voltage drop
Darcy: water flux = hydraulic conductance × pressure drop
Fick-like linear exchange: flux = permeability × concentration difference
So:

node equation = conservation
edge equation = physics of transport
Why keep both p and q as unknowns?

You could eliminate q and get a pure node equation:

B K B^T p + B_b q_b = 0
But keeping q explicit is often better because:

nonlinear flux laws are easier to write
flux boundary conditions fit naturally
later you can mix different edge laws on the same graph
implicit solvers and DAE-like formulations become cleaner
That is why this is a good generic prototype.

3. Transient version
C (p - p_old) / dt + B q + B_b q_b = 0

Now each node can store something.

C is a node storage coefficient, like:

capacitance in electricity
water capacity in hydraulics
compartment volume or buffer capacity in transport
(p - p_old) / dt is the backward-Euler approximation of dp/dt.

So this equation says:

accumulation + net flux balance = 0
or equivalently:

what enters - what leaves = what is stored
This is the standard compartment / control-volume interpretation.

4. What does this mean biologically?

In your context, p does not have to literally mean “electrical potential”. It is just the node variable driving transport. It could be:

xylem pressure
phloem pressure
solute concentration
chemical potential
osmotic potential
And q could be:

axial water flux
axial solute flux
membrane exchange flux
Then:

K is the edge conductance or permeability
B_b q_b represents external forcing
C represents local storage/buffering
5. Why is the left node not balanced in the steady test?

Because the Dirichlet condition replaces its natural conservation row. In the code, the residual row for that node is overwritten by:

p_i - p_i_prescribed = 0
So that node is treated as an imposed boundary value, not as a free conservation equation. That is standard.

If helpful, I can also rewrite that docstring block in the file in a more pedagogical way, with a tiny 3-node numerical example directly under the equations.




6:41 AM
Ok now I get it, nice formulation for a linear system, but what about a general formalisation when the flows equations become non-linear? If that comes to change the strategy, set another test cause I still like having this simple linear system aside.
6:48 AM



8 previous messages
I kept the linear prototype untouched in test_mtg_to_network.py (line 1) and added a nonlinear companion in test_mtg_to_nonlinear_network.py (line 1).

The main point of the new test is that the formulation does not really change when flows become nonlinear. We still write:

a node balance residual
an edge constitutive residual
boundary conditions as extra residual rows or boundary-port terms
the same node/edge unknown layout
What changes is the constitutive map and therefore the solver strategy. In the nonlinear test, the linear law q = K * delta_p becomes the cubic law at test_mtg_to_nonlinear_network.py (line 50):

q = K * (delta_p + beta * delta_p^3)
So the residual becomes state-dependent at test_mtg_to_nonlinear_network.py (line 138), the Jacobian also becomes state-dependent at test_mtg_to_nonlinear_network.py (line 180), and the direct one-shot linear solve is replaced by Newton iterations at test_mtg_to_nonlinear_network.py (line 246).

I also added two checks:

test_nonlinear_system_keeps_same_graph_formulation_but_uses_newton (line 325) verifies convergence and shows the solution differs from the linear one while preserving the same graph structure.
test_nonlinear_jacobian_matches_finite_difference (line 349) checks the analytic Jacobian against a finite-difference fallback, which is the useful bridge before bringing in AD.
I ran the new nonlinear file in the wheat-bridges environment and it passed.

The clean abstraction I’d recommend from here is:

keep the graph API fixed
let the model provide Phi(delta_p, edge_state, parameters, ...)
optionally provide dPhi/d(delta_p)
otherwise use AD or finite differences
choose the solver from the residual type:
linear direct solve if affine
Newton or quasi-Newton if nonlinear
If you want, I can take the next step and refactor both tests into one tiny generic residual_system prototype where the only model-specific piece is the edge law callback.




6:51 AM
1 file changed
Undo

Review
metafspm/test/test_mtg_to_nonlinear_network.py
I don't want a cubic law, I want it to accept any differential equation function working on node and edge states, generically
6:54 AM



16 previous messages
I replaced the cubic-specific companion with a generic callback-based prototype in test_mtg_to_generic_equation_system.py (line 1). The linear reference test in test_mtg_to_network.py (line 1) is untouched.

The generic part is now:

EquationContext at test_mtg_to_generic_equation_system.py (line 61)
This is what user-defined equations receive: graph, node fields, edge fields, current unknowns, previous states, and dt.

EquationBlock at test_mtg_to_generic_equation_system.py (line 92)
One residual block is just a Python callback returning a 1D residual vector.

GenericEquationSystem at test_mtg_to_generic_equation_system.py (line 112)
The system class no longer knows any constitutive law. It only packs unknowns, builds a context, concatenates residual blocks, computes a finite-difference Jacobian, and runs Newton.

Arbitrary user equations:
linear_edge_equation at test_mtg_to_generic_equation_system.py (line 419)
and generic_nonlinear_edge_equation at test_mtg_to_generic_equation_system.py (line 433)
are just examples of callbacks. The framework treats them the same way.

Transient differential terms:
make_transient_free_node_equation_block at test_mtg_to_generic_equation_system.py (line 323)
shows how a time-discretized ODE term can also be just another residual callback.

The two main tests are:

test_generic_equation_system_accepts_swappable_edge_callbacks (line 454)
Same graph, same unknown layout, different equation callbacks.

test_generic_equation_system_can_add_transient_blocks_without_rewriting_solver (line 489)
Same solver, but now one residual block mixes storage and balance.

I also ran the new file in the wheat-bridges environment and it passes.

So the abstraction is now really:

graph API stays fixed
unknown layout stays fixed
modeler supplies arbitrary residual functions on node/edge states
solver uses those residuals generically
If you want, the next step I’d recommend is to move just EquationContext, EquationBlock, and GenericEquationSystem into a tiny reusable helper module so your next prototypes don’t start from the test file.




6:57 AM
4 files changed
+542
-378
Undo

Review
metafspm/test/test_mtg_to_nonlinear_network.py
metafspm/test/test_mtg_to_generic_equation_system.py
metafspm/test/test_mtg_to_generic_equation_system.py
metafspm/test/test_mtg_to_generic_equation_system.py
In a new test file, answer to this: then illustrate how this generic formulation could be used to reimplement the axial_transport_N_arrays method of the Root-CyNAPS model first, and then how I could use metafspm to define all components of the system with decorators, extending on the logic where arbitrary equations were defined just for parallel computations on each node of the data structure without any edge logic
7:02 AM



10 previous messages
Added test_root_cynaps_axial_transport_as_generic_metafspm_system.py (line 1).

The first half shows how to decompose axial_transport_N_arrays into a generic graph system instead of one monolithic method: a Root-CyNAPS-like node balance block at line 119 (line 119), an edge transport block at line 149 (line 149), and one generic assembly at line 184 (line 184). The test at line 210 (line 210) checks that this residual-based formulation solves cleanly.

The second half sketches how MetaFSPM could extend its current decorator/signature logic to graph-wide systems: prototype decorators are at line 240 (line 240), the decorated component is at line 269 (line 269), the signature-to-residual bridge is at line 339 (line 339), and the builder that collects decorated methods into one GenericEquationSystem is at line 380 (line 380). The comparison test at line 418 (line 418) verifies that the decorator-declared system matches the manual generic-block version.

I ran conda run -n wheat-bridges python metafspm/test/test_root_cynaps_axial_transport_as_generic_metafspm_system.py successfully. No production code was changed.




7:07 AM
1 file changed
Undo

Review
metafspm/test/test_root_cynaps_axial_transport_as_generic_metafspm_system.py
Now do a second test, and adapt for non genericity on the way, to  transpose the water_transport_munch_arrays method of Root-CyNAPS to this logic, knowing that here for the newton raphson resolution approach we had to analytically define derivatives, but this is not always defined in the general case. What do these two reimplementations teach us?
7:08 AM



10 previous messages
Added test_root_cynaps_water_transport_as_generic_metafspm_system.py (line 1).

The manual transposition of water_transport_munch_arrays is in the node-only water residual blocks at line 308 (line 308) and line 346 (line 346). The key adaptation is that, unlike the nitrogen prototype, this one does not force edge unknowns: the graph is still used for axial coupling, but the unknown layout is just xylem_pressure and phloem_pressure on nodes at line 404 (line 404). I also added an optional analytic Jacobian hook at line 53 (line 53) and line 373 (line 373), with a finite-difference fallback still available. The first test at line 494 (line 494) checks that both paths converge to the same solution.

The decorator-based MetaFSPM-style declaration is in DecoratedRootCyNAPSLikeWaterComponent (line 559), with the new optional @axial_jacobian hook at line 554 (line 554). The builder at line 712 (line 712) shows how MetaFSPM could keep its signature-driven dependency resolution while extending from per-node functors to graph-scale residual systems.

What the two reimplementations teach us is captured directly in the comparison test (line 788): the common kernel can stay the same, but it should not over-constrain process shape. Nitrogen wants node + edge unknowns and is happy with a generic residual solve; water wants coupled node-only unknowns and benefits from an optional analytic Jacobian. So the right MetaFSPM direction looks like: shared graph extraction, shared residual assembly, shared solver interface, plus optional process-specific extensions rather than one rigid “generic” form.

I verified the new file by running conda run -n wheat-bridges python metafspm/test/test_root_cynaps_water_transport_as_generic_metafspm_system.py. No production code was changed.




7:16 AM
1 file changed
Undo

Review
metafspm/test/test_root_cynaps_water_transport_as_generic_metafspm_system.py
Now I put a third external model mecha_class.py in the test folder, since this will be another use case. Regarding their resolution strategy, assuming the root anatomy network is represented as in the simple test_mtg.py, would this resolution strategy also be adaptable to our general framework here? What do those 3 network-based examples confirm and call for regarding your proposal of a generic system formalisation and solving strategy?
7:21 AM



7 previous messages
Yes. If the anatomy network is encoded on the MTG with heterogeneous node and edge types like in test_mtg.py (line 58), then Mecha’s resolution strategy fits the same general framework, and in some ways more cleanly than the Root-CyNAPS axial cases.

In mecha_class.py (line 572), Mecha builds a global hydraulic matrix and RHS, then solves a linear system in solve (line 595) and solve_W (line 993). Boundary conditions and osmotic terms are assembled into the RHS and a few diagonal edits in initialize_scenarios (line 665) and water_flux (line 896). Then edge fluxes are recovered afterward from the solved node pressures in _calculate_edge_fluxes (line 1088). That is absolutely compatible with the framework we have been prototyping: it is a graph system with node unknowns, edge conductances, boundary operators, and post-processing outputs.

What changes is the shape of the specialization:

Nitrogen is a mixed transient system: node unknowns plus edge unknowns.
Root water is a coupled node-only system: two node fields, optional analytic Jacobian.
Mecha is mostly a linear node-only tissue network: one large heterogeneous graph, direct sparse solve, then derived edge fluxes.
So the three examples confirm the same core idea, but they also show that the core must stay broader than “axial transport”. What they call for is:

A generic GraphView, not a tree-only adjacency object.
It must support arbitrary heterogeneous graphs, typed node/edge masks, and boundary node sets.
A FieldLayout where unknowns may live on nodes, edges, boundaries, or only some of them.
A SystemSpec that can be declared either as residual blocks or, when available, as a direct linear operator A(theta) u = b(theta).
A SolverPolicy with multiple backends:
linear_direct, newton_fd, newton_jacobian, later maybe time integrators and DAEs.
Optional specialization hooks:
analytic Jacobian, direct matrix assembly, custom post-processing.
Clear separation between:
topology cache,
coefficient assembly,
solve,
write-back/output derivation.
So my proposal becomes a bit sharper now: the generic layer should be “graph system formalization”, not “generic axial solver”. After Mecha, even the decorator names should probably move from axial_* toward something like @graph_equation, @graph_jacobian, @graph_linear_operator, and @graph_output.

The short version is: the three cases strongly support a common framework, but they argue against one rigid algebraic template. The right abstraction is a shared graph-system kernel with optional fast paths for linear assembly and analytic derivatives.

I didn’t modify code for this step. If you want, the next useful move is for me to write a fourth prototype test that recasts the Mecha Doussan solve itself as a LinearGraphSystem on the test_mtg.py anatomy graph.




7:23 AM
Yes, for the next step actually write a first v0 of the prototyped generalized system formalisation utility, and create 3 tests to illustrate its use for each of the 3 use cases I have introduced to you here.
9:49 AM



10 previous messages
I added a reusable v0 prototype utility in graph_system_v0.py (line 1). The core pieces are GraphView (line 128), which turns an MTG subset into a solver-facing graph with typed node/edge data, and GraphSystem (line 265), which supports both residual-based Newton solves and direct linear solves, with optional analytic Jacobian and post-processing output hooks. I kept it in the test area on purpose so it stays a low-risk prototype rather than premature production API.

The three illustrative tests are in test_graph_system_v0_use_cases.py (line 1). The nitrogen-style mixed node/edge transport case is at line 121 (line 121), the water-style coupled node-only system with optional analytic Jacobian is at line 423 (line 423), and the Mecha-style heterogeneous linear hydraulic network is at line 498 (line 498). Together they show the same formal kernel can cover mixed transport, coupled axial hydraulics, and tissue-scale linear flow, while still allowing process-specific specialization where needed.

I verified the whole prototype by running conda run -n wheat-bridges python metafspm/test/test_graph_system_v0_use_cases.py successfully.




10:14 AM
2 files changed
+1085
-0
Undo

Review
metafspm/test/graph_system_v0.py
metafspm/test/test_graph_system_v0_use_cases.py
make an alternative to test_mtg.build_three_cell_mtg() because it generates a mtg with three cells under the "segment" scale, but it does not generate a realistic plant seeding for testing for upper scale for a full testing of the resolution. So create a generate_mtg.py that will also generate 1) just one plant node as already, 2) two plant axes, 3 shoot metamets per axis and only 1 for roots, stem and leaf organs for each shoot metamer and just a single root system for each axis / metamer, and few segments per organ in each shoot organ, and a simple primary root per root organ with only one branching and corresponding axis. Then bellow the segment scale, generate the three cells as already done for each segment. This mtg should then be used in tests rather than the previous one as it would represent an actual seedling more realistically.
3:01 PM



49 previous messages
I added the new seedling generator in generate_mtg.py (line 1). It now builds one plant, two main shoot axes, three metamers per main axis, stem/leaf/root organs per shoot metamer, multi-segment shoot organs, primary root segments, and one simple lateral root branch axis per root organ. Under every segment, it reuses the same three-cell anatomy motif as before, so the lower-scale network stays compatible with the graph-based solver tests. I also added get_representative_segment_id (line 500) so tests can extract one clean local segment from the full seedling when they still need the tiny 3-cell chain.

I updated the tests to use that seedling MTG instead of the old isolated one-segment builder:

test_mtg.py (line 1) now validates the seedling structure and keeps the old single-segment builder only as backward compatibility.
test_mtg_to_network.py (line 461) now extracts its local open graph from one representative seedling segment.
test_graph_system_v0_use_cases.py (line 32) now uses the seedling both for local segment transport cases and for the full Mecha-like anatomy case.
test_root_cynaps_water_transport_as_generic_metafspm_system.py (line 108) now also extracts its 3-cell chain from the seedling.
The nitrogen axial prototype continues to work through the updated build_cell_symplast_open_graph (line 461).
I verified the updated setup by running:

test_mtg seedling assertions directly
metafspm/test/test_mtg_to_network.py
metafspm/test/test_graph_system_v0_use_cases.py
metafspm/test/test_root_cynaps_axial_transport_as_generic_metafspm_system.py
metafspm/test/test_root_cynaps_water_transport_as_generic_metafspm_system.py
One modeling assumption I made explicitly: “two plant axes” was interpreted as two main shoot axes, and each root organ also gets one simple lateral root branch axis so the seedling contains realistic upper-scale branching while still staying small enough for tests.




3:12 PM
11 files changed
+802
-445
Undo

Review
metafspm/test/generate_mtg.py
metafspm/test/test_mtg.py
metafspm/test/test_mtg.py
metafspm/test/test_mtg_to_network.py
metafspm/test/test_root_cynaps_water_transport_as_generic_metafspm_system.py
metafspm/test/test_graph_system_v0_use_cases.py
metafspm/test/test_mtg_to_network.py
metafspm/test/generate_mtg.py
metafspm/test/test_mtg_to_network.py
metafspm/test/test_mtg.py
metafspm/test/test_mtg.py

test_graph_system_v0_use_cases.py
Here  for example an understanding question, why declare K_phloem, the axial conductance of phloem vessels as a node field rather than on an edge, I would have rather thought it to be on the edge as it is a conductance between two nodes.
8:37 AM


Because in that prototype I chose to mirror the way water_transport_munch_arrays thinks about the problem, not the most edge-native formulation.

In the water prototype at test_graph_system_v0_use_cases.py (line 338), K_xylem and K_phloem are stored as node fields because each value is really attached to a conducting segment, and in the rooted-tree interpretation each non-root segment defines exactly one axial connection to its parent. Then the actual edge conductance is recovered by indexing with the child node, for example in upward_axial_divergence (line 229) with axial_conductance_by_node[graph.head].

So the logic is:

biological object: one vessel segment
MTG support: one segment/node
axial link: “this segment connects to its parent”
implementation shortcut: store K on the segment/node, then map node -> parent edge using the child index
That matches Root-CyNAPS fairly well, and it also makes the collar term easy because the root segment uses its own K[root] for the boundary connection.

But your intuition is right: in a more generic graph formalism, axial conductance is often better modeled as an edge field. That is the cleaner choice when:

the quantity belongs to the relation between two nodes
the graph is not a rooted tree
one node may connect through several distinct axial links
the edge law depends on edge geometry directly
So for the general framework, I would not enforce node storage here. I would support both:

node-attached axial property:
useful for tree/segment models like current Root-CyNAPS
edge-attached axial property:
more natural for general graph solvers
A good generic rule is:

if the parameter belongs to an entity, store it on that entity
if it belongs to an interaction, store it on the edge
Here I used the first style for compatibility with the current Root-CyNAPS decomposition, but the long-term generic API should absolutely allow K_phloem as an edge field.

If you want, I can refactor the water example into a second variant where K_xylem and K_phloem are declared on edges instead, so you can compare both conventions side by side.




8:42 AM
Yes do refactor
8:47 AM




17 previous messages
Refactored in test_graph_system_v0_use_cases.py (line 235) so the water example now treats internal axial conductances as edge fields, which matches the “conductance belongs to a relation” interpretation better.

The key changes are in upward_axial_divergence (line 235), water_xylem_balance (line 264), water_phloem_balance (line 292), and build_water_system (line 366): K_xylem / K_phloem became K_xylem_edge / K_phloem_edge in edge_fields, while the collar conductances are now separate boundary parameters since that link is not an internal MTG edge. I also kept the water test checking that the analytic Jacobian and finite-difference Jacobian still agree, and re-ran metafspm/test/test_graph_system_v0_use_cases.py successfully.




9:37 AM
2 files changed
+58
-27
Undo

Review
metafspm/test/test_graph_system_v0_use_cases.py
metafspm/test/test_graph_system_v0_use_cases.py

