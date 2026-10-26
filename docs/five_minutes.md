# metafspm in five minutes

This page is for those who want to **use** a plant model built with metafspm: students, experimenters, modellers new
to these models. It explains the few ideas needed to understand what the models' tutorials do. No programming beyond
running a notebook is needed. To **build** or change models, the developer pages come after (end of this page).

## Plant models and FSPMs

A **functional-structural plant model** (FSPM) describes a plant as it grows in space. Its **structure** is its
organs: roots, stems, leaves, each cut into small elements. Its **functions** are the processes in them: growth, the
flows of water and sugars, the light absorbed. Each process is described by equations, with **parameters**, numbers
measured or calibrated for a species.

Many FSPMs exist, each specialised: one for the root system's architecture, one for the shoot's, one for water flow
in a root, one for light in a canopy. Studying a whole plant means making them work together. That is what metafspm
is for.

## Components

In metafspm, each model is a **component**: a piece of a plant model that does one job. Examples are the growth of
the roots, the hydraulics of a root section, or the light absorbed by the leaves. A component states:

- the **variables** it computes (a root's length, a cell's water potential), each with its unit;
- the **inputs** it needs from other components (the sugar a root receives from the shoot, the soil's temperature);
- its **parameters** (an elongation rate, a conductivity), with their default values;
- its **processes**, the equations that compute the variables at each time step.

A component works on the plant's **data structure**: a table of the plant's elements (each root segment, each leaf
piece) and their variables, organised by scale (plant, axis, organ, element).

## Plant models and scenes

Components are combined in two layers:

1. A **plant model** gathers the components of one kind of plant and **couples** them: what one component computes
   becomes another's input. A root architecture and a carbon model, for example, exchange the sugar offered and the
   mass grown.
2. A **scene** plants plants of these models at their positions, each with its own **scenario** (its parameter
   values). It adds the **environment**: a soil, the light, the weather. At each time step it advances every plant
   and the environment together.

Each ported package offers a simple entry point, `simulate`, which builds a plant model with its component alone
and a scene with one plant. It runs the simulation and returns tables of results. It is the way to start. The
packages' developer tutorials show the full use, with several components and plants.

## The models available

Each package has **user tutorials** (`example/tutorials/user/`) that start from `simulate`:

| package | what it models |
|---|---|
| [ArchiSimple](https://github.com/openalea/archisimple) | the architecture of a root system: its roots' elongation, branching, thickening and death |
| [Adel](https://github.com/openalea/adel) | the architecture of wheat shoots: their leaves, stems and tillers, in 3D |
| [Caribu](https://github.com/openalea/caribu) | the light absorbed by each element of a canopy |
| [GRANAP](https://github.com/openalea-incubator/GRANAP-dev) | the anatomy of plant organs' cross sections: their tissues and cells |
| [MECHA](https://github.com/openalea-incubator/MECHA-dev) | the flow of water (and solutes) across a root's or a needle's tissues |

## For developers

- The **framework guide** ({doc}`user`) describes each part: data structures, components, structure, coupling,
  scenes, performance, and how to port a model.
- metafspm's tutorial `examples/soil_plant_atmosphere/tutorial_fspm2026_general_usage.ipynb` builds components
  step by step and couples them with a soil.
- Each package's **developer tutorials** (`example/tutorials/developer/`) read its component through this
  formalism.
