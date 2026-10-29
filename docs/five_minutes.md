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

The models ported to metafspm are in their own packages, on their development branches (the links below). Each has
**user tutorials** (`example/tutorials/user/`) that start from `simulate`, and **developer tutorials**
(`example/tutorials/developer/`).

| package (branch) | what it models | its components |
|---|---|---|
| [ArchiSimple](https://github.com/openalea/archisimple/tree/develop) (`develop`) | the architecture of a root system: its roots' elongation, branching, thickening and death | `RootArchitecture` (structural) |
| [Adel](https://github.com/openalea/adel/tree/to_metafspm) (`to_metafspm`) | the architecture of wheat shoots: their leaves, stems and tillers, in 3D | `ShootArchitecture` (structural), `ScenarioDevelopment` (the organs' lengths from a scenario); the plant model `ScenarioShoot` couples them |
| [Caribu](https://github.com/openalea/caribu/tree/to_metafspm) (`to_metafspm`) | the light absorbed by each element of a canopy | `CaribuLight` (functional), `CaribuLightModel` (the scene's light, an environment model), `SimplePlant` (a simple plant to light) |
| [GRANAP](https://github.com/openalea-incubator/GRANAP-dev/tree/to_metafspm) (`to_metafspm`) | the anatomy of plant organs' cross sections: their tissues and cells | `GranapAnatomy` (structural: the sections' cells), `RootAxis` and `Needle` (organs to carry the sections) |
| [MECHA](https://github.com/openalea-incubator/MECHA-dev/tree/to_metafspm) (`to_metafspm`) | the flow of water (and solutes) across a root's or a needle's tissues | `MechaHydraulics` (water), `MechaSolutes` (sucrose and starch), `MechaOsmoticPresets` (osmotic potentials); on GRANAP's anatomies |

## Installing them

metafspm and these ports are not published on the `openalea3` channel yet. Each is **cloned on its branch and
installed from source** (`pip install -e`), in an environment created from the package's own
`conda/environment.y(a)ml`, which takes the other dependencies (PlantGL, the MTG, numpy, ...) from `openalea3` and
`conda-forge` with mamba. Clone the repositories side by side, in one folder:

```bash
mkdir fspm && cd fspm
git clone -b data_structure_api https://github.com/openalea/metafspm.git metafspm
```

| package | clone | environment, then install |
|---|---|---|
| ArchiSimple | `git clone -b develop https://github.com/openalea/archisimple.git archisimple` | `mamba env create -f archisimple/conda/environment.yml`, `mamba activate archisimple`, `pip install -e metafspm` (the environment installs ArchiSimple) |
| Adel | `git clone -b to_metafspm https://github.com/openalea/adel.git adel` | `mamba env create -n adel_dev -f adel/conda/environment.yml`, `mamba activate adel_dev`, `pip install -e metafspm` |
| Caribu | `git clone -b to_metafspm https://github.com/openalea/caribu.git caribu` (and Adel's, as above, for the wheat examples) | `mamba env create -n caribu -f caribu/conda/environment.yml`, `mamba activate caribu`, `pip install -e metafspm` (and `pip install -e adel --no-deps`) |
| GRANAP | `git clone -b to_metafspm https://github.com/openalea-incubator/GRANAP-dev.git GRANAP` | `mamba env create -f GRANAP/conda/environment.yaml`, `mamba activate granap`, `pip install -e metafspm`, `pip install -e GRANAP` |
| MECHA | GRANAP's, as above, and `git clone -b to_metafspm https://github.com/openalea-incubator/MECHA-dev.git MECHA-dev` | `mamba env create -f MECHA-dev/conda/environment.yaml`, `mamba activate mecha`, `pip install -e metafspm`, `pip install -e GRANAP`, `pip install -e MECHA-dev` |

Keep the folder names above: the environment files find metafspm (and GRANAP) next to the package. Each package's
README gives the same steps. To update, `git pull` in each repository; the editable installs follow.

Adel and Caribu are also on `openalea3` (`mamba install -c openalea3 -c conda-forge openalea.adel openalea.caribu`),
but these are their released versions, the original models **without** the metafspm components above: for the
components and the tutorials, install from the branches.

## Seeing and sharing the results

A scene can also be saved as **fspm-TF files**, a format shared with other plant modelling platforms (CPlantBox,
GroIMP, VirtualPlantLab). Each file holds the whole scene at one time step: every plant's elements and their
variables, the soil, and the 3D shapes.

- `export_scene("scene.glb", scene)` (from `openalea.metafspm.io.fspm_tf`) saves the scene as it is now.
- A scene built with `output_dirpath="outputs", frame_period=24` saves one file every 24 steps in `outputs/frames`,
  with an index file, `scene.fspm.json`.
- In ParaView, fspm-TF's reader plugin opens a file, or the index to play the time steps. Every variable can then
  colour the plants, the root sections or the soil.

To install it, clone fspm-TF next to metafspm (`git clone https://github.com/openalea/fspm-TF.git`), and add
`pip install -e fspm-TF` to the steps above. ParaView's plugin is the file `fspm-TF/paraview/fspm_tf_reader.py`,
loaded once from ParaView's Tools → Manage Plugins menu.

## For developers

- The **framework guide** ({doc}`user`) describes each part: data structures, components, structure, coupling,
  scenes, performance, and how to port a model.
- metafspm's tutorial `examples/soil_plant_atmosphere/tutorial_fspm2026_general_usage.ipynb` builds components
  step by step and couples them with a soil.
- Each package's **developer tutorials** (`example/tutorials/developer/`) read its component through this
  formalism.
