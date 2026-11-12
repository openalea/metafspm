"""
Upscaling the solved water potentials of the plants from their Compartments to the whole plant, scale by scale:
each scale is the average of the scale below it, a derived variable of the DataStructure (recomputed when the
potentials change, e.g. at each scene step).

    Compartment -> SubOrgan -> Organ -> Phytomer -> GrowthUnit -> Axis -> Plant
"""

SCALES = ("SubOrgan", "Organ", "Phytomer", "GrowthUnit", "Axis", "Plant")


def upscale(data_structure, name="water_potential", scales=SCALES) -> dict:
    """
    Derive *name* at each of *scales* in turn, each the mean of the previous one, on DataStructure *data_structure*. Returns
    {scale: derived variable name}, "Compartment" for the solved one.
    """
    names, source = {"Compartment": name}, name
    for scale in scales:
        target = f"{name}_{scale}"
        if not data_structure.has(target):
            data_structure.derive(target, {source: 1.}, location=scale, aggregation="mean")
        names[scale], source = target, target
    return names
