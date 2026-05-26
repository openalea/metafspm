
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import math
from simple_seedling import seedling

g = seedling.g
leaf        = seedling.leaf
internode   = seedling.internode
meristem    = seedling.meristem
phytomer    = seedling.phytomer
growth_unit = seedling.growth_unit
axis        = seedling.axis
leafelement1 = seedling.leafelement1
leafelement2 = seedling.leafelement2
leafelement3 = seedling.leafelement3


# ── Test fixtures ─────────────────────────────────────────────────────────────

_LEAF_LENGTHS = {leafelement1: 1.0, leafelement2: 2.0, leafelement3: 3.0}
_TOTAL = sum(_LEAF_LENGTHS.values())   # 6.0
_MEAN  = _TOTAL / len(_LEAF_LENGTHS)   # 2.0


def _set_lengths():
    """Initialise `length` at SubOrgan scale; zero out all coarser vertices."""
    props = g.property('length')
    props.update(_LEAF_LENGTHS)
    for v in [leaf, internode, meristem, phytomer, growth_unit, axis]:
        props[v] = 0.0


# ── integrate_at_scale tests ──────────────────────────────────────────────────

def test_sum_all_intermediate_scales():
    """One call from SubOrgan to Axis fills every intermediate scale automatically."""
    _set_lengths()
    g.integrate_at_scale('length', from_scale=g.scales.SubOrgan, target_scale=g.scales.Axis)

    length = g.property('length')
    assert math.isclose(length[leaf],        _TOTAL)
    assert math.isclose(length[internode],   0.0)   # no SubOrgan components
    assert math.isclose(length[meristem],    0.0)
    assert math.isclose(length[phytomer],    _TOTAL)
    assert math.isclose(length[growth_unit], _TOTAL)
    assert math.isclose(length[axis],        _TOTAL)


def test_sum_missing_source_values_default_to_zero():
    """Vertices absent from the property dict contribute 0 to the sum."""
    _set_lengths()
    del g.property('length')[leafelement2]
    g.integrate_at_scale('length', from_scale=g.scales.SubOrgan, target_scale=g.scales.Organ)
    assert math.isclose(g.property('length')[leaf], 1.0 + 3.0)


# ── average_at_scale tests ────────────────────────────────────────────────────

def test_average_unweighted():
    """Plain mean of leaf element lengths (1, 2, 3) → 2.0 up to Phytomer scale.

    Only tested up to Phytomer: the seedling has additional SubOrgan vertices
    (a second leaf + root segments) that have no length value and would dilute
    the unweighted mean at GrowthUnit and above.
    """
    _set_lengths()
    g.average_at_scale('length', from_scale=g.scales.SubOrgan, target_scale=g.scales.Phytomer)

    length = g.property('length')
    assert math.isclose(length[leaf],      _MEAN)   # (1+2+3)/3
    assert math.isclose(length[internode], 0.0)      # no SubOrgan descendants → W=0
    assert math.isclose(length[meristem],  0.0)
    # Phytomer: relayed (S=6, W=3) from Leaf; internode/meristem W=0 → mean=2.0
    assert math.isclose(length[phytomer],  _MEAN)


def test_average_weighted_concentration():
    """Mass-weighted mean concentration: amount = concentration × mass, then renormalise.

    Setup:
        leafelement1: concentration=10, mass=2  → amount=20
        leafelement2: concentration=20, mass=3  → amount=60
        leafelement3: concentration=30, mass=5  → amount=150
        bulk = (20+60+150) / (2+3+5) = 230/10 = 23.0
    """
    conc = g.property('concentration')
    mass = g.property('mass')
    conc.update({leafelement1: 10.0, leafelement2: 20.0, leafelement3: 30.0})
    mass.update({leafelement1:  2.0, leafelement2:  3.0, leafelement3:  5.0})
    for v in [leaf, internode, meristem, phytomer]:
        conc[v] = 0.0

    g.average_at_scale('concentration', from_scale=g.scales.SubOrgan,
                       target_scale=g.scales.Phytomer, normalization_property='mass')

    c = g.property('concentration')
    assert math.isclose(c[leaf],      23.0)
    assert math.isclose(c[internode], 0.0)   # no mass → W=0
    assert math.isclose(c[meristem],  0.0)
    # Phytomer: relayed (S=230, W=10) from Leaf; internode/meristem W=0
    assert math.isclose(c[phytomer],  23.0)


if __name__ == "__main__":
    for fn in [
        test_sum_all_intermediate_scales,
        test_sum_missing_source_values_default_to_zero,
        test_average_unweighted,
        test_average_weighted_concentration,
    ]:
        fn()
        print(f"{fn.__name__} passed")
    print("All tests passed.")
