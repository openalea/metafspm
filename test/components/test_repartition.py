"""
Repartition of functional variables when the structure changes, and the active mask. Expected values are computed by hand with the rules of
rhizodep's post_growth_updating.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, StructuralComponent, state_variable
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.solve.decorator import actual, potential, rate, segmentation

from growth import DOC, CarbonProbe, RootGrowthProbe, make_chain



@dataclass
class Pools(FunctionalComponent):
    """One variable of each kind, owned by a functional component."""
    conc: float = state_variable(**DOC, default=3., scale=scales.SubOrgan, kind="massic_concentration")
    amount: float = state_variable(**DOC, default=4., scale=scales.SubOrgan, kind="extensive")
    temperature: float = state_variable(**DOC, default=20., scale=scales.SubOrgan, kind="intensive")
    uptake: float = state_variable(**DOC, default=5., scale=scales.SubOrgan,
                                   kind="non_inertial_extensive")
    tag: float = state_variable(**DOC, default=7., scale=scales.SubOrgan, kind="descriptor")


def _plant(hexose=1., repartition="after_mpg_steps"):
    mtg, data_structure, vids = make_chain()
    growth = RootGrowthProbe(data_structure=data_structure)
    growth.partition_weight = "struct_mass"
    growth.repartition = repartition
    pools = Pools(data_structure=data_structure)
    CarbonProbe(data_structure=data_structure)
    data_structure.set("C_hexose_root", hexose)
    return mtg, data_structure, vids, growth, pools


def _by_vid(data_structure, name):
    return dict(zip(data_structure.entity_ids("node").tolist(), data_structure.get(name).tolist()))


@pytest.mark.parametrize("repartition", ["after_mpg_steps", "end_of_call"])
def test_elongation_dilutes_concentrations_and_keeps_amounts(repartition):
    _, data_structure, vids, growth, _ = _plant(hexose=1., repartition=repartition)
    growth()                                             # apex 0.5 -> 0.9: weight 1.0 -> 1.8
    apex = vids[-1]
    assert _by_vid(data_structure, "conc")[apex] == pytest.approx(3. * 1.0 / 1.8)
    assert _by_vid(data_structure, "conc")[vids[0]] == pytest.approx(3.)    # not grown
    for name, value in (("amount", 4.), ("temperature", 20.), ("uptake", 5.), ("tag", 7.)):
        assert _by_vid(data_structure, name)[apex] == pytest.approx(value)


@pytest.mark.parametrize("repartition", ["after_mpg_steps", "end_of_call"])
def test_segmentation_splits_amounts_by_weight_and_conserves_them(repartition):
    _, data_structure, vids, growth, _ = _plant(hexose=1.5, repartition=repartition)
    growth()                                             # apex 0.5 -> 1.1, then cut into 1.0 + a new apex of 0.1
    apex, new = vids[-1], int(data_structure.entity_ids("node")[data_structure.tips()[0]])
    w = _by_vid(data_structure, "struct_mass")
    assert (w[apex], w[new]) == (pytest.approx(2.0), pytest.approx(0.2))
    f = 0.2 / 2.2
    conc, amount = _by_vid(data_structure, "conc"), _by_vid(data_structure, "amount")
    assert conc[apex] == pytest.approx(3. / 2.2) and conc[new] == pytest.approx(3. / 2.2)
    assert conc[apex] * w[apex] + conc[new] * w[new] == pytest.approx(3. * 1.0)   # the amount c·w of the apex
    assert (amount[apex], amount[new]) == (pytest.approx((1 - f) * 4.), pytest.approx(f * 4.))
    assert _by_vid(data_structure, "temperature")[new] == 20.
    assert (_by_vid(data_structure, "uptake")[new], _by_vid(data_structure, "uptake")[apex]) == (pytest.approx(f * 5.), 5.)
    assert _by_vid(data_structure, "tag")[new] == 7.                 # descriptor: its on_grow default


def test_without_a_partition_weight_growth_only_applies_on_grow():
    _, data_structure, vids, growth, _ = _plant(hexose=1.5)
    growth.partition_weight = None
    growth()
    new = int(data_structure.entity_ids("node")[data_structure.tips()[0]])
    assert _by_vid(data_structure, "amount")[new] == 4. and _by_vid(data_structure, "amount")[vids[-1]] == 4.


@dataclass
class ThickeningProbe(RootGrowthProbe):
    """RootGrowthProbe whose structural mass also grows in an array step (as rhizodep's actual growth and root
    hairs do), keeping its mass at the call's start as the basis of the repartition."""
    initial_struct_mass: float = state_variable(**DOC, default=0., scale=scales.SubOrgan,
                                                kind="descriptor")
    partition_weight = "struct_mass"
    partition_basis = "initial_struct_mass"
    repartition = "end_of_call"

    @potential
    def _initial_struct_mass(self, struct_mass):
        return np.array(struct_mass)

    @actual
    def _struct_mass(self, struct_mass, is_apex):
        return struct_mass * np.where(is_apex == 0., 1.5, 1.)          # the segments thicken


def test_end_of_call_dilutes_what_array_steps_grew():
    """With "end_of_call", the weight an array step changed dilutes the concentrations too, against the basis; with
    "after_mpg_steps" only the MPG-style steps' changes count."""
    for repartition, expected in (("end_of_call", 3. / 1.5), ("after_mpg_steps", 3.)):
        mtg, data_structure, vids = make_chain()
        growth = ThickeningProbe(data_structure=data_structure)
        growth.repartition = repartition
        Pools(data_structure=data_structure)
        CarbonProbe(data_structure=data_structure)
        data_structure.set("C_hexose_root", 0.)                                    # no elongation, no segmentation
        growth()
        conc, mass = _by_vid(data_structure, "conc"), _by_vid(data_structure, "struct_mass")
        assert mass[vids[0]] == pytest.approx(2. * 1.5)
        assert conc[vids[0]] == pytest.approx(expected)
        assert _by_vid(data_structure, "amount")[vids[0]] == 4.


def test_end_of_call_conserves_through_growth_and_segmentation():
    mtg, data_structure, vids = make_chain()
    growth = ThickeningProbe(data_structure=data_structure)
    Pools(data_structure=data_structure)
    CarbonProbe(data_structure=data_structure)
    data_structure.set("C_hexose_root", 1.5)
    for _ in range(3):
        before = (np.sum(data_structure.get("conc") * data_structure.get("struct_mass")), np.sum(data_structure.get("amount")), data_structure.n_nodes())
        growth()
        assert np.sum(data_structure.get("conc") * data_structure.get("struct_mass")) == pytest.approx(before[0])
        assert np.sum(data_structure.get("amount")) == pytest.approx(before[1])
    assert data_structure.n_nodes() > len(vids)


def test_the_repartition_option_is_checked():
    _, data_structure, _, growth, _ = _plant()
    growth.repartition = "sometimes"
    with pytest.raises(ValueError, match="after_mpg_steps"):
        growth()


# ---------------------------------------------------------------- inactive primordia

@dataclass
class Branching(StructuralComponent):
    struct_mass: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")
    partition_weight = "struct_mass"
    active = {"struct_mass": ">0"}

    @segmentation
    def _form_primordium(self):
        mtg = self.mtg
        if getattr(self, "primordium", None) is None:
            self.primordium = mtg.add_child(self.carrier, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='+',
                                                                      label=mtg.labels.SubOrgan.RootSegment,
                                                                      struct_mass=0.))

    @actual
    def _emerge(self):
        if getattr(self, "primordium", None) is not None and getattr(self, "emerge", False):
            self.mtg.set_values("struct_mass", [self.primordium], [0.5])


def _branching_plant():
    mtg, data_structure, vids = make_chain()
    branching = Branching(data_structure=data_structure)
    branching.carrier = vids[1]
    Pools(data_structure=data_structure)
    return mtg, data_structure, vids, branching


def test_a_primordium_copies_concentrations_and_holds_no_amount():
    _, data_structure, vids, branching = _branching_plant()
    branching()
    p = branching.primordium
    assert not data_structure.mask("active")[data_structure.index_of(p)]
    assert (_by_vid(data_structure, "conc")[p], _by_vid(data_structure, "temperature")[p]) == (3., 20.)
    assert (_by_vid(data_structure, "amount")[p], _by_vid(data_structure, "uptake")[p]) == (0., 0.)
    assert _by_vid(data_structure, "amount")[vids[1]] == 4.          # the carrier is not reduced


def test_a_primordium_is_split_from_its_carrier_when_it_emerges():
    _, data_structure, vids, branching = _branching_plant()
    branching()                                          # primordium formed, inactive
    branching.emerge = True
    branching()                                          # it gains mass 0.5: active
    p, carrier = branching.primordium, vids[1]
    w = _by_vid(data_structure, "struct_mass")
    f = 0.5 / (0.5 + w[carrier])
    assert _by_vid(data_structure, "amount")[p] == pytest.approx(f * 4.)
    assert _by_vid(data_structure, "amount")[carrier] == pytest.approx((1 - f) * 4.)
    conc = _by_vid(data_structure, "conc")
    assert conc[p] * w[p] + conc[carrier] * w[carrier] == pytest.approx(3. * w[carrier])


def test_an_entity_becoming_active_keeps_its_own_amount():
    """An existing entity that becomes active (e.g. a primordium holding an amount) adds its share of its parent's
    to its own: the amounts are conserved."""
    _, data_structure, vids, branching = _branching_plant()
    branching()
    p, carrier = branching.primordium, vids[1]
    amount = np.array(data_structure.get("amount"))
    amount[data_structure.index_of(p)] = 2.
    data_structure.set("amount", amount)
    total = data_structure.get("amount").sum()
    branching.emerge = True
    branching()
    w = _by_vid(data_structure, "struct_mass")
    f = 0.5 / (0.5 + w[carrier])
    assert _by_vid(data_structure, "amount")[p] == pytest.approx(2. + f * 4.)
    assert data_structure.get("amount").sum() == pytest.approx(total)


# ---------------------------------------------------------------- the active mask on vectorised steps

@dataclass
class Uptake(FunctionalComponent):
    root_uptake: float = state_variable(**DOC, default=-1., scale=scales.SubOrgan)
    everywhere: float = state_variable(**DOC, default=-1., scale=scales.SubOrgan)
    struct_mass: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)

    @rate
    def _root_uptake(self, struct_mass):
        return 2. * struct_mass

    @rate(include_inactive=True)
    def _everywhere(self, struct_mass):
        return np.ones_like(struct_mass)


def test_vectorised_steps_compute_on_the_active_entities_only():
    _, data_structure, vids, branching = _branching_plant()
    branching()                                          # an inactive primordium
    uptake = Uptake(data_structure=data_structure)
    uptake()
    active = data_structure.mask("active")
    np.testing.assert_allclose(data_structure.get("root_uptake")[active], 2. * data_structure.get("struct_mass")[active])
    np.testing.assert_array_equal(data_structure.get("root_uptake")[~active], -1.)   # inactive entities keep their values
    np.testing.assert_array_equal(data_structure.get("everywhere"), 1.)              # where=None opts out


def test_masks_follow_their_variables():
    mtg, data_structure, _ = make_chain()
    for name in ("length", "is_apex"):
        data_structure.register(name, data_structure._mtg_to_node_array(name), location="node")
    data_structure.define_mask("long", {"length": ">0", "is_apex": 0.})
    assert data_structure.mask("long").sum() == 2
    version = data_structure.mask_version("long")
    data_structure.set("is_apex", 0.)
    assert data_structure.mask("long").sum() == 3 and data_structure.mask_version("long") == version + 1
    data_structure.set("is_apex", 0.)                                # same values: the version does not change
    assert data_structure.mask_version("long") == version + 1
    with pytest.raises(KeyError, match="mask 'missing' is not defined"):
        data_structure.mask("missing")
    data_structure.define_mask("bad", {"nowhere": ">0"})
    with pytest.raises(KeyError, match="variable 'nowhere' is not registered"):
        data_structure.mask("bad")
