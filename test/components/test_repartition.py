"""
Repartition of functional variables when the structure changes, and the active mask. Expected values are computed by hand with the rules of
rhizodep's post_growth_updating.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, StructuralComponent, state_variable
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.solve.decorator import actual, rate, segmentation

from growth import DOC, CarbonProbe, RootGrowthProbe, make_chain



@dataclass
class Pools(FunctionalComponent):
    """One variable of each kind, owned by a functional component."""
    conc: float = state_variable(**DOC, initialize=3., scale=scales.SubOrgan, state_variable_type="massic_concentration")
    amount: float = state_variable(**DOC, initialize=4., scale=scales.SubOrgan, state_variable_type="extensive")
    temperature: float = state_variable(**DOC, initialize=20., scale=scales.SubOrgan, state_variable_type="intensive")
    uptake: float = state_variable(**DOC, initialize=5., scale=scales.SubOrgan,
                                   state_variable_type="NonInertialExtensive")
    tag: float = state_variable(**DOC, initialize=7., scale=scales.SubOrgan, state_variable_type="descriptor")


def _plant(hexose=1.):
    g, ds, vids = make_chain()
    growth = RootGrowthProbe(data_structure=ds)
    growth.partition_weight = "struct_mass"
    pools = Pools(data_structure=ds)
    CarbonProbe(data_structure=ds)
    ds.set("C_hexose_root", hexose)
    return g, ds, vids, growth, pools


def _by_vid(ds, name):
    return dict(zip(ds.entity_ids("node").tolist(), ds.get(name).tolist()))


def test_elongation_dilutes_concentrations_and_keeps_amounts():
    _, ds, vids, growth, _ = _plant(hexose=1.)
    growth()                                             # apex 0.5 -> 0.9: weight 1.0 -> 1.8
    apex = vids[-1]
    assert _by_vid(ds, "conc")[apex] == pytest.approx(3. * 1.0 / 1.8)
    assert _by_vid(ds, "conc")[vids[0]] == pytest.approx(3.)    # not grown
    for name, value in (("amount", 4.), ("temperature", 20.), ("uptake", 5.), ("tag", 7.)):
        assert _by_vid(ds, name)[apex] == pytest.approx(value)


def test_segmentation_splits_amounts_by_weight_and_conserves_them():
    _, ds, vids, growth, _ = _plant(hexose=1.5)
    growth()                                             # apex 0.5 -> 1.1, then cut into 1.0 + a new apex of 0.1
    apex, new = vids[-1], int(ds.entity_ids("node")[ds.tips()[0]])
    w = _by_vid(ds, "struct_mass")
    assert (w[apex], w[new]) == (pytest.approx(2.0), pytest.approx(0.2))
    f = 0.2 / 2.2
    conc, amount = _by_vid(ds, "conc"), _by_vid(ds, "amount")
    assert conc[apex] == pytest.approx(3. / 2.2) and conc[new] == pytest.approx(3. / 2.2)
    assert conc[apex] * w[apex] + conc[new] * w[new] == pytest.approx(3. * 1.0)   # the amount c·w of the apex
    assert (amount[apex], amount[new]) == (pytest.approx((1 - f) * 4.), pytest.approx(f * 4.))
    assert _by_vid(ds, "temperature")[new] == 20.
    assert (_by_vid(ds, "uptake")[new], _by_vid(ds, "uptake")[apex]) == (pytest.approx(f * 5.), 5.)
    assert _by_vid(ds, "tag")[new] == 7.                 # descriptor: its on_grow default


def test_without_a_partition_weight_growth_only_applies_on_grow():
    _, ds, vids, growth, _ = _plant(hexose=1.5)
    growth.partition_weight = None
    growth()
    new = int(ds.entity_ids("node")[ds.tips()[0]])
    assert _by_vid(ds, "amount")[new] == 4. and _by_vid(ds, "amount")[vids[-1]] == 4.


# ---------------------------------------------------------------- inactive primordia

@dataclass
class Branching(StructuralComponent):
    struct_mass: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="extensive")
    partition_weight = "struct_mass"
    active = {"struct_mass": ">0"}

    @segmentation
    def _form_primordium(self):
        g = self.mtg
        if getattr(self, "primordium", None) is None:
            self.primordium = g.add_child(self.carrier, **PropsConfig(scale=g.scales.SubOrgan, edge_type='+',
                                                                      label=g.labels.SubOrgan.RootSegment,
                                                                      struct_mass=0.))

    @actual
    def _emerge(self):
        if getattr(self, "primordium", None) is not None and getattr(self, "emerge", False):
            self.mtg.property("struct_mass")[self.primordium] = 0.5


def _branching_plant():
    g, ds, vids = make_chain()
    branching = Branching(data_structure=ds)
    branching.carrier = vids[1]
    Pools(data_structure=ds)
    return g, ds, vids, branching


def test_a_primordium_copies_concentrations_and_holds_no_amount():
    _, ds, vids, branching = _branching_plant()
    branching()
    p = branching.primordium
    assert not ds.mask("active")[ds.index_of(p)]
    assert (_by_vid(ds, "conc")[p], _by_vid(ds, "temperature")[p]) == (3., 20.)
    assert (_by_vid(ds, "amount")[p], _by_vid(ds, "uptake")[p]) == (0., 0.)
    assert _by_vid(ds, "amount")[vids[1]] == 4.          # the carrier is not reduced


def test_a_primordium_is_split_from_its_carrier_when_it_emerges():
    _, ds, vids, branching = _branching_plant()
    branching()                                          # primordium formed, inactive
    branching.emerge = True
    branching()                                          # it gains mass 0.5: active
    p, carrier = branching.primordium, vids[1]
    w = _by_vid(ds, "struct_mass")
    f = 0.5 / (0.5 + w[carrier])
    assert _by_vid(ds, "amount")[p] == pytest.approx(f * 4.)
    assert _by_vid(ds, "amount")[carrier] == pytest.approx((1 - f) * 4.)
    conc = _by_vid(ds, "conc")
    assert conc[p] * w[p] + conc[carrier] * w[carrier] == pytest.approx(3. * w[carrier])


# ---------------------------------------------------------------- the active mask on vectorised steps

@dataclass
class Uptake(FunctionalComponent):
    root_uptake: float = state_variable(**DOC, initialize=-1., scale=scales.SubOrgan)
    everywhere: float = state_variable(**DOC, initialize=-1., scale=scales.SubOrgan)
    struct_mass: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)

    @rate
    def _root_uptake(self, struct_mass):
        return 2. * struct_mass

    @rate(include_inactive=True)
    def _everywhere(self, struct_mass):
        return np.ones_like(struct_mass)


def test_vectorised_steps_compute_on_the_active_entities_only():
    _, ds, vids, branching = _branching_plant()
    branching()                                          # an inactive primordium
    uptake = Uptake(data_structure=ds)
    uptake()
    active = ds.mask("active")
    np.testing.assert_allclose(ds.get("root_uptake")[active], 2. * ds.get("struct_mass")[active])
    np.testing.assert_array_equal(ds.get("root_uptake")[~active], -1.)   # inactive entities keep their values
    np.testing.assert_array_equal(ds.get("everywhere"), 1.)              # where=None opts out


def test_masks_follow_their_variables():
    g, ds, _ = make_chain()
    for name in ("length", "is_apex"):
        ds.register(name, ds._mtg_to_node_array(name), location="node")
    ds.define_mask("long", {"length": ">0", "is_apex": 0.})
    assert ds.mask("long").sum() == 2
    version = ds.mask_version("long")
    ds.set("is_apex", 0.)
    assert ds.mask("long").sum() == 3 and ds.mask_version("long") == version + 1
    ds.set("is_apex", 0.)                                # same values: the version does not change
    assert ds.mask_version("long") == version + 1
    with pytest.raises(KeyError, match="mask 'missing' is not defined"):
        ds.mask("missing")
    ds.define_mask("bad", {"nowhere": ">0"})
    with pytest.raises(KeyError, match="variable 'nowhere' is not registered"):
        ds.mask("bad")
