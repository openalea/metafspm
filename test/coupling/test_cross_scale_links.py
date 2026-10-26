"""
Cross-scale links between components sharing a DataStructure: default mappings from the provider's
state_variable_type, kind agreement, and mappings between coarse scales.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.declaration import DeclarationError
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from simple_seedling import generate_simple_mpg_seedling

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])



@dataclass
class Segments(FunctionalComponent):
    """Outputs per segment, one of each kind."""
    uptake: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan, state_variable_type="extensive")
    potential: float = state_variable(**DOC, initialize=2., scale=scales.SubOrgan, state_variable_type="intensive")
    weighted_potential: float = state_variable(**DOC, initialize=2., scale=scales.SubOrgan,
                                               state_variable_type="intensive", weight="mass")
    concentration: float = state_variable(**DOC, initialize=3., scale=scales.SubOrgan,
                                          state_variable_type="massic_concentration")
    mass: float = state_variable(**DOC, initialize=0.5, scale=scales.SubOrgan, state_variable_type="extensive")
    tag: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)


@dataclass
class Organs(FunctionalComponent):
    """Inputs per organ, and an organ output read back by the segments' scale."""
    organ_uptake: float = input_variable(**DOC, by="Segments", initialize=0., scale=scales.Organ)
    organ_potential: float = input_variable(**DOC, by="Segments", initialize=0., scale=scales.Organ)
    organ_weighted_potential: float = input_variable(**DOC, by="Segments", initialize=0., scale=scales.Organ)
    organ_concentration: float = input_variable(**DOC, by="Segments", initialize=0., scale=scales.Organ)
    organ_tag: float = input_variable(**DOC, by="Segments", initialize=0., scale=scales.Organ)
    organ_supply: float = state_variable(**DOC, initialize=6., scale=scales.Organ, state_variable_type="extensive")
    organ_temperature: float = state_variable(**DOC, initialize=20., scale=scales.Organ, state_variable_type="intensive")


@dataclass
class SegmentInputs(FunctionalComponent):
    segment_temperature: float = input_variable(**DOC, by="Organs", initialize=0., scale=scales.SubOrgan)
    segment_supply: float = input_variable(**DOC, by="Organs", initialize=0., scale=scales.SubOrgan)
    segment_share: float = input_variable(**DOC, by="Organs", initialize=0., scale=scales.SubOrgan, weight="mass")
    wrong_kind: float = input_variable(**DOC, by="Segments", initialize=0., scale=scales.Organ,
                                       state_variable_type="intensive")


def _couple(links, before=None):
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    components = (Segments(data_structure=ds), Organs(data_structure=ds), SegmentInputs(data_structure=ds))
    if before is not None:
        before(g, ds)
    model = CompositeModel()
    model.components = list(components)
    model._couple_on_data_structures(links)
    return g, ds


def _per_organ(g, ds, values_per_node):
    organs = ds.entity_ids("Organ")
    owner = {int(v): g.complex_at_scale(int(v), g.scales.Organ) for v in ds.entity_ids("node")}
    return {o: [values_per_node[i] for i, v in enumerate(ds.entity_ids("node")) if owner[int(v)] == o] for o in organs}


def _link(receiver, provider, variable, source, **options):
    return {receiver: {provider: {variable: {"sources": {source: 1.}, **options}}}}


def test_extensive_outputs_are_summed_up_by_default():
    g, ds = _couple(_link("Organs", "Segments", "organ_uptake", "uptake"))
    per_organ = _per_organ(g, ds, ds.get("uptake"))
    np.testing.assert_allclose(ds.get("organ_uptake"), [sum(per_organ[o]) for o in ds.entity_ids("Organ")])


def test_intensive_outputs_are_averaged_up_by_their_declared_weight():
    """An intensive variable's mean needs its capacity's weight: the link's, else the one it declares."""
    with pytest.raises(DeclarationError, match="give the weight of its mean"):
        _couple(_link("Organs", "Segments", "organ_potential", "potential"))
    _, ds = _couple(_link("Organs", "Segments", "organ_potential", "potential", aggregation="mean"))
    np.testing.assert_allclose(ds.get("organ_potential"), 2.)
    _, ds = _couple(_link("Organs", "Segments", "organ_weighted_potential", "weighted_potential"))
    np.testing.assert_allclose(ds.get("organ_weighted_potential"), 2.)


def test_extensive_inputs_are_split_down_by_their_declared_weight():
    """An organ's amount split among its segments in proportion to their weight: the organs' totals conserved."""
    g, ds = _couple(_link("SegmentInputs", "Organs", "segment_share", "organ_supply"))
    ds.set("mass", np.arange(1., ds.n_nodes() + 1.))
    per_organ = _per_organ(g, ds, ds.get("segment_share"))
    np.testing.assert_allclose([sum(per_organ[o]) for o in ds.entity_ids("Organ")], 6.)
    masses = _per_organ(g, ds, ds.get("mass"))
    for o in ds.entity_ids("Organ"):
        np.testing.assert_allclose(per_organ[o], 6. * np.array(masses[o]) / sum(masses[o]))


def test_a_targeted_split_conserves_the_amount_on_the_target():
    """Split over the target's entities only; an organ with an amount but no target entity raises (it would be
    lost)."""
    def roots_only(g, ds):
        _root_mask(g, ds)
        rooted = np.unique(ds.owner("Organ")[ds.mask("roots")])
        ds.set("organ_supply", np.where(np.isin(np.arange(len(ds.entity_ids("Organ"))), rooted), 6., 0.))

    with pytest.raises(ValueError, match="no weight on their entities"):
        _couple(_link("SegmentInputs", "Organs", "segment_share", "organ_supply", target="roots"), before=_root_mask)
    g, ds = _couple(_link("SegmentInputs", "Organs", "segment_share", "organ_supply", target="roots"),
                    before=roots_only)
    roots = ds.mask("roots")
    assert (ds.get("segment_share")[~roots] == 0.).all()
    owner = ds.owner("Organ")
    for i in np.unique(owner[roots]):
        np.testing.assert_allclose(ds.get("segment_share")[(owner == i) & roots].sum(), 6.)


def test_massic_concentrations_need_a_weight_up():
    with pytest.raises(DeclarationError, match="give the weight of its mean"):
        _couple(_link("Organs", "Segments", "organ_concentration", "concentration"))
    _, ds = _couple(_link("Organs", "Segments", "organ_concentration", "concentration", weight="mass"))
    np.testing.assert_allclose(ds.get("organ_concentration"), 3.)


def test_intensive_inputs_are_broadcast_down_and_extensive_ones_need_a_mapping():
    _, ds = _couple(_link("SegmentInputs", "Organs", "segment_temperature", "organ_temperature"))
    np.testing.assert_allclose(ds.get("segment_temperature"), 20.)
    with pytest.raises(DeclarationError, match="extensive and goes from a coarse scale down"):
        _couple(_link("SegmentInputs", "Organs", "segment_supply", "organ_supply"))


def test_an_untyped_provider_needs_an_explicit_mapping():
    with pytest.raises(DeclarationError, match=r"Organs\.organ_tag <- Segments.*does not imply one"):
        _couple(_link("Organs", "Segments", "organ_tag", "tag"))
    _, ds = _couple(_link("Organs", "Segments", "organ_tag", "tag", aggregation="mean"))
    np.testing.assert_allclose(ds.get("organ_tag"), 1.)


def test_kinds_must_agree():
    with pytest.raises(ValueError, match=r"SegmentInputs\.wrong_kind \(intensive\) <- Segments\.uptake \(extensive\)"):
        _couple(_link("SegmentInputs", "Segments", "wrong_kind", "uptake"))


def test_links_between_coarse_scales():
    g, ds = _couple({})
    ds.derive("axis_supply", {"organ_supply": 1.}, location="Axis", aggregation="sum")
    axes = ds.entity_ids("Axis")
    organ_axis = [g.complex_at_scale(int(o), g.scales.Axis) for o in ds.entity_ids("Organ")]
    np.testing.assert_allclose(ds.get("axis_supply"), [6. * organ_axis.count(a) for a in axes])
    ds.derive("organ_axis_supply", {"axis_supply": 1.}, location="Organ", aggregation="broadcast")
    np.testing.assert_allclose(ds.get("organ_axis_supply"), [6. * organ_axis.count(a) for a in organ_axis])


def test_couplability_reports_kind_conflicts_and_missing_mappings():
    from openalea.metafspm.coupling.translator import Translator
    from openalea.metafspm.testing import couplability_problems
    g, ds = _couple({})
    translator = Translator.from_dict({**_link("SegmentInputs", "Segments", "wrong_kind", "uptake"),
                                       "Organs": {"Segments": {"organ_tag": {"sources": {"tag": 1.}}}}})
    problems = couplability_problems(SegmentInputs, translator, data_structure=ds)
    assert any("the kinds do not agree" in p for p in problems)
    problems = couplability_problems(Organs, translator, data_structure=ds)
    assert any("organ_tag <- Segments" in p and "does not imply one" in p for p in problems)


# ---------------------------------------------------------------- link scales and targets

def test_link_scales_are_checked_against_the_declarations():
    _couple(_link("Organs", "Segments", "organ_uptake", "uptake", scale="Organ", source_scale="SubOrgan"))
    with pytest.raises(ValueError, match="states that 'organ_uptake' is at scale 6 .* declared at Organ"):
        _couple(_link("Organs", "Segments", "organ_uptake", "uptake", scale=scales.SubOrgan))
    with pytest.raises(ValueError, match="states that 'uptake' is at scale 5 .* declared at node"):
        _couple(_link("Organs", "Segments", "organ_uptake", "uptake", source_scale=scales.Organ))


def _root_mask(g, ds):
    ds.register("label", ds._mtg_to_node_array("label"), location="node")
    ds.define_mask("roots", {"label": g.labels.SubOrgan.RootSegment})


def test_a_targeted_link_maps_to_the_mask_entities_only():
    g, ds = _couple(_link("SegmentInputs", "Organs", "segment_temperature", "organ_temperature", target="roots"),
                    before=_root_mask)
    roots = ds.mask("roots")
    np.testing.assert_allclose(ds.get("segment_temperature")[roots], 20.)
    np.testing.assert_allclose(ds.get("segment_temperature")[~roots], 0.)   # the receiver's default


def test_targets_in_python_translators_and_unknown_masks():
    from openalea.metafspm.coupling.translator import Translator
    translator = Translator().link("SegmentInputs", "segment_temperature", "Organs", {"organ_temperature": 1.},
                                   target="roots")
    assert translator.links[0].target == "roots" and translator.links[0].kind == "derived"
    with pytest.raises(KeyError, match="target mask 'roots' is not defined"):
        _couple(_link("SegmentInputs", "Organs", "segment_temperature", "organ_temperature", target="roots"))


def test_python_translators_keep_link_options_through_the_composite(tmp_path):
    """A .py translator with aggregation, weight and formula links, through the public entry point."""
    module = tmp_path / "translator.py"
    module.write_text(
        "from openalea.metafspm.coupling.translator import Translator\n"
        "from openalea.metafspm.data_structure.configs import ScalesConfig as scales\n"
        "translator = (Translator()\n"
        "    .link('Organs', 'organ_uptake', 'Segments', {'uptake': 1.}, scale=scales.Organ)\n"
        "    .link('Organs', 'organ_concentration', 'Segments', {'concentration': 1.}, aggregation='weighted_mean',\n"
        "          weight='mass')\n"
        "    .link('Organs', 'organ_tag', 'Segments', ('uptake', 'mass'), formula=lambda u, m: u * m, scale=scales.Organ,\n"
        "          aggregation='sum'))\n")
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    components = (Segments(data_structure=ds), Organs(data_structure=ds))
    CompositeModel().declare_data_and_couple_components(translator_path=str(module), components=components)
    per_organ = _per_organ(g, ds, np.ones(ds.n_nodes()))
    np.testing.assert_allclose(ds.get("organ_uptake"), [len(per_organ[o]) for o in ds.entity_ids("Organ")])
    np.testing.assert_allclose(ds.get("organ_concentration"), 3.)
    np.testing.assert_allclose(ds.get("organ_tag"), 0.5 * ds.get("organ_uptake"))


def test_a_link_between_different_units_needs_a_conversion_factor():
    from openalea.metafspm.coupling.declaration import units_agree

    @dataclass
    class InHectopascal(FunctionalComponent):
        p_hpa: float = input_variable(**{**DOC, "unit": "hPa"}, by="InMegapascal", initialize=0.,
                                      scale=scales.SubOrgan)

    @dataclass
    class InMegapascal(FunctionalComponent):
        p_mpa: float = state_variable(**{**DOC, "unit": "MPa"}, initialize=1., scale=scales.SubOrgan,
                                      state_variable_type="intensive")

    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    model = CompositeModel()
    model.components = [InHectopascal(data_structure=ds), InMegapascal(data_structure=ds)]
    with pytest.raises(ValueError, match="does not convert 'MPa' into 'hPa'"):
        model._couple_on_data_structures(_link("InHectopascal", "InMegapascal", "p_hpa", "p_mpa"))
    with pytest.raises(ValueError, match="does not convert"):                  # a wrong conversion
        model._couple_on_data_structures({"InHectopascal": {"InMegapascal": {"p_hpa": {"sources": {"p_mpa": 1e3}}}}})
    model._couple_on_data_structures({"InHectopascal": {"InMegapascal": {"p_hpa": {"sources": {"p_mpa": 1e4}}}}})
    np.testing.assert_allclose(ds.get("p_hpa"), 1e4)
    model._couple_on_data_structures({"InHectopascal": {"InMegapascal": {"p_hpa": {"sources": {"p_mpa": 0.5},
                                                                                 "unit_check": False}}}})
    assert units_agree("cm3 d-1", "cm3.d-1") and units_agree("", "MPa") and not units_agree("MPa", "hPa")


def test_units_are_read_as_quantities():
    from openalea.metafspm.coupling.units import canonical, check_link, conversion_factor, parse_unit
    assert parse_unit("mol.m-2.s-1") == parse_unit("mol m-2 s-1") == ((-2, 0, -1, 1, 0), 1.)
    assert conversion_factor("cm", "m") == pytest.approx(100.) and conversion_factor("mm3 s-1", "m3 s-1") == pytest.approx(1e9)
    assert parse_unit("gC.g-1.s-1")[0] == (0, 0, -1, 0, 0) and parse_unit("mm/h")[0] == (1, 0, -1, 0, 0)
    assert parse_unit("adim") == parse_unit("-") == parse_unit("dimensionless") == ((0,) * 5, 1.)
    assert parse_unit("") is None and parse_unit("furlong") is None
    assert canonical("mol of hexose.s-1") == "mol s-1" and canonical("-") == "adim" and canonical(".m-1") == "m-1"
    with pytest.raises(ValueError, match="not the same quantity"):
        check_link("m", "s", 1., "a link")
    check_link("", "MPa", 1., "undeclared")                                   # agrees with anything
    check_link("furlong", "furlong", 1., "unreadable, spelled alike")
    with pytest.raises(ValueError):
        conversion_factor("m", "kg")
