"""
Derived variables resolved at read: per-variable write
counters, recomputation at get() when a source changed, and read-only derived variables.
"""
import numpy as np
import pytest

from openalea.metafspm.data_structure.data_api import ArrayDataStructure


def _ds(n=4):
    data_structure = ArrayDataStructure(shape=(n, 1, 1))
    data_structure.register("amino_acids", np.arange(n, dtype=float).reshape(n, 1, 1), location="cell")
    data_structure.register("nitrate", 2., location="cell")
    return data_structure


class Counting:
    """Formula that counts its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, a, b):
        self.calls += 1
        return a + 0.5 * b


def test_a_derived_variable_is_recomputed_when_read_after_a_source_write():
    data_structure = _ds()
    view = data_structure.derive("nitrogen_status", {"amino_acids": 1., "nitrate": 0.5})
    data_structure.set("nitrate", 4.)
    assert data_structure.is_stale("nitrogen_status")
    status = data_structure.get("nitrogen_status")
    assert status is view   # recomputed in place: earlier views stay valid
    np.testing.assert_allclose(status.ravel(), np.arange(4.) + 2.)
    assert not data_structure.is_stale("nitrogen_status")


def test_nothing_is_recomputed_when_no_source_changed():
    data_structure, formula = _ds(), Counting()
    data_structure.derive("status", ("amino_acids", "nitrate"), formula=formula)
    assert formula.calls == 1
    for _ in range(3):
        data_structure.get("status")
    assert formula.calls == 1
    data_structure.set("amino_acids", 1.)
    data_structure.get("status")
    data_structure.get("status")
    assert formula.calls == 2


def test_refresh_forces_a_recomputation():
    data_structure, formula = _ds(), Counting()
    data_structure.derive("status", ("amino_acids", "nitrate"), formula=formula)
    data_structure.refresh("status")
    data_structure.refresh()
    assert formula.calls == 3


def test_chains_are_recomputed_in_dependency_order():
    data_structure = _ds()
    data_structure.derive("status", {"amino_acids": 1., "nitrate": 0.5})
    data_structure.derive("doubled_status", {"status": 2.})
    data_structure.set("amino_acids", 10.)
    assert data_structure.is_stale("doubled_status")
    np.testing.assert_allclose(data_structure.get("doubled_status"), 2. * (10. + 1.))
    assert not data_structure.is_stale("status")   # recomputed on the way


def test_aliases_of_and_to_derived_variables():
    data_structure = _ds()
    data_structure.alias("N", "nitrate")
    data_structure.derive("status", {"amino_acids": 1., "N": 0.5})
    data_structure.alias("status_alias", "status")
    data_structure.set("N", 6.)                          # a write through the alias counts for the source
    np.testing.assert_allclose(data_structure.get("status_alias").ravel(), np.arange(4.) + 3.)
    assert data_structure.write_count("N") == data_structure.write_count("nitrate")


def test_derived_variables_are_read_only():
    data_structure = _ds()
    data_structure.derive("status", {"amino_acids": 1., "nitrate": 0.5})
    with pytest.raises(ValueError, match="derived from .* write its sources instead"):
        data_structure.set("status", 0.)
    data_structure.alias("status_alias", "status")
    with pytest.raises(ValueError, match="write its sources instead"):
        data_structure.set("status_alias", 0.)


def test_set_cannot_write_a_derived_variable():
    from simple_seedling import generate_simple_mpg_seedling
    from openalea.metafspm.data_structure.data_api import MPGDataStructure
    mtg, _ = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    data_structure.register("x", 1., location="node")
    data_structure.derive("y", {"x": 2.})
    with pytest.raises(ValueError, match="write its sources instead"):
        data_structure.set("y", np.zeros(data_structure.n_nodes()))


def test_writes_through_a_view_need_mark_written():
    """Unsupported unless declared: the write counter only sees set() and mark_written()."""
    data_structure = _ds()
    data_structure.derive("status", {"amino_acids": 1., "nitrate": 0.5})
    data_structure.get("nitrate")[...] = 8.
    np.testing.assert_allclose(data_structure.get("status").ravel(), np.arange(4.) + 1.)   # stale: the write was not seen
    data_structure.mark_written("nitrate")
    np.testing.assert_allclose(data_structure.get("status").ravel(), np.arange(4.) + 4.)


def test_write_counters():
    data_structure = _ds()
    before = data_structure.write_count("nitrate")
    data_structure.set("nitrate", 1.)
    data_structure.register("nitrate", 3., location="cell")
    assert data_structure.write_count("nitrate") == before + 2
    assert data_structure.write_count("unknown") == 0


def test_a_derived_value_follows_a_source_written_by_another_component():
    """The receiver's pre-step refresh is no longer what makes it current: any read is."""
    data_structure = _ds()
    data_structure.derive("status", {"amino_acids": 1., "nitrate": 0.5})
    data_structure.set("amino_acids", 5.)                                    # e.g. PlantNitrogen's step
    assert float(data_structure._export(["status"])["status"].ravel()[0]) == 6.   # e.g. a Logger reading after the step
