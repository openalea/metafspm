"""
Derived variables resolved at read: per-variable write
counters, recomputation at get() when a source changed, and read-only derived variables.
"""
import numpy as np
import pytest

from openalea.metafspm.data_structure.data_api import ArrayDataStructure


def _ds(n=4):
    ds = ArrayDataStructure(shape=(n, 1, 1))
    ds.register("amino_acids", np.arange(n, dtype=float).reshape(n, 1, 1), location="cell")
    ds.register("nitrate", 2., location="cell")
    return ds


class Counting:
    """Formula that counts its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, a, b):
        self.calls += 1
        return a + 0.5 * b


def test_a_derived_variable_is_recomputed_when_read_after_a_source_write():
    ds = _ds()
    view = ds.derive("nitrogen_status", {"amino_acids": 1., "nitrate": 0.5})
    ds.set("nitrate", 4.)
    assert ds.is_stale("nitrogen_status")
    status = ds.get("nitrogen_status")
    assert status is view   # recomputed in place: earlier views stay valid
    np.testing.assert_allclose(status.ravel(), np.arange(4.) + 2.)
    assert not ds.is_stale("nitrogen_status")


def test_nothing_is_recomputed_when_no_source_changed():
    ds, formula = _ds(), Counting()
    ds.derive("status", ("amino_acids", "nitrate"), formula=formula)
    assert formula.calls == 1
    for _ in range(3):
        ds.get("status")
    assert formula.calls == 1
    ds.set("amino_acids", 1.)
    ds.get("status")
    ds.get("status")
    assert formula.calls == 2


def test_refresh_forces_a_recomputation():
    ds, formula = _ds(), Counting()
    ds.derive("status", ("amino_acids", "nitrate"), formula=formula)
    ds.refresh("status")
    ds.refresh()
    assert formula.calls == 3


def test_chains_are_recomputed_in_dependency_order():
    ds = _ds()
    ds.derive("status", {"amino_acids": 1., "nitrate": 0.5})
    ds.derive("doubled_status", {"status": 2.})
    ds.set("amino_acids", 10.)
    assert ds.is_stale("doubled_status")
    np.testing.assert_allclose(ds.get("doubled_status"), 2. * (10. + 1.))
    assert not ds.is_stale("status")   # recomputed on the way


def test_aliases_of_and_to_derived_variables():
    ds = _ds()
    ds.alias("N", "nitrate")
    ds.derive("status", {"amino_acids": 1., "N": 0.5})
    ds.alias("status_alias", "status")
    ds.set("N", 6.)                          # a write through the alias counts for the source
    np.testing.assert_allclose(ds.get("status_alias").ravel(), np.arange(4.) + 3.)
    assert ds.write_count("N") == ds.write_count("nitrate")


def test_derived_variables_are_read_only():
    ds = _ds()
    ds.derive("status", {"amino_acids": 1., "nitrate": 0.5})
    with pytest.raises(ValueError, match="derived from .* write its sources instead"):
        ds.set("status", 0.)
    ds.alias("status_alias", "status")
    with pytest.raises(ValueError, match="write its sources instead"):
        ds.set("status_alias", 0.)


def test_graph_setters_cannot_write_a_derived_variable():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))
    from simple_seedling import generate_simple_mpg_seedling
    from openalea.metafspm.data_structure.data_api import MPGDataStructure
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    ds.register("x", 1., location="node")
    ds.derive("y", {"x": 2.})
    with pytest.raises(ValueError, match="write its sources instead"):
        ds.set_node_property("y", np.zeros(ds.n_nodes()))


def test_writes_through_a_view_need_mark_written():
    """Unsupported unless declared: the write counter only sees set() and mark_written()."""
    ds = _ds()
    ds.derive("status", {"amino_acids": 1., "nitrate": 0.5})
    ds.get("nitrate")[...] = 8.
    np.testing.assert_allclose(ds.get("status").ravel(), np.arange(4.) + 1.)   # stale: the write was not seen
    ds.mark_written("nitrate")
    np.testing.assert_allclose(ds.get("status").ravel(), np.arange(4.) + 4.)


def test_write_counters():
    ds = _ds()
    before = ds.write_count("nitrate")
    ds.set("nitrate", 1.)
    ds.register("nitrate", 3., location="cell")
    assert ds.write_count("nitrate") == before + 2
    assert ds.write_count("unknown") == 0


def test_a_derived_value_follows_a_source_written_by_another_component():
    """The receiver's pre-step refresh is no longer what makes it current: any read is."""
    ds = _ds()
    ds.derive("status", {"amino_acids": 1., "nitrate": 0.5})
    ds.set("amino_acids", 5.)                                    # e.g. PlantNitrogen's step
    assert float(ds.export(["status"])["status"].ravel()[0]) == 6.   # e.g. a Logger reading after the step
