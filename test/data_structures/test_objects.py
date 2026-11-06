"""
The objects a DataStructure's components share by content (ds.objects: data_structure.objects.ObjectRegistry): one
registry per DataStructure (an MPG-backed one's on its MPG), keyed by content, checkpointed once per object (Shared).
"""
import os

import numpy as np
import pytest

from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure  # noqa: F401
from openalea.metafspm.scene.shared import SharedStore
from openalea.metafspm.solve.memo import content_key

from growth import make_chain


class Section:
    """A costly object built from its parameters (a template), with a part a checkpoint leaves out (its organ)."""

    built = 0

    def __init__(self, parameters):
        Section.built += 1
        self.parameters, self.organ = dict(parameters), object()


def saved_section(section):
    """The form a checkpoint saves: without its organ."""
    copy = Section.__new__(Section)
    copy.parameters, copy.organ = section.parameters, None
    return copy


def _plant():
    return make_chain(3)[1]


def test_objects_are_built_once_per_content_and_numbered():
    ds = _plant()
    sections = ds.objects("sections", saved=saved_section)
    built = Section.built
    first = sections.get(content_key({"layers": 3}), lambda: Section({"layers": 3}))
    again = sections.get(content_key({"layers": 3}), lambda: Section({"layers": 3}))
    other = sections.get(content_key({"layers": 4}), lambda: Section({"layers": 4}))
    assert first is again and other is not first and Section.built == built + 2
    assert sections.index(content_key({"layers": 4})) == 1 and sections.by_index(1) is other
    assert ds.objects("sections") is sections is ds.structure.objects("sections")     # the MPG's
    with pytest.raises(KeyError):
        sections.by_index(2)


def test_each_data_structure_has_its_own_registry():
    """Two DataStructures (two Scenes') never share objects: what one builds is freed with it."""
    one, two = _plant(), ArrayDataStructure(shape=(2,))
    one.objects("sections").add("key", 1.)
    assert "key" not in two.objects("sections") and len(two.objects("sections")) == 0


def test_a_checkpoint_saves_each_object_once_in_its_saved_form(tmp_path):
    ds = _plant()
    sections = ds.objects("sections", saved=saved_section)
    sections.get("a", lambda: Section({"layers": 3}))
    sections.get("b", lambda: Section({"layers": 4}))
    store = SharedStore(str(tmp_path / "shared"))
    ds.checkpoint(str(tmp_path / "c1"), shared=store)
    ds.checkpoint(str(tmp_path / "c2"), shared=store)
    assert len([name for name in os.listdir(tmp_path / "shared") if name.startswith("sections-")]) == 2
    restored = MPGDataStructure.restore(str(tmp_path / "c2"))
    loaded = restored.objects("sections")
    assert loaded.keys() == ["a", "b"] and loaded.by_index(1).parameters == {"layers": 4}
    assert loaded.by_index(0).organ is None                          # its saved form
    assert sections.by_index(0).organ is not None                    # the live one untouched
    grid = ArrayDataStructure(shape=(2,))
    grid.objects("tables").add("t", np.arange(3.))
    grid.checkpoint(str(tmp_path / "grid"))
    np.testing.assert_array_equal(ArrayDataStructure.restore(str(tmp_path / "grid")).objects("tables")["t"],
                                  np.arange(3.))


def test_an_mpg_shares_another_ones_registries():
    """A reduction's library population is built on its plants' objects (MPG.share_objects)."""
    plants, library = make_chain(2)[0], make_chain(1)[0]
    plants.objects("sections").add("a", 1.)
    library.share_objects(plants)
    assert library.objects("sections")["a"] == 1.
    library.objects("sections").add("b", 2.)
    assert plants.objects("sections")["b"] == 2.
