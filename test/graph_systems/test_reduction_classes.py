"""
solve.reduction's kind classes (Relative(width, interpolate=True)), its library and its kinds' batches and
checkpoints, on toy sections (test/helpers/layered_anatomy.py) whose membranes' conductance grows with the section's
length: the interior's Kron reduction is a rational function of the length, so the class kinds' blend is exact at the
class values and within the linear interpolation's bound between them; the kinds saturate (each class evaluated once);
batches, the library kept between evaluations and a checkpoint's kinds give the same kinds as one evaluation.
"""
import numpy as np
import pytest

from openalea.metafspm.compute import Compute
from openalea.metafspm.scene.population import Planting, planting, population_data_structure
from openalea.metafspm.solve import reduction as reduction_module
from openalea.metafspm.solve.reduction import Relative

from reduced_sections import N_SEGMENTS, SYSTEM, _length_plant

def _model(classes):
    """The toy plant (reduced_sections): *classes* None (exact) or a Relative; False: the whole network."""
    return _length_plant(classes)


def _population(classes, lengths):
    """A plant of the toy sections of *lengths* (one per section), not solved."""
    model = _model(classes)
    population = population_data_structure(planting([Planting()], model, {}), model)
    ds = population.data_structure
    owner = np.asarray(ds.owner("SubOrgan"))
    ds.set("length", np.asarray(lengths, dtype=float)[owner])
    return population


def _kinds(population) -> dict:
    """{group: the plant system's Kind} of every reduced group."""
    ds = population.data_structure
    reduction = ds.structure.reduction
    return {i: by_system[SYSTEM] for i, by_system in
            reduction.group_kinds(ds, reduction.reduced_groups(ds), release=True).items()}


class _Counter:
    """The kinds _evaluate is asked for (monkeypatched)."""

    def __init__(self, monkeypatch):
        self.kinds, self.calls = 0, 0
        evaluate = reduction_module._evaluate

        def counted(reduction, parameters, requests):
            self.kinds += len(requests)
            self.calls += 1
            return evaluate(reduction, parameters, requests)
        monkeypatch.setattr(reduction_module, "_evaluate", counted)


WIDTH = 0.1
CLASS = Relative(WIDTH)


def _class(k) -> float:
    return float(np.exp(k * np.log1p(WIDTH)))


def test_a_value_has_its_two_neighbouring_classes_weighted_by_its_place_between_them():
    low, high = _class(3), _class(4)
    value = 0.25 * low + 0.75 * high
    (w_low, c_low), (w_high, c_high) = CLASS.corners(value)
    assert (c_low, c_high) == (low, high)
    np.testing.assert_allclose((w_low, w_high), (0.25, 0.75), rtol=1e-12)
    assert CLASS.corners(low) == ((1., low),)                       # a class value: its class alone
    assert Relative(WIDTH, interpolate=False).corners(value) == ((1., CLASS.classify(value)),)
    assert CLASS.corners(0.) == ((1., 0.),) and CLASS.corners(-2.) == ((1., -2.),)
    with pytest.raises(ValueError, match="positive"):
        Relative(0.)


def test_the_blend_at_a_class_value_is_that_class_kind():
    """A section of a class value has its class's kind alone: the exact kind of that length."""
    lengths = [_class(k) for k in range(N_SEGMENTS)]
    classed, exact = _kinds(_population(CLASS, lengths)), _kinds(_population(None, lengths))
    for i in classed:
        assert classed[i].parts is None
        np.testing.assert_allclose(classed[i].D, exact[i].D, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(classed[i].e, exact[i].e, rtol=1e-13, atol=1e-15)


def test_between_classes_the_blend_is_within_the_linear_interpolations_bound():
    """Between two classes the blend's D is the linear interpolation of the class kinds': its error to the exact
    kind is below h²/8 max|D''| (h the class interval, D'' from exact kinds), far below the nearest class's; the
    potentials of the plant solved with classes are the exact ones within that bound."""
    k = 5
    low, high = _class(k), _class(k + 1)
    h = high - low
    lengths = [low + fraction * h for fraction in (0.1, 0.3, 0.5, 0.62, 0.8, 0.95)]
    classed, exact = _kinds(_population(CLASS, lengths)), _kinds(_population(None, lengths))
    nearest = _kinds(_population(Relative(WIDTH, interpolate=False), lengths))
    step = h / 4
    probes = np.linspace(low, high, 9)
    second = []
    for centre in probes:                     # D'' of the exact kinds across the interval (central differences)
        around = _kinds(_population(None, [centre - step, centre, centre + step] * 2))
        second.append(np.abs(around[0].D - 2 * around[1].D + around[2].D) / step ** 2)
    bound = h ** 2 / 8 * np.max(second, axis=0)
    for i in classed:
        assert classed[i].parts is not None and len(classed[i].parts) == 2
        error = np.abs(classed[i].D - exact[i].D)
        assert np.all(error <= 1.05 * bound + 1e-15)
        assert error.max() < 0.1 * np.abs(nearest[i].D - exact[i].D).max()
    scale = np.abs(exact[0].D).max()
    assert max(np.abs(classed[i].D - exact[i].D).max() for i in classed) / scale < WIDTH ** 2 / 8

    solved = {}
    for name, classes in (("classes", CLASS), ("exact", None)):
        population = _population(classes, lengths)
        population.instance.run()
        solved[name] = np.asarray(population.data_structure.get("water_potential"))
    np.testing.assert_allclose(solved["classes"], solved["exact"], rtol=WIDTH ** 2 / 8)


def test_the_blends_interior_is_its_class_kinds_weighted():
    """A blended group's interior recovered (recover, recover_interiors) is its class kinds' interiors weighted."""
    lengths = [_class(2) * (1 + 0.04 * i) for i in range(N_SEGMENTS)]
    population = _population(CLASS, lengths)
    population.instance.run()
    ds = population.data_structure
    reduction = ds.structure.reduction
    kinds = _kinds(population)
    every = reduction.recover_interiors(ds)
    for i, kind in kinds.items():
        kept = np.flatnonzero(np.asarray(ds.owner("SubOrgan")) == i)
        values = np.asarray(ds.get("water_potential"))[kept]
        rows, interior = reduction.recover(ds, i, values)
        parts = kind.parts or ((1., kind),)
        expected = sum(weight * part.reduction.recover(values, -part.interior_source) for weight, part in parts)
        np.testing.assert_allclose(interior, expected, rtol=1e-12)
        np.testing.assert_allclose(every[i][4], expected, rtol=1e-12)
        np.testing.assert_array_equal(every[i][3], rows)


def test_the_kinds_saturate_each_class_evaluated_once(monkeypatch):
    """Sections growing through the classes: each class's kind is evaluated once (kept while its family is in
    use), so once every class is seen no kind is evaluated again; the exact kinds are evaluated at every length."""
    counter = _Counter(monkeypatch)
    population = _population(CLASS, [_class(0)] * N_SEGMENTS)
    ds = population.data_structure
    owner = np.asarray(ds.owner("SubOrgan"))
    growth = np.linspace(0., 12., 40)
    for cycle in range(2):
        for t in growth:
            lengths = np.array([_class(0) * (1 + WIDTH) ** (t + i / N_SEGMENTS) for i in range(N_SEGMENTS)])
            ds.set("length", lengths[owner])
            _kinds(population)
        if cycle == 0:
            first = counter.kinds
    assert first <= 14 + 1                    # the 14 classes from (1.1)^0 to (1.1)^13, once each
    assert counter.kinds == first             # the second growth: none evaluated again
    reduction = ds.structure.reduction
    assert len(reduction._kinds) == first


def test_batches_and_the_kept_library_give_the_same_kinds_as_one_evaluation(monkeypatch):
    """Kinds evaluated in batches of at most Compute.kind_batch interior cells (one batch per kind here), in the
    library kept between batches (each batch's groups replacing the former's), are those of one evaluation in a
    fresh library, bit for bit."""
    lengths = [0.3 + 0.37 * i for i in range(N_SEGMENTS)]
    reference = _kinds(_population(None, lengths))
    counter = _Counter(monkeypatch)
    population = _population(None, lengths)
    reduction = population.data_structure.structure.reduction
    interior = reduction._interior_cells(next(iter(reduction._full)))
    with Compute(kind_batch=interior).limits():
        batched = _kinds(population)
    assert counter.calls == N_SEGMENTS and counter.kinds == N_SEGMENTS
    assert len(reduction._libraries) == 1
    library = next(iter(reduction._libraries.values()))
    assert library.population is not None
    for i in reference:
        np.testing.assert_array_equal(batched[i].D, reference[i].D)
        np.testing.assert_array_equal(batched[i].e, reference[i].e)
    population.data_structure.set("length", np.asarray(population.data_structure.get("length")) * 1.5)
    again = _kinds(population)                                   # the same library, its groups replaced
    fresh = _kinds(_population(None, [1.5 * value for value in lengths]))
    assert len(reduction._libraries) == 1 and next(iter(reduction._libraries.values())) is library
    for i in fresh:
        np.testing.assert_array_equal(again[i].D, fresh[i].D)


def test_a_checkpoint_keeps_the_kinds_a_resumed_plant_evaluates_none(tmp_path, monkeypatch):
    """The kinds are in the reduction's checkpoint (shared parts, written once): a population loaded from it solves
    as before without evaluating a kind (no library built)."""
    lengths = [_class(1) * (1 + 0.07 * i) for i in range(N_SEGMENTS)]
    population = _population(CLASS, lengths)
    population.instance.run()
    from openalea.metafspm.scene.shared import SharedStore
    solved = population.data_structure
    shared = tmp_path / "shared"
    store = SharedStore(str(shared))
    solved.checkpoint(str(tmp_path / "checkpoint"), shared=store)
    names = sorted(path.name for path in shared.iterdir())
    kinds = len(solved.structure.reduction._kinds)
    assert sum(name.startswith("kind-") for name in names) == kinds
    solved.checkpoint(str(tmp_path / "checkpoint_2"), shared=store)
    assert sorted(path.name for path in shared.iterdir()) == names     # written once

    counter = _Counter(monkeypatch)
    model = _model(CLASS)
    fresh = population_data_structure(planting([Planting()], model, {}), model)
    fresh.data_structure.load_checkpoint(str(tmp_path / "checkpoint"))
    reduction = fresh.data_structure.structure.reduction
    assert len(reduction._kinds) == kinds and not reduction._libraries
    fresh.instance.run()
    assert counter.kinds == 0 and not reduction._libraries
    np.testing.assert_allclose(fresh.data_structure.get("water_potential"), solved.get("water_potential"),
                               rtol=1e-12)


class _Shapes:
    """An object parameter of the default repr (its address), as Adel's LeafShapes."""

    def __init__(self, values):
        self.values = values


def test_the_signature_of_an_object_parameter_is_its_contents_not_its_address():
    """A plant's object parameter of the default repr is digested by its contents: a copy (a checkpoint's load in
    another process) keys the same kinds, another content others."""
    import copy
    import pickle
    from openalea.metafspm.solve.reduction import Reduction, _signature
    reduction = Reduction()                                  # the pickles kept on it
    shapes = _Shapes({"blade": np.linspace(0., 1., 5)})
    assert _signature(reduction, {"leaves": shapes}) == _signature(reduction, {"leaves": copy.deepcopy(shapes)})
    assert _signature(reduction, {"leaves": shapes}) == \
        _signature(reduction, {"leaves": pickle.loads(pickle.dumps(shapes))})
    assert _signature(reduction, {"leaves": np.array([shapes, None], dtype=object)}) == \
        _signature(reduction, {"leaves": np.array([copy.deepcopy(shapes), None], dtype=object)})
    assert _signature(reduction, {"leaves": shapes}) != \
        _signature(reduction, {"leaves": _Shapes({"blade": np.linspace(0., 2., 5)})})
