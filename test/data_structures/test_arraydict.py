from openalea.metafspm.data_structure.arraydict import ArrayDict  # adjust import to your file name

# This test checks the invariance of ArrayDict operations
def test_arraydict():
    ad = ArrayDict({3: 0.3, 1: 0.1, 2: 0.2})
    assert list(ad.keys()) == [1, 2, 3]

    ad[5] = 0.5        # insertion
    ad[4] = 0.4
    assert list(ad.keys()) == [1, 2, 3, 4, 5]

    ad.update({2: 9, 10: 1, 0: 0})
    assert list(ad.keys()) == [0, 1, 2, 3, 4, 5, 10]

    ad.assign_at([0, 2], [7, 8])  # value updates leave order intact
    ad.__delitem__(3)             # deletion
    assert ad._check_invariant()   # confirms sorted order and mapping consistency


def test_delete_many_equals_deleting_one_key_at_a_time():
    import numpy as np
    from openalea.metafspm.data_structure.arraydict import ArrayDict
    keys = np.arange(0, 400, 3)
    bulk, single = ArrayDict(), ArrayDict()
    for d in (bulk, single):
        d.update({int(k): float(k) / 2 for k in keys})
    doomed = [int(k) for k in keys[::4]] + [10**6]            # an absent key is ignored
    assert bulk.delete_many(doomed) == len(keys[::4])
    for k in doomed[:-1]:
        del single[k]
    assert list(bulk) == list(single) and bulk.values_array().tolist() == single.values_array().tolist()
    assert all(bulk[k] == single[k] for k in single)
    bulk[1] = 5.                                                # inserts still keep the order
    assert list(bulk)[:2] == [1, 3]


def test_delete_range_equals_deleting_its_keys():
    """delete_range(start, stop): the keys in [start, stop) gone (an instance's block of ids), as delete_many."""
    import numpy as np
    from openalea.metafspm.data_structure.arraydict import ArrayDict
    keys = np.arange(0, 400, 3)
    ranged, many = ArrayDict(), ArrayDict()
    for d in (ranged, many):
        d.update({int(k): float(k) / 2 for k in keys})
    assert ranged.delete_range(100, 250) == many.delete_many(np.arange(100, 250))
    assert list(ranged) == list(many) and ranged.values_array().tolist() == many.values_array().tolist()
    assert ranged.delete_range(1000, 2000) == 0 and ranged.delete_range(5, 5) == 0
    assert ranged[99] == many[99] and 102 not in ranged
    ranged[101] = 1.                                            # inserts still keep the order
    assert list(ranged)[list(ranged).index(99) + 1] == 101


def test_bulk_construction_equals_inserting_each_key():
    import numpy as np
    from openalea.metafspm.data_structure.arraydict import ArrayDict
    rng = np.random.default_rng(0)
    keys = rng.permutation(500)[:300].tolist() + [7, 7]          # unsorted, a key repeated (the last value wins)
    values = rng.normal(size=len(keys)).tolist()
    values[-1] = None                                           # None becomes NaN, as an insert makes it
    inserted = ArrayDict()
    for k, v in zip(keys, values):
        inserted[k] = v
    bulk = ArrayDict.from_items(keys, values)
    assert list(bulk) == list(inserted) == sorted(set(keys))
    np.testing.assert_array_equal(bulk.values_array(), inserted.values_array())
    assert ArrayDict(dict(zip(keys, values))).keys_array().tolist() == bulk.keys_array().tolist()
    bulk[1000] = 2.                                             # and it keeps working as an ArrayDict
    del bulk[keys[0]]
    assert bulk[1000] == 2. and keys[0] not in bulk


def test_bulk_operations_hold_no_key_dict():
    """Bulk writes and reads find keys by binary search: no dict of the keys until a key is used on its own."""
    import numpy as np
    keys = np.arange(0, 3000, 3)
    d = ArrayDict.from_items(keys, keys / 2.)
    d.scatter(keys[::5].tolist(), np.zeros(keys[::5].size))
    d.insert_many(np.array([1, 4, 7000]), np.array([1., 4., 7.]))
    d.update({10: 5., 9: 4.5})
    assert d._index is None
    assert 4 in d and 5 not in d and "x" not in d and d._index is None
    assert d[4] == 4. and d._index is not None                 # a key on its own: the dict, built once
    assert d._check_invariant()


def test_insert_many_equals_inserting_each_key():
    import numpy as np
    rng = np.random.default_rng(0)
    existing = np.unique(rng.integers(0, 1000, 200))
    new = np.setdiff1d(np.unique(rng.integers(0, 1200, 150)), existing)
    rng.shuffle(new)
    bulk = ArrayDict.from_items(existing, existing * 1.)
    single = ArrayDict.from_items(existing, existing * 1.)
    bulk.insert_many(new, new * 2.)
    for k in new.tolist():
        single[k] = k * 2.
    assert list(bulk.items()) == list(single.items())
    assert bulk._check_invariant() and single._check_invariant()
    with __import__("pytest").raises(KeyError):
        bulk.indices_of([-1])
