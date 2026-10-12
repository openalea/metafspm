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
    assert ad.check_invariant()   # confirms sorted order and mapping consistency


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
