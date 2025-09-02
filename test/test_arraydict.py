import numpy as np
from openalea.metafspm.utils import ArrayDict  # adjust import to your file name

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
