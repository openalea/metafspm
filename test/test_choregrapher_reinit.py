from utils import deep_reload_package
deep_reload_package("openalea")
from openalea.metafspm.component_factory import Choregrapher


def test_reinit_Choregrapher():
    c1 = Choregrapher()
    c1.potential[1] = 1

    Choregrapher._instance = None

    c2 = Choregrapher()
    c2.potential[2] = 2

    assert id(c1) != id(c2)
    assert c2.potential == {2:2}


if __name__ == '__main__':
    test_reinit_Choregrapher()