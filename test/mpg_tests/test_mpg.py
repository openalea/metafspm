from openalea.metafspm.data_structure.mpg import MPG


def test_mpg_init():
    g = MPG()

    vid = g.add_component(g.anchors[g.scales["axis"]], test_prop=1)
    print(vid)

    g.remove_anchors()
    print(g.vertices())


if __name__ == "__main__":
    test_mpg_init()
