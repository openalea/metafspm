from simple_seedling import seedling

def test_partial_multiscale():
    g = seedling.g
    assert len(g.vertices()) == 1 + 10 + 29 # root + number of scales + number of created elements


if __name__ == "__main__":
    test_partial_multiscale() 
