from simple_seedling import generate_simple_mpg_seedling


g, _ = generate_simple_mpg_seedling()
def test_filtering():
    # filter in
    filtered = g.array_filtering("label", filter_in=dict(label=g.labels.SubOrgan.LeafElement)) 
    assert ((g.labels.SubOrgan.LeafElement in filtered) and (g.labels.Organ.Leaf not in filtered))
    # empty filter test, all except root which does not received default properties
    assert len(g.vertices()) - 1 == len(g.array_filtering("label"))
    # filter out
    filtered = g.array_filtering("label", filter_out=dict(label=g.labels.Multiscale.Anchor))
    assert ((g.labels.Multiscale.Anchor not in filtered) and (g.labels.SubOrgan.LeafElement in filtered)) 
    # filter in and out
    filtered = g.array_filtering("label", filter_in=dict(scale=g.scales.SubOrgan), filter_out=dict(label=g.labels.Multiscale.Anchor))
    assert ((g.labels.SubOrgan.LeafElement in filtered) and (g.labels.Multiscale.Anchor not in filtered))

if __name__ == '__main__':
    test_filtering()
