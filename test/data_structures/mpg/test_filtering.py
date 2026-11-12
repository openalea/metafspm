from simple_seedling import generate_simple_mpg_seedling


mtg, _ = generate_simple_mpg_seedling()
def test_filtering():
    # filter in
    filtered = mtg.array_filtering("label", filter_in=dict(label=mtg.labels.SubOrgan.LeafElement)) 
    assert ((mtg.labels.SubOrgan.LeafElement in filtered) and (mtg.labels.Organ.Leaf not in filtered))
    # empty filter test, all except root which does not received default properties
    assert len(mtg.vertices()) - 1 == len(mtg.array_filtering("label"))
    # filter out
    filtered = mtg.array_filtering("label", filter_out=dict(label=mtg.labels.Multiscale.Anchor))
    assert ((mtg.labels.Multiscale.Anchor not in filtered) and (mtg.labels.SubOrgan.LeafElement in filtered)) 
    # filter in and out
    filtered = mtg.array_filtering("label", filter_in=dict(scale=mtg.scales.SubOrgan), filter_out=dict(label=mtg.labels.Multiscale.Anchor))
    assert ((mtg.labels.SubOrgan.LeafElement in filtered) and (mtg.labels.Multiscale.Anchor not in filtered))

if __name__ == '__main__':
    test_filtering()
