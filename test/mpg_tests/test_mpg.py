from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.data_structure.configs import PropsConfig


def test_partial_multiscale():
    g = MPG()
    axis = g.add_system_root_at_scale(g.scales.Axis, label=g.labels.Axis.Shoot)
    growth_unit = g.add_component(axis, **PropsConfig(scale=g.scales.GrowthUnit, edge_type='/', label=g.labels.GrowthUnit.Shoot))
    phytomer = g.add_component(growth_unit, **PropsConfig(scale=g.scales.Phytomer, edge_type='/', label=g.labels.Phytomer.Shoot))
    internode = g.add_component(phytomer, **PropsConfig(scale=g.scales.Organ, edge_type='/', label=g.labels.Organ.StemInternode))
    meristem = g.add_child(internode, **PropsConfig(scale=g.scales.Organ, edge_type='/', label=g.labels.Organ.Meristem))
    leaf = g.add_child(internode, **PropsConfig(scale=g.scales.Organ, edge_type='/', label=g.labels.Organ.Leaf))
    leafelement1 = g.add_component(leaf, **PropsConfig(scale=g.scales.SubOrgan, edge_type='/', label=g.labels.SubOrgan.LeafElement))
    leafelement2 = g.add_child(leafelement1, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.LeafElement))
    leafelement3 = g.add_child(leafelement2, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.LeafElement))
    
    assert len(g.vertices()) == 1 + 10 + 9 # root + number of scales + number of created elements


if __name__ == "__main__":
    test_partial_multiscale() 
