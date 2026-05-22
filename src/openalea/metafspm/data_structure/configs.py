
from dataclasses import dataclass, field, fields
import inspect


@dataclass
class ScalesConfig:
    Plant:      int = field(default=1, metadata=dict(
        description="Whole plant shoot and root axes",
    ))
    Axis:       int = field(default=2, metadata=dict(
        description="branched shoot or root axes rooted at shoot-root junction",
    ))
    GrowthUnit: int = field(default=3, metadata=dict(
        description="Set of phytomers between consecutive meristem winter dormancy.",
    )) 
    Phytomer:   int = field(default=4, metadata=dict(
        description="Valid for shoot and root: Set of (i) internode, including terminal bud if this is the terminal phytomer, (ii) lateral node if not emerged or branching point if emerged, (iii, shoot only) leaf.",
    ))
    Organ:      int = field(default=5, metadata=dict(
        description="Plant organ specialized for a specific autotrohy functiono (C aquisition, transport, organogenesis, hydromineral acquisition).",
    ))
    SubOrgan:   int = field(default=6, metadata=dict(
        description="Discretization of an organ as longitudinal segments, girth, domains.",
    ))
    Layer:      int = field(default=7, metadata=dict(
        description="Concentric layering of a SubOrgan.",
    ))
    Cell:       int = field(default=8, metadata=dict(
        description="Explicit unitary cells",
    ))
    Compartment:int = field(default=9, metadata=dict(
        description="Storing nodes of a graph defined from all abovementionned scales",
    )) 
    Connection: int = field(default=10, metadata=dict(
        description="Storing edges of a graph defined from all abovementionned scales",
    ))

    def __init__(self, replace: dict = {}, insert: dict = {}):
        self.anchors = {}
    
    @property
    def translator(self):
        tranlator = {f.default: f.name for f in fields(self)}
        return dict(sorted(tranlator.items(), key=lambda x: x[0]))

    @property
    def name_translator(self):
        tranlator = {f.name: f.default for f in fields(self)}
        return dict(sorted(tranlator.items(), key=lambda x: x[1])) 

    @property
    def names(self):
        return list(self.translator.values())

    @property
    def values(self):
        return list(self.translator.keys())

    def __iter__(self):
        return iter(self.values)


@dataclass
class PropsConfig:
    """
    Configuration class for variables that need to be systematically assigned to MPG's vertices
    """
    # Multiscale positioning properties
    isanchor: bool = field(default=False, metadata=dict(
        description="boolean filter to select all MPG elements that are not structural scale anchors"
    ))
    scale: int = field(default=1, metadata=dict(
        description="Scale number managed by openalea.metafspm.data_structure.mpg.ScalesConfig"
    ))
    up_scale_parent: int = field(default=None, metadata=dict(
        description="Upper scale parent vertex (current scale - 1) spatially containing current vertex, stored to avoid multiple MPG traversals."
    ))
    down_scale_children: list = field(default=None, metadata=dict(
        description="Lower scale children vertices (current scale + 1) spatially contained within current vertex, stored to avoid multiple mtg traversals."
    ))

    # Vertex description properties
    label: int = field(default=0, metadata=dict(
        description="Vertex label stored as unique integer to enable vectorized elemet filtering, common scale specific examples are proposed by ScalesConfig"
    ))
    type: int = field(default=0, metadata=dict(
        description="Vertex type, i.e. label subcategory, stored as unique integer to enable vectorized elemet filtering, common scale specific examples are proposed by ScalesConfig"
    ))
    # parent: int = field(default=None, metadata=dict(
    #     description="Current vertex topological parent identifier within the same scale, stored to avoid multiple MPG traversals"
    # ))
    # label: list = field(default=None, metadata=dict(
    #     description="Current vertex topological children identifiers at same scale, stored to avoid multiple MPG traversals"
    # ))

    def __new__(cls, **kwargs):
        instance = object.__new__(cls)
        # manually init dataclass defaults since __init__ won't run
        props_config = {f.name: f.default for f in fields(cls)}
        props_config.update(kwargs)
        return props_config


class LabelsConfig:
    class Plant:
        scale = ScalesConfig.Plant
        Wheat: int = 'Wheat'
        Maize: int = 'Maize'
        FabaBean: int = 'FabaBean'
        OilseedRape: int = 'OilseedRape'

    class Axis:
        scale = ScalesConfig.Axis
        Shoot: int = "ShootAxis"
        Root: int = "RootAxis"

    class GrowthUnit:
        scalle = ScalesConfig.GrowthUnit
        Shoot: int = "ShootGrowthUnit"
        Root: int = "RootGrowthUnit"

    class Phytomer:
        scale = ScalesConfig.Phytomer
        Shoot: int = "ShootPhytomer"
        Root: int = "RootPhytomer"

    class Organ:
        scale = ScalesConfig.Organ
        Meristem: int = "Meristem"
        StemInternode: int = "StemInternode"
        RootInternode: int = "RootInternode"
        Leaf: int = "Leaf"
        Flower: int = "Flower"
        Fruit: int = "Fruit"

    class SubOrgan:
        scale = ScalesConfig.SubOrgan
        LeafElement: int = "LeafElement"
        StemElement: int = "StemElement"
        RootSegment: int = "RootSegment"

    class Layer:
        scale = ScalesConfig.Layer
        # Models not accounting for anatomy but radial layering of SubOrgans
        GreenArea: int = "GreenArea"
        HiddenZone: int = "HiddenZone"
        # Explicit anatomical models
        Epidermis: int = "Epidermis"
        Exodermis: int = "Exodermis"
        CorticalParenchyma: int = "CorticalParenchyma"
        Endodermis: int = "Endodermis"
        StelarParenchyma: int = "StelarParenchyma"
        ProtoXylem: int = "ProtoXylem"
        MetaXylem: int = "MetaXylem"
        ProtoPhloem: int = "ProtoPhloem"
        Phloem: int = "Phloem"
        Cambium: int = "Cambium"
        Aerenchyma: int = "Aerenchyma"

    class Cell:
        scale = ScalesConfig.Cell
        Epidermis: int = "EpidermisCell"
        Exodermis: int = "ExodermisCell"
        CorticalParenchyma: int = "CorticalParenchymaCell"
        Endodermis: int = "EndodermisCell"
        StelarParenchyma: int = "StelarParenchymaCell"
        ProtoXylem: int = "ProtoXylemCell"
        MetaXylem: int = "MetaXylemCell"
        ProtoPhloem: int = "ProtoPhloemCell"
        Phloem: int = "PhloemCell"
        Cambium: int = "CambiumCell"
        Aerenchyma: int = "AerenchymaCell"

    class Compartment:
        scale = ScalesConfig.Compartment
        Symplastic: int = "SymplasticNode"
        Apoplastic: int = "ApoplasticNode"

    class Connection:
        scale = ScalesConfig.Connection
        Transmembrane: int = "TransmembraneEdge"
        Symplastic: int = 'SymplasticEdge'
        Apoplastic: int = "ApoplasticNode"
    
    def __init__(self):
        self.filters = {}
        for name, value in inspect.getmembers(self.__class__):
            if inspect.isclass(value) and not name.startswith('_'):
                for attr, val in vars(value).items():
                    if not attr.startswith('_') and isinstance(val, str):
                        setattr(value, attr, self.filter_as_unique_int(val))

    def filter_as_unique_int(self, filter: str):
        if filter in self.filters.keys():
            integer = self.filters[filter]
        else:
            if len(self.filters) == 0:
                integer = 1
            else:
                integer = int(max(list(self.filters.values())) + 1)
            self.filters[filter] = integer
        return integer
