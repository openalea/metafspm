# The carbon model of test_coupling.py in VirtualPlantLab (PlantGraphs), as a participant of an encoding3d coupling
# session (encoding3d.jl): launched by the coordinator, it refines each of metafspm's root segments into two layers, grown
# by a graph rewriting rule, and gives each layer a hexose concentration, (rank + 1) / (1 + its segment's structural
# mass), computed on the graph by a query. Its graph is sent with encoding3d's PlantGraphs extension (from_graph).
using encoding3d
using PlantGraphs
using ZMQ

struct Tissue <: Node end
struct Bud <: Node end
mutable struct Layer <: Node
    segment::Int64                                  # its segment's id in the session
    rank::Int
    hexose::Float64
end
mutable struct NewSegments                          # the graph's data: the segments to refine at the next rewriting
    segments::Vector{Int64}
end

refine = Rule(Bud, rhs = bud -> begin
    segments = data(graph(bud)).segments
    isempty(segments) ? Bud() : Bud() + Tuple(Layer(segment, 0, 0.0) + Layer(segment, 1, 0.0) for segment in segments)
end)
tissue = Graph(axiom = Tissue() + Bud(), rules = refine, data = NewSegments(Int64[]))

declaration = Declaration("carbon"; platform = "VirtualPlantLab",
    entity_sets = [EntitySetSpec("plant", "SubOrgan"), EntitySetSpec("plant", "Layer"; rights = "refines")],
    variables = [VariableSpec("plant", "Layer", "struct_mass", "requires"; unit = "g", kind = "extensive"),
                 VariableSpec("plant", "Layer", "hexose", "provides"; unit = "mol.g-1",
                              kind = "massic_concentration")])
participant = Participant(declaration; timeout = 300)
offset = participant.id_offset
refined = Set{Int64}()

for turn in turns(participant)
    plant = data_structure(turn.frame, "plant")
    # each segment's structural mass, from its layers' (copied down by the coordinator)
    mass = Dict{Int64,Float64}()
    if any(set -> set.name == "Layer", plant.entity_sets)
        received = entity_set(plant, "Layer")
        written = entity_set(from_graph(tissue; id = "plant", entity_set = "Layer", rank = 7,
                                        keep = node -> node isa Layer), "Layer")
        segment_of = Dict(id + offset => segment for (id, segment) in
                          zip(written.columns["id"], written.columns["segment"]))
        for (layer, value) in zip(received.columns["id"], received.columns["struct_mass"])
            isnan(value) || (mass[segment_of[layer]] = value)
        end
    end
    # the new segments refined by the rule, then every layer's hexose by a query
    data(tissue).segments = [segment for segment in entity_set(plant, "SubOrgan").columns["id"]
                             if !(segment in refined)]
    union!(refined, data(tissue).segments)
    rewrite!(tissue)
    for layer in apply(tissue, Query(Layer))
        layer.hexose = (layer.rank + 1) / (1 + get(mass, layer.segment, 0.0))
    end
    # the layers as the session's entities: ids in the participant's range, their segment as complex
    layers = entity_set(from_graph(tissue; id = "plant", entity_set = "Layer", rank = 7,
                                   keep = node -> node isa Layer), "Layer")
    columns = encoding3d.Columns(
        "id" => layers.columns["id"] .+ offset,
        "complex" => Vector{Int64}(layers.columns["segment"]),
        "parent" => [parent == NO_ID ? NO_ID : parent + offset for parent in layers.columns["parent"]],
        "edge_type" => layers.columns["edge_type"],
        "label" => layers.columns["label"],
        "hexose" => Vector{Float64}(layers.columns["hexose"]))
    send_update(participant, Frame(data_structures = [DataStructure("plant";
        entity_sets = [EntitySet("Layer"; columns = columns, rank = 7)])]))
end
