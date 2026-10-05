"""
Test doubles shared by the wrapper tests:
- TRANSLATOR: a coupling translator with one entry per link kind, under the names RootCarbon and RootNitrogen
  (doubles_ds.translator() gives it with the DataStructure components' names, PlantCarbon and PlantNitrogen).
The DataStructure-backed plant and soil doubles are in doubles_ds.py.
"""
import yaml

TIME_STEP = 3600

SEGMENT_LENGTH = 0.02
SOIL_VOXEL_SIDE = 0.05


# ---------------------------------------------------------------- translator

# translator[receiver][provider][receiver_variable] = {provider_variable: factor}
# One entry per link kind exercised by the wrapper tests.
TRANSLATOR = {
    "RootCarbon": {
        "RootCarbon": {},
        "RootNitrogen": {
            "nitrogen_status": {"amino_acids": 1.0, "nitrate": 0.5},  # multi-source weighted sum
        },
        "SoilModel": {
            "soil_temperature": {"soil_temperature": 1.0},  # soil output, identity
        },
    },
    "RootNitrogen": {
        "RootCarbon": {
            "hexose": {"hexose": 1.0},  # identity
            "sugar": {"hexose": 1.0},  # alias, different name
            "carbon_supply": {"hexose_exudation": 2.0},  # numeric factor
        },
        "RootNitrogen": {},
        "SoilModel": {
            "C_hexose_soil": {"C_hexose_soil": 1.0},  # soil output, identity
        },
    },
    "SoilModel": {
        "RootCarbon": {
            "hexose_exudation_massic": {"hexose_exudation": "12 * 6"},  # string expression, _massic rename
        },
        "RootNitrogen": {
            "amino_acids_exudation": {"amino_acids_exudation": 5.0},  # same name with factor != 1
        },
        "SoilModel": {},
    },
}


def write_translator(path, translator=None):
    with open(path, "w") as f:
        yaml.dump(TRANSLATOR if translator is None else translator, f)
    return str(path)
