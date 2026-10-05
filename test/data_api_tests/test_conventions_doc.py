"""
docs/conventions.md is checked against the code (design note datastructure_contract §7, step 1f): its tables of
locations, mappings and default mappings must name exactly what the declaration resolver implements.
"""
import os
import re

import pytest

from openalea.metafspm.coupling import declaration
from openalea.metafspm.coupling.declaration import DeclarationError, default_mapping

PAGE = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "conventions.md")


def _table(header: str) -> list:
    """Rows of the markdown table whose header starts with *header*, as lists of cells."""
    with open(PAGE, encoding="utf-8") as f:
        lines = f.read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {header}"))
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def _code(cell: str) -> list:
    return re.findall(r"`([^`]+)`", cell)


def test_the_locations_table_names_the_solver_locations():
    assert {_code(row[0])[0] for row in _table("Location")} == set(declaration.SOLVER_LOCATIONS)


def test_the_mappings_table_names_every_mapping():
    documented = {_code(row[0])[0] for row in _table("Mapping")}
    assert documented == set(declaration.UP_MAPPINGS) | set(declaration.DOWN_MAPPINGS) | set(declaration.EDGE_MAPPINGS)


def test_former_edge_mapping_names_are_documented():
    rows = {_code(row[0])[0]: row[1] for row in _table("Mapping")}
    for former, current in declaration._EDGE_MAPPING_RENAMES.items():
        assert f"formerly `{former}`" in rows[current]


def _documented_default(cell: str):
    names = _code(cell)
    return names[0] if names else None   # "error" cells name no mapping


@pytest.mark.parametrize("row", _table("`state_variable_type`"), ids=lambda row: row[0])
def test_the_default_mapping_table_matches_the_resolver(row):
    kinds = [kind.strip() for kind in row[0].split(",")]
    kinds = [None if kind == "none" else kind for kind in kinds]
    for kind in kinds:
        for direction, cell in (("up", row[1]), ("down", row[2])):
            expected = _documented_default(cell)
            if expected is None:
                with pytest.raises(DeclarationError):
                    default_mapping(kind, direction, "x", weight="w")
            else:
                assert default_mapping(kind, direction, "x", weight="w") == expected
