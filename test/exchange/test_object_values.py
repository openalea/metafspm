"""Object variables in frames: tables (pandas DataFrames, e.g. a plant's tables) written as JSON and read back."""
import pandas as pd

from openalea.metafspm.io.fspm_tf import _decoded, _encoded


def test_a_table_round_trips():
    table = pd.DataFrame({"axe": ["MS", "T1"], "nf": [11, 9]}, index=[3, 4])
    axes, phytomers, number = _decoded(_encoded((table, table, 2)))
    assert axes.equals(table) and phytomers.equals(table) and number == 2
