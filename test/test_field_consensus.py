from generic_fspm.field_consensus import model_var


def test_field_consensus():
    f: float = model_var(default=0., min_value=0., max_value=1., unit="mol.s-1", unit_comment="test", description="",
                  value_comment="", references="", DOI="", variable_type="", by="", state_variable_type="", edit_by="")
    assert f.default == 0.