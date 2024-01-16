from dataclasses import field

def model_var(default, min_value: float, max_value: float, unit: str, unit_comment: str, description: str, value_comment: str, references: str, DOI: list,
              variable_type: str, by: str, state_variable_type: str, edit_by: str):
    """
    This function is used to constrain component variables declaration in a dataclass in a commonly admitted way

    :param default: Default value of the attribute if not superimposed by scenario of coupling.
    :param unit: International system unit
    :param unit_comment: More precision about unit (example : mol of ... per g of ...)
    :param description: Full description of the variable purpose, scale and related hypotheses.
    :param value_comment: Commentary if default value identified from literature has been overwritten, store original value and explain why it was changed.
    :param references: References for value and hypotheses (format Author et al., Year).
    :param DOI: DOIs of the referred papers, stored in a list of str.
    :param variable_type: variable type according to component standards; -input- will be read as expected input from another model. If not coupled, the model will keep this default. -state_variable- are the segment scale state variables. -plant_scale_state- are the summed totals variables or the plant scale variables. -parameter- are the model parameters.
    :param by: which model is provider of the considered variable?
    :param state_variable_type: intensive (size invariant property) or extensive (size dependant property).
    :param edit_by: Explicit whether the considered default value could be superimposed only by a developer or a user. This can be used to exposed more precisely certain parametrization sensitive variables for user.
    :raise BadDefaultError: If '(type(min_value)==float and default < min_value) or (type(max_value)==float and default > max_value)':
    :raise BadUnitError: If 'unit not in dir(UnitRegistry())'
    :raise BadVariableTypeError: If 'variable_type not in ("input", "state_variable", "plant_scale_state", "parameter")'
    :raise BadStateTypeError: If 'state_variable_type not in ("intensive", "extensive")'
    :raise BadEditError: If 'edit_by not in ("user", "developer")'
    """
    return field(default=default,
                 metadata=dict(unit=unit, unit_comment=unit_comment, description=description, min_value=min_value,
                               max_value=max_value, value_comment=value_comment, references=references, DOI=DOI,
                               variable_type=variable_type, by=by,
                               state_variable_type=state_variable_type, edit_by=edit_by))
