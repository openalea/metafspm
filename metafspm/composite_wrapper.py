import numpy as np
import yaml
from dataclasses import fields
from importlib import import_module, reload
import sys


def recursive_reload(module):
    blacklist = ["openalea.mtg.mtg", "openalea.mtg.tree", "collections", "functools"]
    reload(module)
    for child_module in vars(module).values():
        if isinstance(child_module, type) and module.__name__ != child_module.__module__ and child_module.__module__ not in blacklist:
            module_to_reload = import_module(name=child_module.__module__)
            recursive_reload(module_to_reload)


class CompositeModel:

    def get_documentation(self, filters: dict, models: list):
        """
        Documentation of the RootCyNAPS parameters
        :return: documentation text
        """
        to_print = ""
        for model in models:
            to_print += "MODEL DOCUMENTATION : \n"
            if model.__doc__ == None:
                to_print += "   no documentation"+ "\n\n"
            else:
                to_print += model.__doc__ + "\n\n"
            to_print += "MODEL OUTPUT VARIABLES : \n"

            docu = fields(model)
            first = True
            for f in docu:
                if first:
                    headers = f.metadata.keys()
                    max_format = "{:.30} "
                    width_format = "{:<31}"
                    to_print += width_format.format(max_format.format("name")) + " | "
                    for header in headers:
                        if header == "description":
                            max_format = "{:.90} "
                            width_format = "{:<91}"
                        else:
                            max_format = "{:.30} "
                            width_format = "{:<31}"
                        to_print += width_format.format(max_format.format(header)) + " | "
                    to_print += "\n\n"
                    first = False
                filtering = [f.metadata[k] in v for k, v in filters.items()]
                if False not in filtering or len(filtering) == 0:
                    to_print += width_format.format(max_format.format(f.name)) +  " | "
                    values = list(f.metadata.values())
                    for value in values:
                        if values.index(value) == 2:
                            max_format = "{:.90} "
                            width_format = "{:<91}"
                            if len(value) > 90:
                                value = value.replace(value[87:], "...")
                        else:
                            max_format = "{:.30} "
                            width_format = "{:<31}"
                        to_print += width_format.format(max_format.format(value)) + " | "
                    to_print += "\n"

        return to_print

    @property
    def documentation(self):
        return self.get_documentation(filters={}, models=self.components)

    @property
    def inputs(self):
        return self.get_documentation(filters=dict(variable_type=["input"]), models=self.components)

    def declare_and_couple_components(self, *args, translator_path: str = ""):
        """
        Description : linker function that will enable properties sharing through MTG.

        Parameters :
        :param translator: list matrix containing translator dictionnaries for each model pair
        :param components: inistances of components that should be coupled as indicated by the coupling_translator.yaml

        Note :  The whole property is transfered, so if only the collar value of a spatial property is needed,
        it will be accessed through the first vertice with the [1] indice. Not spatialized properties like xylem pressure or
        single point properties like collar flows are only stored in the indice [1] vertice.
        """

        self.components = [component for component in args]

        try:
            with open(translator_path + "/coupling_translator.yaml", "r") as f:
                translator = yaml.safe_load(f)
        except FileNotFoundError:
            print("NOTE : You will now have to provide information about shared variables between the modules composing this model :\n")
            translator = self.translator_matrix_builder()
            with open(translator_path + "/coupling_translator.yaml", "w") as f:
                yaml.dump(translator, f)

        L = len(self.components)
        for receiver in self.components:
            for applier in self.components:
                if id(receiver) != id(applier):
                    linker = translator[receiver.__class__.__name__][applier.__class__.__name__]
                    # If a model has been targeted on this position
                    if len(linker.keys()) > 0:
                        applier_name = applier.__class__.__name__
                        # First we create a link to applier class so that created properties refer to it properly
                        setattr(receiver, applier_name, applier)
                        # We set properties with getter method only to retrieve the values dynamically from inputs
                        for name, source_variables in linker.items():
                            formula = ""
                            for source_name, unit_conversion in source_variables.items():                          
                                if type(list(getattr(applier, source_name).values())[0]) != str:
                                    formula += f"(self.{applier_name}.{source_name}[vid]*{unit_conversion})+"
                                else:
                                    formula += f"self.{applier_name}.{source_name}[vid] "
                            # First get the dimensions of the dictionnaries that will be managed, !!! supposing every input has the same dimension
                            iterator = f"self.{applier_name}.{list(source_variables.keys())[0]}.keys()"
                            # Then we sum every targetted variable for every vertex, with same iteration as for keys
                            setattr(receiver.__class__, name, property(eval(f"""lambda self: dict(zip({iterator}, [{formula[:-1]} for vid in {iterator}]))""")))
                            
                        #receiver.available_inputs += [dict(applier=applier, linker=linker)]

    def translator_matrix_builder(self):
        """
        Translator matrix builder utility, to be used if no translator dictionay is available on modules' directory

        :param components: inistances of components that should be coupled as indicated by the coupling_translator.yaml
        """
        L = len(self.components)
        translator = {self.components[i].__class__.__name__:{self.components[k].__class__.__name__:{} for k in range(L)} for i in range(L)}
        for receiver_model in range(L):
            inputs = [f for f in fields(self.components[receiver_model]) if f.metadata["variable_type"] == "input"]
            needed_models = list(set([f.metadata["by"] for f in inputs]))
            needed_models.sort()
            for name in needed_models:
                print([(model + 1, self.components[model].__class__.__name__) for model in range(len(self.components))])
                which = int(input(f"[for {self.components[receiver_model].__class__.__name__}] Which is {name}? (0 for None): ")) - 1
                needed_inputs = [f.name for f in inputs if f.metadata["by"] == name]
                if 0 <= which < L:
                    available = self.get_documentation(filters=dict(variable_type=["state_variable", "plant_scale_state"]), models=[self.components[which]])
                    print(available)
                    for var in needed_inputs:
                        selected = input(f"For {var}, Nothing for same name / enter target names * conversion factor / Separate by ; -> ").split(";")
                        com_dict = {}
                        for expression in selected:
                            if "*" in expression:
                                l = expression.split("*")
                                com_dict[l[0].replace(" ", "")] = float(l[1])
                            elif expression.strip() == "":
                                com_dict[var] = 1.
                            else:
                                com_dict[expression.replace(" ", "")] = 1.
                        translator[self.components[receiver_model].__class__.__name__][self.components[which].__class__.__name__][var] = com_dict

        return translator

    def declare_data_structures(self, shoot=None, root=None, atmosphere=None, soil=None):
        self.data_structures = {}
        if shoot:
            self.data_structures["shoot"] = shoot
        if root:
            self.data_structures["root"] = root
        if atmosphere:
            self.data_structures["atmosphere"] = atmosphere
        if soil:
            self.data_structures["soil"] = soil

    def apply_input_tables(self, tables: dict, to: tuple, when: float):
        if not hasattr(self, "models_data_required"):
            all_available_state_variables = []
            for model in to:
                all_available_state_variables += model.state_variables
            self.models_data_required = [[var for var in tables.keys() if (
                                            # Either the input is not provided by another coupled module
                                            (var in model.inputs) and (var not in all_available_state_variables)) or (
                                            # Or the considered model provides the variable but gets it from input data only
                                            var in model.state_variables)]
                                         for model in to]

        for model in range(len(to)):
            for var in self.models_data_required[model]:
                if hasattr(to[model], "voxels"):
                    # supposed True : if isinstance(getattr(to[model].voxels, var), np.ndarray):
                    to[model].voxels[var].fill(tables[var][when])
                elif isinstance(getattr(to[model], var), dict):
                    setattr(to[model], var, {1: tables[var][when]})
                else:
                    raise TypeError("Unknown data structure to apply input data to")
