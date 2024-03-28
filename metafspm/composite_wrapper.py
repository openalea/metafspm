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
        return self.get_documentation(filters={}, models=self.models)

    @property
    def inputs(self):
        return self.get_documentation(filters=dict(variable_type=["input"]), models=self.models)

    def link_around_mtg(self, translator_path: list):
        """
        Description : linker function that will enable properties sharing through MTG.

        Parameters :
        :param translator: list matrix containing translator dictionnaries for each model pair

        Note :  The whole property is transfered, so if only the collar value of a spatial property is needed,
        it will be accessed through the first vertice with the [1] indice. Not spatialized properties like xylem pressure or
        single point properties like collar flows are only stored in the indice [1] vertice.
        """

        try:
            with open(translator_path + "/coupling_translator.yaml", "r") as f:
                translator = yaml.safe_load(f)
        except FileNotFoundError:
            print("NOTE : You will now have to provide information about shared variables between the modules composing this model :\n")
            translator = self.translator_matrix_builder()
            with open(translator_path + "/coupling_translator.yaml", "w") as f:
                yaml.dump(translator, f)

        L = len(self.models)
        for receiver in self.models:
            for applier in self.models:
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
        # TODO surely not working, debug with a working Root-CyNAPS wrapping
        """
        L = len(self.models)
        translator = {self.models[i].__class__.__name__:{self.models[k].__class__.__name__:{} for k in range(L)} for i in range(L)}
        for receiver_model in range(L):
            inputs = [f for f in fields(self.models[receiver_model]) if f.metadata["variable_type"] == "input"]
            needed_models = list(set([f.metadata["by"] for f in inputs]))
            needed_models.sort()
            for name in needed_models:
                print([(model + 1, self.models[model].__class__.__name__) for model in range(len(self.models))])
                which = int(input(f"[for {self.models[receiver_model].__class__.__name__}] Which is {name}? (0 for None): ")) - 1
                needed_inputs = [f.name for f in inputs if f.metadata["by"] == name]
                if 0 <= which < L:
                    available = self.get_documentation(filters=dict(variable_type=["state_variable", "plant_scale_state"]), models=[self.models[which]])
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
                        translator[self.models[receiver_model].__class__.__name__][self.models[which].__class__.__name__][var] = com_dict

        return translator
