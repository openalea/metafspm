from dataclasses import fields
from importlib import import_module, reload
import sys


class CompositeModel:

    def load(self, model, *args, **kwargs):
        """
        This utility is intended to ensure separated Choregrapher instances between each component
        """
        module = import_module(name=model.__module__)
        del sys.modules["genericmodel.component"]
        reload(module)
        model = getattr(module, model.__name__)
        return model(*args, **kwargs)

    def get_documentation(self, filters: dict, models: list):
        """
        Documentation of the RootCyNAPS parameters
        :return: documentation text
        """
        to_print = ""
        for model in models:
            to_print += "MODEL DOCUMENTATION : \n"
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
                filtering = [f.metadata[k] == v for k, v in filters.items()]
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
        return self.get_documentation(filters=dict(variable_type="input"), models=self.models)

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
        for receiver_index in range(L):
            receiver = self.models[receiver_index]
            for applier_index in range(L):
                if receiver_index != applier_index:
                    applier = self.models[applier_index]
                    linker = translator[receiver_index][applier_index]
                    # If a model has been targeted on this position
                    if len(linker.keys()) > 0:
                        receiver.available_inputs += [dict(applier=applier, linker=linker)]

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
                    available = self.get_documentation(filters=dict(variable_type="state_variable"), models=[self.models[which]])
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
                        break

        return translator

    # TODO later think of a multiprocessing possibility for shoot + soil or C + N
