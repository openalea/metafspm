import yaml
from dataclasses import fields
from importlib import import_module, reload


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


    def declare_data(self, shoot=None, root=None, atmosphere=None, soil=None):
        self.data_structures = {}
        if shoot:
            self.data_structures["shoot"] = shoot
        if root:
            self.data_structures["root"] = root
        if atmosphere:
            self.data_structures["atmosphere"] = atmosphere
        if soil:
            self.data_structures["soil"] = soil

    def couple_components(self, *args, translator_path: str = ""):
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

        translator = self.open_or_create_translator(translator_path)

        soil_name = "SoilModel" # TODO : find a way to generalize this
        self.soil_inputs, self.soil_outputs = self.get_component_inputs_outputs(translator=translator, components_names=[c.__class__.__name__ for c in self.components], target_name=soil_name, names_for_others=False)

        props = self.data_structures["root"].properties()

        for receiver in self.components:
            self.couple_current_with_components_list(receiver=receiver, components=[c.__class__.__name__ for c in self.components] + [soil_name], translator=translator, common_props=props)
            
    def open_or_create_translator(self, translator_path):
        try:
            with open(translator_path + "/coupling_translator.yaml", "r") as f:
                translator = yaml.safe_load(f)
        except FileNotFoundError:
            print("NOTE : You will now have to provide information about shared variables between the modules composing this model :\n")
            translator = self.translator_matrix_builder()
            with open(translator_path + "/coupling_translator.yaml", "w") as f:
                yaml.dump(translator, f)
        
        return translator

    def couple_current_with_components_list(self, receiver, components, translator, common_props=None, subcategory=None):

        if not hasattr(receiver, "pullable_inputs"):
            receiver.pullable_inputs = {}

        if subcategory is not None and subcategory in receiver.pullable_inputs.keys():
            pass
        else:
            if subcategory is not None:
                receiver.pullable_inputs[subcategory] = {}

            for applier in components:
                if receiver.__class__.__name__ != applier:
                    linker = translator[receiver.__class__.__name__][applier]
                    # If a model has been targeted on this position
                    if len(linker.keys()) > 0:
                        # We set properties with getter method only to retrieve the values dynamically from inputs
                        for name, source_variables in linker.items():
                            if len(source_variables.keys()) > 0:
                                # Handling exception where operations are put in the coupling translator for unit conversion
                                for source_name, unit_conversion in source_variables.items():
                                    if isinstance(unit_conversion, str):
                                        source_variables[source_name] = eval(unit_conversion)

                                if len(source_variables.keys()) == 1:
                                    for source_name, unit_conversion in source_variables.items():
                                        # If there is only one variable to associate
                                        if source_name == name:
                                            # Do nothing the coupling should already be done during initialization
                                            continue
                                        else:
                                            # If only the name is different, just create an alias in the dictionnary and then recreate the pointer of the receiver class to this alias.
                                            if unit_conversion == 1. and common_props is not None:
                                                # If not created yet, for example in case of secondary soil initialization, we set default and suppose it will be modified later
                                                if source_name not in common_props.keys():
                                                    common_props[source_name] = {}

                                                common_props[name]  = common_props[source_name]
                                            else:
                                                # NOTE TODO : We will probably need to switch only the the second option later
                                                if subcategory is None:
                                                    receiver.pullable_inputs[name] = {source_name: unit_conversion}
                                                else:
                                                    receiver.pullable_inputs[subcategory][name] = {source_name: unit_conversion}

                                else:
                                    if subcategory is None:
                                        receiver.pullable_inputs[name] = source_variables
                                    else:
                                        receiver.pullable_inputs[subcategory][name] = {source_name: unit_conversion}


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

    def declare_data_and_couple_components(self, shoot=None, root=None, atmosphere=None, soil=None, translator_path: str = "", components: tuple = ()):
        self.declare_data(shoot=shoot, root=root, atmosphere=atmosphere, soil=soil)

        self.couple_components(translator_path=translator_path, *components)


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
                elif hasattr(to[model], "props"):
                    to[model].props[var].update({1: tables[var][when]})
                else:
                    raise TypeError("Unknown data structure to apply input data to")


    def get_component_inputs_outputs(self, translator, components_names, target_name, names_for_others=True):
        expected_inputs = []
        expected_outputs = []

        target_component = translator[target_name]

        for component in components_names:
            if component != target_name:
                # Get outputs from all others
                input_components = translator[component]
                for provider, source_variables in input_components.items():
                    # Among inputs if the target is found
                    if provider == target_name:
                        if names_for_others:
                            expected_outputs += list(source_variables.keys())
                        else:
                            for _, translation in source_variables.items():
                                expected_outputs += list(translation.keys())
            
                # Get inputs from all for target component
                if names_for_others:
                    for _, translation in target_component[component].items():
                                expected_inputs += list(translation.keys())
                else:
                    expected_inputs += list(target_component[component].keys())

        expected_inputs = list(set(expected_inputs))
        expected_outputs = list(set(expected_outputs))

        return expected_inputs, expected_outputs