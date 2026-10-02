import yaml
from dataclasses import fields
from openalea.metafspm.coupling.translator import Translator


def _live_data_structure(component):
    """The component's DataStructure when it supports live reading (get/set/register), else None."""
    ds = getattr(component, "data_structure", None)
    return ds if (ds is not None and hasattr(ds, "register") and hasattr(ds, "get")) else None


class CompositeModel:

    # Name of the soil component in the translator; subclasses coupled with another soil model override it
    soil_name = "SoilModel"

    def get_documentation(self, filters: dict, models: list):
        """
        Documentation of the declared variables of each model, one column per metadata key.
        Fields without declare() metadata (e.g. FunctionalComponent.data_structure) are skipped.

        :param filters: {metadata key: accepted values}, a field is listed only if it matches every filter.
        :return: documentation text
        """
        def cell(value, width=30):
            text = "" if value is None else str(value)
            if len(text) > width:
                text = text[:width - 3] + "..."
            return f"{text:<{width + 1}} | "

        to_print = ""
        for model in models:
            to_print += "MODEL DOCUMENTATION : \n"
            to_print += ("   no documentation" if model.__doc__ is None else model.__doc__) + "\n\n"
            to_print += "MODEL OUTPUT VARIABLES : \n"

            declared = [f for f in fields(model) if "variable_type" in f.metadata]
            if len(declared) == 0:
                continue
            headers = list(declared[0].metadata.keys())
            widths = {header: 90 if header == "description" else 30 for header in headers}
            to_print += cell("name") + "".join(cell(header, widths[header]) for header in headers) + "\n\n"

            for f in declared:
                if all(f.metadata.get(k) in v for k, v in filters.items()):
                    to_print += cell(f.name) + "".join(cell(f.metadata.get(header), widths[header]) for header in headers) + "\n"

        return to_print

    @property
    def documentation(self):
        return self.get_documentation(filters={}, models=getattr(self, "components", []))

    @property
    def inputs(self):
        return self.get_documentation(filters=dict(variable_type=["input"]), models=getattr(self, "components", []))


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
        Couple the DataStructure-backed components *args* through the translator at *translator_path*
        (YAML or Python module): links between them become aliases and derived variables on their
        DataStructure (see _couple_on_data_structures). Links with the soil component (``soil_name``) are
        exchanged by the scene (coupler.Transport / Coupler); ``soil_outputs`` lists the soil variables read
        by the plant components.
        """
        self.components = [component for component in args]
        for component in self.components:
            if _live_data_structure(component) is None:
                raise TypeError(f"{type(component).__name__} is not DataStructure-backed: props-based components were "
                                "removed, see docs/design/downstream_migration.md")

        translator = self.open_or_create_translator(translator_path)
        self.soil_inputs, self.soil_outputs = self.get_component_inputs_outputs(
            translator=translator, components_names=[c.__class__.__name__ for c in self.components],
            target_name=self.soil_name, names_for_others=False)
        self._couple_on_data_structures(translator)

    def _couple_on_data_structures(self, translator: dict) -> None:
        """
        Coupling of DataStructure-backed components (design note §5): links between components sharing a
        DataStructure become name-level aliases or derived variables refreshed by the receiver before its step;
        identities need nothing. Links with components outside this composite (e.g. the soil) are exchanged
        by the scene, not here. Soil outputs are registered on the plant DataStructure, initialised to 0.
        """
        by_name = {component.__class__.__name__: component for component in self.components}
        for component in self.components:
            ds = component.data_structure
            for name in self.soil_outputs:
                if ds.has(name):
                    ds.set(name, 0.)
                else:
                    ds.register(name, location="node")

        for link in Translator.from_dict(translator).links:
            if link.receiver not in by_name or link.provider not in by_name or link.receiver == link.provider:
                continue
            receiver, provider = by_name[link.receiver], by_name[link.provider]
            ds = receiver.data_structure
            if provider.data_structure is not ds:
                raise NotImplementedError(f"{link.receiver}.{link.variable} <- {link.provider}: coupling across "
                                          "DataStructures needs a Coupler (devplan WD.5)")
            if link.kind == "identity":
                continue
            if link.kind == "alias":
                (source,) = link.sources
                if ds.has(link.variable) and link.variable not in ds.aliases():
                    ds.unregister(link.variable)
                ds.alias(link.variable, source)
                continue
            if link.detail == "same_name_factor":
                raise ValueError(f"{link.receiver}.{link.variable} is linked to {link.provider}.{link.variable} with "
                                 f"factor {link.sources[link.variable]}: a same-name link within one data structure "
                                 "must have a factor of 1, rename the receiving variable")
            location = ds.location(link.variable) if ds.has(link.variable) else None
            if link.formula is not None:
                ds.derive(link.variable, link.sources, formula=link.formula, location=location,
                          aggregation=link.aggregation, weight=link.weight)
            else:
                ds.derive(link.variable, dict(link.sources), location=location,
                          aggregation=link.aggregation, weight=link.weight)
            derived_inputs = receiver.__dict__.setdefault("_derived_inputs", [])
            if link.variable not in derived_inputs:
                derived_inputs.append(link.variable)

    def open_or_create_translator(self, translator_path):
        """
        Translator from a YAML file, or from a Python module defining ``translator = Translator(...)`` (.py),
        in the nested {receiver: {provider: {variable: {source: factor}}}} format. A missing YAML file is built
        interactively and written.
        """
        if str(translator_path).endswith(".py"):
            return Translator.from_module(str(translator_path)).to_nested()
        try:
            with open(translator_path, "r") as f:
                translator = yaml.safe_load(f)
        except FileNotFoundError:
            print("NOTE : You will now have to provide information about shared variables between the modules composing this model :\n")
            translator = self.translator_matrix_builder()
            with open(translator_path, "w") as f:
                yaml.dump(translator, f)
        
        return translator

    def translator_matrix_builder(self):
        """
        Translator matrix builder utility, to be used if no translator dictionay is available on modules' directory

        :param components: inistances of components that should be coupled as indicated by the coupling_translator.yaml
        """
        L = len(self.components)
        translator = {self.components[i].__class__.__name__:{self.components[k].__class__.__name__:{} for k in range(L)} for i in range(L)}
        for receiver_model in range(L):
            # Fields without declare() metadata (e.g. FunctionalComponent.data_structure) are not variables
            inputs = [f for f in fields(self.components[receiver_model]) if f.metadata.get("variable_type") == "input"]
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
        # The coupled DataStructures must be consistent (shapes, aliases, derivations; design note DS11)
        checked = []
        for component in getattr(self, "components", ()):
            ds = getattr(component, "data_structure", None)
            if ds is not None and hasattr(ds, "validate_variables") and not any(ds is other for other in checked):
                ds.validate_variables()
                checked.append(ds)


    def apply_input_tables(self, tables: dict, to: tuple, when: float):
        if tables is not None:
            # Selection depends on the targeted models and the provided tables, recompute it when they change
            selection_key = (tuple(id(model) for model in to), tuple(tables.keys()))
            if getattr(self, "_models_data_required_key", None) != selection_key:
                self._models_data_required_key = selection_key
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
                    if _live_data_structure(to[model]) is None:
                        raise TypeError("Unknown data structure to apply input data to")
                    # The table value applies to the whole variable
                    to[model].data_structure.set(var, tables[var][when])


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
