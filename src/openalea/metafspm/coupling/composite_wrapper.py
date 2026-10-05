import yaml
from dataclasses import fields
from openalea.metafspm.coupling.translator import Translator


def _live_data_structure(component):
    """The component's DataStructure when it supports live reading (get/set/register), else None."""
    ds = getattr(component, "data_structure", None)
    return ds if (ds is not None and hasattr(ds, "register") and hasattr(ds, "get")) else None


class CompositeModel:

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
        DataStructure (see _couple_on_data_structures). Links with components on other DataStructures (the
        environment) are exchanged by the Scene (coupling.cross.Exchanges).
        """
        self.components = [component for component in args]
        for component in self.components:
            if _live_data_structure(component) is None:
                raise TypeError(f"{type(component).__name__} is not DataStructure-backed: props-based components were "
                                "removed, see docs/design/downstream_migration.md")

        translator = self.open_or_create_translator(translator_path)
        self._couple_on_data_structures(translator)

    def _couple_on_data_structures(self, translator: dict) -> None:
        """
        Coupling of DataStructure-backed components: links between components sharing a
        DataStructure become name-level aliases or derived variables refreshed by the receiver before its step;
        identities need nothing. Links with components outside this composite (e.g. the soil) are exchanged
        by the Scene, not here.
        """
        by_name = {component.__class__.__name__: component for component in self.components}
        for link in Translator.from_dict(translator).links:
            if link.receiver not in by_name or link.provider not in by_name or link.receiver == link.provider:
                continue
            receiver, provider = by_name[link.receiver], by_name[link.provider]
            ds = receiver.data_structure
            if provider.data_structure is not ds:
                raise NotImplementedError(f"{link.receiver}.{link.variable} <- {link.provider}: coupling across "
                                          "DataStructures goes through the Scene (coupling.cross.Exchanges)")
            self._check_link_kinds(link, receiver, provider)
            self._check_link_scales(link, ds)
            if link.kind == "identity":
                continue
            crosses_locations = (link.kind == "alias" and ds.has(link.variable)
                                 and ds.location(link.variable) != ds.location(next(iter(link.sources))))
            if link.kind == "alias" and not crosses_locations:
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
            aggregation = link.aggregation
            if aggregation is None and location is not None:
                aggregation = self._default_link_mapping(ds, link, location)
            options = dict(location=location, aggregation=aggregation, weight=link.weight, target=link.target)
            if link.target is not None and ds.has(link.variable):
                options["default"] = ds._variable_meta().get(link.variable, {}).get("default", 0.)
            if link.formula is not None:
                ds.derive(link.variable, link.sources, formula=link.formula, **options)
            else:
                ds.derive(link.variable, dict(link.sources), **options)
            derived_inputs = receiver.__dict__.setdefault("_derived_inputs", [])
            if link.variable not in derived_inputs:
                derived_inputs.append(link.variable)

    @staticmethod
    def _declared_kind(component, ds, name):
        spec = getattr(component, "_variable_specs", {}).get(name)
        if spec is not None and spec.kind is not None:
            return spec.kind
        return ds._variable_meta().get(name, {}).get("kind") if hasattr(ds, "_variable_meta") else None

    def _check_link_kinds(self, link, receiver, provider) -> None:
        """A receiver declaring a state_variable_type must agree with its provider's."""
        from openalea.metafspm.coupling.declaration import kinds_agree
        ds = receiver.data_structure
        received = getattr(receiver, "_variable_specs", {}).get(link.variable)
        received = received.kind if received is not None else None
        for source in link.sources:
            provided = self._declared_kind(provider, ds, source)
            if not kinds_agree(received, provided):
                raise ValueError(f"{link.receiver}.{link.variable} ({received}) <- {link.provider}.{source} "
                                 f"({provided}): the kinds do not agree (extensive with extensive, intensive or "
                                 "massic with intensive or massic)")

    @staticmethod
    def _check_link_scales(link, ds) -> None:
        """A link stating its scales must agree with the declared locations (the declarations rule)."""
        from openalea.metafspm.coupling.declaration import location_of_scale
        checks = [(link.variable, link.scale)] + [(source, link.source_scale) for source in link.sources]
        for name, scale in checks:
            if scale is None or not ds.has(name):
                continue
            expected, actual = location_of_scale(ds, scale), ds.location(name)
            if expected != actual:
                raise ValueError(f"{link.receiver}.{link.variable} <- {link.provider}: the link states that '{name}' "
                                 f"is at scale {scale} ({expected}), but it is declared at {actual}")

    @staticmethod
    def _default_link_mapping(ds, link, location):
        """
        Mapping of a link between two locations that gives no aggregation, from its sources' state_variable_type
        None when the locations are the same.
        """
        from openalea.metafspm.coupling.declaration import DeclarationError, default_mapping, link_direction
        source_locations = {ds.location(source) for source in link.sources}
        if len(source_locations) != 1 or location in source_locations:
            return None
        (source_location,) = source_locations
        name = f"{link.receiver}.{link.variable} <- {link.provider}"
        direction = link_direction(ds, source_location, location)
        if direction is None:
            raise DeclarationError(f"{name}: from {source_location} to {location} has no default mapping, give "
                                   "aggregation=")
        kinds = {ds._variable_meta().get(ds._resolve(source), {}).get("kind") for source in link.sources}
        if len(kinds) != 1:
            raise DeclarationError(f"{name}: its sources have different kinds {sorted(map(str, kinds))}, give "
                                   "aggregation=")
        (kind,) = kinds
        try:
            return default_mapping(kind, direction, name, weight=link.weight)
        except DeclarationError as error:
            raise DeclarationError(f"{error} (link from {source_location} to {location})") from None

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
        # The coupled DataStructures must be consistent (shapes, aliases, derivations)
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
