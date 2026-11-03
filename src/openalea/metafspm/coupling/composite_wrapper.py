from typing import Optional

from openalea.metafspm.coupling.declaration import check_link
from openalea.metafspm.coupling.translator import Translator


def _live_data_structure(component):
    """The component's DataStructure when it supports live reading (get/set/register), else None."""
    ds = getattr(component, "data_structure", None)
    return ds if (ds is not None and hasattr(ds, "register") and hasattr(ds, "get")) else None


class CompositeModel:
    """
    Components sharing one DataStructure, coupled by couple(): an input with the name of another component's
    variable reads it (the DataStructure is the coupling); the translator's links state the rest, a rename, a
    conversion factor, a change of scale or a formula. couplings() lists how each input is coupled.

    translator: the model's coupling contract (a class attribute, anything Translator.load reads): couple() uses it
    when given none, and its links between initiators apply while they build each plant (coupling.initiation).
    """
    translator = None

    def couple(self, *components, translator=None) -> None:
        """
        Couple the DataStructure-backed *components*: the links of *translator* (Translator.load: a Translator, a
        nested dict, a .py module or a YAML file; None: no link, every input shared by its name) between them become
        aliases and derived variables on their DataStructure; links with components on other DataStructures (the
        environment) are exchanged by the Scene (coupling.cross.Exchanges). Their DataStructures are then checked
        (shapes, aliases, derivations).
        """
        self.components = list(components)
        for component in self.components:
            if _live_data_structure(component) is None:
                raise TypeError(f"{type(component).__name__} is not DataStructure-backed: props-based components were "
                                "removed, see docs/migration.md")
        self.translator = Translator.load(translator if translator is not None else type(self).translator)
        self._couple_on_data_structures(self.translator)
        checked = []
        for component in self.components:
            ds = component.data_structure
            if hasattr(ds, "validate_variables") and not any(ds is other for other in checked):
                ds.validate_variables()
                checked.append(ds)

    def run_together(self, *components) -> None:
        """
        Call *components* as one: their steps in one schedule (each step group at its schedule position, the
        components in the given order within a position), e.g. a development's potential lengths, the architecture's
        segmentation, then the development's postsegmentation (the new organs' lengths) before the architecture's.
        Each component's inputs are pulled first and its after-call work (write-back, clock, repartition) done last.
        """
        components = components or tuple(self.components)
        contexts = [component._before_call() for component in components]
        for component in components:
            component.pull_available_inputs()
        components[0].choregrapher.run_together(components)
        for component, context in zip(components, contexts):
            component._after_call(context)

    def couplings(self):
        """
        A pandas DataFrame, one row per input of the coupled components: receiver, variable, provider and coupling,
        "shared" (another component of the composite declares it: the same array), "alias" or "derived" (a link of
        the translator: a rename, or a factor, change of scale or formula), "scene" (a link with a component on
        another DataStructure, exchanged by the Scene), "planting" (a planting variable: planting_variable) or "not
        provided" (its initial value, or what the Scene writes: a forcing, an event).
        """
        import pandas as pd
        components = getattr(self, "components", [])
        translator = getattr(self, "translator", None) or Translator()
        scene = self.__dict__.get("_scene_translator")          # the Scene's links (with the environment models)
        if scene is not None:
            merged = Translator()
            merged.links.extend(translator.links)
            merged.links.extend(link for link in scene.links if link not in translator.links)
            translator = merged
        names = {type(component).__name__ for component in components}
        provided = {}
        for component in components:
            table = type(component).declarations()
            if table.empty:
                continue
            for name in table.loc[~table["role"].isin(["input", "parameter", "setting", "forcing"]), "name"]:
                provided.setdefault(name, type(component).__name__)
        rows = []
        for component in components:
            receiver = type(component).__name__
            table = type(component).declarations()
            if table.empty:
                continue
            for name in table.loc[table["role"] == "geometry", "name"]:
                rows.append(dict(receiver=receiver, variable=name, provider="the plants' structural components",
                                 coupling="geometry", sources="triangles()"))
            for name in table.loc[table["role"] == "input", "name"]:
                links = [link for link in translator.links_of(receiver) if link.variable == name
                         and link.provider != receiver]
                if links:
                    link = links[0]
                    coupling = "scene" if link.provider not in names else link.coupling
                    sources = ", ".join(f"{source}" if factor in (1, 1.) else f"{source} * {factor}"
                                        for source, factor in link.sources.items()) if link.formula is None \
                        else "formula"
                    rows.append(dict(receiver=receiver, variable=name, provider=link.provider,
                                     coupling=coupling, sources=sources))
                elif name in provided and provided[name] != receiver:
                    rows.append(dict(receiver=receiver, variable=name, provider=provided[name], coupling="shared",
                                     sources=name))
                elif _is_planting(type(component), name):
                    rows.append(dict(receiver=receiver, variable=name, provider="Planting", coupling="planting",
                                     sources=name))
                else:
                    rows.append(dict(receiver=receiver, variable=name, provider=None, coupling="not provided",
                                     sources=None))
        return pd.DataFrame(rows, columns=["receiver", "variable", "provider", "coupling", "sources"])

    def _couple_on_data_structures(self, translator) -> None:
        """
        Coupling of DataStructure-backed components: links between components sharing a
        DataStructure become name-level aliases or derived variables refreshed by the receiver before its step;
        identities need nothing. Links with components outside this composite (e.g. the soil) are exchanged
        by the Scene, not here.
        """
        by_name = {component.__class__.__name__: component for component in self.components}
        for link in Translator.load(translator).links:
            if link.receiver not in by_name or link.provider not in by_name or link.receiver == link.provider:
                continue
            receiver, provider = by_name[link.receiver], by_name[link.provider]
            # views of one DataStructure at the components' scales (LocatedView) are that DataStructure: linked on it,
            # their locations named as there
            ds = _base_of(receiver.data_structure)
            if _base_of(provider.data_structure) is not ds:
                raise NotImplementedError(f"{link.receiver}.{link.variable} <- {link.provider}: coupling across "
                                          "DataStructures goes through the Scene (coupling.cross.Exchanges)")
            check_link(link, receiver, provider)
            self._check_link_scales(link, ds)
            detail = link.mapped_detail
            if detail == "identity":
                continue
            crosses_locations = (detail == "alias" and ds.has(link.variable)
                                 and ds.location(link.variable) != ds.location(next(iter(link.sources))))
            if detail == "alias" and not crosses_locations:
                (source,) = link.sources
                if ds.has(link.variable) and link.variable not in ds.aliases():
                    ds.unregister(link.variable)
                ds.alias(link.variable, source)
                continue
            if detail == "same_name_factor":
                raise ValueError(f"{link.receiver}.{link.variable} is linked to {link.provider}.{link.variable} with "
                                 f"factor {link.sources[link.variable]}: a same-name link within one data structure "
                                 "must have a factor of 1, rename the receiving variable")
            location = ds.location(link.variable) if ds.has(link.variable) else self._receiving_location(link,
                                                                                                         receiver, ds)
            aggregation, weight = link.aggregation, link.weight
            if location is not None and (aggregation is None or (weight is None and aggregation in
                                                                 ("weighted_mean", "split"))):
                default, weight = self._default_link_mapping(ds, link, location)
                aggregation = aggregation or default
            options = dict(location=location, aggregation=aggregation, weight=weight, where=link.target,
                           missing=link.missing, default=self._receiving_default(link, receiver, ds))
            if link.formula is not None:
                ds.derive(link.variable, link.sources, formula=link.formula, **options)
            else:
                ds.derive(link.variable, dict(link.sources), **options)
            derived_inputs = receiver.__dict__.setdefault("_derived_inputs", [])
            if link.variable not in derived_inputs:
                derived_inputs.append(link.variable)

    @staticmethod
    def _receiving_location(link, receiver, ds) -> Optional[str]:
        """
        Where a link derives its receiving variable not yet on *ds* (an optional input nothing registered): the
        link's scale, else the receiver's declaration (on the view it works on, named as on *ds*); both given, they
        must agree. None when neither says (the sources' location).
        """
        from openalea.metafspm.coupling.declaration import location_of_scale
        stated = location_of_scale(ds, link.scale) if link.scale is not None else None
        spec = receiver.variable_specs.get(link.variable)
        declared = None
        if spec is not None:
            declared, at = spec.location, receiver.data_structure.__dict__.get("_at")
            if receiver.data_structure is not ds and at is not None and declared in (None, "node"):
                declared = at                           # a view's nodes: the entities of its location (LocatedView)
        if stated is not None and declared is not None and stated != declared:
            raise ValueError(f"{link.receiver}.{link.variable} <- {link.provider}: the link states scale "
                             f"{link.scale} ({stated}), but {link.receiver} declares it at {declared}")
        return stated or declared

    @staticmethod
    def _receiving_default(link, receiver, ds) -> float:
        """The receiving variable's default (its value without a provider): the receiver's declaration's, else what
        *ds* holds of it, else 0."""
        spec = receiver.variable_specs.get(link.variable)
        default = spec.default if spec is not None else None
        if default is None:
            default = ds._variable_meta().get(ds._resolve(link.variable) if ds.has(link.variable) else link.variable,
                                              {}).get("default")
        try:
            return 0. if default is None else float(default)
        except (TypeError, ValueError):
            return 0.

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
        (mapping, weight) of a link between two locations that gives no aggregation (or a weighted one without its
        weight), from its sources' kind; the weight is the link's, else the one declared on the
        variable whose entities it weighs: the source's for a mean (up), the receiver's for a split (down).
        (None, the link's weight) when the locations are the same.
        """
        from openalea.metafspm.coupling.declaration import DeclarationError, default_mapping, link_direction
        source_locations = {ds.location(source) for source in link.sources}
        if len(source_locations) != 1 or location in source_locations:
            return None, link.weight
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
        weight = link.weight
        if weight is None:
            meta = ds._variable_meta()
            weighed = [ds._resolve(source) for source in link.sources] if direction == "up" else [link.variable]
            declared = {meta.get(name_, {}).get("weight") for name_ in weighed}
            weight = declared.pop() if len(declared) == 1 else None
        try:
            return default_mapping(kind, direction, name, weight=weight), weight
        except DeclarationError as error:
            raise DeclarationError(f"{error} (link from {source_location} to {location})") from None


def _base_of(data_structure):
    """The DataStructure a view (LocatedView) is of, or *data_structure* itself."""
    return data_structure.__dict__.get("_base", data_structure)


def _is_planting(cls, name) -> bool:
    """Whether *name* is one of *cls*'s planting variables (planting_variable: an input by the planting)."""
    from dataclasses import fields
    field = next((f for f in fields(cls) if f.name == name), None)
    return field is not None and field.metadata.get("by") == "Planting"
