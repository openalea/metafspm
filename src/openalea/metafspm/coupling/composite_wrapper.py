from contextvars import ContextVar
from typing import Optional

from openalea.metafspm.coupling.declaration import check_link
from openalea.metafspm.coupling.translator import Translator, matches


def _live_data_structure(component):
    """The component's DataStructure when it supports live reading (get/set/register), else None."""
    ds = getattr(component, "data_structure", None)
    return ds if (ds is not None and hasattr(ds, "register") and hasattr(ds, "get")) else None


_NESTED = ContextVar("metafspm_nested_model", default=False)     # a model built as another's part


class Part:
    """A part built with constructor keywords (its settings at construction):
    ``parts = {"hydraulics": Part(MechaHydraulics, soil_boundary="node_dirichlet")}``."""

    def __init__(self, cls, **keywords):
        self.cls, self.keywords = cls, keywords

    def __repr__(self) -> str:
        return f"Part({self.cls.__name__}, {', '.join(f'{k}={v!r}' for k, v in self.keywords.items())})"


class CompositeModel:
    """
    A model: components sharing one DataStructure, declared as its parts and run by its schedule
    (coupling.schedule)::

        class WheatPlant(CompositeModel):
            initiators = (WheatCrown, ScenarioShoot, RootArchitecture, GranapAnatomy)
            parts = {"crown": WheatCrown, "shoot": ScenarioShoot, "rule": NodalRootEmergence, ...}
            schedule = ("shoot", "rule", "roots", Step("anatomy", cadence="1 d"), ...)
            translator = PLANT_TRANSLATOR

    *parts* ({name: class, or Part(class, **keywords): a component built with constructor keywords}): the
    components, or nested models (a CompositeModel: its parts become this model's
    components, its schedule one step of this one's, its translator's links merged), built in order on the model's
    DataStructure in its time step (a component class declaring its own time_step: in it, its sub-step), each the
    model's attribute of its name; then coupled once: an input with the name of another
    component's variable reads it (the DataStructure is the coupling); the translator's links state the rest, a
    rename, a conversion factor, a change of scale or a formula. couplings() lists how each input is coupled.
    *schedule* (coupling.schedule: part names, methods, Step, Together; default: the parts in order) is what run()
    runs; steps() gives its steps bound, with their reads and writes. A model's own work besides its parts' (a mask
    defined on its DataStructure) goes in its __init__, after super().__init__.

    translator: the model's coupling contract (a class attribute, anything Translator.load reads): its links between
    initiators apply while they build each plant (coupling.initiation). initiators: the StructuralComponents building
    each plant, in order (a nested model's initiators where the model is named).

    An environment model (scene.environment.EnvironmentModel) builds its components on its own DataStructures in its
    __init__ (couple(), or its components list), its schedule naming them.
    """
    translator = None
    parts = {}
    schedule = None

    def __init__(self, data_structure=None, time_step=None):
        cls = type(self)
        self.data_structure, self.time_step = data_structure, time_step
        if data_structure is None:                  # a model given its components (couple: an environment model's)
            if cls.parts:
                raise TypeError(f"{cls.__name__}(): its parts are built on its DataStructure, give it")
            self.components = []
            return
        components = []
        for name, part in cls.parts.items():
            keywords = {}
            if isinstance(part, Part):
                part, keywords = part.cls, part.keywords
            if isinstance(part, type) and issubclass(part, CompositeModel):
                token = _NESTED.set(True)
                try:
                    built = part(data_structure, time_step)
                finally:
                    _NESTED.reset(token)
                components.extend(built.components)
            elif getattr(part, "time_step", None) is not None:      # a component of its own (sub-)step
                built = part(data_structure=data_structure, **keywords)
                components.append(built)
            else:
                built = part(data_structure=data_structure, **{"time_step": time_step, **keywords})
                components.append(built)
            setattr(self, name, built)
        if _NESTED.get():
            self.components = components            # coupled by the model it is a part of
        else:
            self.couple(*components, translator=_merged_translator(cls))
        self.steps()                                # its schedule checked now

    @classmethod
    def initiator_classes(cls) -> tuple:
        """The model's initiators, a nested model's expanded into its own (in place)."""
        found = []
        for initiator in getattr(cls, "initiators", ()):
            if isinstance(initiator, type) and issubclass(initiator, CompositeModel):
                found.extend(initiator.initiator_classes())
            else:
                found.append(initiator)
        return tuple(found)

    def couple(self, *components, translator=None) -> None:
        """
        The model's *components* coupled (a model's parts are, at its construction; an environment model couples
        those it builds on its DataStructures): the links of *translator* (Translator.load: a Translator, a nested
        dict, a .py module or a YAML file; None: the model's, else no link, every input shared by its name) between
        them become aliases and derived variables on their DataStructure; links with components on other
        DataStructures (the environment) are exchanged by the Scene (coupling.cross.Exchanges). Their DataStructures
        are then checked (shapes, aliases, derivations).
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

    def steps(self) -> list:
        """The model's schedule bound (coupling.schedule.ScheduledStep: name, kind, components, stages, cadence,
        reads, writes), the cadences its scenario's "cadence" sets included."""
        steps = self.__dict__.get("_steps")
        if steps is None:
            from openalea.metafspm.coupling.schedule import bind
            steps = self.__dict__["_steps"] = bind(self)
        return steps

    def set_cadences(self, cadences) -> None:
        """The cadences of the model's steps, {step name: duration, Cadence or None (every step)}: its scenario's
        "cadence" key (a duration keeping the declared Cadence's "at")."""
        if not isinstance(cadences, dict):
            raise TypeError(f"{type(self).__name__}: its scenario's cadence is {{step name: duration}}, got "
                            f"{cadences!r}")
        self.__dict__["_cadences"] = {**self.__dict__.get("_cadences", {}), **cadences}
        self.__dict__.pop("_steps", None)
        self.steps()                                    # checked now: an unknown step raises

    def run(self) -> None:
        """One run of the model: its schedule's steps, in order, each component's clock at the run's start for each
        of its steps (a component scheduled twice, for two stages, reads the step's forcings both times); the
        components that ran counted once (Component.calls: the scene steps they ran in)."""
        clocks = [(component, component.__dict__.get("_clock", 0.)) for component in self.components]
        for step in self.steps():
            for component, clock in clocks:
                component.__dict__["_clock"] = clock
            step.run()
        for component in self.components:
            if component.__dict__.pop("_ran", False):
                component.__dict__["_calls"] = component.__dict__.get("_calls", 0) + 1

    def run_together(self, *components) -> None:
        """
        Call *components* as one: their steps in one schedule (each step group at its schedule position, the
        components in the given order within a position), e.g. a development's potential lengths, the architecture's
        segmentation, then the development's postsegmentation (the new organs' lengths) before the architecture's.
        Each component's inputs are pulled first and its after-call work (write-back, clock, repartition) done last.
        A schedule's Together(...) runs it; a model's method step (ScenarioShoot's grow) calls it.
        """
        _run_together(components or tuple(self.components), [None] * len(components or self.components))

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
                links = [link for link in translator.links if link.variable == name and matches(link.receiver, component)
                         and not matches(link.provider, component)]
                if links:
                    link = links[0]
                    coupling = "scene" if not any(matches(link.provider, other) for other in components) \
                        else link.coupling
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
        for component in self.components:          # their links are this translator's: load_checkpoint keeps them
            _base_of(component.data_structure).__dict__["_links_coupled"] = True
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
            link = self._geometric_sources(link, receiver, ds)
            self._check_link_scales(link, ds)
            detail = link.mapped_detail
            if link.names:                          # a receiver-private name: always the link's own variable
                self._derive_private(link, receiver, ds)
                continue
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

    def _geometric_sources(self, link, receiver, ds):
        """The link with its geometric sources (coupling.geometry.FractionIn, CentreIn) replaced by the geometric
        variables *ds* derives for them (DataStructure.derive_geometry), at the receiving location."""
        from dataclasses import replace
        from openalea.metafspm.coupling.geometry import GEOMETRIC_SOURCES
        if not any(isinstance(source, GEOMETRIC_SOURCES) for source in link.sources):
            return link
        location = ds.location(link.variable) if ds.has(link.variable) else self._receiving_location(link, receiver,
                                                                                                    ds)
        renamed = {}
        for source in link.sources:
            if isinstance(source, GEOMETRIC_SOURCES):
                if not ds.has(source.name):
                    ds.derive_geometry(source.name, source, location=location)
                renamed[source] = source.name
        if link.formula is not None:
            return replace(link, sources=tuple(renamed.get(source, source) for source in link.sources))
        return replace(link, sources={renamed.get(source, source): factor for source, factor in link.raw_factors.items()})

    def _derive_private(self, link, receiver, ds) -> None:
        """A link with a receiver-private name (Link.names): its values derived under that name, which the receiver
        reads as its variable (DataStructure.private_names around its calls), with the variable's declared metadata;
        the variable of that name the other components share untouched."""
        private = link.private_name
        if ds.has(private) and private not in ds.derived():
            raise ValueError(f"{link.receiver}.{link.variable} <- {link.provider}: its private name '{private}' is "
                             "already a variable of the DataStructure")
        location = ds.location(link.variable) if ds.has(link.variable) else self._receiving_location(link, receiver,
                                                                                                    ds)
        renamed = type(link)(link.receiver, private, link.provider, link.sources if link.formula is not None
                             else link.raw_factors, scale=link.scale, source_scale=link.source_scale,
                             aggregation=link.aggregation, weight=link.weight, target=link.target,
                             formula=link.formula, unit_check=link.unit_check, missing=link.missing)
        aggregation, weight = link.aggregation, link.weight
        if location is not None and (aggregation is None or (weight is None and aggregation in
                                                             ("weighted_mean", "split"))):
            default, weight = self._default_link_mapping(ds, renamed, location)
            aggregation = aggregation or default
        options = dict(location=location, aggregation=aggregation, weight=weight, where=link.target,
                       missing=link.missing, default=self._receiving_default(link, receiver, ds))
        if link.formula is not None:
            ds.derive(private, link.sources, formula=link.formula, **options)
        else:
            ds.derive(private, dict(link.sources), **options)
        meta = ds._variable_meta()
        declared = meta.get(ds._resolve(link.variable), {}) if ds.has(link.variable) else {}
        meta.setdefault(private, {}).update({key: value for key, value in declared.items() if key != "derived"})
        names = receiver.__dict__.setdefault("_private_names", {})
        names[link.variable] = private
        derived_inputs = receiver.__dict__.setdefault("_derived_inputs", [])
        if private not in derived_inputs:
            derived_inputs.append(private)

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


def _run_together(components, overrides) -> None:
    """The components' steps in one schedule (CompositeModel.run_together), *overrides* their private names'."""
    contexts = [component._before_call() for component in components]
    with components[0].private_names(components[1:], overrides=overrides[0], component_overrides=overrides[1:]):
        for component in components:
            component.pull_available_inputs()
        components[0].choregrapher.run_together(components)
    for component, context in zip(components, contexts):
        component._after_call(context)
        component.__dict__["_ran"] = True


def _merged_translator(cls):
    """The links of a model class's translator and of its nested models' (its parts' model classes, theirs too),
    each once: the model's coupling contract (coupling, initiation)."""
    merged = Translator()
    translators = [Translator.load(cls.translator)] if getattr(cls, "translator", None) is not None else []
    translators += [_merged_translator(part) for part in getattr(cls, "parts", {}).values()
                    if isinstance(part, type) and issubclass(part, CompositeModel)]
    for translator in translators:
        for link in translator.links:
            if link not in merged.links:
                merged.links.append(link)
    return merged


def model_translator(model):
    """A plant model class's coupling contract: its translator's links and its nested models' (a model without
    parts, its translator)."""
    if isinstance(model, type) and issubclass(model, CompositeModel):
        return _merged_translator(model)
    return getattr(model, "translator", None)


def model_initiators(model) -> tuple:
    """A plant model class's initiators, a nested model's expanded into its own."""
    if isinstance(model, type) and issubclass(model, CompositeModel):
        return model.initiator_classes()
    return tuple(getattr(model, "initiators", ()))


def _base_of(data_structure):
    """The DataStructure a view (LocatedView) is of, or *data_structure* itself."""
    return data_structure.__dict__.get("_base", data_structure)


def _is_planting(cls, name) -> bool:
    """Whether *name* is one of *cls*'s planting variables (planting_variable: an input by the planting)."""
    from dataclasses import fields
    field = next((f for f in fields(cls) if f.name == name), None)
    return field is not None and field.metadata.get("by") == "Planting"
