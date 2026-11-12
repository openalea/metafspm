"""
The translator at initiation: the links between a plant model's initiators, applied while they build each plant.

Initiators build every plant before any component exists. A later initiator's input linked from an earlier one (e.g.
GRANAP's SubOrgan stem_radius from Adel's Organ diameter) is written to its MPG property before the later initiator
builds the plant: the sources' properties at their scale, the link's factors (a weighted sum) or formula, mapped to
the receiver's scale by the link's aggregation. After initiation the coupling derives the link on the DataStructure
as usual, from the same values.
"""
from dataclasses import fields, is_dataclass

import numpy as np

from openalea.metafspm.coupling.declaration import DOWN_AGGREGATIONS, UP_AGGREGATIONS, default_aggregation
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import ScalesConfig


def initiation_links(translator, initiators) -> dict:
    """
    {index of an initiator in *initiators*: the links of *translator* into it from an earlier initiator}, its input or
    parameter the receiver. A link into an initiator from a later one raises: the initiators build in order.
    """
    if translator is None:
        return {}
    names = [getattr(initiator, "__name__", str(initiator)) for initiator in initiators]
    out = {}
    for link in Translator.load(translator).links:
        if link.receiver not in names or link.provider not in names or link.receiver == link.provider:
            continue
        receiver, provider = names.index(link.receiver), names.index(link.provider)
        meta = _metadata(initiators[receiver], link.variable)
        if meta is None or meta.get("role") not in ("input_variable", "parameter"):
            continue
        if provider > receiver:
            raise ValueError(f"{link.receiver}.{link.variable} <- {link.provider}: the initiators build in order, "
                             f"{names}: a link between initiators goes from an earlier one to a later one")
        if link.target is not None:
            raise ValueError(f"{link.receiver}.{link.variable} <- {link.provider}: a link with a target mask needs a "
                             "DataStructure, it cannot apply at initiation")
        out.setdefault(receiver, []).append((link, initiators[provider]))
    return out


def apply_link(mtg, plant: int, link, receiver, provider) -> None:
    """Write *link*'s values (receiver.variable <- provider's sources) on *plant*'s vertices of the receiver's scale."""
    from openalea.metafspm.coupling.declaration import resolve_equivalences
    link = resolve_equivalences(link, receiver, provider)
    scale = _scale(receiver, link.variable, link)
    sources = list(link.sources)
    source_scales = {_scale(provider, source, link) for source in sources}
    if len(source_scales) != 1:
        raise ValueError(f"{_name(link)}: its sources span several scales")
    source_scale = source_scales.pop()
    for given, declared in ((link.scale, scale), (link.source_scale, source_scale)):
        if given is not None and given != declared:
            raise ValueError(f"{_name(link)}: the link's scale {given} is not the declared one ({declared})")

    source_vids = _vertices(mtg, plant, source_scale)
    arrays = [_values(mtg, source, source_vids) for source in sources]
    if link.formula is not None:
        values = np.asarray(link.formula(*arrays), dtype=float)
    else:
        values = sum(factor * array for factor, array in zip(link.sources.values(), arrays))

    if scale == source_scale:
        vids, out = source_vids, values
    else:
        vids = _vertices(mtg, plant, scale)
        direction = "down" if scale > source_scale else "up"
        kind = (_metadata(provider, sources[0]) or {}).get("kind")
        aggregation = link.aggregation or default_aggregation(kind, direction, link.variable, link.weight)
        allowed = DOWN_AGGREGATIONS if direction == "down" else UP_AGGREGATIONS
        if aggregation not in allowed:
            raise ValueError(f"{_name(link)}: aggregation '{aggregation}' does not go {direction} (one of {allowed})")
        if direction == "down":
            owner = _index_in(source_vids, _owners(mtg, vids, source_scale))
            out = np.where(owner >= 0, values[owner.clip(0)], np.nan)
            if aggregation == "split":
                weight = _values(mtg, link.weight, vids)
                total = np.bincount(owner.clip(0), weights=np.where(owner >= 0, weight, 0.), minlength=values.size)
                out = out * weight / total[owner.clip(0)]
        else:
            owner = _index_in(vids, _owners(mtg, source_vids, scale))
            kept = (owner >= 0) & np.isfinite(values)
            weight = _values(mtg, link.weight, source_vids) if aggregation == "weighted_mean" else np.ones(values.size)
            total = np.bincount(owner[kept], weights=(values * weight)[kept], minlength=vids.size)
            count = np.bincount(owner[kept], weights=weight[kept], minlength=vids.size)
            out = total if aggregation == "sum" else np.where(count > 0, total / np.where(count > 0, count, 1.), np.nan)
            if aggregation in ("max", "min"):
                out = np.full(vids.size, np.nan)
                (np.fmax if aggregation == "max" else np.fmin).at(out, owner[kept], values[kept])
            out = np.where(np.bincount(owner[kept], minlength=vids.size) > 0, out, np.nan)
    out = np.asarray(out, dtype=float)
    written = np.isfinite(out)                       # no source value: none written (the declared default applies)
    mtg.set_values(link.variable, np.asarray(vids, dtype=np.int64)[written], out[written].tolist())


def _metadata(component, name):
    if not is_dataclass(component):
        return None
    field = next((f for f in fields(component) if f.name == name), None)
    return None if field is None else field.metadata


def _scale(component, name, link) -> int:
    meta = _metadata(component, name)
    if meta is None:
        raise ValueError(f"{_name(link)}: {getattr(component, '__name__', component)} declares no '{name}'")
    scale = meta.get("scale")
    if isinstance(scale, str):
        scale = getattr(ScalesConfig, scale, None)
    if scale is None or scale in (ScalesConfig.Compartment, ScalesConfig.Connection):
        raise ValueError(f"{_name(link)}: '{name}' has no MTG scale coarser than the Compartments, the link cannot "
                         "apply at initiation")
    return int(scale)


def _vertices(mtg, plant: int, scale: int) -> np.ndarray:
    """*plant*'s vertices at *scale* (the plant itself at its own scale), the MPG's anchors excluded."""
    if scale == mtg.scales.Plant:
        return np.array([plant], dtype=np.int64)
    vids = np.asarray([int(v) for v in mtg.components_at_scale(plant, scale=scale)], dtype=np.int64)
    anchor = mtg.topology_arrays()["is_anchor"]
    return vids[~anchor[vids]] if vids.size else vids


def _values(mtg, name, vids) -> np.ndarray:
    unique, inverse = np.unique(np.asarray(vids, dtype=np.int64), return_inverse=True)
    found, values = mtg.values_at(name, unique)
    out = np.full(unique.size, np.nan)
    known = np.array([value is not None for value in values.tolist()], dtype=bool)
    out[np.searchsorted(unique, found[known])] = np.asarray(values[known], dtype=float)
    return out[inverse]


def _owners(mtg, vids, scale) -> list:
    """Each vertex's complex at *scale* (-1: none)."""
    out = []
    for v in vids.tolist():
        owner = mtg.complex_at_scale(v, scale)
        out.append(-1 if owner is None else int(owner))
    return out


def _index_in(known: np.ndarray, ids) -> np.ndarray:
    """Position of each id in *known* (-1: not there)."""
    ids = np.asarray(ids, dtype=np.int64)
    if known.size == 0:
        return np.full(ids.size, -1, dtype=np.int64)
    order = np.argsort(known)
    position = np.searchsorted(known[order], ids).clip(0, known.size - 1)
    return np.where(known[order][position] == ids, order[position], -1)


def _name(link) -> str:
    return f"{link.receiver}.{link.variable} <- {link.provider}"
