"""
A run's provenance: what produced it, recorded in run.json, in every checkpoint's scene.json and in every frame's
writer, so that a run can be reproduced from its folder::

  provenance(scene)            {"packages": {distribution: {"version", "commit", "dirty"}}, "python", "start",
                                "coupling", "links", "seeds", "parameters": {"hash", "non_default"}, "checks"}
  mismatches(recorded, current)
                               ["openalea.mecha: commit 9bcb3f9 (clean) then, 1a2b3c4 (modified) now", ...]

packages: python's and every distribution the scene's models and components come from (metafspm, each port), with
numpy, scipy, pandas, numba, openalea.mtg and encoding3d; an editable install (a clone) adds its git commit and
whether its tracked files were modified (dirty). seeds: each component's random_seed (Component.random's streams).
parameters: the hash of every component's parameters and settings as the scene runs them, and those differing from
their declared default. checks: the self-checks the run ran (Compute.checks). links: the scene translator's links (Translator.to_nested). The forcings applied are
summarised apart (ForcingSummary: the mean, least and greatest of each forcing's per-step means), in run.json.

Scene.load_checkpoint compares the checkpoint's provenance with the scene's and warns with each difference (another
commit, version, parameter value, link, seed, the variables a component declares now and the checkpoint lacks); it
refuses only a checkpoint of another format (data_api.check_checkpoint_format).
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution, distributions

import numpy as np

# The packages every run records, whatever its models
_ALWAYS = ("openalea.metafspm", "numpy", "scipy", "pandas", "numba", "openalea.mtg", "openalea.encoding3d")


@lru_cache(maxsize=None)
def _editable() -> tuple:
    """((name, version, source folder)) of the editable installs (pip install -e: a clone)."""
    found = []
    for installed in distributions():
        try:
            direct = json.loads(installed.read_text("direct_url.json") or "{}")
        except (ValueError, OSError):
            continue
        url = direct.get("url", "")
        if direct.get("dir_info", {}).get("editable") and url.startswith("file://"):
            found.append((installed.metadata["Name"], installed.version, os.path.abspath(url[len("file://"):])))
    return tuple(found)


@lru_cache(maxsize=None)
def _git(folder: str) -> dict:
    """{"commit", "dirty"} of the git clone *folder* is in ({} when none, or without git)."""
    try:
        commit = subprocess.run(["git", "-C", folder, "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=10)
        if commit.returncode != 0:
            return {}
        status = subprocess.run(["git", "-C", folder, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return {}
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


@lru_cache(maxsize=None)
def package(module: str):
    """(distribution name, {"version", and for a clone "commit", "dirty"}) of the installed module *module*
    ("openalea.mecha.component"), or None when no distribution has it."""
    loaded = sys.modules.get(module)
    path = os.path.abspath(getattr(loaded, "__file__", None) or "") if loaded is not None else ""
    for name, version, folder in _editable():         # a clone: the one whose folder holds the module's file
        if path and path.startswith(folder + os.sep):
            return name, {"version": version, **_git(folder)}
    parts = module.split(".")
    for end in range(len(parts), 0, -1):             # "openalea.cnwgrass.metafspm": openalea.cnwgrass's
        for name in (".".join(parts[:end]), "-".join(parts[:end])):
            try:
                found = distribution(name)
            except PackageNotFoundError:
                continue
            return found.metadata["Name"], {"version": found.version}
    return None


def packages(scene=None) -> dict:
    """{distribution: {"version", "commit", "dirty"}}: the always recorded ones and those of *scene*'s models and
    components."""
    modules = list(_ALWAYS)
    if scene is not None:
        modules += [population.model.__module__ for population in scene.populations]
        modules += [type(model).__module__ for model in scene.environment]
        modules += [type(component).__module__ for component in scene.components]
    found = {}
    for module in dict.fromkeys(modules):
        described = package(module)
        if described is not None:
            found.setdefault(described[0], described[1])
    return dict(sorted(found.items()))


def _json(value):
    """*value* as JSON (the recorder's description: arrays as lists, classes and functions by name)."""
    from openalea.metafspm.scene.recorder import _described
    return _described(value)


def _parameter_value(component, name: str, default):
    """The value *component* runs with: a per-plant parameter's from its DataStructure (one number when every
    plant has the same, else the plants' values), else the component's attribute."""
    data_structure = getattr(component, "data_structure", None)
    try:
        stored = data_structure is not None and data_structure.has(name)
    except Exception:
        stored = False
    if stored:
        values = np.asarray(data_structure.get(name))
        if values.dtype.kind in "biuf" and values.size and np.all(values == values.flat[0]):
            return values.flat[0]
        return values
    return getattr(component, name, default)


def parameters(scene) -> dict:
    """{"hash": the sha256 of every component's parameters and settings, "non_default": {component: {name: value}}
    those differing from their declared default}."""
    values, non_default = {}, {}
    for component in scene.components:
        declarations = type(component).declarations()
        if declarations.empty:
            continue
        name = component.__dict__.get("_scene_name", type(component).__name__)
        rows = declarations[declarations["role"].isin(("parameter", "setting"))]
        own = {row.name: _json(_parameter_value(component, row.name, row.default))
               for row in rows.itertuples(index=False)}
        values[name] = own
        changed = {key: value for key, value, default in zip(own, own.values(), rows["default"])
                   if value != _json(default)}
        if changed:
            non_default[name] = changed
    text = json.dumps(values, sort_keys=True, default=repr)
    return {"hash": hashlib.sha256(text.encode()).hexdigest(), "non_default": non_default}


def _coupling(coupling) -> dict:
    """A FixedPoint's description: its class and fields (its steps by name)."""
    from dataclasses import fields, is_dataclass
    if not is_dataclass(coupling):
        return {"class": type(coupling).__name__}
    return {"class": type(coupling).__name__,
            **{field.name: _json(getattr(coupling, field.name)) for field in fields(coupling)}}


def provenance(scene, variables: bool = False) -> dict:
    """What produced *scene*'s run (see the module); with *variables*, each component's declared variables too
    (a checkpoint's, compared at its load)."""
    coupling = getattr(scene, "coupling", "explicit")
    content = {"python": platform.python_version(), "packages": packages(scene),
               "start": None if scene.start is None else scene.start.isoformat(),
               "coupling": coupling if isinstance(coupling, str) else _coupling(coupling),
               "links": _json(scene.translator.to_nested()),
               "seeds": {component.__dict__.get("_scene_name", type(component).__name__): _json(component.random_seed)
                         for component in scene.components if hasattr(component, "random_seed")},
               "parameters": parameters(scene),
               "checks": list(scene.compute.checks)}
    if variables:
        content["variables"] = {component.__dict__.get("_scene_name", type(component).__name__):
                                sorted(type(component).declarations()["name"])
                                for component in scene.components if hasattr(type(component), "declarations")}
    return content


def frame_writer(scene) -> dict:
    """The provenance a frame's writer carries: python's and the packages' versions and commits."""
    return {"python": platform.python_version(), "packages": packages(scene)}


def _package_label(described: dict) -> str:
    label = described.get("version", "?")
    if "commit" in described:
        label += f" commit {described['commit'][:9]} ({'modified' if described.get('dirty') else 'clean'})"
    return label


def mismatches(recorded: dict, current: dict, parameters: bool = True) -> list:
    """Each difference between the *recorded* provenance (a checkpoint's) and the *current* one, in words; [] when
    none (or nothing recorded). *parameters* False: the parameters left out (a run resumed with changed ones)."""
    if not recorded:
        return []
    found = []
    if recorded.get("python") != current.get("python"):
        found.append(f"python {recorded.get('python')} then, {current.get('python')} now")
    before, now = recorded.get("packages", {}), current.get("packages", {})
    for name in sorted(set(before) | set(now)):
        if name not in now:
            found.append(f"{name}: {_package_label(before[name])} then, not used now")
        elif name not in before:
            found.append(f"{name}: not used then, {_package_label(now[name])} now")
        elif _package_label(before[name]) != _package_label(now[name]):
            found.append(f"{name}: {_package_label(before[name])} then, {_package_label(now[name])} now")
    for key in ("start", "coupling"):
        if recorded.get(key) != current.get(key):
            found.append(f"{key}: {recorded.get(key)!r} then, {current.get(key)!r} now")
    if recorded.get("links") != current.get("links"):
        found.append("the scene's links differ (scene.translator)")
    seeds_before, seeds_now = recorded.get("seeds", {}), current.get("seeds", {})
    for name in sorted(set(seeds_before) | set(seeds_now)):
        if seeds_before.get(name) != seeds_now.get(name):
            found.append(f"{name}.random_seed: {seeds_before.get(name)} then, {seeds_now.get(name)} now")
    parameters_before, parameters_now = recorded.get("parameters", {}), current.get("parameters", {})
    if parameters and parameters_before.get("hash") != parameters_now.get("hash"):
        changed_before = parameters_before.get("non_default", {})
        changed_now = parameters_now.get("non_default", {})
        names = sorted({f"{component}.{name}" for side in (changed_before, changed_now)
                        for component, values in side.items() for name in values
                        if changed_before.get(component, {}).get(name) != changed_now.get(component, {}).get(name)})
        found.append("the parameters differ" + (f": {', '.join(names)}" if names else
                                                 " (their defaults: another code version)"))
    variables_before, variables_now = recorded.get("variables"), current.get("variables")
    if variables_before is not None and variables_now is not None:
        for name in sorted(variables_now):
            if name not in variables_before:
                continue
            new = sorted(set(variables_now[name]) - set(variables_before[name]))
            if new:
                found.append(f"{name} declares variables the checkpoint lacks: {', '.join(new)}")
    return found


class ForcingSummary:
    """The forcings a scene applied: for each, the mean, least and greatest of its per-step means (a spatial
    forcing's averaged over its entities), and the steps counted; run.json's "forcings", kept by checkpoints."""

    def __init__(self, state: dict = None):
        self.state = {name: dict(values) for name, values in (state or {}).items()}

    def add(self, name: str, value) -> None:
        mean = float(np.mean(value))
        entry = self.state.setdefault(name, {"sum": 0., "steps": 0, "min": mean, "max": mean})
        entry["sum"] += mean
        entry["steps"] += 1
        entry["min"], entry["max"] = min(entry["min"], mean), max(entry["max"], mean)

    def as_dict(self) -> dict:
        return {name: {"mean": entry["sum"] / entry["steps"], "min": entry["min"], "max": entry["max"],
                       "steps": entry["steps"]} for name, entry in sorted(self.state.items()) if entry["steps"]}
