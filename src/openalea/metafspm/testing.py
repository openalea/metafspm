"""
Checks for downstream packages: run them in a package's own tests to make sure its components
stay couplable with the translators it ships.
"""
from dataclasses import fields

# Framework fields that are not model variables
_FRAMEWORK_FIELDS = {"data_structure", "time_step"}


def couplability_problems(component_cls, translator, name: str = None, data_structure=None) -> list:
    """
    Problems preventing *component_cls* from being coupled through *translator* (a coupling.translator.Translator):

      * a declared field without declare() metadata;
      * a link whose receiving variable the component does not declare;
      * a link reading a variable of the component that it does not declare;
      * with a *data_structure*: a declaration that does not resolve on it (scale, location, mapping), and an
        inconsistent DataStructure (DataStructure.validate).

    *name* is the component name used in the translator (default: the class name).
    """
    name = name or component_cls.__name__
    declared = {f.name: f for f in fields(component_cls) if f.name not in _FRAMEWORK_FIELDS}
    problems = [f"{name}.{f.name} has no declare() metadata (variable_type, by, unit, ...)"
                for f in declared.values() if "variable_type" not in f.metadata]
    for link in translator.links:
        if link.receiver == name and link.variable not in declared:
            problems.append(f"{name}.{link.variable} is linked from {link.provider} but {name} does not declare it")
        if link.provider == name:
            for source in link.sources:
                if source not in declared:
                    problems.append(f"{link.receiver}.{link.variable} reads {name}.{source}, which {name} does not declare")
    if data_structure is not None:
        from openalea.metafspm.coupling.composite_wrapper import CompositeModel
        from openalea.metafspm.coupling.declaration import DeclarationError, declared_specs, kinds_agree
        try:
            specs = declared_specs(component_cls, data_structure)
        except DeclarationError as error:
            problems.append(str(error))
            specs = {}
        # Links into this component whose providers are already on the DataStructure: kinds and scale mappings
        ds = data_structure
        for link in translator.links:
            spec = specs.get(link.variable)
            if link.receiver != name or spec is None or not all(ds.has(source) for source in link.sources):
                continue
            for source in link.sources:
                provided = ds._variable_meta().get(ds._resolve(source), {}).get("kind")
                if not kinds_agree(spec.kind, provided):
                    problems.append(f"{name}.{link.variable} ({spec.kind}) <- {link.provider}.{source} ({provided}): "
                                    "the kinds do not agree")
            if link.aggregation is None:
                try:
                    CompositeModel._default_link_mapping(ds, link, spec.location)
                except DeclarationError as error:
                    problems.append(str(error))
        if hasattr(data_structure, "validate"):
            try:
                data_structure.validate()
            except ValueError as error:
                problems.append(str(error))
    return problems


def assert_component_couplable(component_cls, translator, name: str = None, data_structure=None) -> None:
    problems = couplability_problems(component_cls, translator, name=name, data_structure=data_structure)
    if problems:
        raise AssertionError(f"{name or component_cls.__name__} is not couplable:\n  " + "\n  ".join(problems))


# ── Comparisons with a legacy ─────────────────────────────────────────────────
#
# A port keeps its legacy frozen (`<package>.legacy`, or the original in another environment) and compares with it:
# `<package>/comparison/legacy_run.py` runs the legacy, `port_run.py` the port on the same inputs, and
# `test/comparison/` asserts their outputs equal, bit for bit or within stated tolerances. The helpers below are
# the pieces the ports shared.

def run_executables(steps, files: dict, outputs=()) -> dict:
    """{output name: text} of executables run one after the other in a temporary directory holding *files*
    ({name: text}, written as given: no newline translation on Windows); *steps* [(executable, [arguments])] (an executable found on the PATH); outputs ending with a
    "/"-prefixed folder name or not decodable are returned as bytes."""
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        for name, text in files.items():
            path = Path(directory, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text) if isinstance(text, bytes) else path.write_text(text, newline="")
        for executable, arguments in steps:
            command = shutil.which(executable) or executable
            subprocess.run([command, *map(str, arguments)], cwd=directory, capture_output=True, check=True)
        found = {}
        for name in outputs:
            path = Path(directory, name)
            if path.exists():
                data = path.read_bytes()
                try:
                    found[name] = data.decode()
                except UnicodeDecodeError:
                    found[name] = data
        return found


def run_in_env(env: str, script: str, cwd=None, timeout=None, variables=None) -> str:
    """The standard output of Python *script* (its source) run in conda environment *env* (a legacy's own, e.g. an
    original package that cannot share the port's); raises with its error output when it fails."""
    import json
    import os
    import subprocess
    conda = os.environ.get("CONDA_EXE", "conda")
    prefixes = json.loads(subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True,
                                         check=True).stdout)["envs"]
    prefix = next((p for p in prefixes if os.path.basename(p) == env), None)
    if prefix is None:
        raise ValueError(f"no conda environment '{env}'")
    python = os.path.join(prefix, "python.exe") if os.name == "nt" else os.path.join(prefix, "bin", "python")
    done = subprocess.run([python, "-c", script], cwd=cwd, capture_output=True,
                          text=True, timeout=timeout, env=dict(os.environ, **(variables or {})))
    if done.returncode != 0:
        raise RuntimeError(f"the script failed in '{env}':\n{done.stderr[-3000:]}")
    return done.stdout


def text_differences(expected: str, actual: str, limit: int = 5) -> list:
    """The first *limit* rows where two texts differ: [(row number from 1, expected row, actual row)]; a missing
    row is None."""
    first, second = expected.splitlines(), actual.splitlines()
    found = []
    for number in range(max(len(first), len(second))):
        a = first[number] if number < len(first) else None
        b = second[number] if number < len(second) else None
        if a != b:
            found.append((number + 1, a, b))
            if len(found) == limit:
                break
    return found


def assert_texts_equal(expected: str, actual: str, name: str = "text") -> None:
    """Two outputs equal as text, row by row (the legacy's files against the port's), the first differences shown."""
    found = text_differences(expected, actual)
    if found:
        rows = "\n".join(f"  row {number}:\n    expected {a!r}\n    actual   {b!r}" for number, a, b in found)
        raise AssertionError(f"{name} differs:\n{rows}")


def array_differences(expected, actual, rtol: float = 0., atol: float = 0.) -> dict:
    """{"count", "first", "largest"} of the entries where two arrays differ beyond *rtol*, *atol* (NaNs equal at the
    same places); {} when equal. Shapes must agree."""
    import numpy as np
    a, b = np.asarray(expected), np.asarray(actual)
    if a.shape != b.shape:
        raise AssertionError(f"shapes differ: {a.shape} expected, {b.shape} actual")
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        both_nan = np.isnan(a64) & np.isnan(b64)
        with np.errstate(invalid="ignore"):
            close = (a64 == b64) | (np.abs(a64 - b64) <= atol + rtol * np.abs(a64))
        differ = ~(close | both_nan)
        gaps = np.where(differ, np.abs(a64 - b64), 0.)
        largest = float(np.nanmax(gaps)) if differ.any() else 0.
    else:
        differ = a != b
        largest = None
    if not differ.any():
        return {}
    first = tuple(int(i) for i in np.argwhere(differ)[0])
    return {"count": int(differ.sum()), "first": first, "largest": largest}


def assert_arrays_equal(expected, actual, rtol: float = 0., atol: float = 0., name: str = "array") -> None:
    """Two arrays equal (bitwise by default, within *rtol*, *atol* when given): the number of differing entries,
    the first one and the largest gap shown."""
    import numpy as np
    found = array_differences(expected, actual, rtol, atol)
    if found:
        index = found["first"]
        raise AssertionError(f"{name}: {found['count']} entries differ (largest gap {found['largest']}); the first at "
                             f"{index}: expected {np.asarray(expected)[index]!r}, actual {np.asarray(actual)[index]!r}")


class worktree:
    """A context manager: *ref* of git repository *repo* checked out apart, in a temporary folder (its path), and
    removed after: to run an earlier version beside the current one without stash, reset or checkout."""

    def __init__(self, repo, ref: str):
        self.repo, self.ref, self.path = str(repo), ref, None

    def __enter__(self) -> str:
        import subprocess
        import tempfile
        self.path = tempfile.mkdtemp(prefix="worktree_")
        subprocess.run(["git", "-C", self.repo, "worktree", "add", "-q", "--detach", self.path, self.ref], check=True,
                       capture_output=True)
        return self.path

    def __exit__(self, *exception):
        import subprocess
        subprocess.run(["git", "-C", self.repo, "worktree", "remove", "--force", self.path], check=False,
                       capture_output=True)
        return False
