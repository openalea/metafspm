"""
A port's corrections of its legacy's behaviours, by name: each port reproduces its legacy first (the regression
reference), then corrects its behaviours one by one, each correction a name a user chooses.

    CORRECTIONS = Corrections({"voxel_floor": "the receiver's voxel floor((G - O) / side)", ...})
    chosen = CORRECTIONS.resolve("recommended")          # a frozenset of names; "legacy" or () for none
    CORRECTIONS.table()                                   # name, description, presets: for the docs

Presets: "legacy" (none) and "recommended" (all, unless given); a port may add its own. With a *precision* (the
ports that reproduce their legacy's printed digits), corrections are refused with the legacy's precision, which
stays the bit-for-bit reference.
"""


class Corrections:
    """The named corrections of a port (*names*: {name: description}) and its presets ({preset: names})."""

    def __init__(self, names: dict, presets: dict = None):
        self.names = dict(names)
        self.presets = {"legacy": frozenset(), "recommended": frozenset(self.names)}
        for preset, chosen in (presets or {}).items():
            self.presets[preset] = self._checked(chosen)

    def _checked(self, chosen) -> frozenset:
        resolved = frozenset(chosen)
        unknown = sorted(resolved - set(self.names))
        if unknown:
            raise ValueError(f"unknown corrections {unknown}: {sorted(self.names)} or a preset {sorted(self.presets)}")
        return resolved

    def resolve(self, chosen, precision=None) -> frozenset:
        """The names of *chosen*: a preset's name, None or () for none, or names; refused with precision
        "legacy"."""
        if chosen is None:
            return frozenset()
        if isinstance(chosen, str):
            resolved = self.presets[chosen] if chosen in self.presets else self._checked((chosen,))
        else:
            resolved = self._checked(chosen)
        if resolved and precision == "legacy":
            raise ValueError("corrections apply with precision 'full': precision 'legacy' is the legacy's numbers")
        return resolved

    def table(self):
        """A pandas DataFrame: each correction's name, description and the presets holding it."""
        import pandas as pd
        return pd.DataFrame([dict(name=name, description=description,
                                  presets=", ".join(preset for preset, chosen in self.presets.items() if name in chosen))
                             for name, description in self.names.items()])

    def rst(self) -> str:
        """The corrections as a reStructuredText list table, for a port's documentation."""
        rows = [".. list-table::", "   :header-rows: 1", "", "   * - correction", "     - what it corrects"]
        for name, description in self.names.items():
            rows += [f"   * - ``{name}``", f"     - {description}"]
        return "\n".join(rows) + "\n"
