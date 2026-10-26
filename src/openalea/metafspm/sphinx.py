"""
A component's reference page from its declarations, for Sphinx: in a package's docs `conf.py`,

    extensions = [..., "openalea.metafspm.sphinx"]

then in a page

    .. metafspm-component:: openalea.caribu.component.CaribuLight

renders its docstring, its variables by kind (inputs, forcings, state variables, parameters, settings: name, scale,
unit, description, default or provider) and its schedule (the steps in the order they run, with their docstrings).
`component_rst(cls)` gives the same text for other uses.
"""
import importlib
import inspect

KINDS = (("input", "Inputs"), ("forcing", "Forcings"), ("state_variable", "State variables"),
         ("plant_scale_state", "Plant-scale states"), ("parameter", "Parameters"), ("setting", "Settings"))


def _cell(value) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.split()).replace("|", "\\|") or " "


def component_rst(cls, heading: str = "-") -> str:
    """The reference text of component class *cls* (reStructuredText), its section titles underlined with
    *heading*; a model without declarations (an EnvironmentModel) by its docstring and its methods'."""
    lines = []
    doc = inspect.getdoc(cls)
    if doc:
        lines += [doc, ""]
    if not hasattr(cls, "declarations"):              # a model (an EnvironmentModel, a CompositeModel)
        interval = getattr(cls, "interval", None)
        if interval is not None:
            lines += [f"Runs fully every {interval:g} s.", ""]
        methods = [(name, inspect.getdoc(member)) for name, member in vars(cls).items()
                   if inspect.isfunction(member) and not name.startswith("_")]
        lines += [f"- ``{name}``: {' '.join(method_doc.split())}" for name, method_doc in methods if method_doc]
        return "\n".join(lines) + "\n"
    table = cls.declarations()
    for kind, title in KINDS:
        rows = table[table["kind"] == kind]
        if rows.empty:
            continue
        last = "provider" if kind == "input" else "default"
        lines += [title, heading * len(title), "", ".. list-table::", "   :header-rows: 1", "",
                  "   * - name", "     - scale", "     - unit", "     - description", f"     - {last}"]
        for row in rows.itertuples(index=False):
            lines += [f"   * - ``{row.name}``", f"     - {_cell(row.scale)}", f"     - {_cell(row.unit)}",
                      f"     - {_cell(row.description)}",
                      f"     - {_cell(row.by if kind == 'input' else row.default)}"]
        lines.append("")
    schedule = cls.schedule()
    if schedule:
        lines += ["Schedule", heading * len("Schedule"), ""]
        for stages, steps in schedule:
            lines.append(f"- **{', '.join(stages)}**: " + ", ".join(f"``{step}``" for step in steps))
            for step in steps:
                method = getattr(cls, step, None)
                step_doc = inspect.getdoc(method) if method is not None else None
                if step_doc:
                    lines.append(f"  ``{step}``: {' '.join(step_doc.split())}")
        lines.append("")
    return "\n".join(lines)


def _load(path: str):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def setup(app):
    """The Sphinx extension: the directive metafspm-component."""
    from docutils import nodes
    from docutils.statemachine import StringList
    from sphinx.util.docutils import SphinxDirective

    class ComponentDirective(SphinxDirective):
        required_arguments = 1
        has_content = False

        def run(self):
            cls = _load(self.arguments[0])
            title = cls.__name__
            text = f"{title}\n{'=' * len(title)}\n\n" + component_rst(cls)
            container = nodes.section()
            container.document = self.state.document
            self.state.nested_parse(StringList(text.splitlines()), self.content_offset, container, match_titles=True)
            return container.children

    app.add_directive("metafspm-component", ComponentDirective)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
