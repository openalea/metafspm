import numpy as np
import inspect as ins
from typing import get_type_hints, get_origin, get_args


# General process resolution method
class Functor:
    def __init__(self, fun, iteraring: bool = False, total: bool = False):
        self.fun = fun
        self.name = self.fun.__name__[1:]
        # The class qualified name (the function's without its own name) and module identify the step's class
        self.class_qualname = self.fun.__qualname__.rsplit('.', 1)[0]
        self.class_name = self.class_qualname.rsplit('.', 1)[-1]
        self.family = f"{self.fun.__module__}:{self.class_qualname}"
        self.timing_name = f"functor.{self.class_name}.{self.name}"     # run.json's "compute"/"timing" entry
        self.iterating = iteraring
        self.total = total
        self.input_names = self.inputs(self.fun)
        self.supplementary_outputs = int((self.num_outputs(fun) - 1) / 2)
        self.returns_dict = self._annotated_dict(fun)
        if len(self.input_names) == 0:
            self.iterating = True

    def inputs(self, fun):
        arguments = ins.getfullargspec(fun)[0]
        arguments.remove("self")
        return arguments
    
    @staticmethod
    def _annotated_dict(func):
        """Whether the step is annotated `-> dict` (its outputs named by the keys it returns)."""
        returned = get_type_hints(func).get("return")
        return returned is dict or get_origin(returned) is dict

    def num_outputs(self, func):
        """
        Convention: multiple outputs must be annotated as `-> tuple[...]`.
        Returns an int, "variadic" for tuple[T, ...], or 1 for non-tuple returns.
        Returns None if there's no return annotation.
        """
        ret = get_type_hints(func).get("return")
        # print(ret)
        if ret is None:
            return 1  # no annotation → unknown

        elif get_origin(ret) is tuple:
            return len(get_args(ret))
        
        else:
            return 1
        

    def _call_on_data_structure(self, instance, ds):
        """
        Evaluate the step on DataStructure arrays and write the outputs in place.
        Vectorised by default: one call with whole arrays; functions marked vectorized=False are called per element.
        When the DataStructure defines the step's mask (default "active"), arguments at the mask's location are restricted to the selected entities and outputs at that location
        are written back on them only: the other entities keep their values.
        """
        args, locations = self._arguments(instance, ds)
        mask, mask_location = self._mask(instance, ds)
        if mask is not None:
            args = [a[mask] if location == mask_location else a for a, location in zip(args, locations)]
        key = self._memo_key(instance, ds, args, mask) if getattr(self.fun, "__memo__", False) else None
        kept = instance.__dict__.get("_step_memos", {}).get(self.name) if key is not None else None
        if kept is not None and kept[0] == key:
            out = _copied(kept[1])                      # the same values: the last outputs
        else:
            instance.__dict__["_in_equation"] = True        # self.<parameter> is refused inside steps
            try:
                out = self._evaluate(instance, args)
            finally:
                instance.__dict__["_in_equation"] = False
            if key is not None:
                instance.__dict__.setdefault("_step_memos", {})[self.name] = (key, _copied(out))
        if out is None:            # a step returning None writes nothing (e.g. an inherited step disabled)
            return
        self._write_outputs(instance, ds, out, args, mask, mask_location)

    def _memo_key(self, instance, ds, args, mask) -> str:
        """A memoized step's key: the structure, its component's parameters (their versions) and settings, the mask
        and its arguments' values."""
        from openalea.metafspm.solve.memo import content_key
        cls = type(instance)
        structure = ds.structure_stamp() if hasattr(ds, "structure_stamp") else getattr(ds, "topology_version", None)
        parameters = tuple(ds.write_count(name) if ds.has(name) else -1
                           for name in (cls.declared_parameters() if hasattr(cls, "declared_parameters") else ()))
        settings = tuple(repr(getattr(instance, name)) for name in (
            cls.declared_settings() if hasattr(cls, "declared_settings") else ()))
        return content_key(repr((structure, parameters, settings)), 0 if mask is None else mask,
                           *(a for a in args if a is not None))

    def _arguments(self, instance, ds):
        """
        The step's arguments and their locations. Parameters stored per plant (or as scalars) come broadcast to the
        step's location, the location of its other arguments (nodes, or cells on grids), as read-only views. A
        variable the component reads at the nodes from a coarser scale (declared mapped down, e.g. a SubOrgan's
        length broadcast to its Compartments) while another component holds it at that scale comes mapped too.
        """
        specs = getattr(instance, "_variable_specs", {})
        settings = set(type(instance).declared_settings()) if hasattr(type(instance), "declared_settings") else set()
        optional = getattr(type(instance), "_optional", lambda name: False)
        missing = {name for name in self.input_names if name in specs and not ds.has(name) and optional(name)}

        def where(name):                       # an optional input nothing provides: at its declared location
            return specs[name].location if name in missing else ds.location(name)

        is_parameter = [name not in settings and getattr(specs.get(name), "variable_type", None) == "parameter"
                        and where(name) not in ("node", "edge", "cell") for name in self.input_names]
        from openalea.metafspm.coupling.declaration import held_at_scale, mapped_read
        mapped = [name not in settings and not p and held_at_scale(specs.get(name), ds)
                  for name, p in zip(self.input_names, is_parameter)]
        others = [specs[name].location if m else where(name)
                  for name, p, m in zip(self.input_names, is_parameter, mapped) if not p and name not in settings]
        stores = ds._var_stores()
        step_location = next((loc for loc in others if loc in ("node", "edge", "cell")), None)
        if step_location is None:              # no argument per node: its outputs' location (e.g. the plants'), else nodes
            step_location = self._outputs_location(instance, specs, stores) or ("node" if "node" in stores else "cell")
        args, locations = [], []
        for name, parameter, down in zip(self.input_names, is_parameter, mapped):
            if name in settings:                        # a setting: the value itself
                args.append(getattr(instance, name))
                locations.append(None)
            elif down:
                args.append(mapped_read(ds, specs[name]))
                locations.append(specs[name].location)
            elif name in missing:
                spec = specs[name]                        # an optional input nothing provides: its default
                args.append(np.full(tuple(ds._location_shape(spec.location)) + tuple(spec.shape or ()),
                                    np.nan if spec.default is None else spec.default))
                locations.append(spec.location)
            elif parameter:
                args.append(ds.parameter_view(name, step_location))
                locations.append(step_location)
            else:
                args.append(ds.get(name))
                locations.append(ds.location(name))
        return args, locations

    def _outputs_location(self, instance, specs, stores):
        """The one location of the step's declared outputs (its method's name, its declared output locations), or None:
        a step of coarse outputs (a plant's) gets its parameters there."""
        declared = getattr(self.fun, "__output_locations__", {})
        locations = {specs[name].location for name in [self.name, *declared] if name in specs}
        locations |= set(declared.values())
        if len(locations) == 1:
            (location,) = locations
            if location in stores:
                return location
        return None

    def _evaluate(self, instance, args):
        if getattr(self.fun, "__vectorized__", True):
            out = self.fun(instance, *args)
        else:
            arrays = [isinstance(a, np.ndarray) and a.ndim > 0 for a in args]      # settings stay whole
            shapes = [a.shape for a, array in zip(args, arrays) if array]
            size = shapes[0][0] if shapes else 1
            per_element = [self.fun(instance, *(a[i] if array else a for a, array in zip(args, arrays)))
                           for i in range(size)]
            if per_element and isinstance(per_element[0], dict):
                out = {name: np.asarray([values[name] for values in per_element]) for name in per_element[0]}
            elif self.supplementary_outputs:
                out = tuple(zip(*per_element))
                out = (np.asarray(out[0]),) + tuple(
                    x[0] if k % 2 == 0 else np.asarray(x) for k, x in enumerate(out[1:]))
            else:
                out = np.asarray(per_element)
        return out

    def _write_outputs(self, instance, ds, out, args, mask, mask_location):
        if isinstance(out, dict):                       # named outputs: {name: values}
            outputs = list(out.items())
        elif self.returns_dict:
            raise ValueError(f"{self.class_name}.{self.name} is annotated -> dict but returns {type(out).__name__}")
        else:
            if self.supplementary_outputs and (not isinstance(out, tuple)
                                               or len(out) != 2 * self.supplementary_outputs + 1):
                given = len(out) if isinstance(out, tuple) else 1
                raise ValueError(f"{self.class_name}.{self.name} returns {given} items, its annotation declares "
                                 f"{2 * self.supplementary_outputs + 1} (its own output, then name, values pairs)")
            outputs = [(self.name, out[0] if self.supplementary_outputs else out)]
            for s in range(self.supplementary_outputs):
                outputs.append((out[2 * s + 1], out[2 * s + 2]))
        # Nodes created by the step (a NewNodes pair, StructuralComponent): the repartition's basis is taken before any
        # output is written, the nodes are created after the outputs on the existing ones
        from openalea.metafspm.coupling.component import NewNodes
        new_nodes = [values for _, values in outputs if isinstance(values, NewNodes)]
        outputs = [(name, values) for name, values in outputs if not isinstance(values, NewNodes)]
        if new_nodes and not hasattr(instance, "_create_nodes"):
            raise TypeError(f"{self.class_name}.{self.name} returns NewNodes: only a StructuralComponent creates nodes")
        before = instance._before_new_nodes() if new_nodes else None
        declared = getattr(self.fun, "__output_locations__", {})
        for name, values in outputs:
            values = _output_array(values, ds.get(name).dtype if ds.has(name) else None)
            masked = (mask is not None and values.ndim >= 1 and values.shape[0] == int(mask.sum())
                      and (values.ndim == 1 or (ds.has(name) and ds.get(name).shape[1:] == values.shape[1:])))
            if not ds.has(name):
                if masked and name not in declared:
                    from openalea.metafspm.solve.decorator import _undeclared_output
                    _undeclared_output(f"{type(instance).__name__}.{self.name}", name)
                else:
                    location = self._output_location(instance, ds, name, values, declared)
                ds.register(name, location=location)
            if masked and ds.location(name) == mask_location:
                full = np.array(ds.get(name), dtype=object if values.dtype == object else float)
                full[mask] = values
                values = full
            ds.set(name, values)
        for new in new_nodes:
            instance._create_nodes(new, before)
            before = instance._before_new_nodes() if before is not None else None

    def _mask(self, instance, ds):
        """
        (mask, location) restricting this step, or (None, None): the step's filters, and the DataStructure's "active"
        mask unless include_inactive (both at the same location; an edge filter is not restricted by the active nodes).
        """
        if not hasattr(ds, "has_mask"):
            return None, None
        masks = ds.__dict__.get("_masks", {})
        active = None
        if not getattr(self.fun, "__include_inactive__", False) and ds.has_mask("active"):
            active = (ds.mask("active"), masks["active"]["location"])
        rule = getattr(self.fun, "__filters__", None)
        if rule is None:
            return active if active is not None else (None, None)
        from openalea.metafspm.solve.decorator import _Filters
        location = self._filters_location(ds, rule)
        name = _Filters(rule, f"{type(instance).__name__}.{self.name}", location).mask_name(ds)
        mask, location = ds.mask(name), ds.__dict__["_masks"][name]["location"]
        if active is not None and active[1] == location:
            mask = mask & active[0]
        return mask, location

    @staticmethod
    def _filters_location(ds, rule):
        """The entities a step's filters select: edges when every key is an edge variable, else nodes (or cells)."""
        if isinstance(rule, dict) and all(ds.has(key) and ds.location(key) == "edge" for key in rule):
            return "edge"
        return "node"

    def _output_location(self, instance, ds, name, values, declared):
        """
        Location of an output that is not a registered variable: given by
        the step decorator, "scalar" for total steps and 0-d values, else an error (its location must be declared).
        """
        if name in declared:
            location = declared[name]
            if hasattr(ds, "_mtg"):
                from openalea.metafspm.coupling.declaration import resolve_location
                location = resolve_location(ds, location)
            return location
        if self.total or values.ndim == 0:
            return "scalar"
        from openalea.metafspm.solve.decorator import _undeclared_output
        _undeclared_output(f"{type(instance).__name__}.{self.name}", name)

    def __call__(self, instance, data, *args):
        """Run the step on *data*, the component's DataStructure (iterating steps only receive the instance). Its
        wall time and calls are counted under functor.<Class>.<step> (Compute.timed: run.json's "compute"/"timing"),
        inclusive of the steps it runs inside (e.g. a reduction's library components evaluating its archetypes)."""
        from openalea.metafspm.compute import Compute
        with Compute.current().timed(self.timing_name):
            self._run(instance, data)

    def _run(self, instance, data):
        if self.iterating:
            # Steps without arguments of a StructuralComponent edit the MPG: the component synchronises around them
            run_mpg_step = getattr(instance, "_run_mpg_step", None)
            if run_mpg_step is not None:
                run_mpg_step(lambda: self.fun(instance))
            else:
                self.fun(instance)
        elif hasattr(data, "get") and hasattr(data, "register") and hasattr(data, "location"):
            self._call_on_data_structure(instance, data)
        else:
            raise TypeError(f"Step {self.class_name}.{self.name} needs a DataStructure (got {type(data).__name__}); "
                            "props-based components were removed, see docs/migration.md")


def _copied(out):
    """A step's outputs with their arrays copied (kept, or given again, without sharing them)."""
    if isinstance(out, np.ndarray):
        return out.copy()
    if isinstance(out, dict):
        return {name: _copied(value) for name, value in out.items()}
    if isinstance(out, tuple):
        return tuple(_copied(value) for value in out)
    return out


def _output_array(values, stored_dtype=None) -> np.ndarray:
    """A step's output as an array: floats, or objects (one record per entity: a tuple, a string, a dict) for a
    variable of dtype object or values that are objects."""
    if stored_dtype == object or (isinstance(values, np.ndarray) and values.dtype == object):
        if isinstance(values, np.ndarray) and values.dtype == object:
            return values
        out = np.empty(len(values), dtype=object)
        out[:] = list(values)
        return out
    return np.asarray(values, dtype=float)
