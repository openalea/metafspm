import numpy as np
import inspect as ins
from typing import get_type_hints, get_origin, get_args


# General process resolution method
class Functor:
    def __init__(self, fun, iteraring: bool = False, total: bool = False):
        self.fun = fun
        self.name = self.fun.__name__[1:]
        # The class qualified name (the function's without its own name) and module identify the step's class (DS13)
        self.class_qualname = self.fun.__qualname__.rsplit('.', 1)[0]
        self.class_name = self.class_qualname.rsplit('.', 1)[-1]
        self.family = f"{self.fun.__module__}:{self.class_qualname}"
        self.iterating = iteraring
        self.total = total
        self.input_names = self.inputs(self.fun)
        self.supplementary_outputs = int((self.num_outputs(fun) - 1) / 2)
        if len(self.input_names) == 0:
            self.iterating = True

    def inputs(self, fun):
        arguments = ins.getfullargspec(fun)[0]
        arguments.remove("self")
        return arguments
    
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
        Evaluate the step on DataStructure arrays and write the outputs in place (design note §8).
        Vectorised by default: one call with whole arrays; functions marked vectorized=False are called per element.
        When the DataStructure defines the step's mask (default "active", design note structure_and_boundaries §4,
        D15), arguments at the mask's location are restricted to the selected entities and outputs at that location
        are written back on them only: the other entities keep their values.
        """
        args, locations = self._arguments(instance, ds)
        mask, mask_location = self._mask(instance, ds)
        if mask is not None:
            args = [a[mask] if location == mask_location else a for a, location in zip(args, locations)]
        instance.__dict__["_in_equation"] = True        # self.<parameter> is refused inside steps (QH2)
        try:
            out = self._evaluate(instance, args)
        finally:
            instance.__dict__["_in_equation"] = False
        self._write_outputs(instance, ds, out, args, mask, mask_location)

    def _arguments(self, instance, ds):
        """
        The step's arguments and their locations. Parameters stored per plant (or as scalars) come broadcast to the
        step's location, the location of its other arguments (nodes, or cells on grids), as read-only views.
        """
        specs = getattr(instance, "_variable_specs", {})
        is_parameter = [getattr(specs.get(name), "variable_type", None) == "parameter"
                        and ds.location(name) not in ("node", "edge", "cell") for name in self.input_names]
        others = [ds.location(name) for name, p in zip(self.input_names, is_parameter) if not p]
        stores = ds._var_stores()
        step_location = next((loc for loc in others if loc in ("node", "edge", "cell")),
                             "node" if "node" in stores else "cell")
        args, locations = [], []
        for name, parameter in zip(self.input_names, is_parameter):
            if parameter:
                args.append(ds.parameter_view(name, step_location))
                locations.append(step_location)
            else:
                args.append(ds.get(name))
                locations.append(ds.location(name))
        return args, locations

    def _evaluate(self, instance, args):
        if getattr(self.fun, "__vectorized__", True):
            out = self.fun(instance, *args)
        else:
            shapes = [a.shape for a in args if a.ndim > 0]
            size = shapes[0][0] if shapes else 1
            per_element = [self.fun(instance, *(a if a.ndim == 0 else a[i] for a in args)) for i in range(size)]
            if self.supplementary_outputs:
                out = tuple(zip(*per_element))
                out = (np.asarray(out[0]),) + tuple(
                    x[0] if k % 2 == 0 else np.asarray(x) for k, x in enumerate(out[1:]))
            else:
                out = np.asarray(per_element)
        return out

    def _write_outputs(self, instance, ds, out, args, mask, mask_location):
        outputs = [(self.name, out[0] if self.supplementary_outputs else out)]
        for s in range(self.supplementary_outputs):
            outputs.append((out[2 * s + 1], out[2 * s + 2]))
        declared = getattr(self.fun, "__output_locations__", {})
        for name, values in outputs:
            values = np.asarray(values, dtype=float)
            masked = mask is not None and values.shape == (int(mask.sum()),)
            if not ds.has(name):
                if masked and name not in declared:
                    from openalea.metafspm.solve.decorator import infer_output_location
                    location = infer_output_location(f"{type(instance).__name__}.{self.name}", name, values.shape,
                                                     {mask_location: values.shape})
                else:
                    location = self._output_location(instance, ds, name, values, declared)
                ds.register(name, location=location)
            if masked and ds.location(name) == mask_location:
                full = np.array(ds.get(name), dtype=float)
                full[mask] = values
                values = full
            ds.set(name, values)

    def _mask(self, instance, ds):
        """(mask, location) restricting this step, or (None, None): "active" by default, where=None opts out."""
        where = getattr(self.fun, "__where__", "active")
        if where is None or not hasattr(ds, "has_mask"):
            return None, None
        if not ds.has_mask(where):
            if where != "active":
                raise KeyError(f"{type(instance).__name__}.{self.name}: mask '{where}' is not defined on the "
                               "DataStructure")
            return None, None
        return ds.mask(where), ds.__dict__["_masks"][where]["location"]

    def _output_location(self, instance, ds, name, values, declared):
        """
        Location of an output that is not a registered variable (design note datastructure_contract §5): given by
        the step decorator, "scalar" for total steps and 0-d values, else inferred from its shape when unambiguous.
        """
        if name in declared:
            location = declared[name]
            if hasattr(ds, "_mtg"):
                from openalea.metafspm.coupling.declaration import resolve_location
                location = resolve_location(ds, location)
            return location
        if self.total or values.ndim == 0:
            return "scalar"
        from openalea.metafspm.solve.decorator import infer_output_location
        stores = ds._var_stores()
        shapes = {location: ds._location_shape(location) for location in ("node", "edge", "cell") if location in stores}
        return infer_output_location(f"{type(instance).__name__}.{self.name}", name, values.shape, shapes)

    def __call__(self, instance, data, *args):
        """Run the step on *data*, the component's DataStructure (iterating steps only receive the instance)."""
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
                            "props-based components were removed, see docs/design/downstream_migration.md")
