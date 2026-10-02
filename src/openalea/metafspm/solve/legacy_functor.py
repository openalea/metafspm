import numpy as np
import inspect as ins
from typing import get_type_hints, get_origin, get_args


# General process resolution method
class Functor:
    def __init__(self, fun, iteraring: bool = False, total: bool = False):
        self.fun = fun
        self.name = self.fun.__name__[1:]
        self.class_name = self.fun.__qualname__.split('.')[0]
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
        """
        args = [ds.get(name) for name in self.input_names]
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

        outputs = [(self.name, out[0] if self.supplementary_outputs else out)]
        for s in range(self.supplementary_outputs):
            outputs.append((out[2 * s + 1], out[2 * s + 2]))
        declared = getattr(self.fun, "__output_locations__", {})
        for name, values in outputs:
            values = np.asarray(values, dtype=float)
            if not ds.has(name):
                ds.register(name, location=self._output_location(instance, ds, name, values, declared))
            ds.set(name, values)

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
            self.fun(instance)
        elif hasattr(data, "get") and hasattr(data, "register") and hasattr(data, "location"):
            self._call_on_data_structure(instance, data)
        else:
            raise TypeError(f"Step {self.class_name}.{self.name} needs a DataStructure (got {type(data).__name__}); "
                            "props-based components were removed, see docs/design/downstream_migration.md")
