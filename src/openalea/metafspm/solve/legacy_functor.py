import inspect as ins
from typing import get_type_hints, get_origin, get_args


# General process resolution method
class Functor:
    numba_speedup = False
    
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
        

    def __call__(self, instance, data, data_type="<class 'dict'>", *args):
        if self.iterating:
            self.fun(instance)
        elif data_type == "<class 'dict'>":
            if self.total:
                data[self.name].update(
                    {1: self.fun(instance, *(data[arg] for arg in self.input_names))})
            else:
                if self.supplementary_outputs != 0:
                    outputs = [{} for _ in range(self.supplementary_outputs + 1)]
                    supplementary_names = []
                    for vid in data["focus_elements"]:
                        out = self.fun(instance, *(data[arg][vid] for arg in self.input_names))
                        outputs[0][vid] = out[0]
                        for s in range(self.supplementary_outputs):
                            output_name = out[2*s + 1]
                            if output_name not in supplementary_names:
                                supplementary_names.append(output_name)
                            outputs[s+1][vid] = out[2*s + 2]
                            
                    data[self.name].update(outputs[0])
                    for k, name in enumerate(supplementary_names):
                        data[name].update(outputs[k+1])
                else:
                    # print(self.name, [arg for arg in self.input_names if 254 not in data[arg].keys()])
                    data[self.name].update(
                        {vid: self.fun(instance, *(data[arg][vid] for arg in self.input_names)) for vid in data["focus_elements"]})
                
        elif data_type == "<class 'openalea.metafspm.data_structure.arraydict.ArrayDict'>":
            if self.total:
                data[self.name].update(
                    {1: self.fun(instance, *(data[arg] for arg in self.input_names))})
            else:
                # Array based and numba accelerated computations if compatible
                if self.numba_speedup:
                    mask = data["vertex_index"].indices_of(data["focus_elements"])
                    
                    if self.supplementary_outputs != 0:
                        out = self.fun(*(data[arg].values_array()[mask] for arg in self.input_names))
                        data[self.name].assign_at(mask, out[0])
                        for s in range(self.supplementary_outputs):
                            data[out[2*s + 1]].assign_at(mask, out[2*s + 2])
                    else:
                        # print(self.name, {arg: data[arg] for arg in self.input_names})
                        data[self.name].assign_at(mask, self.fun(*(data[arg].values_array()[mask] for arg in self.input_names)))

                # Else per element dictionnary-based computations
                else:
                    if self.supplementary_outputs != 0:
                        outputs = [{} for _ in range(self.supplementary_outputs + 1)]
                        supplementary_names = []
                        for vid in data["focus_elements"]:
                            out = self.fun(instance, *(data[arg][vid] for arg in self.input_names))
                            outputs[0][vid] = out[0]
                            for s in range(self.supplementary_outputs):
                                output_name = out[2*s + 1]
                                if output_name not in supplementary_names:
                                    supplementary_names.append(output_name)
                                outputs[s+1][vid] = out[2*s + 2]
                        data[self.name].update(outputs[0])
                        for k, name in enumerate(supplementary_names):
                            data[name].update(outputs[k+1])

                    
                    else:
                        # print(self.name, [arg for arg in self.input_names])
                        data[self.name].update(
                                {vid: self.fun(instance, *(data[arg][vid] for arg in self.input_names)) for vid in data["focus_elements"]})
            # data[self.name].assign_all(self.fun(instance, *(data[arg].values_array() for arg in self.input_names)))
                
        elif data_type == "<class 'numpy.ndarray'>":
            data[self.name] = self.fun(instance, *(data[arg] for arg in self.input_names))
