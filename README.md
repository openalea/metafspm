# MetaFSPM

## Purpose 

This package is intended to provide minimal tools to assist Functionnal Structural Plant Modellers (FSPM) so that they can make their production more readable regarding scientific content. It is also intended as a minimal constraint in object oriented programming to ease the model coupling in OpenAlea around a Multiscale Tree Graph (MTG). However the idea is to remain very generic in order to accept a wide variety of model types. You can't impose formalism on others, that's how great civilizations die. Our aim here is rather to favor research projects among different scientific communities.

## Main principles

## Installation

Prerequisites to installation :
- miniconda (https://docs.anaconda.com/free/miniconda/miniconda-install/) 
- git (https://git-scm.com/downloads)

In a terminal, optionally install mamba for faster installation
```
conda install -y -c conda-forge mamba
```

- From terminal, clone this package and then , then run the command :
```
git clone https://github.com/GeraultTr/metafspm.git
```

- Create an environment dedicated to your model with the necessary requirements
```
mamba create -n your_model_env -c conda-forge -c openalea3 --strict-channel-priority --file requirements.txt
mamba activate your_model_env
```

- Finally, run the setup.py :
```
python -m setup develop
```

### Second option TODO : when package is released, just create in:
```
mamba install -c conda-forge -c openalea3 metafspm
```

## Example use

### For model design

- First, in a single python .py file the model has to be packaged as a class and decorated by @dataclass from the dataclasses module.

- Then, you must import utilities :  
```
from metafspm.component import Model, declare
from metafspm.component_factory import *
```

- Make your model class inherit the imported "Model" class to benefit from the utilities of this package.

- In the class attributes, define your model variables (inputs, state_variables, plant_scale_state_variables, parameters) under the following format using the imported function declare : 
```
@dataclass
class YourModel(Model):
    variable: float = declare(default=0., unit="", unit_comment="", descripition="",
                                min_value="", max_value="", value_comment="", reference="", DOI="",
                                variable_type="", by="", state_variable_type="", edit_by="user")
    def __init__(self, g):
        ...
```
All the fields of the declare function have to be filled, see declare() docstring for more informations on field constraints.

- Then, build the __init__ method of you class. Provide the inputs that are not model parameters or value input. In our example it is the instantiated MTG object that can be imported from file or provided by a growth model if coupled.
- The dictionnary of properties should be stored in the self.props attribute. If you use MTG, it means :
```
    def __init__(self, g, **scenario):
        self.g = g
        self.props = self.g.properties()
```
- Then add data to your model inherited choregrapher with filters depending on which elements this model should focus (example exclude null mass or dead elements)
```
        self.choregrapher.add_data(instance=self, data_name="props", filter={"property":["positive_filter"]})
```
- Before any other operation, we apply the provided scenario by changing default parameters and initialization
```
        self.apply_scenario(**scenario)
```
- Finish the init by binding mtg interest properties to your class instance, self : 
```
        self.link_self_to_mtg()
```

- Finally implement your model processes in individual methods and decorate them with available decorators. For example :

```
    @rate
    def hexose_exudation(self, hexose_root, hexose_soil, root_surface):
        return self.parameter * (hexose_root - hexose_soil) * root_surface

    @state
    def hexose(self, hexose, hexose_exudation):
        return hexose + (self.time_step / struct_mass) * 
            (-hexose_exudation 
             + ...
            )
```

- Choregrapher will then call method groups according to the consensus scheduling table presented above.

### For model wrapping:

- Create a python file with the name of your wrapp (the published name of the model)
- Import your model class and the composite_wrapper utility :
```
from your_package.your_model import YourModel
from composite_wrapper import CompositeModel
```

- Wrapp your model in the following format :
```
class WrappedModel(CompositeModel):
    def__init__(self, g, **scenario):
        self.your_model = self.load(YourModel, g, **scenario)
    def run(self):
        self.your_model()
```

Two commentaries here : 
First, you need to use the load function here as your model execution depends on the Choregrapher class, which is a singleton. However you don't want this single Choregrapher instance to be shared among all instances of your model (imagine two independant plants).
Second commentary is the use of self.your_model() : this refers to the __call__ method you inherited during model design from the "Model" class. This simply calls choregrapher execution.

- If you intend to couple several models after importing them, two steps have to be added :
```
import your_wrap_package
from your_model_packages import FirstModel, SecondModel

class WrappedModel(CompositeModel):
    def__init__(self, g, **scenario):
        self.model_1 = self.load(FirstModel, g, **scenario)
        self.model_2 = self.load(SecondModel, g, **scenario)
        
        # Store the list of instances
        self.models = (self.model_1, self.model_2)

        # LINKING MODULES
        self.link_around_mtg(translator_path=your_wrap_package.__path__[0])

        # Some initialization must be performed AFTER linking modules
        [m.post_coupling_init() for m in self.models]

    def run(self):
        # Run the models sequence according to your scheduling hypotheses
        self.model_1()
        self.model_2()
```     
2 comments : 
- link_around_mtg(translator_path) searches for the "coupling_translator.yaml" configuration file that explicits which state variables from a source model can be considered as input for a receiver model. If it doesn't exist, you will be guided through a step-by-step guide to build this coupling transltor. It is based on which state variables are flaged as input and state_variable in the model file. Usually the coupling_translator.yaml file is already provided by the modeller.
- post_coupling_init() is a method inherited from genericmodel.component.Model that creates a dynamic pointer to mtg properties in the model self instance. Thus each time a property is modified as self.hexose[vertex_id] = 0., it is also modified in mtg.properties(). Don't hesitate to superimpose this method if you need additionnal operations after models initialization and coupling.

Note : If a growth model is included, you need to define a "post_growth_updating()" method in each model to be called after the growth so that the length of the properties managed by each non-growth model matches the actualized number of elements actualized by the growth model.



### For model use




For a more practical example, please see Root_CyNAPS package : https://github.com/GeraultTr/Root_CyNAPS

## Contributing

We are open to contributions on the develop branch of this package.

## Authors and acknowledgment

Gerault T., Rees F., Farra S., Barillot R., Pradal C.

## License

