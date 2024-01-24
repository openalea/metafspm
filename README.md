# Generic_FSPM

## Purpose 

This package is intended to provide minimal tools to assist Functionnal Structural Plant Modellers (FSPM) so that they can make their production more readable regarding scientific content. It is also intended as a minimal constraint in object oriented programming to ease the model coupling in OpenAlea around a Multiscale Tree Graph (MTG). However the idea is to remain very generic in order to accept a wide variety of model types. You can't impose formalism on others, that's how great civilizations die. Our aim here is rather to favor research projects among different scientific communities.

## Main principles

## Installation

- Intall the lastest version of conda : https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html 

- Create an environment dedicated to your model
```
conda create -n your_model python==3.10

```
Then :
### First option : install from the requirements.txt
- Make sure you have git installed : https://git-scm.com/downloads

- From terminal, clone this package and then , then run the command :
```
git clone https://github.com/GeraultTr/genericmodel.git
```

- Then, cd into the directory and install necessary packages with the following requirements files : 
```
```

- Finally, run the setup.py :
```
python -m setup develop
```

### Second option TODO : when package is released, just create a new environment :
```
conda install -c conda-forge genericfspm
```

## Example use

### For model design

- First, the model has to be packaged as a class and decorated by @dataclass from the dataclasses module.

- Then, you must import utilities :  
```
from genericfspm.component import Model, declare
from genericfspm.component_factory import *
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


- Finally implement your model processes in individual methods

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



### For model use




For a more practical example, please see Root_CyNAPS package : https://github.com/GeraultTr/Root_CyNAPS

## Contributing

We are open to contributions on the develop branch of this package.

## Authors and acknowledgment

Gerault T., Rees F., Farra S., Barillot R., Pradal C.

## License

