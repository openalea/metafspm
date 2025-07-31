import pandas as pd

# Edited models
from dummy_components import Carbon, Nitrogen, SoilModel
from openalea.mtg import MTG

# Utilities
from openalea.metafspm.composite_wrapper import CompositeModel
from openalea.metafspm.component_factory import Choregrapher


class Model(CompositeModel):
    """
    Root-BRIDGES model

    Use guideline :
    1. store in a variable Model(g, time_step) to initialize the model, g being an openalea.MTG() object and time_step an time interval in seconds.

    2. print Model.documentation for more information about editable model parameters (optional).

    3. Use Model.scenario(**dict) to pass a set of scenario-specific parameters to the model (optional).

    4. Use Model.run() in a for loop to perform the computations of a time step on the passed MTG File
    """

    def __init__(self, time_step: int, translator_path, **scenario):
        """
        DESCRIPTION
        ----------
        __init__ method of the model. Initializes the thematic modules and link them.

        :param g: the openalea.MTG() instance that will be worked on. It must be representative of a root architecture.
        :param time_step: the resolution time_step of the model in seconds.
        """

        # DECLARE GLOBAL SIMULATION TIME STEP
        Choregrapher().add_simulation_time_step(time_step)
        self.time = 0

        parameters = scenario["parameters"]
        self.input_tables = scenario["input_tables"]

        # Should have been created by an architecture model init here
        g_properties = {"struct_mass": {1:0.001}, "type": {1: "Normal_root_after_emergence"}, "label": {1: "Apex"}}
        g = MTG()
        props = g.properties()
        props.update(g_properties)

        self.carbon = Carbon(g_properties=props, time_step=time_step, **parameters)
        self.nitrogen = Nitrogen(g_properties=props, time_step=time_step, **parameters)
        self.soil = SoilModel(g_properties=props, time_step=time_step, **parameters)


        # LINKING MODULES
        self.declare_data_and_couple_components(root=g,
                                                translator_path=translator_path,
                                                components=(self.carbon, self.nitrogen, self.soil))


    def run(self):
        self.apply_input_tables(tables=self.input_tables, to=self.components, when=self.time)

        self.carbon()
        self.nitrogen()
        self.soil()

        self.time += 1


def test_dummy_model_assembly():
    input_table = pd.read_csv("inputs/dummy_temperatures.csv", index_col="t")
    scenario = {"parameters": {},
                "input_tables": {"temperature": input_table["temperature"]}}
    model = Model(time_step=3600, translator_path="inputs", **scenario)

    model.run()

    assert model.soil.props["DOC"][1] != 0