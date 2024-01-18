from generic_fspm.composite_wrapper import CompositeModel


class Model(CompositeModel):
    def __init__(self) -> None:
        super().__init__()

def test_import_composite():
    composite_model = Model()
