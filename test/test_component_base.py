from utils import deep_reload_package
deep_reload_package("openalea")
from openalea.metafspm.component import Model


class TestComponent:
    def test_bare_component(self):
        model = Model()
