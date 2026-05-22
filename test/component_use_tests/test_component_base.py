from utils import deep_reload_package
deep_reload_package("openalea")
from openalea.metafspm.coupling.component import Component


class TestComponent:
    def test_bare_component(self):
        model = Component()
