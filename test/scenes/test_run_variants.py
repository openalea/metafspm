"""
scenarios.run_variants: one scene per variant in processes of their own, the results in the variants' order and
equal to the runs in this process; the processes bounded (variant_processes: a scene's "auto" threads shared when
several variants run at once, an oversubscription refused); a failing variant raises with its name.
"""
import os

import numpy as np
import pytest

from openalea.metafspm.compute import Compute
from openalea.metafspm.scene import scenarios as scenarios_module
from openalea.metafspm.scene.scenarios import run_variants, variant_processes

from variant_runs import exudation_run, failing_run

VARIANTS = {f"rate_{rate}": {"exudation_rate": rate} for rate in (0.05, 0.1, 0.2)}


def test_the_variants_in_processes_equal_the_runs_here():
    here = run_variants(exudation_run, VARIANTS, processes=1)
    in_processes = run_variants(exudation_run, VARIANTS, processes=2, compute=Compute())
    assert list(in_processes) == list(VARIANTS)
    assert {pid for _, pid, _, _ in here.values()} == {os.getpid()}
    assert os.getpid() not in {pid for _, pid, _, _ in in_processes.values()}
    assert len({pid for _, pid, _, _ in in_processes.values()}) == 3          # a fresh process per variant
    for name in VARIANTS:
        assert in_processes[name][2]["threads"] != "auto"                   # "auto" within a variant: shared
        for one, other in zip(here[name][3], in_processes[name][3]):
            np.testing.assert_array_equal(one, other)
    assert not np.array_equal(here["rate_0.05"][3][0], here["rate_0.2"][3][0])


def test_spawned_variant_processes():
    in_processes = run_variants(exudation_run, dict(list(VARIANTS.items())[:2]), processes=2, start_method="spawn")
    assert list(in_processes) == list(VARIANTS)[:2]


def test_the_processes_are_bounded(monkeypatch):
    monkeypatch.setattr(scenarios_module.os, "cpu_count", lambda: 8)
    assert variant_processes("auto", variants=20)[0] == 7
    assert variant_processes("auto", variants=3)[0] == 3
    assert variant_processes(2, Compute(threads=4))[1].threads == 4
    with pytest.raises(ValueError, match="oversubscribe"):
        variant_processes(4, Compute(threads=4))
    assert variant_processes(4, Compute())[1].threads == 2                     # "auto": the processors shared
    assert variant_processes(1, Compute())[1].threads == "auto"
    with pytest.raises(ValueError):
        variant_processes(0)


def test_a_failing_variant_raises_with_its_name():
    with pytest.raises(RuntimeError, match="'b' failed"):
        run_variants(failing_run, {"a": {}, "b": {"fail": True}, "c": {}}, processes=2)
