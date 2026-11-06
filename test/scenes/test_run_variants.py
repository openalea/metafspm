"""
scenarios.run_variants: one scene per variant in a process pool, the results in the variants' order and equal to
the runs in this process; the processes bounded (variant_processes: a scene's "auto" pool replaced by 1 when
several variants run at once, an oversubscription refused); a failing variant raises with its name.
"""
import os

import numpy as np
import pytest

from openalea.metafspm.compute import Compute
from openalea.metafspm.scene import scenarios as scenarios_module
from openalea.metafspm.scene.scenarios import run_variants, variant_processes

from variant_runs import exudation_run, failing_run, pool_run

VARIANTS = {f"rate_{rate}": {"exudation_rate": rate} for rate in (0.05, 0.1, 0.2)}


def test_the_variants_in_processes_equal_the_runs_here():
    here = run_variants(exudation_run, VARIANTS, workers=1)
    pooled = run_variants(exudation_run, VARIANTS, workers=2, compute=Compute(workers="auto"))
    assert list(pooled) == list(VARIANTS)
    assert {pid for _, pid, _, _ in here.values()} == {os.getpid()}
    assert os.getpid() not in {pid for _, pid, _, _ in pooled.values()}
    assert len({pid for _, pid, _, _ in pooled.values()}) == 3          # a fresh process per variant
    for name in VARIANTS:
        assert pooled[name][2]["workers"] == 1                        # "auto" within a variant: 1
        for one, other in zip(here[name][3], pooled[name][3]):
            np.testing.assert_array_equal(one, other)
    assert not np.array_equal(here["rate_0.05"][3][0], here["rate_0.2"][3][0])


def test_a_spawned_pool():
    pooled = run_variants(exudation_run, dict(list(VARIANTS.items())[:2]), workers=2, start_method="spawn")
    assert list(pooled) == list(VARIANTS)[:2]


def test_a_variants_scene_may_have_its_own_pool():
    pooled = run_variants(pool_run, {"a": {}, "b": {}}, workers=2, compute=Compute(workers=2))
    assert all(info["workers"] == 2 for info in pooled.values())


def test_the_processes_are_bounded(monkeypatch):
    monkeypatch.setattr(scenarios_module.os, "cpu_count", lambda: 8)
    assert variant_processes("auto", variants=20)[0] == 7
    assert variant_processes("auto", variants=3)[0] == 3
    processes, compute = variant_processes(4, Compute(workers="auto"))
    assert processes == 4 and compute.workers == 1
    assert variant_processes(1, Compute(workers="auto"))[1].workers == "auto"    # one at a time: its own pool
    assert variant_processes(2, Compute(workers=4))[1].workers == 4
    with pytest.raises(ValueError, match="oversubscribe"):
        variant_processes(3, Compute(workers=4))
    with pytest.raises(ValueError, match="oversubscribe"):
        variant_processes(4, Compute(threads=4))
    with pytest.raises(ValueError):
        variant_processes(0)


def test_a_failing_variant_raises_with_its_name():
    with pytest.raises(RuntimeError, match="'b' failed"):
        run_variants(failing_run, {"a": {}, "b": {"fail": True}, "c": {}}, workers=2)
