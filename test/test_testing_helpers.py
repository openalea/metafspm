"""metafspm.testing's comparison helpers and metafspm.corrections."""
import subprocess

import numpy as np
import pytest

from openalea.metafspm.corrections import Corrections
from openalea.metafspm.testing import (array_differences, assert_arrays_equal, assert_texts_equal, run_executables,
                                       text_differences, worktree)


def test_texts_and_arrays():
    assert text_differences("a\nb\nc", "a\nB\nc\nd") == [(2, "b", "B"), (4, None, "d")]
    assert_texts_equal("a\nb", "a\nb")
    with pytest.raises(AssertionError, match="row 2"):
        assert_texts_equal("a\nb", "a\nc")
    assert array_differences([1., np.nan, 3.], [1., np.nan, 3.]) == {}
    found = array_differences([1., 2., 3.], [1., 2.5, 3.1])
    assert found["count"] == 2 and found["first"] == (1,) and found["largest"] == pytest.approx(0.5)
    assert array_differences([1., 2.], [1., 2. + 1e-9], rtol=1e-6) == {}
    with pytest.raises(AssertionError, match="1 entries differ"):
        assert_arrays_equal(np.arange(3), [0, 1, 5])


def test_run_executables(tmp_path):
    out = run_executables([("cp", ["in.txt", "out.txt"])], {"in.txt": "hello\n"}, outputs=("out.txt", "none.txt"))
    assert out == {"out.txt": "hello\n"}


def test_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for command in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(repo), *command], check=True)
    (repo / "f.txt").write_text("one")
    subprocess.run(["git", "-C", str(repo), "add", "f.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "one"], check=True)
    (repo / "f.txt").write_text("two")                              # uncommitted: left as it is
    with worktree(repo, "HEAD") as path:
        assert open(f"{path}/f.txt").read() == "one"
    assert (repo / "f.txt").read_text() == "two"


def test_corrections():
    corrections = Corrections({"a": "first", "b": "second"}, presets={"only_a": ("a",)})
    assert corrections.resolve("recommended") == {"a", "b"} and corrections.resolve(None) == frozenset()
    assert corrections.resolve("legacy") == frozenset() and corrections.resolve("only_a") == {"a"}
    assert corrections.resolve("b") == {"b"} and corrections.resolve(["a"]) == {"a"}
    with pytest.raises(ValueError):
        corrections.resolve(["c"])
    with pytest.raises(ValueError):
        corrections.resolve("recommended", precision="legacy")
    assert corrections.table()["presets"].tolist() == ["recommended, only_a", "recommended"]
    assert "``a``" in corrections.rst()


def test_run_in_env():
    import os
    import shutil
    import sys
    from openalea.metafspm.testing import run_in_env
    import json
    import subprocess
    conda = os.environ.get("CONDA_EXE", "conda")
    if shutil.which(conda) is None:
        pytest.skip("conda is not available")
    listed = json.loads(subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True,
                                       check=True).stdout)["envs"]
    if os.path.realpath(sys.prefix) not in {os.path.realpath(prefix) for prefix in listed}:
        pytest.skip("this environment is not one conda lists (conda-build's test environment)")
    assert run_in_env(os.path.basename(sys.prefix), "print(6 * 7)").strip() == "42"
