# Running on a GPU

metafspm runs on the CPU by default, and every model runs without a GPU. A GPU is optional: `Compute(device="cuda")`
moves the scene's heaviest linear algebra to it. This page says what moves, when it pays, how to install CuPy and
how to check that it works.

**The GPU path is experimental: it should work on suitable hardware but has not been tested on a GPU yet.** Its
tests run on numba's CUDA simulator and on a fake CuPy. On a GPU, results agree with the CPU's within round-off, not
bit for bit: the device sums in another order, and its compiler may fuse multiplies and adds.

```python
from openalea.metafspm import Compute, Scene

scene = Scene(planting, ..., compute=Compute(device="cuda"))
```

## What runs on the GPU

`device="cuda"` hands the arrays to one small backend (`openalea.metafspm.backend`). There are two backends:
numpy and scipy (`"cpu"`) and CuPy (`"cuda"`). The backend takes the following work:

- **A reduction's kinds** (their Kron reductions, `KronReduction`). Each kind solves its interior once per kept node
  (X = A_II⁻¹ A_IK, a few dozen right-hand sides over tens of thousands of cells), and this is where the evaluation
  spends its time. (This is the factorisation of the reduced system's backend, `LinearSolver.factorize`: SuperLU's
  on the device's backend, under `"cupy"` and `"superlu"` alike; a symmetric system mapped to `"cholmod"` is
  factorised by CHOLMOD on the host.)
  - The factorisation is SuperLU's, on the host, the same one the CPU uses (cupyx's own `splu` also factorises on
    the host).
  - Its L and U factors are copied to the device once, and the right-hand sides are solved there.
  - A_IK goes to the device sparse and is made dense there; X comes back once, because the kind keeps it on the
    host for its injection, its volume shares and its display recovery.
  - A right-hand side smaller than `backend.DEVICE_MIN_VALUES` (200 000 values) stays on the host: copying it would
    cost more than solving it.
- **The graph systems' linear solves**, with `linear_solver="cupy"` (the default when `device="cuda"`):
  - A system of at least `backend.DEVICE_MIN_NONZEROS` (100 000) entries is solved on the device as a whole. The
    independent blocks that the CPU solves one by one (the plants of a population) go together in one call: one
    batch, with one copy each way.
  - The solver is cuDSS when `nvmath-python` is installed (the factorisation also runs on the GPU), and cupyx's
    `spsolve` otherwise.
  - The matrix's pattern (`indptr`, `indices`) is copied once and kept while it does not change. Each solve copies
    only the values and the right-hand side.
  - Smaller systems are solved by SuperLU on the host.
- **Caribu's near field.** The components' setting named `device` defaults to the Compute's, so Caribu's radiosity
  near field runs on the GPU too, through its own numba.cuda kernels. A scenario can still give Caribu
  `device="cpu"`.

The rest stays on the CPU, as before. This includes the steps, the Newton loops, the tree kernels, the exchanges,
`condense=` (`condensed_solve`) and MECHA's equivalent conductances.

The only crossings between host and device are `to_device` and `to_host`. Each one is counted (factor copies
included), and `run.json` records them:

```json
"compute": {"device": {"name": "cuda", "solver": "cupyx", "transfers": {"to_device": 78, "to_host": 21,
                                                                         "bytes_to_device": 262620276,
                                                                         "bytes_to_host": 616061888}}}
```

Nothing imports CuPy unless the device is `"cuda"`. A scene asking for `"cuda"` raises when it is built if CuPy is
missing (an `ImportError` with the install line) or if no CUDA device answers (a `RuntimeError`). `run_variants`'
processes are then spawned rather than forked (CUDA does not survive a fork), and each one holds its own device
context.

## When it pays

- **A single run on a laptop: the CPU.** Use case 1's work is many small sparse solves (a kind takes about 0.16 s on
  one core). Copies and kernel launches cost about as much as they save, and the reduction's kinds saturate after a
  few days (see the framework guide, "Performance notes").
- **Repeated runs (sensitivity analyses, calibrations): several CPU processes first.** `scenarios.run_variants` runs
  one scene per variant in parallel processes. Its gain is the number of cores, with no new dependency:

  ```python
  from openalea.metafspm import Compute, run_variants, variants

  scenarios = variants(base, {"EL": None, "IPD": None}, design="sampling", samples=40, seed=1, declared=[RootGrowth])
  results = run_variants(run_one, scenarios, processes="auto", compute=Compute())
  ```

  `run_one(name, scenario, compute)` is a module-level function. It builds its scene with `compute=compute`, runs
  it, finishes it and returns what is kept (a table, a path). Each variant runs in a fresh process, and the results
  come back in the scenarios' order.
- **Large scenes or many plants on a GPU workstation.** Batches of large kinds (stems and blades of tens of
  thousands of cells each, at the start of a run or when organs emerge) and whole-population systems above 100 000
  entries are what the device takes. Measure it before relying on it (below).

### Recommended setups for sensitivity analyses

| machine | setup | why |
|---|---|---|
| laptop (4–16 cores) | `run_variants(run_one, scenarios, processes="auto", compute=Compute())` | one variant per core, processors − 1 at most; each scene's threads `processors // processes` |
| workstation, many cores | `processes=k`, `compute=Compute(threads=w)` with k × w ≤ processors | a few large scenes, each with w threads |
| GPU workstation | `run_variants(..., processes=1 or 2, compute=Compute(device="cuda"))` | the device shared by few processes (each has its own context and memory); the other cores free for Caribu's threads |

`variant_processes` checks the bound before anything starts:
- several variant processes turn a scene's `threads="auto"` into `processors // processes` (the processors shared);
- an integer is kept if `processes × Compute.threads ≤ processors` (numba's kernels, the solves' blocks and the
  kinds);
- anything else raises "oversubscribe".

## Installing CuPy

CuPy matching the machine's CUDA (`nvidia-smi` shows the driver's CUDA version):

```bash
pip install cupy-cuda12x            # CUDA 12 (cupy-cuda11x for CUDA 11)
# or
conda install -c conda-forge cupy
# optional: cuDSS, the GPU's sparse direct solver (its factorisation on the device too)
pip install "nvmath-python[cu12]"
```

Caribu's GPU near field needs numba's CUDA support (`numba.cuda.is_available()`; the `cuda-nvcc` and
`cuda-nvrtc` conda packages, or the CUDA toolkit).

## Checking it

1. **The device answers:**
   `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), cupy.cuda.Device(0).attributes['MultiProcessorCount'])"`.
2. **The backend's tests:** `pytest test/graph_systems/test_device_backend.py`. Without a GPU these tests run on a
   fake CuPy (`test/helpers/fake_cupy.py`: numpy under cupy's interface). Its device arrays are refused by numpy,
   and every crossing is logged. The tests cover:
   - the dispatch;
   - the transfers (the factors copied once, a pattern once, only the values after);
   - results equal to the CPU path (the fake's numpy and the same SuperLU: bit for bit; a real GPU agrees within
     round-off);
   - the errors.
3. **Use case 1 on the device, against the CPU:**

   ```bash
   cd examples/use_case_1
   python benchmark.py --plants 2 --days 3 --label cpu_2p3d
   python benchmark.py --plants 2 --days 3 --device cuda --label cuda_2p3d
   ```

   Both entries go to `outputs/benchmarks/<commit>.json`. Compare each step's `results`: with cupyx, the kinds
   should agree to round-off, because the triangular solves on the device sum in another order. Compare also the
   step times and the `device` entry (its transfers). `--linear-solver superlu` keeps the linear solves on the
   host and moves only the kinds. Run it under a memory bound, one run at a time
   (`systemd-run --user --scope -p MemoryMax=8G ...`).

None of the CuPy tests run in CI, and the GPU path has not been run on a GPU yet: it is experimental.
