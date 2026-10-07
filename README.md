# Numerical results: finite-horizon DP for hydro reservoirs

Implementation of the algorithms in *Finite Horizon Dynamic Programming for
Hydro Reservoirs* (C. Won, M2 Optimization, IP Paris), following
Carpentier, Chancelier, Leclère and Pacaud, *Stochastic decomposition applied
to large-scale hydro valleys management*, EJOR 270(3):1086–1098, 2018.

```
hydro.py         Algorithms 1 to 3 of the report, vectorised
instances.py     the 2, 4 and 6 dam valleys
tests.py         regression tests against the report's hand computations
experiments.py   E1 to E5
run_all.py       driver
analysis.py      diagnostics used while writing Section IV
```

The paper itself is in this repository as `paper.pdf`. Equation and section
numbers in the comments, such as (23) or Section III-C1, refer to it, and the
table below says which experiment produced which of its tables and figures.

Requires `numpy`, and `matplotlib` for the figures.

## How to run

```bash
cd Numerical_results
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib

python run_all.py --calibrate      # measure this machine, predict every run
python run_all.py --tests          # must pass before anything else
python run_all.py --all --heavy    # everything reported in the paper
```

`--all --heavy` is what produced the published numbers, and takes about four
and a half hours on the machine below. Without `--heavy` the largest six-dam
run is skipped.

| experiment | output | where it appears in the paper |
|---|---|---|
| E1 | `results/e1_interpolation_vs_rounding.csv` | Tables II and III |
| E2 | `results/e2a_fixed_resolution.csv`, `e2b_finest_affordable.csv` | Table IV |
| E3 | `results/e3_estimator.csv` | Table V |
| E5 | `results/e5_dimension_at_fixed_resolution.csv` | Table VI |
| E4 | `figures/e4b_trajectories.pdf`, `e4c_payoff_distribution.pdf` | Figures 2 and 3 |

`figures/e4a_value_function.pdf` is produced but not used in the paper.

All cores are used by default. `--workers N` sets the number of processes and
`--workers 1` runs serially; `HYDRO_WORKERS` does the same through the
environment. Backward passes are cached in `cache/`, so re-running is cheap.
Delete `cache/` after changing anything in `instances.py`: the cache key is
only `(d, n, m, rule)`.

To push the experiments further, edit the configuration block at the top of
`experiments.py`. Cost scales as `(2nm)^d`, so doubling both `n` and `m` costs
a factor of `4^d`, which at `d = 6` is 4096.

## Correctness

| test | what it checks |
|---|---|
| `test_worked_example` | Section III-A: at `x = (50,25)`, wet noise, `u = (20,40)`, `t = T-1`, `Eval` returns `-46.25`, the target point is `(60,20)`, and the admissible set at the empty state under the dry noise is the single control `(0,0)` |
| `test_one_dam` | the one-dam study: `V2 = (25,0,0)`, `V1 = (10,-30,-40)`, `V0 = (-8,-54,-76)`, `J = -62` by exhaustive enumeration of all `L^T` scenarios |
| `test_interpolation_identities` | equations (22) and (23): the weights sum to one and reproduce the target point in expectation |
| `test_parallel_matches_serial` | `backward(workers>1)` reproduces `backward(workers=1)` bit for bit, at `d = 2, 4, 6` and under both rules |
| `test_simulate_blocking` | `simulate(chunk=c)` reproduces `simulate(chunk=N)` bit for bit, and every simulated volume stays inside `[0, xmax]` to within `1e-6` |

## The instance

Carpentier et al. publish no instance data for their academic valleys. Their
whole specification is that the dams have roughly equal maximal volume, that
turbine capacity grows and inflow falls with depth, that the inflow laws are
discrete with finite support, and that prices are deterministic. Only the
topologies are reproduced, from their Fig. 4:

| | `P(i)` |
|---|---|
| 2-dam | `-, {1}` |
| 4-dam | `-, {1}, {2}, {3}` |
| 6-dam | `-, {1}, -, {2,3}, {4}, {5}` |

Every number is ours, and each is traceable to one of their sentences:

| quantity | value | source |
|---|---|---|
| `T` | 12 monthly steps, one year | theirs |
| `xmax` | 100, all dams | "more or less the same maximal volume" |
| local mean inflow | 8 at a source down to 3 at the bottom, linear in depth | "more inflow for an upstream dam" |
| `umax` | `round(2 x C)`, `C` the mean inflow of the dam's catchment | "more capacity for a downstream dam" |
| inflow law | `L = 5` outcomes, outcome `l` scaling every dam's mean inflow by `nu_l`, `nu = (0.4, 0.7, 1.0, 1.4, 2.0)`; the probabilities change with the month | "discrete laws with finite support" |
| prices | deterministic, 1.3 at a source falling to 0.7 at the bottom | "deterministic market prices" |
| `eps` | `1e-3` | operating cost of the turbine |
| `xhat`, `x0` | 50, half full | ours |
| `alpha` | 0.1 | ours |

One multiplier scales all dams at once, so the whole valley is wet or dry
together. Independent outcomes per dam would need `L = 5^d` elements, 15,625 at
`d = 6`, multiplying every runtime by 3125: `L` enters the cost linearly, so
joint outcomes are cheap and independent ones are not.

**Why `umax` follows the catchment and not the depth.** A dam must pass the
accumulated flow of everything above it. A naive ramp (20 upstream to 40
downstream) leaves the six-dam valley receiving 71.5 units a month at the
bottom and able to discharge 40, so it spills about 44% of all water whatever
the policy does. The multiplier was set by measurement. At `d = 4`, the share
of releases sitting exactly at the cap is:

| multiplier | 1.0 | 1.2 | 1.4 | 1.6 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|
| saturated | 97.6% | 89.8% | 75.7% | 56.9% | 18.1% | 1.0% |

At 1.0 the policy is trivial, at 3.0 the cap never binds. 2.0 is the smallest
value at which it binds in a minority of decisions while still binding.

**Spill is rare here, and that is not avoidable.** With all capacities at 100
and a one-year horizon, spill becomes active only once a reservoir holds less
than about a month of catchment flow, at which point it can no longer carry
water between seasons. At `d = 4`, a reservoir of 4.1 months of flow spills in
0.0% of (month, scenario, dam) triples, 2.2 months 0.0%, 1.3 months 0.1%, 0.8
months 0.8%. The `max` of (2) is therefore dormant in the reported runs; it is
exercised by `test_one_dam`, where the reservoir does overflow.

**Seasonality is carried by the probabilities.** Section III-A3 requires the
support `W` to be the same every month, with only `rho_{t,l}` varying, so the
five inflow vectors are fixed and `rho` interpolates between a dry and a wet
law over the year, peaking in May. For the same reason prices are constant in
`t`: a seasonal price would make `W` depend on `t`. The profile itself is
arbitrary and no conclusion depends on its shape.

## Parallelism

The entries `(V_t)_k` are computed independently: at fixed `t`, every call to
`Eval` reads the same array `V_{t+1}` and writes one entry of `V_t`. That loop
is embarrassingly parallel and `backward_parallel` splits it across processes.
NumPy does not help by itself, since it multi-threads only inside BLAS and
LAPACK and this computation is elementwise arithmetic, gathers and reductions.

`V_{t+1}` is the only shared quantity. It is placed in a shared memory segment
the workers map once rather than pickled to every task, with `Pool.map`
providing the synchronisation; everything else a worker needs it rebuilds at
start-up. Two limits are worth knowing: memory multiplies by the worker count
(`HYDRO_MEM_MB`, default 256, caps the working set per worker), and a constant
factor does not beat an exponential, since an 8x speedup at `d = 6` lets `nm`
grow only by `8^(1/6) = 1.41`.

## Implementation notes

Points that belong to the code rather than to the mathematics, and were kept
out of the report.

**Minimisation over `U` rather than `C_t(x,w)`.** `Eval` returns `+inf` at
inadmissible candidates, so the minimum over the control grid equals the
minimum over the admissible set. This is in the report, equation (54).

**The cell index is clipped from below.** Equation (16) needs no truncation
from below, because an admissible control guarantees `y >= xlow`. The
vectorised form evaluates every candidate before masking the inadmissible
ones, so for an inadmissible candidate the floor can be negative and would
read entries of `V` belonging to no cell. In floating point an admissible
candidate at exactly its bound can also land a hair below `xlow`, and that case
is not removed by the masking. Clipping the index into `[0, n-2]` handles both
and leaves admissible candidates untouched.

**The admissibility tolerance.** Equation (5) asks for
`u^i <= x^i + q^i + z^i`; the code tests `u^i <= avail + 1e-9`. `avail` is
built from a grid point, an inflow, and upstream releases and spills that are
themselves differences and maxima, while `u^i` comes from a different
`linspace`, so when the constraint holds with equality the two sides can land a
few bits apart and a legal release would be dropped. The slack is eleven orders
of magnitude below the volumes. It also admits releases infeasible by up to a
billionth, so the cost was measured rather than argued, by running the same
code with the tolerance at zero:

| | candidate pairs | admitted only by the tolerance | share | largest change in `V` |
|---|---|---|---|---|
| `d=2, n=41` | 840,500 | 8 | 0.001% | 0 |
| `d=2, n=161` | 12,960,500 | 27 | 0.000% | 0 |
| `d=4, n=5` | 253,125 | 84 | 0.033% | 0 |

None of the admitted candidates is ever the minimiser, so removing the
tolerance reproduces every `V_t` bit for bit. It is kept as cheap insurance
against a configuration in which "release everything" does become attractive.

**Algorithm 3 does not clip the stored state into `[0, xmax]`.** A clip would
repair a state that had left the feasible set and leave no trace;
`test_simulate_blocking` asserts the bound instead, so a future instance that
leaves it is reported rather than quietly patched.

**Blocking.** Arrays indexed by (grid point, candidate, dam) hold
`n^d x m^d x d` entries, beyond memory for all but the smallest valleys.
`backward` partitions the grid into blocks of at most `B` points. Entries for
different `k` are independent, so the result does not depend on the partition;
`B` trades peak memory against the number of passes.

**Tie-breaking.** `np.argmin` returns the first minimiser, which is the rule
fixed in Section III-C1: smallest index in the enumeration of `U`.

**Common random numbers.** `draw_scenarios` is seeded, so every discretisation
is evaluated on the same `N` scenarios and differences between rows are
differences between policies, not sampling noise.

**A self-test not in the paper.** Replacing the exact state update
`X <- f_t(X,U,w)` in Algorithm 3 by a draw of the vertex `G(f_t(X,U,w), xi)`
simulates the discretized process instead. Its estimator converges to
`vbar_0(x_0)`, which Algorithm 2 computes exactly, so a discrepancy exceeding
the confidence interval (60) indicates a bug in the backward recursion.

## Machine

Apple M4 Pro, 14 cores (10 performance and 4 efficiency), 24 GB, macOS 26.6.2,
Python 3.9.6, NumPy 2.0.2. Carpentier et al. report their timings on a 3.4 GHz
four-core Intel Xeon E3, so absolute times are not comparable between the two
papers; the growth in the number of dams is.
