"""
Driver for the numerical results of the report.

    python run_all.py --calibrate      measure this machine's throughput and
                                       print the predicted time of every run
    python run_all.py --tests          the regression tests against the report
    python run_all.py --e1 --e2 ...    one or more experiments
    python run_all.py --all            tests, then E1 to E5
    python run_all.py --all --heavy    the same, with the largest d = 6 run
                                       added (hours; intended to be left
                                       running overnight)

Results go to results/*.csv and figures/*.pdf.  Backward passes are cached in
cache/, so re-running is cheap; delete cache/ to force a recomputation.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time

import numpy as np

import experiments as X
from hydro import (PARALLEL_MIN_WORK, Discretisation, backward, draw_scenarios,
                   simulate, work_units)
from instances import valley


def machine() -> str:
    return (f"{platform.system()} {platform.release()} / {platform.machine()}"
            f" / Python {platform.python_version()} / NumPy {np.__version__}")


def sim_units(d, m, N, T) -> float:
    """N T m^d d, a proxy for the cost of Algorithm 3.

    Algorithm 3 evaluates every candidate at every scenario at every step, and
    holds (scenarios in one noise group) x m^d x d arrays while it does so.
    The noise index does not enter: the L groups partition the N scenarios.
    """
    return float(N) * T * (m ** d) * d


# Probe sizes, per dimension: one small run to time a single process, one
# larger run to time `workers` of them, and a short simulation on the second.
# Throughput is NOT the same at every d: the 2^d gathers of (45) grow faster
# than the arithmetic they feed, so a rate measured at d = 2 and applied to
# d = 6 is optimistic by a factor of two or more.  Each d is measured.
PROBES = {2: {"serial": (41, 10), "par": (161, 10), "simN": 2000},
          4: {"serial": (5, 3), "par": (6, 6), "simN": 300},
          6: {"serial": (3, 2), "par": (3, 3), "simN": 300}}


def _time(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def calibrate(verbose: bool = True, workers: int | None = None):
    """Measure this machine at each d, then predict every configured run."""
    if workers is None:
        workers = X.WORKERS
    ser, par, srate = {}, {}, {}

    if verbose:
        print(f"machine   {machine()}")
        print(f"cores     {os.cpu_count()} reported, using {workers} worker(s)")
        print("\nthroughput, measured separately at each d")
        print(f"  {'d':>2} | {'1 process':>12} | {'%d processes' % workers:>13}"
              f" {'speedup':>8} | {'Algorithm 3':>13}")

    for d, p in PROBES.items():
        inst = valley(d)
        n, m = p["serial"]
        ds = Discretisation(inst, n=n, m=m)
        backward(ds, workers=1)                                   # warm up
        ser[d] = (work_units(d, n, m, inst.L, inst.T)
                  / _time(lambda: backward(ds, workers=1)))

        n2, m2 = p["par"]
        dp = Discretisation(inst, n=n2, m=m2)
        Vp = [None]
        s = _time(lambda: Vp.__setitem__(
            0, backward(dp, workers=workers, min_work=0)))
        par[d] = work_units(d, n2, m2, inst.L, inst.T) / s

        scen = draw_scenarios(inst, p["simN"], seed=1)
        srate[d] = (sim_units(d, m2, p["simN"], inst.T)
                    / _time(lambda: simulate(dp, Vp[0], scen)))
        if verbose:
            print(f"  {d:>2} | {ser[d]:>10.2e}/s | {par[d]:>11.2e}/s "
                  f"{par[d]/ser[d]:>7.1f}x | {srate[d]:>11.2e}/s")

    if verbose:
        print("  Throughput falls with d, so a rate measured at d = 2 and")
        print("  applied to d = 6 understates the cost by a factor of two or")
        print("  more.  Each row below uses the rate of its own d, and the")
        print("  single-process rate wherever the run is actually serial:")
        print(f"  E2a is, and so is any pass below {PARALLEL_MIN_WORK:.0e}"
              f" work units.\n")

    def secs_b(d, n, m, force_serial=False):
        inst = valley(d)
        w = work_units(d, n, m, inst.L, inst.T)
        parallel = workers > 1 and w >= PARALLEL_MIN_WORK and not force_serial
        return w / (par[d] if parallel else ser[d])

    if not verbose:
        return par

    # backward passes, deduplicated: E1 and E3 share many grids
    print("predicted time of each BACKWARD pass:")
    seen, total_b, cache_mb = set(), 0.0, 0.0
    rows = [("E1", X.E1_D, n, X.E1_M, rule, False)
            for n in X.E1_NS + [X.E1_NREF] for rule in (0, 1)
            if not (n == X.E1_NREF and rule)]
    rows += [("E2a", d, n, m, 0, True) for (d, n, m) in X.E2A]
    rows += [("E2b", d, n, m, 0, False) for (d, n, m) in X.e2b_rows()]
    rows += [("E3", d, n, m, 0, False) for (d, n, m, _) in X.e3_rows()]
    rows += [("E5", d, n, m, 0, False) for (d, n, m) in X.E5]
    for tag, d, n, m, rule, serial in rows:
        key = (d, n, m, rule)
        if key in seen and tag not in ("E2a", "E2b"):
            continue          # E2a and E2b force a recomputation (use_cache)
        seen.add(key)
        s = secs_b(d, n, m, serial)
        total_b += s
        cache_mb += (valley(d).T + 1) * (n ** d) * 8 / 1e6
        if s > 1.0:
            print(f"  {tag:>4}  d={d}  n={n:>5}  m={m:>3}"
                  f"{'  (rounding)' if rule else '':>12}"
                  f"{'  1 proc' if serial else '':>8}   {X._fmt_secs(s):>10}")
    print("        passes shared between experiments are computed once;"
          " those under a second are not listed")
    print(f"  backward total   {X._fmt_secs(total_b)}"
          f"      cache grows to about {cache_mb/1e3:.1f} GB uncompressed\n")

    # simulations, at the rate of their own d
    print("predicted time of each SIMULATION:")
    total_s = 0.0
    sims = [("E1", X.E1_D, X.E1_M, X.E1_N_SCEN * (2 * len(X.E1_NS) + 1))]
    sims += [("E3", d, m, N) for (d, n, m, N) in X.e3_rows()]
    sims += [("E4b", X.E4_PATHS[0], X.E4_PATHS[2], X.E4_PATHS[3]),
             ("E4c", X.E4_HIST[0], X.E4_HIST[2], X.E4_HIST[3])]
    sims += [("E5", d, m, X.E5_N) for (d, n, m) in X.E5]
    for tag, d, m, N in sims:
        s = sim_units(d, m, N, valley(d).T) / srate[d]
        total_s += s
        if s > 1.0:
            print(f"  {tag:>4}  d={d}  m={m:>3}  N={N:>9,}   "
                  f"{X._fmt_secs(s):>10}")
    print(f"  simulation total {X._fmt_secs(total_s)}\n")
    print(f"  GRAND TOTAL      {X._fmt_secs(total_b + total_s)}"
          f"   {'(--heavy)' if X.HEAVY else '(without --heavy)'}")
    return par


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--e1", action="store_true")
    ap.add_argument("--e2", action="store_true")
    ap.add_argument("--e3", action="store_true")
    ap.add_argument("--e4", action="store_true")
    ap.add_argument("--e5", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--heavy", action="store_true",
                    help="include the largest d = 6 run (hours)")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker processes (default: all cores)")
    a = ap.parse_args(argv)
    if a.workers:
        X.WORKERS = a.workers
    X.HEAVY = a.heavy
    if not any(v for k, v in vars(a).items() if k not in ("workers", "heavy")):
        ap.print_help()
        return 0

    if a.calibrate:
        calibrate()
        return 0

    print(f"machine   {machine()}")
    print(f"workers   {X.WORKERS}")
    print(f"heavy     {X.HEAVY}\n")

    t0 = time.perf_counter()
    if a.tests or a.all:
        import tests
        tests.test_worked_example()
        tests.test_one_dam()
        tests.test_interpolation_identities()
        tests.test_parallel_matches_serial()
        tests.test_simulate_blocking()

    if a.e1 or a.all:
        X.e1_interpolation_vs_rounding()
    if a.e2 or a.all:
        X.e2_complexity()
    if a.e3 or a.all:
        X.e3_estimator()
    if a.e4 or a.all:
        X.e4_figures()
    if a.e5 or a.all:
        X.e5_dimension_at_fixed_resolution()
    print(f"\ntotal wall clock {X._fmt_secs(time.perf_counter() - t0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
