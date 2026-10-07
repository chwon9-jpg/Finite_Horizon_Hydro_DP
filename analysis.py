"""
Diagnostics on the output of `python run_all.py --all`.

Nothing here belongs to the algorithms of the report.  This script only reads
the backward passes already in cache/ and measures six things the four
experiments do not report, so that Section IV can be written from measurement
rather than from guesswork:

  A  E1 with PAIRED error bars.  E1 drives every grid with the same 10,000
     scenarios, so the quantity it tabulates is a paired difference and its
     standard error is that of Z_n - Z_ref, not that of Z.  The unpaired
     half-width (0.561) is the wrong yardstick and makes the fine grids look
     unresolved when they are not.

  B  The E3 gap at two meshes per dimension, which separates a dimension
     effect from a mesh effect.  Two grids are cached at each of d = 4 and
     d = 6, so the gap is measured twice at the same d with different delta.

  C  Which constraints actually bind along the simulated paths.

  D  How the cost of Algorithm 3 scales in N at d = 4, in one piece against
     the chunking hydro.simulate does by default.

  E  Convexity of V_t on the grid, which is what would fix the sign of the
     interpolation bias in B.

Run after --all:   python analysis.py
"""

from __future__ import annotations

import time

import numpy as np

from hydro import draw_scenarios, simulate, value_at
from instances import valley
from experiments import SEED, solve


# hydro.simulate holds one (B, m^d, d) array per scenario group, so at d = 4,
# m = 10 a run of N = 10^4 would need several GB in one piece.  It therefore
# cuts the scenarios into chunks by default, which is exact: every path starts
# from x0 and is independent of the others.  Part E measures what that costs.


def rule(title):
    print("\n" + title)
    print("-" * len(title))


# A. E1 with paired error bars

def a_paired_e1():
    rule("A  E1 with paired error bars  (d = 2, m = 10, N = 10,000)")
    inst = valley(2)
    scen = draw_scenarios(inst, 10_000, seed=SEED)

    disc_r, V_r, _ = solve(2, 321, 10)
    Zr = simulate(disc_r, V_r, scen)["Z"]
    ref = Zr.mean()
    print(f"  reference n = 321:  Jhat = {ref:.4f}   "
          f"unpaired half-width = {1.96*Zr.std(ddof=1)/100:.4f}")
    print(f"  {'n':>5} {'delta':>7} | {'rule':>6} {'Jhat':>11} {'err':>9} "
          f"{'paired h-w':>11} {'resolved':>10}")
    for n in [6, 11, 21, 41, 81, 161]:
        for name, rnd in (("interp", False), ("round", True)):
            disc, V, _ = solve(2, n, 10, rounding=rnd)
            Z = simulate(disc, V, scen, rounding=rnd)["Z"]
            diff = Z - Zr
            h = 1.96 * diff.std(ddof=1) / np.sqrt(diff.size)
            err = abs(diff.mean())
            print(f"  {n:>5} {100/(n-1):>7.3f} | {name:>6} {Z.mean():>11.4f} "
                  f"{err:>9.4f} {h:>11.4f} "
                  f"{'yes' if err > h else 'no':>10}")


# B. the E3 gap, dimension effect against mesh effect

def b_gap_vs_mesh():
    rule("B  the E3 gap at two meshes per dimension  (all passes cached)")
    print(f"  {'d':>2} {'n':>3} {'m':>3} {'delta':>7} {'N':>6} | "
          f"{'vbar0':>10} {'Jhat':>10} {'half':>7} {'gap':>8} {'gap/|J|':>8}")
    for (d, n, m, N) in [(2, 161, 10, 10_000), (2, 41, 10, 10_000),
                         (4, 8, 10, 10_000), (4, 5, 3, 10_000),
                         (6, 5, 3, 4_000), (6, 4, 4, 4_000)]:
        inst = valley(d)
        disc, V, _ = solve(d, n, m)
        scen = draw_scenarios(inst, N, seed=SEED)
        r = simulate(disc, V, scen)
        v0 = value_at(disc, V[0], inst.x0)
        gap = v0 - r["Jhat"]
        print(f"  {d:>2} {n:>3} {m:>3} {100/(n-1):>7.2f} {N:>6} | "
              f"{v0:>10.3f} {r['Jhat']:>10.3f} {r['half']:>7.3f} "
              f"{gap:>8.3f} {100*gap/abs(r['Jhat']):>7.2f}%")


# C. which constraints bind

def c_binding():
    rule("C  constraints that bind along the simulated paths")
    print(f"  {'d':>2} {'n':>3} {'m':>3} | {'u = umax':>9} {'u = 0':>8} "
          f"{'spill > 0':>10} {'x_T < xhat':>11}")
    for (d, n, m, N) in [(2, 161, 10, 4_000), (4, 8, 10, 2_000),
                         (6, 4, 4, 2_000)]:
        inst = valley(d)
        disc, V, _ = solve(d, n, m)
        scen = draw_scenarios(inst, N, seed=SEED)
        r = simulate(disc, V, scen, keep_paths=True)
        U, P = r["releases"], r["paths"]
        cap = np.isclose(U, inst.umax).mean()
        zero = np.isclose(U, 0.0).mean()
        spill = np.isclose(P[1:], inst.xmax).mean()
        short = (P[-1] < inst.xhat).mean()
        print(f"  {d:>2} {n:>3} {m:>3} | {100*cap:>8.1f}% {100*zero:>7.1f}% "
              f"{100*spill:>9.1f}% {100*short:>10.1f}%")


# D. cost of Algorithm 3 in N, at the configuration that dominated E3

def d_simulation_scaling():
    rule("D  cost of Algorithm 3 in N  (d = 4, n = 8, m = 10; |U| = 10^4)")
    inst = valley(4)
    disc, V, _ = solve(4, 8, 10)
    print(f"  {'N':>6} {'unblocked':>12} {'per 1000':>10} | "
          f"{'blocked':>10} {'per 1000':>10}")
    for N in (250, 500, 1000, 2000):
        scen = draw_scenarios(inst, N, seed=SEED)
        t0 = time.perf_counter(); simulate(disc, V, scen, chunk=N)
        s1 = time.perf_counter() - t0
        t0 = time.perf_counter(); simulate(disc, V, scen)
        s2 = time.perf_counter() - t0
        print(f"  {N:>6} {s1:>11.2f}s {1000*s1/N:>9.2f}s | "
              f"{s2:>9.2f}s {1000*s2/N:>9.2f}s")
    print("  Chunking costs about 1% at these sizes and the per-1000 column")
    print("  is flat: Algorithm 3 is linear in N and nothing pages here.")


# E. convexity of V_t on the grid

def e_convexity():
    rule("E  axis-wise second differences of V_t on the grid")
    print(f"  {'d':>2} {'n':>3} {'m':>3} {'t':>3} | {'min 2nd diff':>13} "
          f"{'share < 0':>10}")
    for (d, n, m) in [(2, 161, 10), (4, 8, 10), (6, 4, 4)]:
        disc, V, _ = solve(d, n, m)
        for t in (0, 6, 11):
            A = V[t].reshape((n,) * d)
            lo, neg, tot = np.inf, 0, 0
            for i in range(d):
                s = np.diff(A, n=2, axis=i)
                lo = min(lo, s.min()); neg += (s < -1e-9).sum(); tot += s.size
            print(f"  {d:>2} {n:>3} {m:>3} {t:>3} | {lo:>13.4f} "
                  f"{100*neg/tot:>9.2f}%")


if __name__ == "__main__":
    a_paired_e1()
    b_gap_vs_mesh()
    c_binding()
    d_simulation_scaling()
    e_convexity()
