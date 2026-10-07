"""
The four experiments of Section IV.

  E1  interpolation against nearest-neighbour rounding, as the grid is refined
      (discharges the claim of Section II-C5)
  E2  cost of the backward pass against the number of dams
      (a) at fixed resolution, (b) at the finest resolution affordable
      (discharges Section III-D)
  E3  the Monte Carlo estimator beside the value of the discretized process,
      at several meshes per dimension, so that the gap between them is
      measured against delta rather than against d
      (discharges Section III-C)
  E4  three figures

Every experiment writes a CSV into results/ and prints a table.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np

from hydro import (Discretisation, backward, simulate, draw_scenarios,
                   value_at, work_units)
from instances import valley

WORKERS = int(os.environ.get("HYDRO_WORKERS", "0")) or (os.cpu_count() or 1)
HEAVY = False               # set by run_all.py --heavy

RESULTS = "results"
FIGURES = "figures"
CACHE = "cache"
SEED = 20261003

# configurations
# Time scales as (2nm)^d, so doubling n and m at d = 6 costs a factor of 4096.
# The predicted cost of every row is printed by `run_all.py --calibrate`.

E1_D = 2
E1_M = 10
E1_NS = [6, 11, 21, 41, 81, 161, 321, 641]
E1_NREF = 1281              # the reference is itself a discretisation; its own
E1_N_SCEN = 10_000          # error is the floor the interp column settles on

E2A = [(2, 5, 3), (4, 5, 3), (6, 5, 3)]          # fixed resolution, serial

E2B = [(2, 1281, 10), (4, 16, 10), (6, 6, 4)]            # finest affordable
E2B_HEAVY = [(2, 1281, 10), (4, 16, 10), (6, 7, 4)]

# (d, n, m, N).  Several meshes per d: the point of E3 is the trend of the gap
# in delta at FIXED d, which is what separates a mesh effect from a dimension
# effect.  N is set so that the half-width (60) stays below the gap being
# measured; at d = 2 the finest mesh needs a large N because its gap is small.
#
# N must be the SAME for every row at a given d.  draw_scenarios returns the
# same scenarios for the same (seed, N), so rows that share N are driven by
# identical weather and the CHANGE in the gap between two meshes is a paired
# difference, which is what the "drop" column below reports, and the only
# way to resolve a change in the gap once the gap itself is near the noise.
E3 = [(2, 41, 10, 1_000_000), (2, 81, 10, 1_000_000),
      (2, 161, 10, 1_000_000), (2, 321, 10, 1_000_000),
      (4, 5, 10, 10_000), (4, 8, 10, 10_000),
      (4, 12, 10, 10_000), (4, 16, 10, 10_000),
      (6, 4, 4, 4_000), (6, 5, 4, 4_000), (6, 6, 4, 4_000)]
E3_HEAVY = [(6, 7, 4, 4_000)]

# E5: the dimension effect with delta AND the control grid both held fixed.
# E3's equal-mesh comparison puts d = 4 at m = 10 beside d = 6 at m = 4, so the
# ratio it gives mixes the number of dams with the coarseness of the control
# grid, and is only an upper bound on the effect of d.  Here n = 5 and m = 4 in
# every row: delta = 25 and the number of release levels per dam are identical,
# and d is the only thing that varies.  (6, 5, 4) is already cached by E3.
E5 = [(2, 5, 4), (4, 5, 4), (6, 5, 4)]
E5_N = 10_000

E4_HEATMAP = (2, 321, 10)
E4_PATHS = (6, 6, 4, 2_000)
E4_HIST = (2, 321, 10, 1_000_000)   # matches the E3 row for this grid, so that
                                    # the gap in the figure and the gap in the
                                    # table are the same number

PALETTE = {"blue": "#2f6fd0", "orange": "#d4731c", "ink": "#1a1a1a",
           "muted": "#6f6f6b", "grid": "#e5e5e2", "surface": "#fcfcfb",
           "green": "#1d7a6c"}


def e2b_rows():
    return E2B_HEAVY if HEAVY else E2B


def e3_rows():
    return E3 + E3_HEAVY if HEAVY else E3


# helpers

def _ensure_dirs():
    for p in (RESULTS, FIGURES, CACHE):
        os.makedirs(p, exist_ok=True)


def solve(d, n, m, rounding=False, use_cache=True, workers=None,
          repeat=False):
    """Backward pass, cached on disk.  Returns (disc, V, seconds).

    `workers` processes are used; the result does not depend on it (see
    tests.test_parallel_matches_serial), so it is not part of the cache key.

    use_cache=False forces a recomputation.  The timing experiments must do
    this, since a cached `secs` is whenever the cache was built, possibly a
    different run, a different worker count, or a different machine.

    repeat=True re-runs a solve that took less than 50 ms and reports the mean,
    so that the figure measures the computation rather than the fixed cost of
    setting it up.
    """
    _ensure_dirs()
    tag = f"d{d}_n{n}_m{m}" + ("_round" if rounding else "")
    path = os.path.join(CACHE, tag + ".npz")
    inst = valley(d)
    disc = Discretisation(inst, n=n, m=m)
    if use_cache and os.path.exists(path):
        z = np.load(path)
        return disc, z["V"], float(z["secs"])
    w = WORKERS if workers is None else workers
    t0 = time.perf_counter()
    V = backward(disc, rounding=rounding, workers=w)
    secs = time.perf_counter() - t0
    if repeat and secs < 0.05:
        reps = max(2, int(0.2 / max(secs, 1e-6)))
        t0 = time.perf_counter()
        for _ in range(reps):
            backward(disc, rounding=rounding, workers=w)
        secs = (time.perf_counter() - t0) / reps
    np.savez_compressed(path, V=V, secs=secs)
    return disc, V, secs


def write_csv(name, header, rows):
    _ensure_dirs()
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"    -> {path}")


def _fmt_secs(s):
    return f"{s:.3f} s" if s < 300 else (f"{s/60:.1f} min" if s < 7200
                                         else f"{s/3600:.2f} h")


# E1

def e1_interpolation_vs_rounding():
    """Both rules, the same scenarios, increasingly fine grids.

    Two tables.  The first is accuracy: how far the realised cost of the
    induced strategy sits from the reference.  Because every grid is driven by
    the SAME scenarios (common random numbers, Section III-C), what is
    tabulated is a paired difference Z_n - Z_ref, and its half-width is that of
    the difference, not that of Z.  Reporting the unpaired half-width of Z
    would be an error of two to three orders of magnitude on the fine grids.

    The second is honesty: vbar_0(x_0), what the discretized model PREDICTS the
    strategy will cost, beside what it actually costs.  The sign of that gap is
    the point of the experiment.
    """
    print("\nE1  interpolation vs rounding, d = %d, m = %d, N = %d"
          % (E1_D, E1_M, E1_N_SCEN))
    inst = valley(E1_D)
    scen = draw_scenarios(inst, E1_N_SCEN, seed=SEED)   # common random numbers

    disc_r, V_r, _ = solve(E1_D, E1_NREF, E1_M)
    Zref = simulate(disc_r, V_r, scen)["Z"]
    ref = float(Zref.mean())
    print(f"  reference n = {E1_NREF}:  Jhat = {ref:.4f}   "
          f"vbar0 = {value_at(disc_r, V_r[0], inst.x0):.4f}")

    rows = []
    print("\n  accuracy  (err is |Jhat - Jhat_ref|; +- is the PAIRED 95% "
          "half-width)")
    print(f"  {'n':>5} {'delta':>7} | {'interp Jhat':>12} {'err':>9} {'+-':>8} "
          f"| {'round Jhat':>12} {'err':>9} {'+-':>8}")
    for n in E1_NS:
        out = {}
        for rule in (False, True):
            disc, V, _ = solve(E1_D, n, E1_M, rounding=rule)
            r = simulate(disc, V, scen, rounding=rule)
            dif = r["Z"] - Zref
            v0 = value_at(disc, V[0], inst.x0)
            out[rule] = (r["Jhat"], abs(float(dif.mean())),
                         1.96 * float(dif.std(ddof=1)) / np.sqrt(dif.size),
                         v0, v0 - r["Jhat"], r["half"])
        delta = inst.xmax[0] / (n - 1)
        (ji, ei, hi, vi, gi, si), (jr, er, hr, vr, gr, sr) = out[False], out[True]
        print(f"  {n:>5} {delta:>7.3f} | {ji:>12.4f} {ei:>9.4f} {hi:>8.4f} "
              f"| {jr:>12.4f} {er:>9.4f} {hr:>8.4f}")
        rows.append([n, delta, ji, ei, hi, vi, gi, si,
                     jr, er, hr, vr, gr, sr])

    print("\n  what the model promises against what it delivers")
    print("  (+- is the UNPAIRED half-width (60): vbar0 is deterministic, so")
    print("   the gap cannot be paired and a gap below +- is not resolved)")
    print(f"  {'n':>5} {'delta':>7} | {'interp vbar0':>13} {'gap':>9} "
          f"{'+-':>7} | {'round vbar0':>13} {'gap':>10} {'+-':>7}")
    for r in rows:
        print(f"  {r[0]:>5} {r[1]:>7.3f} | {r[5]:>13.4f} {r[6]:>+9.4f} "
              f"{r[7]:>7.4f} | {r[11]:>13.4f} {r[12]:>+10.4f} {r[13]:>7.4f}")
    print("  gap = vbar0(x0) - Jhat.  Positive means the discretized model")
    print("  under-promises what its own strategy achieves, negative that it")
    print("  over-promises.  The two rules differ in sign.")

    write_csv("e1_interpolation_vs_rounding.csv",
              ["n", "delta", "Jhat_interp", "err_interp", "paired_half_interp",
               "vbar0_interp", "gap_interp", "half_interp",
               "Jhat_round", "err_round", "paired_half_round",
               "vbar0_round", "gap_round", "half_round"], rows)
    return ref


# E2

def e2_complexity():
    print("\nE2a  cost at fixed resolution (single process, so that the")
    print("     ratios are not distorted by process start-up)")
    rows_a = []
    print(f"  {'d':>2} {'n':>4} {'m':>3} {'|E|':>10} {'|U|':>10} "
          f"{'work':>10} {'time':>10} {'ratio':>8} {'units/s':>10}")
    prev = None
    for (d, n, m) in E2A:
        inst = valley(d)
        disc, V, secs = solve(d, n, m, use_cache=False, workers=1, repeat=True)
        w = work_units(d, n, m, inst.L, inst.T)
        ratio = secs / prev if prev else float("nan")
        print(f"  {d:>2} {n:>4} {m:>3} {n**d:>10,} {m**d:>10,} "
              f"{w:>10.2e} {_fmt_secs(secs):>10} {ratio:>8.1f} {w/secs:>10.2e}")
        rows_a.append([d, n, m, n ** d, m ** d, w, secs, w / secs,
                       value_at(disc, V[0], inst.x0)])
        prev = secs
    write_csv("e2a_fixed_resolution.csv",
              ["d", "n", "m", "E", "U", "work_units", "secs", "units_per_sec",
               "vbar0"], rows_a)
    print("     A row whose units/s is far below the others never reached full")
    print("     speed: at that size the fixed cost of one backward pass, the")
    print("     grid, the array allocations, 12 x L Python-level calls --")
    print("     outweighs the arithmetic, so its time ratio understates the")
    print("     true growth.  No fixed resolution avoids this: one that is")
    print("     compute-bound at d=2 is out of reach at d=6.")

    print(f"\nE2b  cost at the finest resolution affordable "
          f"({WORKERS} process(es))")
    rows_b = []
    print(f"  {'d':>2} {'n':>4} {'m':>3} {'n*m':>6} {'delta':>7} {'|E|':>12} "
          f"{'work':>10} {'time':>10} {'units/s':>10}")
    for (d, n, m) in e2b_rows():
        inst = valley(d)
        disc, V, secs = solve(d, n, m, use_cache=False)
        w = work_units(d, n, m, inst.L, inst.T)
        print(f"  {d:>2} {n:>4} {m:>3} {n*m:>6} "
              f"{inst.xmax[0]/(n-1):>7.2f} {n**d:>12,} "
              f"{w:>10.2e} {_fmt_secs(secs):>10} {w/secs:>10.2e}")
        rows_b.append([d, n, m, n * m, inst.xmax[0] / (n - 1), n ** d, w, secs,
                       w / secs, value_at(disc, V[0], inst.x0)])
    write_csv("e2b_finest_affordable.csv",
              ["d", "n", "m", "nm", "delta", "E", "work_units", "secs",
               "units_per_sec", "vbar0"], rows_b)


# E3

def e3_estimator():
    """The gap vbar_0(x_0) - Jhat_N at several meshes per dimension.

    One row per (d, n, m) is a comparison of the value of the discretized
    process with the realised cost of the strategy it induces.  Several rows at
    the same d make the trend in delta visible, which is what decides whether
    the gap is an artefact of the mesh or of the dimension.
    """
    print("\nE3  the estimator (59)-(60) beside vbar0(x0), at several meshes")
    rows = []
    print(f"  {'d':>2} {'n':>4} {'m':>3} {'delta':>7} {'N':>9} | "
          f"{'vbar0':>11} {'Jhat':>11} {'half-width':>11} {'gap':>9} "
          f"{'gap/|J|':>8} | {'drop':>9} {'+-':>7}")
    last_d, prev = None, None
    for (d, n, m, N) in e3_rows():
        if last_d is not None and d != last_d:
            print()
            prev = None                 # no pairing across dimensions
        last_d = d
        inst = valley(d)
        disc, V, _ = solve(d, n, m)
        scen = draw_scenarios(inst, N, seed=SEED)
        r = simulate(disc, V, scen)
        v0 = value_at(disc, V[0], inst.x0)
        gap = v0 - r["Jhat"]
        delta = inst.xmax[0] / (n - 1)

        # How much the gap FELL since the previous (coarser) mesh at this d.
        # vbar0 is deterministic, so the gap itself cannot be paired; but the
        # two Jhat's are averages over the SAME scenarios, so their difference
        # is paired and its half-width is that of the difference.  Once the gap
        # is near the noise floor this is the only statement about the trend
        # that the data can still support.
        if prev is not None and prev["N"] == N:
            dz = prev["Z"] - r["Z"]
            drop = (prev["v0"] - v0) - float(dz.mean())
            dhalf = 1.96 * float(dz.std(ddof=1)) / np.sqrt(dz.size)
            cols = f"| {drop:>+9.3f} {dhalf:>7.3f}"
        else:
            drop = dhalf = float("nan")
            cols = f"| {'':>9} {'':>7}"
        prev = {"Z": r["Z"], "v0": v0, "N": N}

        print(f"  {d:>2} {n:>4} {m:>3} {delta:>7.2f} {N:>9,} | "
              f"{v0:>11.3f} {r['Jhat']:>11.3f} {r['half']:>11.3f} "
              f"{gap:>+9.3f} {100*gap/abs(r['Jhat']):>7.2f}% {cols}")
        rows.append([d, n, m, delta, N, v0, r["Jhat"], r["sd"],
                     r["ci"][0], r["ci"][1], gap,
                     100 * gap / abs(r["Jhat"]), drop, dhalf])
    write_csv("e3_estimator.csv",
              ["d", "n", "m", "delta", "N", "vbar0", "Jhat", "sd", "ci_lo",
               "ci_hi", "gap", "gap_pct", "drop_vs_prev", "drop_half"], rows)
    print("  gap is measured against the unpaired half-width, so a gap below")
    print("  it is only an upper bound, which is the honest reading once the")
    print("  mesh is fine enough that the gap has nearly closed.")
    print("  drop is the fall in the gap since the row above, and IS paired:")
    print("  both rows are driven by the same scenarios, so its half-width is")
    print("  that of the difference and is smaller by orders of magnitude.")
    print("  The trend in delta is therefore resolved even where the gap on an")
    print("  individual row is not.")


# E5

def e5_dimension_at_fixed_resolution():
    """The gap against d, with delta and m both held fixed.

    E3's equal-mesh comparison puts d = 4 at m = 10 beside d = 6 at m = 4, so
    the ratio it reports mixes the number of dams with the coarseness of the
    control grid and is only an upper bound on the effect of d.  Here n = 5 and
    m = 4 in every row, so delta = 25 and the release levels per dam are the
    same throughout and d is the only thing that varies.

    The gaps are put on a common scale by dividing by |Jhat|: a six-dam valley
    is worth about twice what a two-dam one is, so the same proportional error
    would otherwise read as a larger absolute number.  That normalisation is a
    choice, and the paper should say so; gap per dam would give another number.
    """
    print("\nE5  the dimension effect at fixed mesh and fixed control grid")
    print("    delta = 25.00 and m = 4 in every row; only d changes")
    rows, base = [], None
    print(f"  {'d':>2} {'n':>4} {'m':>3} {'delta':>7} {'N':>8} | "
          f"{'vbar0':>11} {'Jhat':>11} {'half-width':>11} {'gap':>9} "
          f"{'gap/|J|':>8} {'vs d=2':>8}")
    for (d, n, m) in E5:
        inst = valley(d)
        disc, V, _ = solve(d, n, m)
        scen = draw_scenarios(inst, E5_N, seed=SEED)
        r = simulate(disc, V, scen)
        v0 = value_at(disc, V[0], inst.x0)
        gap = v0 - r["Jhat"]
        pct = 100 * gap / abs(r["Jhat"])
        base = pct if base is None else base
        delta = inst.xmax[0] / (n - 1)
        print(f"  {d:>2} {n:>4} {m:>3} {delta:>7.2f} {E5_N:>8,} | "
              f"{v0:>11.3f} {r['Jhat']:>11.3f} {r['half']:>11.3f} "
              f"{gap:>+9.3f} {pct:>7.2f}% {pct/base:>7.2f}x")
        rows.append([d, n, m, delta, E5_N, v0, r["Jhat"], r["sd"],
                     r["ci"][0], r["ci"][1], gap, pct, pct / base])
    write_csv("e5_dimension_at_fixed_resolution.csv",
              ["d", "n", "m", "delta", "N", "vbar0", "Jhat", "sd", "ci_lo",
               "ci_hi", "gap", "gap_pct", "ratio_vs_d2"], rows)
    print("    The last column is the growth of the relative gap with d and")
    print("    nothing else moving: a measurement, not an upper bound.")


# E4

def _style(ax):
    ax.grid(True, color=PALETTE["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#c2c2bd")
        s.set_linewidth(0.8)


def e4_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.facecolor": PALETTE["surface"],
                         "axes.facecolor": PALETTE["surface"],
                         "mathtext.fontset": "cm"})
    _ensure_dirs()
    print("\nE4  figures")

    # (a) the value function at d = 2
    d, n, m = E4_HEATMAP
    inst = valley(d)
    disc, V, _ = solve(d, n, m)
    Z = V[0].reshape(n, n)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(Z.T, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, inst.xmax[0], 0, inst.xmax[1]])
    cs = ax.contour(np.linspace(0, inst.xmax[0], n),
                    np.linspace(0, inst.xmax[1], n), Z.T,
                    levels=10, colors="white", linewidths=0.6, alpha=0.65)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    ax.plot(inst.x0[0], inst.x0[1], "o", ms=8, mfc="none",
            mec=PALETTE["orange"], mew=2)
    ax.annotate("$x_0$", inst.x0, xytext=(8, 6), textcoords="offset points",
                color=PALETTE["orange"], fontsize=10)
    ax.set_xlabel("$x^1$  (dam 1)")
    ax.set_ylabel("$x^2$  (dam 2)")
    ax.set_title(r"$\bar v_0$ on the grid,  $d=2$,  $n=%d$,  $m=%d$" % (n, m),
                 loc="left", fontsize=11, pad=10)
    fig.colorbar(im, ax=ax, label=r"$\bar v_0(x)$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "e4a_value_function.pdf"))
    fig.savefig(os.path.join(FIGURES, "e4a_value_function.png"), dpi=200)
    plt.close(fig)
    # the slopes quoted in the text: a unit stored upstream is sold again at
    # every dam below it, so d(vbar0)/dx^1 should be about -(p^1 + p^2) and
    # d(vbar0)/dx^2 about -p^2.  Nothing in the code enforces this.
    g1 = -np.median(np.gradient(Z, disc.delta[0], axis=0))
    g2 = -np.median(np.gradient(Z, disc.delta[1], axis=1))
    print(f"    marginal water value   dam 1 {g1:.3f} (p1+p2 = "
          f"{inst.p[0].sum():.2f}),  dam 2 {g2:.3f} (p2 = {inst.p[0][1]:.2f})")
    print(f"    -> {FIGURES}/e4a_value_function.pdf")

    # (b) mean volume trajectories at d = 6
    d, n, m, N = E4_PATHS
    inst = valley(d)
    disc, V, _ = solve(d, n, m)
    scen = draw_scenarios(inst, N, seed=SEED)
    r = simulate(disc, V, scen, keep_paths=True)
    P = r["paths"]                                   # (T+1, N, d)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    _style(ax)
    cmap = plt.get_cmap("viridis")
    for i in range(inst.d):
        col = cmap(0.1 + 0.75 * i / max(inst.d - 1, 1))
        mean = P[:, :, i].mean(1)
        lo, hi = np.percentile(P[:, :, i], [10, 90], axis=1)
        ax.fill_between(range(inst.T + 1), lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(mean, color=col, lw=2, label=f"dam {i+1}")
    ax.axhline(inst.xhat[0], color=PALETTE["ink"], lw=1, ls="--")
    ax.annotate(r"$\hat x$", (0.2, inst.xhat[0]), xytext=(0, 5),
                textcoords="offset points", fontsize=9, color=PALETTE["ink"])
    # the grid of the discretisation, which the mean paths visibly track
    for gp in disc.g[0][1:-1]:
        ax.axhline(gp, color=PALETTE["muted"], lw=0.6, ls=":", alpha=0.7,
                   zorder=1)
    ax.set_xlabel("month $t$")
    ax.set_ylabel("volume")
    ax.set_xlim(0, inst.T)
    ax.set_title(r"Simulated volumes, $d=6$, $n=%d$ (mean, 10-90%% band), "
                 r"$N=%d$" % (n, N), loc="left", fontsize=11, pad=10)
    ax.legend(frameon=False, fontsize=8.5, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "e4b_trajectories.pdf"))
    fig.savefig(os.path.join(FIGURES, "e4b_trajectories.png"), dpi=200)
    plt.close(fig)
    print(f"    (dotted lines are the interior grid points g_k; the mean paths")
    print(f"     track them, which is what n = {n} costs in resolution)")
    print(f"    -> {FIGURES}/e4b_trajectories.pdf")

    # (c) payoff distribution, comparable in form to their Fig. 7
    d, n, m, N = E4_HIST
    inst = valley(d)
    disc, V, _ = solve(d, n, m)
    scen = draw_scenarios(inst, N, seed=SEED)
    r = simulate(disc, V, scen)
    v0 = value_at(disc, V[0], inst.x0)
    gap = v0 - r["Jhat"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _style(ax)
    ax.hist(r["Z"], bins=60, color=PALETTE["blue"], alpha=0.85,
            edgecolor=PALETTE["surface"], linewidth=0.4)
    # Two different quantities, and the figure must not be read as showing one.
    # Jhat_N (59) is the mean of the bars: what the strategy actually costs.
    # vbar_0(x_0) is what the discretized model predicted before any scenario
    # was drawn.  Their distance is the gap of E3.  The model line is dashed
    # and drawn last so that both stay visible when, as here, they coincide.
    ax.axvline(r["Jhat"], color=PALETTE["orange"], lw=2.2, zorder=4,
               label=r"$\hat J_N = %.2f$   (simulated)" % r["Jhat"])
    ax.axvline(v0, color=PALETTE["green"], lw=2.2, ls=(0, (5, 4)), zorder=5,
               label=r"$\bar v_0(x_0) = %.2f$   (model)" % v0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.annotate(r"gap $=%+.3f \pm %.3f$" % (gap, r["half"]), xy=(0.015, 0.63),
                xycoords="axes fraction", fontsize=9,
                color=PALETTE["muted"])
    ax.set_xlabel(r"realized cost $Z^{(r)}$")
    ax.set_ylabel("count")
    ax.set_title(r"Payoff distribution, $d=2$, $n=%d$, $N=%d$" % (n, N),
                 loc="left", fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "e4c_payoff_distribution.pdf"))
    fig.savefig(os.path.join(FIGURES, "e4c_payoff_distribution.png"), dpi=200)
    plt.close(fig)
    print(f"    sd = {r['sd']:.2f}, i.e. {100*r['sd']/abs(r['Jhat']):.0f}% of "
          f"the mean: vbar0 is an expectation, not a guarantee")
    print(f"    Jhat_N = {r['Jhat']:.3f}   vbar0 = {v0:.3f}   gap = {gap:+.3f}"
          f"   (half-width {r['half']:.3f})")
    print(f"    -> {FIGURES}/e4c_payoff_distribution.pdf")
