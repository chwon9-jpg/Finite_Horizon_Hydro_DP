"""
Regression tests against the two hand computations in the report.

If both pass, the implementation agrees with the algorithms as written:

  test_worked_example  the worked example of Section III-A, which evaluates
                          one candidate by hand and obtains -46.25
  test_one_dam         the one-dam study document, which runs Algorithms 1 to 3
                          to completion and obtains V_2, V_1, V_0 and
                          J^(T,pi) = -62

Run with:  python tests.py
"""

import numpy as np

from hydro import (Instance, Discretisation, backward, draw_scenarios,
                   eval_candidates, simulate, simulate_exhaustive)


def test_worked_example():
    """Section III-A: x = (50,25), wet noise, u = (20,40), t = T-1.

    The report computes, by hand,
        target point      \\tilde y = (60, 20)
        cell index        k = (2, 0),  (lambda^1, lambda^2) = (0.4, 0.8)
        interpolation     I(V_T, \\tilde y) = 13.75
        running cost      c = -60
        Eval              = -46.25
    """
    inst = Instance(
        d=2, parents=[[], [0]],
        xmax=np.array([100.0, 100.0]), umax=np.array([40.0, 40.0]),
        q=np.array([[10.0, 5.0], [30.0, 15.0]]),
        p=np.array([[1.0, 1.0], [1.0, 1.0]]),
        rho=np.array([[0.7, 0.3]]),                 # T = 1: only V_T is needed
        eps=0.0,
        xhat=np.array([75.0, 50.0]), alpha=np.array([0.01, 0.01]),
        x0=np.array([50.0, 25.0]), name="III-A worked example")
    disc = Discretisation(inst, n=5, m=3)

    VT = inst.phi(disc.grid_points())

    # the four entries listed in the report
    gp = disc.grid_points()
    want = {(50., 0.): 31.25, (50., 25.): 12.5, (75., 0.): 25., (75., 25.): 6.25}
    for (a, b), v in want.items():
        idx = int(np.nonzero((gp[:, 0] == a) & (gp[:, 1] == b))[0][0])
        assert abs(VT[idx] - v) < 1e-12, f"phi({a},{b}) = {VT[idx]}, want {v}"

    x = np.array([[50.0, 25.0]])
    val, y = eval_candidates(disc, x, l=1, Vnext_flat=VT)     # l = 1 is wet
    j = int(np.nonzero((disc.U[:, 0] == 20) & (disc.U[:, 1] == 40))[0][0])

    assert np.allclose(y[0, j], [60.0, 20.0]), f"y = {y[0, j]}, want (60,20)"
    assert abs(val[0, j] - (-46.25)) < 1e-10, f"Eval = {val[0, j]}, want -46.25"

    # the inadmissible candidate of the report: at x = (0,0) under the dry
    # noise, C_t(x,w) is the single element (0,0)
    val0, _ = eval_candidates(disc, np.zeros((1, 2)), l=0, Vnext_flat=VT)
    adm = np.nonzero(np.isfinite(val0[0]))[0]
    assert adm.size == 1 and tuple(disc.U[adm[0]]) == (0.0, 0.0), \
        f"admissible set at the empty state = {disc.U[adm]}"
    print("  worked example (III-A)        Eval = -46.25            OK")


def test_one_dam():
    """The one-dam study document: d = 1, n = 3, T = 2.

    Expected:  V_2 = (25, 0, 0),  V_1 = (10, -30, -40),
               V_0 = (-8, -54, -76),  J^(T,pi) = -62.
    """
    inst = Instance(
        d=1, parents=[[]],
        xmax=np.array([100.0]), umax=np.array([40.0]),
        q=np.array([[10.0], [30.0]]), p=np.array([[1.0], [1.0]]),
        rho=np.array([[0.5, 0.5], [0.5, 0.5]]),
        eps=0.0,
        xhat=np.array([50.0]), alpha=np.array([0.01]),
        x0=np.array([50.0]), name="one-dam example")
    disc = Discretisation(inst, n=3, m=3)
    V = backward(disc)

    for t, want in [(2, [25., 0., 0.]), (1, [10., -30., -40.]),
                    (0, [-8., -54., -76.])]:
        assert np.allclose(V[t], want), f"V_{t} = {V[t]}, want {want}"

    J = simulate_exhaustive(disc, V)
    assert abs(J - (-62.0)) < 1e-10, f"J = {J}, want -62"

    v0 = float(V[0][1])                     # x_0 = 50 = g_1
    assert abs(v0 - (-54.0)) < 1e-12
    print("  one-dam example               V_0 = (-8,-54,-76), J = -62   OK")


def test_interpolation_identities():
    """(22) sum of weights = 1 and (23) mean exactness, on random targets."""
    inst = Instance(
        d=3, parents=[[], [0], [1]],
        xmax=np.array([100.0, 80.0, 120.0]), umax=np.array([30.0, 30.0, 30.0]),
        q=np.array([[10.0, 8.0, 6.0]]), p=np.ones((1, 3)),
        rho=np.ones((1, 1)), eps=0.0,
        xhat=np.array([50.0, 40.0, 60.0]), alpha=np.full(3, 0.01),
        x0=np.zeros(3), name="identity check")
    disc = Discretisation(inst, n=7, m=3)
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 1, size=(500, 3)) * inst.xmax
    ki = np.minimum((y / disc.delta).astype(int), disc.n - 2)
    lam = y / disc.delta - ki
    tot = np.zeros(500)
    mean = np.zeros((500, 3))
    for kp in disc.kappa:
        w = np.ones(500)
        for i in range(3):
            w *= lam[:, i] if kp[i] else (1 - lam[:, i])
        tot += w
        mean += w[:, None] * ((ki + kp) * disc.delta)
    assert np.allclose(tot, 1.0), "weights do not sum to one"
    assert np.allclose(mean, y), "interpolation is not mean exact"
    print("  interpolation identities      (22) and (23)             OK")


def test_parallel_matches_serial():
    """backward(workers > 1) must reproduce backward(workers = 1) exactly.

    The two differ only in the order the blocks are visited; each entry of
    V_t is computed by the same arithmetic in the same order, so the results
    must agree bit for bit, not merely to within a tolerance.
    """
    from instances import valley
    from hydro import Discretisation

    for (d, n, m) in [(2, 21, 10), (4, 4, 3), (6, 3, 2)]:
        disc = Discretisation(valley(d), n=n, m=m)
        for rounding in (False, True):
            Vs = backward(disc, rounding=rounding, workers=1)
            Vp = backward(disc, rounding=rounding, workers=4)
            assert np.array_equal(Vs, Vp), (
                f"d={d} n={n} m={m} rounding={rounding}: parallel differs, "
                f"max |diff| = {np.abs(Vs - Vp).max()}")
    print("  parallel == serial            bit for bit              OK")


def test_simulate_blocking():
    """simulate(chunk = c) must reproduce simulate(chunk = N) exactly, and
    every simulated volume must stay inside the box [xlow, xmax].

    Blocking.  The N paths of Algorithm 3 are independent and all start at
    x_0, so cutting the scenarios into blocks changes the order the paths are
    visited and nothing else.  Each path is then built by the same arithmetic
    in the same order, so the costs must agree bit for bit, not merely to
    within a tolerance, and so must the estimator (59), which sums the same
    array.

    Bounds.  Algorithm 3 stores the state unclipped, on purpose.  The
    admissibility test of eval_candidates carries a 1e-9 tolerance (see the
    README), so a candidate that is infeasible by a billionth could in
    principle be selected and drive a volume just below xlow.  Measurement
    says such candidates are never selected: removing the tolerance
    reproduces every V_t bit for bit, and this assertion is what would
    report it if some future instance changed that.  Clipping the state here
    instead would repair the symptom and hide the cause.  The threshold is
    1e-6, eight orders of magnitude below the volumes themselves, so it
    ignores floating-point dust and catches anything of consequence.
    """
    from instances import valley

    for (d, n, m, N) in [(2, 21, 10, 600), (4, 4, 3, 400)]:
        inst = valley(d)
        disc = Discretisation(inst, n=n, m=m)
        V = backward(disc)
        scen = draw_scenarios(inst, N, seed=1)
        whole = simulate(disc, V, scen, chunk=N, keep_paths=True)
        split = simulate(disc, V, scen, chunk=37, keep_paths=True)
        for key in ("Z", "paths", "releases"):
            assert np.array_equal(whole[key], split[key]), (
                f"d={d} n={n} m={m}: blocked {key} differs, max |diff| = "
                f"{np.abs(whole[key] - split[key]).max()}")
        assert whole["Jhat"] == split["Jhat"] and whole["sd"] == split["sd"]

        X = whole["paths"]
        below, above = float(X.min()), float((X - inst.xmax).max())
        assert below > -1e-6 and above < 1e-6, (
            f"d={d} n={n} m={m}: a simulated volume left [0, xmax]; "
            f"lowest {below:.3e}, largest excess over xmax {above:.3e}")
    print("  blocked == unblocked          bit for bit, in [0,xmax]  OK")


if __name__ == "__main__":
    print("regression tests against the report")
    test_worked_example()
    test_one_dam()
    test_interpolation_identities()
    test_parallel_matches_serial()
    test_simulate_blocking()
    print("all tests passed")
