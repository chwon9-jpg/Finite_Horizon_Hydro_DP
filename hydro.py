"""
Finite-horizon dynamic programming for hydro valley management.

Implementation of the algorithms of the report "Finite Horizon Dynamic
Programming for Hydro Reservoirs":

    Algorithm 1   Eval        evaluation of one candidate control
    Algorithm 2   backward    backward induction over the grid
    Algorithm 3   simulate    Monte Carlo simulation of the induced strategy

Equation and section numbers in the comments refer to the report.

Conventions:
* Dams are 0-indexed here and 1-indexed in the report.
* The minimal volume and the minimal release are zero:  xlow^i = ulow^i = 0.
* Costs are MINIMISED, so selling water gives a negative running cost.
* Arrays V_t are stored flat, in C order over the multi-index set K; the
  bijection k <-> g_k of (43) is the usual row-major ravel of shape (n,)*d.
"""

from __future__ import annotations

import itertools
import os
import warnings
from dataclasses import dataclass

import numpy as np

__all__ = ["Instance", "Discretisation", "backward", "simulate",
           "simulate_exhaustive", "eval_candidates", "work_units",
           "auto_block", "auto_chunk"]


# Model

@dataclass
class Instance:
    """A hydro valley: everything in Section II of the report except the grids.

    parents[i] is the set P(i+1) of (0-based) indices of the dams flowing
    directly into dam i; every element must be < i, as required in II-A1.
    """
    d: int
    parents: list            # list of lists of 0-based indices, all < i
    xmax: np.ndarray         # (d,)   \bar x^i
    umax: np.ndarray         # (d,)   \bar u^i
    q: np.ndarray            # (L, d) inflows of the L outcomes of W   (eq. 46)
    p: np.ndarray            # (L, d) prices of the L outcomes of W
    rho: np.ndarray          # (T, L) rho_{t,l}                        (eq. 47)
    eps: float               # \varepsilon of (6)
    xhat: np.ndarray         # (d,)   target volumes of (7)
    alpha: np.ndarray        # (d,)   terminal weights of (7)
    x0: np.ndarray           # (d,)   initial volumes
    name: str = ""

    def __post_init__(self):
        for i, par in enumerate(self.parents):
            if any(j >= i for j in par):
                raise ValueError(
                    f"dam {i+1}: parents must have smaller index (II-A1)")
        if self.q.min() < 0:
            raise ValueError("inflows must be nonnegative (II-A1)")
        if not np.allclose(self.rho.sum(1), 1.0):
            raise ValueError("each row of rho must be a probability vector")

    @property
    def T(self) -> int:
        return self.rho.shape[0]

    @property
    def L(self) -> int:
        return self.q.shape[0]

    def phi(self, x: np.ndarray) -> np.ndarray:
        """Terminal cost (7).  x has shape (..., d)."""
        short = np.maximum(0.0, self.xhat - x)
        return (self.alpha * short ** 2).sum(-1)


@dataclass
class Discretisation:
    """The grids of Section II-B and the index sets of Section III-A."""
    inst: Instance
    n: int                   # grid points per dam          -> |E| = n^d
    m: int                   # control levels per dam       -> |U| = m^d

    def __post_init__(self):
        d, n, m = self.inst.d, self.n, self.m
        self.delta = self.inst.xmax / (n - 1)                 # (d,)   eq. (12)
        self.g = [np.linspace(0.0, self.inst.xmax[i], n) for i in range(d)]
        self.U = np.array(list(itertools.product(
            *[np.linspace(0.0, self.inst.umax[i], m) for i in range(d)])))
        self.M = self.U.shape[0]                              # m^d
        self.kappa = np.array(list(itertools.product([0, 1], repeat=d)))
        self.stride = np.array([n ** (d - 1 - i) for i in range(d)])
        self.NG = n ** d

    def grid_points(self) -> np.ndarray:
        """The n^d points g_k of E, in the order of the flat index."""
        mesh = np.meshgrid(*self.g, indexing="ij")
        return np.stack(mesh, axis=-1).reshape(-1, self.inst.d)

    def grid_points_range(self, b0: int, b1: int) -> np.ndarray:
        """Rows b0:b1 of grid_points(), without materialising the whole array.

        A worker process needs only its own block, and at d = 6 the full array
        is large enough that giving every worker a copy is wasteful.
        """
        idx = np.unravel_index(np.arange(b0, b1), (self.n,) * self.inst.d)
        return np.stack([self.g[i][idx[i]] for i in range(self.inst.d)], -1)

    def bytes_per_state(self) -> int:
        """Working set of eval_candidates per state, in bytes.

        Five arrays of shape (B, M, d) (s, y, yc, ki, lam) and about six of
        shape (B, M); used to size the blocks against a memory budget.
        """
        return self.M * (40 * self.inst.d + 48)


def work_units(d, n, m, L, T) -> float:
    """T L n^d m^d (d + 2^d), a proxy for the cost, used only for calibration.

    This is NOT (62), which counts arithmetic operations and so carries d*2^d
    per candidate rather than d + 2^d.  The dominant real cost is not the
    arithmetic but the 2^d gathers from V_{t+1}, which are random accesses that
    defeat the cache, and empirically d + 2^d tracks measured throughput across
    d far better than d*2^d does.  The paper's (62) and this proxy answer
    different questions and both are right for their own.
    """
    return float(T) * L * (n ** d) * (m ** d) * (d + 2 ** d)


# Algorithm 1, vectorised over grid points and candidates

def eval_candidates(disc: Discretisation, x: np.ndarray, l: int,
                    Vnext_flat: np.ndarray, rounding: bool = False):
    """Algorithm 1 for every (state in x) x (candidate in U), one noise.

    x          : (B, d) states, NOT required to be grid points (Section III-C
                 evaluates this at points of X that are not in E)
    l          : index of the noise outcome w_(l)
    Vnext_flat : flat array of V_{t+1}
    rounding   : if True, replace the interpolation (45) by nearest-neighbour
                 rounding, used only by experiment E1

    Returns (val, y) with val of shape (B, M), equal to +infinity at
    inadmissible candidates (line 5), and y of shape (B, M, d) the target
    points f_t(x,u,w) of (4).
    """
    inst, n = disc.inst, disc.n
    d, M, U = inst.d, disc.M, disc.U
    B = x.shape[0]
    q, p = inst.q[l], inst.p[l]

    s = np.empty((B, M, d))
    y = np.empty((B, M, d))
    ok = np.ones((B, M), dtype=bool)

    for i in range(d):
        # line 3: z^i = sum over the dams directly upstream
        if inst.parents[i]:
            z = np.zeros((B, M))
            for j in inst.parents[i]:
                z = z + U[:, j][None, :] + s[:, :, j]
        else:
            z = 0.0
        avail = x[:, i][:, None] + q[i] + z          # x^i + q^i + z^i
        # line 4: admissibility, u^i <= x^i + q^i + z^i - xlow^i   (eq. 5)
        ok &= U[:, i][None, :] <= avail + 1e-9
        # line 7 and line 8
        s[:, :, i] = np.maximum(0.0, avail - U[:, i][None, :] - inst.xmax[i])
        y[:, :, i] = avail - U[:, i][None, :] - s[:, :, i]

    # Inadmissible candidates are carried through the interpolation and only
    # masked afterwards, so the cell index must be clipped from below as well
    # as from above; see the README.  Admissible candidates are unaffected.
    yc = np.clip(y, 0.0, inst.xmax)

    if rounding:
        ki = np.rint(yc / disc.delta).astype(np.intp)
        np.clip(ki, 0, n - 1, out=ki)
        flat = (ki * disc.stride).sum(-1)
        acc = Vnext_flat[flat]
    else:
        ki = np.minimum((yc / disc.delta).astype(np.intp), n - 2)  # eq. (16)
        np.clip(ki, 0, n - 2, out=ki)
        lam = np.clip(yc / disc.delta - ki, 0.0, 1.0)              # eq. (18)
        acc = np.zeros((B, M))
        for kp in disc.kappa:                                      # eq. (45)
            w = np.ones((B, M))
            for i in range(d):
                w *= lam[:, :, i] if kp[i] else (1.0 - lam[:, :, i])
            flat = ((ki + kp) * disc.stride).sum(-1)
            acc += w * Vnext_flat[flat]

    # line 10: running cost (6); independent of the state
    c = (-(p * U) + inst.eps * U ** 2).sum(1)                      # (M,)
    val = np.where(ok, c[None, :] + acc, np.inf)
    return val, y


# Algorithm 2

MEM_BUDGET = int(os.environ.get("HYDRO_MEM_MB", "256")) * 1024 * 1024


def auto_block(disc: Discretisation, workers: int = 1,
               budget: int = MEM_BUDGET) -> int:
    """Block size: as large as the memory budget allows, but small enough that
    there are at least a few blocks per worker to balance the load."""
    by_memory = max(1, budget // max(disc.bytes_per_state(), 1))
    by_workers = max(1, -(-disc.NG // max(workers * 2, 1)))
    return max(1, min(disc.NG, by_memory, by_workers))


PARALLEL_MIN_WORK = float(os.environ.get("HYDRO_PARALLEL_MIN_WORK", 1e8))


def _main_is_a_file() -> bool:
    """True when __main__ is a script on disk, which "spawn" requires."""
    import __main__
    f = getattr(__main__, "__file__", None)
    return bool(f) and os.path.exists(f)


def backward(disc: Discretisation, rounding: bool = False,
             block: int | None = None, progress=None,
             workers: int = 1, min_work: float | None = None) -> np.ndarray:
    """Algorithm 2.  Returns V of shape (T+1, n^d), flat over K.

    V[t, k] is (V_t)_k = \\bar v_t(g_k) of (44).

    workers > 1 spreads the work over that many processes; see
    backward_parallel.  The result is identical either way.

    Starting processes costs a second or two, so a problem below `min_work`
    work units is run serially however many workers are asked for: otherwise
    the start-up dominates and a timing of it measures nothing.  Pass
    min_work=0 to parallelise regardless.
    """
    if min_work is None:
        min_work = PARALLEL_MIN_WORK
    inst = disc.inst
    big = work_units(inst.d, disc.n, disc.m, inst.L, inst.T) >= min_work
    if workers > 1 and big:
        return backward_parallel(disc, rounding=rounding, block=block,
                                 progress=progress, workers=workers)

    inst = disc.inst
    T, L, NG = inst.T, inst.L, disc.NG
    if block is None:
        block = auto_block(disc)

    G = disc.grid_points()
    V = np.empty((T + 1, NG))
    V[T] = inst.phi(G)                                        # lines 1-3, (41)

    for t in range(T - 1, -1, -1):
        Vn = V[t + 1]
        out = np.empty(NG)
        for b0 in range(0, NG, block):
            xb = G[b0:b0 + block]
            h = np.empty((L, xb.shape[0]))
            for l in range(L):                                # line 7
                val, _ = eval_candidates(disc, xb, l, Vn, rounding)
                h[l] = val.min(axis=1)
            out[b0:b0 + xb.shape[0]] = (inst.rho[t][:, None] * h).sum(0)
        V[t] = out                                            # line 9, (48)
        if progress:
            progress(t)
    return V


# Algorithm 2, in parallel
#
# Section III-B notes that the entries (V_t)_k are computed independently of
# one another: at a fixed t, every call to Eval reads the same array V_{t+1}
# and writes one entry of V_t.  The loop over (block of grid points, noise) is
# therefore embarrassingly parallel, and that is what is split here.
#
# V_{t+1} is the only quantity shared between the tasks.  It is placed in a
# shared memory segment that the workers map once, rather than being pickled
# to every task: the parent writes it at the start of each time step, and the
# workers only read.  The synchronisation is the Pool.map itself: the parent
# writes before the map and does not touch the buffer until it returns.

_WORKER: dict = {}


def _worker_init(inst, n, m, rounding, shm_name, ng):
    from multiprocessing import shared_memory
    disc = Discretisation(inst, n=n, m=m)
    shm = shared_memory.SharedMemory(name=shm_name)
    _WORKER.update(disc=disc, rounding=rounding, shm=shm,
                   V=np.ndarray((ng,), dtype=np.float64, buffer=shm.buf))


def _worker_task(args):
    b0, b1, l = args
    disc = _WORKER["disc"]
    x = disc.grid_points_range(b0, b1)
    val, _ = eval_candidates(disc, x, l, _WORKER["V"], _WORKER["rounding"])
    return b0, l, val.min(axis=1)


def backward_parallel(disc: Discretisation, rounding: bool = False,
                      block: int | None = None, progress=None,
                      workers: int | None = None) -> np.ndarray:
    """Algorithm 2 over several processes.  Identical output to backward()."""
    import multiprocessing as mp
    from multiprocessing import shared_memory

    inst = disc.inst
    T, L, NG = inst.T, inst.L, disc.NG
    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, min(workers, NG * L))
    if block is None:
        block = auto_block(disc, workers)

    tasks = [(b0, min(b0 + block, NG), l)
             for b0 in range(0, NG, block) for l in range(L)]

    V = np.empty((T + 1, NG))
    V[T] = inst.phi(disc.grid_points())                       # lines 1-3, (41)

    # "spawn" re-imports the parent's __main__, so it needs that to be a real
    # file on disk: it fails in a REPL, in a notebook, and when the script is
    # piped in on stdin.  Testing that by actually starting a process is both
    # slow and, in an unguarded script, recursive, since the child re-runs the
    # script.  Checking __main__ costs nothing and distinguishes the cases.
    if not _main_is_a_file():
        warnings.warn("worker processes need __main__ to be a script on disk; "
                      "running in a single process instead. Put the call "
                      "behind `if __name__ == \"__main__\":` in a .py file to "
                      "use several cores.", RuntimeWarning)
        return backward(disc, rounding=rounding, block=None,
                        progress=progress, workers=1, min_work=float("inf"))

    shm = shared_memory.SharedMemory(create=True, size=NG * 8)
    try:
        shared_V = np.ndarray((NG,), dtype=np.float64, buffer=shm.buf)
        ctx = mp.get_context("spawn")       # the macOS default; explicit here
        with ctx.Pool(workers, initializer=_worker_init,
                      initargs=(inst, disc.n, disc.m, rounding,
                                shm.name, NG)) as pool:
            for t in range(T - 1, -1, -1):
                shared_V[:] = V[t + 1]      # publish V_{t+1} to the workers
                h = np.empty((L, NG))
                for b0, l, hb in pool.imap_unordered(_worker_task, tasks,
                                                     chunksize=1):
                    h[l, b0:b0 + hb.shape[0]] = hb
                V[t] = (inst.rho[t][:, None] * h).sum(0)      # line 9, (48)
                if progress:
                    progress(t)
    finally:
        shm.close()
        shm.unlink()
    return V


def value_at(disc: Discretisation, V0: np.ndarray, x: np.ndarray) -> float:
    """\\bar v_0(x_0): the entry if x_0 is a grid point, the interpolation of
    V_0 at x_0 otherwise (end of Section II-D3)."""
    val, _ = _interp_only(disc, x[None, :], V0)
    return float(val[0])


def _interp_only(disc, x, Vflat):
    """I(V, x) of (45) at arbitrary points x of shape (B, d)."""
    n = disc.n
    xc = np.clip(x, 0.0, disc.inst.xmax)
    ki = np.minimum((xc / disc.delta).astype(np.intp), n - 2)
    np.clip(ki, 0, n - 2, out=ki)
    lam = np.clip(xc / disc.delta - ki, 0.0, 1.0)
    acc = np.zeros(x.shape[0])
    for kp in disc.kappa:
        w = np.ones(x.shape[0])
        for i in range(disc.inst.d):
            w *= lam[:, i] if kp[i] else (1.0 - lam[:, i])
        acc += w * Vflat[((ki + kp) * disc.stride).sum(-1)]
    return acc, None


# Algorithm 3

def draw_scenarios(inst: Instance, N: int, seed: int) -> np.ndarray:
    """N scenarios (56): indices in {0,...,L-1} of shape (N, T).

    The same seed gives the same scenarios for every discretisation, which is
    the common-random-numbers device of Section III-C.
    """
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(inst.L, size=N, p=inst.rho[t])
                     for t in range(inst.T)], axis=1)


def auto_chunk(disc: Discretisation, budget: int = MEM_BUDGET) -> int:
    """Scenarios per block in Algorithm 3.

    One call to eval_candidates holds about five (B, m^d, d) arrays at a time,
    so a group of B scenarios sharing one noise costs bytes_per_state() each.
    Algorithm 3 splits its N scenarios into L groups, one per noise, so a chunk
    of C scenarios peaks at roughly C / L inside a group.
    """
    per_group = max(1, budget // max(disc.bytes_per_state(), 1))
    return max(disc.inst.L, per_group * disc.inst.L)


def _summary(Z: np.ndarray) -> dict:
    """The estimator (59) and the confidence interval (60) from the costs Z."""
    N = Z.size
    sd = float(Z.std(ddof=1)) if N > 1 else 0.0
    half = 1.96 * sd / np.sqrt(N)
    Jhat = float(Z.mean())
    return {"Z": Z, "Jhat": Jhat, "sd": sd, "half": float(half),
            "ci": (Jhat - half, Jhat + half)}


def simulate(disc: Discretisation, V: np.ndarray, scen: np.ndarray,
             rounding: bool = False, keep_paths: bool = False,
             chunk: int | None = None):
    """Algorithm 3, vectorised over the N scenarios.

    Returns a dict with the path costs Z (57)-(58), the estimator (59) and the
    confidence interval (60); optionally the volume trajectories.

    The N paths are independent and all start at x_0, so the scenarios may be
    cut into blocks of `chunk` and the blocks run one after another.  Every
    path then sees exactly the same arithmetic in the same order as it would
    unblocked, so the result is identical and not merely close
    (tests.test_simulate_blocking).  What blocking buys is a bound on memory,
    which is otherwise N / L * m^d * d doubles held at once: at d = 4, m = 10,
    N = 10^4 that is several gigabytes, and a machine that has to page for it
    spends more time waiting than computing.  Algorithm 2 is blocked over grid
    points for the same reason (auto_block); this is the same device applied to
    the other loop.
    """
    if chunk is None:
        chunk = auto_chunk(disc)
    N = scen.shape[0]
    if N <= chunk:
        return _simulate_chunk(disc, V, scen, rounding, keep_paths)

    parts = [_simulate_chunk(disc, V, scen[a:a + chunk], rounding, keep_paths)
             for a in range(0, N, chunk)]
    out = _summary(np.concatenate([p["Z"] for p in parts]))
    if keep_paths:
        out["paths"] = np.concatenate([p["paths"] for p in parts], axis=1)
        out["releases"] = np.concatenate([p["releases"] for p in parts], axis=1)
    return out


def _simulate_chunk(disc: Discretisation, V: np.ndarray, scen: np.ndarray,
                    rounding: bool = False, keep_paths: bool = False):
    """One block of scenarios; see simulate."""
    inst = disc.inst
    N, T, d = scen.shape[0], inst.T, inst.d

    X = np.tile(inst.x0.astype(float), (N, 1))
    Z = np.zeros(N)
    paths = np.empty((T + 1, N, d)) if keep_paths else None
    rels = np.empty((T, N, d)) if keep_paths else None
    if keep_paths:
        paths[0] = X

    for t in range(T):
        Unew = np.empty((N, d))
        for l in range(inst.L):                     # group the scenarios by noise
            sel = np.nonzero(scen[:, t] == l)[0]
            if sel.size == 0:
                continue
            val, y = eval_candidates(disc, X[sel], l, V[t + 1], rounding)
            j = val.argmin(axis=1)                  # line 5; argmin takes the
            Unew[sel] = disc.U[j]                   # first minimiser (III-C1)
            X_new_sel = y[np.arange(sel.size), j]   # line 7: exact dynamics (4)
            p = inst.p[l]
            Z[sel] += (-(p * disc.U[j]) + inst.eps * disc.U[j] ** 2).sum(1)
            X[sel] = X_new_sel
        if keep_paths:
            paths[t + 1] = X
            rels[t] = Unew

    Z += inst.phi(X)                                 # line 9
    out = _summary(Z)
    if keep_paths:
        out["paths"] = paths
        out["releases"] = rels
    return out


def simulate_exhaustive(disc: Discretisation, V: np.ndarray,
                        rounding: bool = False) -> float:
    """J^{(T,pi)} computed exactly by enumerating all L^T scenarios.

    Only usable when L^T is small; used by the regression tests, where it must
    reproduce the hand computation of the worked example.
    """
    inst = disc.inst
    scen = np.array(list(itertools.product(range(inst.L), repeat=inst.T)))
    prob = np.prod([inst.rho[t][scen[:, t]] for t in range(inst.T)], axis=0)
    res = simulate(disc, V, scen, rounding)
    return float((prob * res["Z"]).sum())
