"""
The three valley instances used in Section IV.

Topologies are those of Fig. 4 of Carpentier et al. (2018); every other number
is ours, since the paper reports no instance data for its academic valleys
beyond the qualitative statements quoted in the README.  Each choice below is
traceable to one of those statements.
"""

import numpy as np

from hydro import Instance

# Topologies.  parents[i] holds the 0-based indices of the dams flowing
# directly into dam i+1, i.e. the set P(i+1) of the report.

TOPOLOGY = {
    2: [[], [0]],                              # 1 -> 2
    4: [[], [0], [1], [2]],                    # 1 -> 2 -> 3 -> 4      (Fig. 4)
    6: [[], [0], [], [1, 2], [3], [4]],        # 1->2, 2,3->4, 4->5->6 (Fig. 4)
}

# Shared parameters

T = 12                      # monthly steps over one year
XMAX = 100.0                # "all the dams have more or less the same maximal
                            #  volume"
MU_TOP, MU_BOTTOM = 8.0, 3.0       # "more inflow for an upstream dam"
GAMMA = 2.0                 # turbine capacity, as a multiple of the mean
                            # inflow of the dam's whole catchment.  This is
                            # what makes "more capacity for a downstream dam"
                            # quantitative: a dam must be able to pass the
                            # accumulated flow of everything above it, or the
                            # valley spills whatever the policy does.  GAMMA
                            # exceeds one so that a dam can also draw its
                            # reservoir down.  At gamma = 1.0 the cap binds in
                            # 98% of decisions and the policy is trivial; at
                            # 2.0 it binds in 18%, which leaves an interior
                            # trade-off most of the time.  See the README.

P_TOP, P_BOTTOM = 1.3, 0.7  # value of water per unit volume, by depth.  The
                            # energy produced by turbinating one unit at dam i
                            # is proportional to that dam's head, which is
                            # larger in the steep upper reaches than on the
                            # flatter lower course; normalised to about 1 on
                            # average.

NU = np.array([0.4, 0.7, 1.0, 1.4, 2.0])        # the L = 5 inflow multipliers
RHO_DRY = np.array([0.35, 0.35, 0.20, 0.07, 0.03])
RHO_WET = np.array([0.03, 0.12, 0.25, 0.35, 0.25])
PEAK_MONTH = 4              # t = 0 is January, so the peak is in May

EPS = 1e-3                  # "usually small" operating cost of the turbine
ALPHA = 0.1                 # terminal weight
XHAT_FRAC = 0.5             # target volume, as a fraction of the capacity
X0_FRAC = 0.5               # initial volume, likewise


def depths(parents):
    """Depth of each dam in the valley: 0 at a source, +1 per step downstream."""
    dep = np.zeros(len(parents), dtype=int)
    for i, par in enumerate(parents):
        dep[i] = 0 if not par else 1 + max(dep[j] for j in par)
    return dep


def catchment(parents, mu):
    """Mean inflow of the whole sub-valley above and including each dam."""
    C = np.zeros(len(parents))
    for i, par in enumerate(parents):          # parents have smaller index
        C[i] = mu[i] + sum(C[j] for j in par)
    return C


def seasonal_law(T=T, peak=PEAK_MONTH):
    """rho_{t,l} of (47).

    The support W does not depend on t, as Section III-A3 requires; the whole
    seasonality is carried by the probabilities.  theta_t interpolates between
    a dry law and a wet law over the twelve months, peaking at `peak`.
    """
    t = np.arange(T)
    theta = 0.5 * (1.0 + np.cos(2.0 * np.pi * (t - peak) / T))
    return (1.0 - theta)[:, None] * RHO_DRY + theta[:, None] * RHO_WET


def valley(d: int) -> Instance:
    """The d-dam instance, d in {2, 4, 6}."""
    if d not in TOPOLOGY:
        raise ValueError(f"no topology for d = {d}; have {sorted(TOPOLOGY)}")
    parents = TOPOLOGY[d]
    dep = depths(parents)
    frac = dep / dep.max()

    mu = MU_TOP + (MU_BOTTOM - MU_TOP) * frac            # falls downstream
    umax = np.round(GAMMA * catchment(parents, mu))      # rises downstream
    xmax = np.full(d, XMAX)

    q = NU[:, None] * mu[None, :]                        # (L, d)
    pi = P_TOP + (P_BOTTOM - P_TOP) * frac               # falls downstream
    p = np.tile(pi, (len(NU), 1))                        # deterministic prices

    return Instance(
        d=d, parents=parents, xmax=xmax, umax=umax,
        q=q, p=p, rho=seasonal_law(),
        eps=EPS,
        xhat=XHAT_FRAC * xmax, alpha=np.full(d, ALPHA),
        x0=X0_FRAC * xmax,
        name=f"{d}-dam valley")


def describe(inst: Instance) -> str:
    dep = depths(inst.parents)
    lines = [f"{inst.name}:  d = {inst.d}, T = {inst.T}, L = {inst.L}",
             f"  P(i)     {[[j + 1 for j in p] for p in inst.parents]}",
             f"  depth    {list(dep)}",
             f"  xmax     {list(inst.xmax)}",
             f"  umax     {list(inst.umax)}",
             f"  mean q   {list(np.round(inst.q.mean(0), 2))}",
             f"  price p  {list(np.round(inst.p[0], 2))}",
             f"  xhat     {list(inst.xhat)}   alpha {inst.alpha[0]}"
             f"   eps {inst.eps}   x0 {list(inst.x0)}"]
    return "\n".join(lines)


if __name__ == "__main__":
    for d in (2, 4, 6):
        print(describe(valley(d)), "\n")
    rho = seasonal_law()
    print("rho_{t,.} by month (rows t = 0..11, columns l = 1..5):")
    for t in range(T):
        print(f"  t={t:2d}  " + "  ".join(f"{v:.3f}" for v in rho[t]),
              f"   E[nu] = {(rho[t] * NU).sum():.3f}")
