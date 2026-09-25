"""Reusable-holdout Guess-and-Check (Rogers et al., AISTATS 2020).

Guess on train, check on the holdout. A passed check returns the train
interval. A failed check returns a coarsened holdout estimate and spends
one failure. The stream stops when no valid coarsening remains.
"""
import math
from dataclasses import dataclass, field

from scipy.special import comb

import stats


def _c(j):
    return 6.0 / (math.pi ** 2 * (j + 1) ** 2)


def cp_tau_upper(k, n, delta):
    if n <= 0:
        return 0.0, 1.0
    a = k / float(n)
    tau = max(stats.cp_upper(k, n, delta) - a, 0.0)
    return float(a), float(tau)


def _max_gamma(tau, n_h, beta_i):
    if n_h <= 0 or tau <= 0 or beta_i <= 0:
        return 0.0
    beta_i = min(max(beta_i, 1e-300), 1.0)
    need = math.log(2.0 / beta_i)

    def ok(gam):
        return 2.0 * n_h * (tau - gam) ** 2 >= need

    if not ok(0.0):
        return 0.0
    lo, hi = 0.0, tau
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return min(lo, tau * (1.0 - 1e-12))


def _discretize(y, gamma):
    if gamma <= 0:
        return float(min(1.0, max(0.0, y)))
    return float(min(1.0, max(0.0, round(y / gamma) * gamma)))


@dataclass
class GnCResult:
    answer: float | None
    tau: float
    passed_check: bool
    tau_h: float
    beta_i: float

    @property
    def ucb(self):
        if self.answer is None:
            return None
        return min(1.0, self.answer + self.tau)


@dataclass
class GuessAndCheck:
    beta: float
    n_h: int
    i: int = 0
    f: int = 0
    gammas: list = field(default_factory=list)
    alive: bool = True

    def _beta_i(self):
        nu = 1.0
        if self.f > 0:
            nu = float(comb(self.i, self.f, exact=False)) if self.i >= self.f else 1.0
            for g in self.gammas:
                nu *= math.ceil(1.0 / max(g, 1e-12))
            nu = max(nu, 1.0)
        return (self.beta * _c(self.i) * _c(self.f)) / nu

    def query(self, a_g, tau, k_h, n_h=None):
        n_h = int(self.n_h if n_h is None else n_h)
        a_h = (k_h / float(n_h)) if n_h else 0.0
        beta_i = self._beta_i()
        _, tau_h = cp_tau_upper(k_h, n_h, beta_i)
        if not self.alive:
            return GnCResult(None, tau, False, tau_h, beta_i)

        self.i += 1
        if (a_g + tau) > (a_h + tau_h):
            return GnCResult(float(a_g), tau, True, tau_h, beta_i)

        self.f += 1
        gamma = _max_gamma(tau, n_h, beta_i)
        if gamma > 0:
            self.gammas.append(gamma)
            return GnCResult(_discretize(a_h, gamma), tau, False, tau_h, beta_i)

        self.alive = False
        return GnCResult(None, tau, False, tau_h, beta_i)
