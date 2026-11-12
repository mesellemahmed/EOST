"""IRT calibration for EOST v2.

Two measurement models are produced from one calibration panel, sharing the
same calibrator abilities and the same (empirical-Bayes) item hyperprior:

* **plug-in** (``pi``): joint-MAP item parameters are treated as known, as in
  classical CAT and recent LLM adaptive-evaluation work;
* **marginal** (``mg``): every item keeps its full grid posterior over
  ``(log a, b)`` given the panel, and downstream computations use the expected
  response function (ERF) ``P̄_i(θ) = E[σ(a_i(θ - b_i)) | panel]``
  (Mislevy, Wingersky & Sheehan, 1994).

Item parameters enter only their own item's likelihood, so the target's
marginal likelihood factorises over items: using ``P̄_i`` inside the ability
posterior, the predictive distribution and the mutual-information criterion is
exact given the calibrators' abilities.

Empirical Bayes. The item hyperprior ``log a ~ N(μ_a, σ_a²)``, ``b ~ N(μ_b, σ_b²)``
is estimated by maximising the marginal likelihood of *all* items given the
calibrator abilities (EM on the item grid). Hyperparameters pool information
over hundreds of items, so they are well determined even with N = 4
calibrators; this keeps items that no calibrator solved (or every calibrator
solved) from being shrunk towards an arbitrary fixed prior centre.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp


@dataclass(frozen=True)
class Priors:
    mu_log_a: float = 0.0
    sd_log_a: float = 1.0
    mu_b: float = 0.0
    sd_b: float = 2.0
    sd_theta: float = 1.0


@dataclass
class Calibration:
    model: str                      # "2pl" or "1pl"
    theta_cal: np.ndarray           # (N,) calibrator abilities (MAP)
    a: np.ndarray                   # (I,) MAP discriminations
    b: np.ndarray                   # (I,) MAP difficulties
    priors: Priors                  # item hyperprior actually used (EB or fixed)
    converged: bool
    n_iter: int
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------- MAP fit
def _negpost_2pl(x, Y, M, N, I, pr: Priors):
    th, b, u = x[:N], x[N:N + I], x[N + I:]
    a = np.exp(u)
    z = a[None, :] * (th[:, None] - b[None, :])
    ll = M * (Y * log_expit(z) + (1.0 - Y) * log_expit(-z))
    r = M * (expit(z) - Y)                              # d(-ll)/dz
    f = -ll.sum() + 0.5 * (th ** 2).sum() / pr.sd_theta ** 2 \
        + 0.5 * ((b - pr.mu_b) ** 2).sum() / pr.sd_b ** 2 \
        + 0.5 * ((u - pr.mu_log_a) ** 2).sum() / pr.sd_log_a ** 2
    g_th = (r * a[None, :]).sum(1) + th / pr.sd_theta ** 2
    g_b = -(r * a[None, :]).sum(0) + (b - pr.mu_b) / pr.sd_b ** 2
    g_u = (r * (th[:, None] - b[None, :])).sum(0) * a + (u - pr.mu_log_a) / pr.sd_log_a ** 2
    return f, np.concatenate([g_th, g_b, g_u])


def _negpost_1pl(x, Y, M, N, I, pr: Priors):
    th, b, u = x[:N], x[N:N + I], x[N + I]
    a = np.exp(u)
    z = a * (th[:, None] - b[None, :])
    ll = M * (Y * log_expit(z) + (1.0 - Y) * log_expit(-z))
    r = M * (expit(z) - Y)
    f = -ll.sum() + 0.5 * (th ** 2).sum() / pr.sd_theta ** 2 \
        + 0.5 * ((b - pr.mu_b) ** 2).sum() / pr.sd_b ** 2 + 0.5 * (u - pr.mu_log_a) ** 2 / pr.sd_log_a ** 2
    g_th = a * r.sum(1) + th / pr.sd_theta ** 2
    g_b = -a * r.sum(0) + (b - pr.mu_b) / pr.sd_b ** 2
    g_u = (r * (th[:, None] - b[None, :])).sum() * a + (u - pr.mu_log_a) / pr.sd_log_a ** 2
    return f, np.concatenate([g_th, g_b, [g_u]])


def fit_map(Y: np.ndarray, model: str = "2pl", priors: Priors = Priors(),
            mask: np.ndarray | None = None, maxiter: int = 3000, x0: np.ndarray | None = None) -> Calibration:
    """Joint MAP calibration of abilities and item parameters on a panel (N × I)."""
    Y = np.asarray(Y, dtype=float)
    N, I = Y.shape
    M = np.ones_like(Y) if mask is None else np.asarray(mask, dtype=float)
    if x0 is None:
        p_sys = (M * Y).sum(1) / np.maximum(M.sum(1), 1)
        p_item = (M * Y).sum(0) / np.maximum(M.sum(0), 1)
        th0 = np.clip(np.log((p_sys + 0.5 / I) / (1 - p_sys + 0.5 / I)), -3, 3)
        th0 = (th0 - th0.mean()) / (th0.std() + 1e-9) if N > 1 else th0 * 0
        b0 = np.clip(-np.log((p_item + 0.5 / N) / (1 - p_item + 0.5 / N)), -4, 4)
        x0 = np.concatenate([th0, b0, np.zeros(I) + priors.mu_log_a]) if model == "2pl" \
            else np.concatenate([th0, b0, [priors.mu_log_a]])
    fun = {"2pl": _negpost_2pl, "1pl": _negpost_1pl}[model]
    res = minimize(fun, x0, args=(Y, M, N, I, priors), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "gtol": 1e-5, "ftol": 1e-10, "maxcor": 30})
    x = res.x
    th, b = x[:N], x[N:N + I]
    a = np.exp(x[N + I:]) if model == "2pl" else np.full(I, np.exp(x[N + I]))
    return Calibration(model=model, theta_cal=th, a=a, b=b, priors=priors,
                       converged=bool(res.success), n_iter=int(res.nit),
                       extras={"objective": float(res.fun), "x": x})


# ------------------------------------------------------------- item posteriors
@dataclass
class ItemGrid:
    log_a: np.ndarray   # (G,)
    b: np.ndarray       # (G,)


B_RANGE = (-8.0, 8.0, 129)       # step 0.125
LOG_A_RANGE = (-2.5, 2.5, 26)    # a in [0.08, 12.2], step 0.2 in log a


def make_item_grid(model: str = "2pl", a_common: float | None = None,
                   b_range=B_RANGE, log_a_range=LOG_A_RANGE) -> ItemGrid:
    """Fixed tensor grid over (log a, b), independent of the (estimated) prior."""
    bs = np.linspace(*b_range)
    la = np.array([np.log(a_common)]) if model == "1pl" else np.linspace(*log_a_range)
    LA, B = np.meshgrid(la, bs, indexing="ij")
    return ItemGrid(log_a=LA.ravel(), b=B.ravel())


def grid_logprior(grid: ItemGrid, pr: Priors, model: str = "2pl") -> np.ndarray:
    lp = -0.5 * ((grid.b - pr.mu_b) / pr.sd_b) ** 2
    if model == "2pl":
        lp = lp - 0.5 * ((grid.log_a - pr.mu_log_a) / pr.sd_log_a) ** 2
    return lp - logsumexp(lp)          # normalised over the grid


def grid_loglik(Y: np.ndarray, theta_cal: np.ndarray, grid: ItemGrid,
                mask: np.ndarray | None = None) -> np.ndarray:
    """Log-likelihood of every item at every grid point, (I, G), via two GEMMs."""
    Y = np.asarray(Y, dtype=float)
    M = np.ones_like(Y) if mask is None else np.asarray(mask, dtype=float)
    z = np.exp(grid.log_a)[None, :] * (theta_cal[:, None] - grid.b[None, :])   # (N, G)
    return (M * Y).T @ log_expit(z) + (M * (1.0 - Y)).T @ log_expit(-z)


def _softmax_rows(L):
    L = L - L.max(1, keepdims=True)
    W = np.exp(L)
    return W / W.sum(1, keepdims=True)


def empirical_bayes(LL: np.ndarray, grid: ItemGrid, init: Priors, model: str = "2pl",
                    iters: int = 200, tol: float = 1e-6) -> tuple[Priors, dict]:
    """Type-II ML item hyperprior by EM on the grid (θ_cal held fixed)."""
    pr = init
    prev = -np.inf
    hist = []
    for k in range(iters):
        lp = grid_logprior(grid, pr, model)
        A = LL + lp[None, :]
        mll = float(logsumexp(A, axis=1).sum())
        W = _softmax_rows(A)
        mb = float((W @ grid.b).mean())
        vb = float((W @ grid.b ** 2).mean() - mb ** 2)
        upd = dict(mu_b=float(np.clip(mb, -6, 6)), sd_b=float(np.clip(np.sqrt(max(vb, 1e-6)), 0.3, 4.0)))
        if model == "2pl":
            ma = float((W @ grid.log_a).mean())
            va = float((W @ grid.log_a ** 2).mean() - ma ** 2)
            upd.update(mu_log_a=float(np.clip(ma, -1.5, 1.5)), sd_log_a=float(np.clip(np.sqrt(max(va, 1e-6)), 0.1, 1.5)))
        pr = replace(pr, **upd)
        hist.append(mll)
        if mll - prev < tol * max(1.0, abs(mll)):
            break
        prev = mll
    return pr, {"em_iters": k + 1, "marginal_loglik": hist[-1]}


def fit_items_given_theta(Y: np.ndarray, theta: np.ndarray, pr: Priors, model: str = "2pl",
                          a_common: float | None = None, u0=None, b0=None, iters: int = 50,
                          tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Conditional item MAP given calibrator abilities, vectorised Fisher scoring.

    2PL: parameters (u = log a, b) per item; 1PL: b only with a = a_common.
    Fisher scoring uses the expected (negative definite) Hessian plus the prior
    precision, with step halving to guarantee monotone ascent.
    """
    Y = np.asarray(Y, dtype=float)
    N, I = Y.shape
    b = np.zeros(I) + pr.mu_b if b0 is None else np.array(b0, dtype=float)
    u = (np.zeros(I) + pr.mu_log_a) if u0 is None else np.array(u0, dtype=float)
    if model == "1pl":
        u = np.full(I, np.log(a_common))

    def objective(u_, b_):
        z = np.exp(u_)[None, :] * (theta[:, None] - b_[None, :])
        ll = (Y * log_expit(z) + (1 - Y) * log_expit(-z)).sum(0)
        lp = -0.5 * ((b_ - pr.mu_b) / pr.sd_b) ** 2
        if model == "2pl":
            lp = lp - 0.5 * ((u_ - pr.mu_log_a) / pr.sd_log_a) ** 2
        return ll + lp

    f = objective(u, b)
    for _ in range(iters):
        a = np.exp(u)
        z = a[None, :] * (theta[:, None] - b[None, :])
        p = expit(z)
        r, w = Y - p, p * (1 - p)
        gb = -a * r.sum(0) - (b - pr.mu_b) / pr.sd_b ** 2
        hbb = -a ** 2 * w.sum(0) - 1.0 / pr.sd_b ** 2
        if model == "2pl":
            gu = (r * z).sum(0) - (u - pr.mu_log_a) / pr.sd_log_a ** 2
            huu = -(w * z ** 2).sum(0) - 1.0 / pr.sd_log_a ** 2
            hub = a * (w * z).sum(0)
            det = huu * hbb - hub ** 2
            du = -(hbb * gu - hub * gb) / det
            db = -(-hub * gu + huu * gb) / det
        else:
            du, db = np.zeros(I), -gb / hbb
        step = np.ones(I)
        for _h in range(30):
            u_new = u + step * du if model == "2pl" else u
            b_new = b + step * db
            f_new = objective(u_new, b_new)
            bad = f_new < f - 1e-12
            if not bad.any():
                break
            step = np.where(bad, step * 0.5, step)
        u, b = np.where(bad, u, u_new), np.where(bad, b, b_new)
        f_old, f = f, np.where(bad, f, f_new)
        if np.max(np.abs(f - f_old)) < tol:
            break
    return np.exp(u), b


def calibrate(Y: np.ndarray, model: str = "2pl", priors: Priors = Priors(),
              empirical_bayes_prior: bool = True) -> tuple[Calibration, np.ndarray, ItemGrid]:
    """Full calibration of one panel.

    1. joint MAP (abilities + items) under the default priors fixes θ_cal;
    2. optional empirical-Bayes item hyperprior given θ_cal (EM on a coarse grid);
    3. plug-in item parameters = conditional MAP given θ_cal under that prior;
    4. marginal model = exact grid posterior of each item given θ_cal under that prior.
    Plug-in and marginal models therefore share θ_cal and the item hyperprior;
    they differ only in plugging in the mode versus integrating the posterior.
    """
    cal0 = fit_map(Y, model, priors)
    th = cal0.theta_cal
    a_common = float(cal0.a[0]) if model == "1pl" else None
    grid = make_item_grid(model, a_common=a_common)
    pr, info = priors, {}
    if empirical_bayes_prior:
        coarse = make_item_grid(model, a_common=a_common, b_range=(B_RANGE[0], B_RANGE[1], 65),
                                log_a_range=(LOG_A_RANGE[0], LOG_A_RANGE[1], 14))
        pr, info = empirical_bayes(grid_loglik(Y, th, coarse), coarse, priors, model, iters=60, tol=1e-7)
    a, b = fit_items_given_theta(Y, th, pr, model, a_common=a_common,
                                 u0=np.log(cal0.a), b0=cal0.b)
    LL = grid_loglik(Y, th, grid)
    W = _softmax_rows(LL + grid_logprior(grid, pr, model)[None, :])
    cal = Calibration(model=model, theta_cal=th, a=a, b=b, priors=pr, converged=cal0.converged,
                      n_iter=cal0.n_iter, extras={"objective": cal0.extras["objective"], **info})
    return cal, W, grid


def item_grid_posterior(Y, theta_cal, grid, priors: Priors, model="2pl", mask=None) -> np.ndarray:
    """Posterior weights over item parameters given calibrator abilities, (I, G)."""
    LL = grid_loglik(Y, theta_cal, grid, mask)
    return _softmax_rows(LL + grid_logprior(grid, priors, model)[None, :])


# --------------------------------------------------------- response functions
@dataclass
class MeasurementModel:
    """Item response functions on an ability grid (one row per item)."""
    name: str                 # "pi" (plug-in) or "mg" (marginal / ERF)
    theta: np.ndarray         # (K,) ability grid
    P: np.ndarray             # (I, K) P(y=1 | theta)
    dP: np.ndarray            # (I, K) dP/dtheta
    item_a_eff: np.ndarray | None = None  # diagnostics: 4 × max slope

    @property
    def fisher(self) -> np.ndarray:
        P = np.clip(self.P, 1e-12, 1 - 1e-12)
        return self.dP ** 2 / (P * (1 - P))


def plugin_model(cal: Calibration, theta: np.ndarray) -> MeasurementModel:
    z = cal.a[:, None] * (theta[None, :] - cal.b[:, None])
    P = expit(z)
    return MeasurementModel("pi", theta, P, cal.a[:, None] * P * (1 - P), item_a_eff=cal.a.copy())


def marginal_model(W: np.ndarray, grid: ItemGrid, theta: np.ndarray) -> MeasurementModel:
    a = np.exp(grid.log_a)
    S = expit(a[:, None] * (theta[None, :] - grid.b[:, None]))      # (G, K)
    P = W @ S
    dP = W @ (a[:, None] * S * (1 - S))
    return MeasurementModel("mg", theta, P, dP, item_a_eff=4.0 * dP.max(1))


def default_theta_grid(lo: float = -5.0, hi: float = 5.0, K: int = 201) -> np.ndarray:
    return np.linspace(lo, hi, K)
