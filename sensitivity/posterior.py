# posterior.py
"""
Negative log-posterior J(theta) = Phi(y; theta) + Psi(theta)  (4.19)
for the two inference settings of Section 6.2, with adjoint gradients in PyTorch.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

import config as cfg
from forward_model import PoissonBoltzmannModel
from prior import MaternKLPrior

torch.set_default_dtype(torch.float64)


class _Misfit(torch.autograd.Function):
    """
    Phi = ||B u - y||^2 / (2 sigma^2), u = W^h(c; f_p, f_n)   (4.15).
    The backward pass is a single adjoint solve giving dPhi/dc, dPhi/df_p, dPhi/df_n.
    """

    @staticmethod
    def forward(ctx, c, f_p, f_n, model, y, sigma):
        u, ctx.record = model.solve_recorded(c.detach().numpy(), float(f_p), float(f_n))
        r = model.observe(u) - y
        ctx.model = model
        ctx.dPhi_dobs = r / sigma**2
        return c.new_tensor(0.5 * float(r @ r) / sigma**2)

    @staticmethod
    def backward(ctx, grad_out):
        dPhi_du = float(grad_out) * (ctx.model.B.T @ ctx.dPhi_dobs)
        g_c, g_p, g_n = ctx.model.adjoint_gradient(ctx.record, dPhi_du)
        return torch.from_numpy(g_c), torch.tensor(g_p), torch.tensor(g_n), None, None, None


class Posterior:
    """
    Setting 1 (known plateaus):  theta = z,
        J(z) = Phi(y; F_2(z)) + ||z||^2 / 2.
    Setting 2 (joint):           theta = (z, r_p, r_n),
        c_p = a_p + (b_p - a_p) S(r_p),  c_n = a_n + (b_n - a_n) S(r_n),
        J(theta) = Phi + ||z||^2 / 2 - log S'(r_p) - log S'(r_n),
    where the last two terms make c_p ~ U(a_p, b_p) and c_n ~ U(a_n, b_n).
    """

    def __init__(self, model, prior, y, sigma, plateaus=None, cp_bounds=None, cn_bounds=None):
        self.model = model
        self.prior = prior
        self.y = np.asarray(y, dtype=float)
        self.sigma = float(sigma)
        self.known_plateaus = plateaus is not None
        if self.known_plateaus:
            self.cp, self.cn = map(float, plateaus)
        else:
            self.cp_bounds = tuple(map(float, cp_bounds))
            self.cn_bounds = tuple(map(float, cn_bounds))
        self.n_kl = prior.n_kl
        self.dim = self.n_kl + (0 if self.known_plateaus else 2)

    # ------------------------------------------------------------------
    # Parameter maps
    # ------------------------------------------------------------------
    def plateaus(self, theta):
        """(c_p, c_n) for theta of shape (dim,) or (batch, dim)."""
        if self.known_plateaus:
            return torch.tensor(self.cp), torch.tensor(self.cn)
        theta = torch.as_tensor(theta)
        (a_p, b_p), (a_n, b_n) = self.cp_bounds, self.cn_bounds
        return (
            a_p + (b_p - a_p) * torch.sigmoid(theta[..., -2]),
            a_n + (b_n - a_n) * torch.sigmoid(theta[..., -1]),
        )

    def field(self, theta):
        """Doping c = F_2(theta) (4.12) for theta of shape (dim,) or (batch, dim)."""
        theta = torch.as_tensor(theta)
        cp, cn = self.plateaus(theta)
        return self.prior.pushforward(theta[..., : self.n_kl], cp[..., None], cn[..., None])

    def dirichlet_data(self, cp, cn):
        s = 2.0 * self.model.delta**2
        return torch.asinh(cp / s), torch.asinh(cn / s)

    # ------------------------------------------------------------------
    # Posterior terms
    # ------------------------------------------------------------------
    def neg_log_prior(self, theta, jacobian=True):
        """
        Psi (4.11), up to an additive constant. With jacobian=True it is the density of
        the sampled coordinates (z, r_p, r_n); with jacobian=False the density in the
        paper's coordinates (z, c_p, c_n), where the uniform plateau priors are constant.
        """
        z = theta[: self.n_kl]
        psi = 0.5 * (z @ z)
        if jacobian and not self.known_plateaus:
            r = theta[self.n_kl:]
            psi = psi + torch.sum(F.softplus(r) + F.softplus(-r))   # -log S(r)(1 - S(r))
        return psi

    def misfit(self, theta):
        cp, cn = self.plateaus(theta)
        c = self.prior.pushforward(theta[: self.n_kl], cp, cn)
        f_p, f_n = self.dirichlet_data(cp, cn)
        return _Misfit.apply(c, f_p, f_n, self.model, self.y, self.sigma)

    def potential(self, theta, jacobian=True):
        """
        J(theta) (4.19), differentiable through the adjoint (Algorithm 5.1).
        Sampling uses jacobian=True; the MAP (5.10) in (z, c_p, c_n) uses jacobian=False.
        """
        return self.misfit(theta) + self.neg_log_prior(theta, jacobian)

    def value_and_grad(self, theta, jacobian=True):
        theta = torch.tensor(np.asarray(theta, dtype=float), requires_grad=True)
        J = self.potential(theta, jacobian)
        J.backward()
        return float(J), theta.grad.numpy().copy()

    # ------------------------------------------------------------------
    # Gradient-free evaluations (pCN)
    # ------------------------------------------------------------------
    def neg_log_likelihood(self, theta):
        """Phi(y; theta) (4.15), without recording an adjoint tape; +inf if the PDE solve fails."""
        with torch.no_grad():
            theta = torch.as_tensor(np.asarray(theta, dtype=float))
            cp, cn = self.plateaus(theta)
            c = self.prior.pushforward(theta[: self.n_kl], cp, cn).numpy()
            f_p, f_n = self.dirichlet_data(cp, cn)
        try:
            r = self.model.observe(self.model.solve(c, float(f_p), float(f_n))) - self.y
        except RuntimeError:
            return np.inf
        return 0.5 * float(r @ r) / self.sigma**2

    @staticmethod
    def neg_log_prior_plateaus(r):
        """-log S(r_p)(1 - S(r_p)) - log S(r_n)(1 - S(r_n))."""
        r = np.asarray(r, dtype=float)
        return float(np.sum(np.logaddexp(0.0, r) + np.logaddexp(0.0, -r)))

    def initial_theta(self, rng):
        """
        Starting point of a chain (numpy Generator rng). With config.INIT_FROM_PRIOR a prior
        draw: z ~ N(0, I) and r ~ Logistic(0, 1), i.e. c_p, c_n uniform on their intervals.
        Otherwise z = 1e-2 N(0, I) and the plateaus at the interval midpoints (r = 0).
        """
        theta = np.zeros(self.dim)
        theta[: self.n_kl] = (1.0 if cfg.INIT_FROM_PRIOR else 1e-2) * rng.standard_normal(self.n_kl)
        if not self.known_plateaus and cfg.INIT_FROM_PRIOR:
            theta[self.n_kl:] = rng.logistic(size=2)
        return theta


def starting_point(post, rng, init_file, chain, seed=None):
    """
    Starting point and the initialisation policy that produced it: row `chain` of the .npy file
    init_file (one starting point per chain; policy records the file's SHA-256), or a draw from rng
    (policy: prior draws if config.INIT_FROM_PRIOR, else near zero; seed is the generator's seed).
    """
    if init_file is None:
        kind = "prior_draw" if cfg.INIT_FROM_PRIOR else "near_zero (1e-2 N(0, I))"
        return post.initial_theta(rng), {"kind": kind, "rng_seed": seed}
    import hashlib
    with open(init_file, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    theta0 = np.asarray(np.load(init_file)[chain], dtype=float)
    if theta0.shape != (post.dim,) or not np.all(np.isfinite(theta0)):
        raise ValueError(f"row {chain} of {init_file} is not a finite starting point of dimension {post.dim}")
    return theta0, {"kind": "file", "path": os.path.abspath(init_file), "sha256": sha, "row": chain}


def inversion_mesh_n(data, n=None):
    """Inversion mesh n x n: n, or the data set's inversion mesh. The electrodes have width 1/48, so n must be
    a multiple of 48."""
    n = int(data["mesh_n"]) if n is None else int(n)
    if n < 48 or n % 48:
        raise ValueError(f"the inversion mesh must be a multiple of 48, not {n}")
    return n


def true_field(data, V):
    """The true doping of a data set on the CG1 space V of an n x n inversion mesh. The data set stores it on its
    own inversion mesh only; on another mesh it is evaluated from its formula, after checking that the formula
    reproduces the stored field."""
    from generate_synthetic_data import true_doping
    n = round(V.dim() ** 0.5) - 1
    if n == int(data["mesh_n"]):
        return np.asarray(data["c_true"], dtype=float)
    V0 = PoissonBoltzmannModel(data["mesh_n"], data["eps"], data["delta"], data["electrodes"]).V
    c0 = true_doping(V0, int(data["example"]), data["cp_true"], data["cn_true"])
    if not np.allclose(c0, data["c_true"], rtol=0, atol=1e-12):
        raise ValueError("the true-doping formula does not reproduce the stored field (interface width?)")
    return true_doping(V, int(data["example"]), data["cp_true"], data["cn_true"])


def build_problem(data, setting, k=None, n_kl=None, tau=None, mean_a=None, cp_bounds=None, cn_bounds=None,
                  inversion_mesh=None):
    """Forward model, prior and posterior for a data set; k, n_kl, tau, mean_a and, in the joint setting,
    the plateau bounds cp_bounds and cn_bounds default to config, the inversion mesh to the data set's."""
    model = PoissonBoltzmannModel(inversion_mesh_n(data, inversion_mesh), data["eps"], data["delta"], data["electrodes"])
    prior = MaternKLPrior(model.V, cfg.N_KL if n_kl is None else n_kl, cfg.TAU if tau is None else tau, cfg.ALPHA,
                          cfg.K_SIGMOID if k is None else k, cfg.PRIOR_MEAN_A if mean_a is None else mean_a)
    if cfg.known_plateaus(setting):
        kwargs = dict(plateaus=(data["cp_true"], data["cn_true"]))
    else:
        kwargs = dict(cp_bounds=cfg.CP_BOUNDS if cp_bounds is None else cp_bounds,
                      cn_bounds=cfg.CN_BOUNDS if cn_bounds is None else cn_bounds)
    return Posterior(model, prior, data["y_obs"], data["sigma"], **kwargs)
