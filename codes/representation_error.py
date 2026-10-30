# representation_error.py
"""
Best approximation of the true doping by the pushforward prior (Reviewer 1, Eq. (4.7): "report the best
sigmoid-approximation error at k = 20 or show sensitivity to k"). For each sharpness k and KL truncation
N_KL (production prior otherwise: trend mean a = 2, tau = 16, alpha = 4, known plateaus), find
    z* = argmin_z || c_p + (c_n - c_p) S(k (m + Phi z)) - c_true ||_{L2}^2
(L-BFGS-B, several starts) and report the relative L2 error (RE) and SSIM of the best approximation.
This is the part of the reported RE and SSIM that no data could remove. No PDE solves are involved.
Usage: python representation_error.py [--k 10 14 20 28] [--n_kl 12 48] [--out results/kstudy_Ex1]
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.optimize import minimize

import config as cfg
from diagnostics import mass_matrix, relative_l2_error, ssim
from forward_model import PoissonBoltzmannModel
from io_utils import load_data
from prior import MaternKLPrior


def best_approximation(prior, c_true, M, starts=4, seed=0):
    """Least-squares fit in the M-norm with the analytic gradient
    d/dz ||e||_M^2 = 2 Phi^T [(c_n - c_p) k S'(kX) (M e)],  e = c(z) - c_true,  X = m + Phi z."""
    Phi, m = prior.basis.numpy(), prior.mean.numpy()
    cp, cn, k = cfg.CP_TRUE, cfg.CN_TRUE, prior.k

    def fun(z):
        s = 1.0 / (1.0 + np.exp(-k * (m + Phi @ z)))
        e = cp + (cn - cp) * s - c_true
        Me = M @ e
        return float(e @ Me), 2.0 * Phi.T @ ((cn - cp) * k * s * (1.0 - s) * Me)

    rng = np.random.default_rng(seed)
    z0s = [np.zeros(prior.n_kl)] + [rng.standard_normal(prior.n_kl) for _ in range(starts - 1)]
    best = min((minimize(fun, z0, jac=True, method="L-BFGS-B", options=dict(maxiter=20000, gtol=1e-10, ftol=1e-15))
                for z0 in z0s), key=lambda r: r.fun)
    return prior.pushforward(torch.from_numpy(best.x), cp, cn).numpy(), best


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=float, nargs="+", default=[10, 14, 20, 28])
    ap.add_argument("--n_kl", type=int, nargs="+", default=[12, 48])
    ap.add_argument("--examples", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--out", default=os.path.join(cfg.RESULTS_DIR, "kstudy_Ex1"))
    a = ap.parse_args()
    torch.set_num_threads(1)
    V = PoissonBoltzmannModel(cfg.MESH_N, cfg.EPS, cfg.DELTA).V
    M = mass_matrix(V)
    rows = []
    for n_kl in a.n_kl:
        for k in a.k:
            prior = MaternKLPrior(V, n_kl, cfg.TAU, cfg.ALPHA, k, cfg.PRIOR_MEAN_A)
            for ex in a.examples:
                c_true = load_data(cfg.data_file(ex))["c_true"]
                c_best, res = best_approximation(prior, c_true, M)
                rows.append(dict(example=ex, n_kl=n_kl, k=k, RE=relative_l2_error(M, c_best, c_true),
                                 SSIM=ssim(V, c_best, c_true), z_norm=float(np.linalg.norm(res.x)), converged=bool(res.success)))
    lines = ["| N_KL | k | Example 1: best RE / SSIM | Example 2: best RE / SSIM |", "|---|---|---|---|"]
    for n_kl in a.n_kl:
        for k in a.k:
            cells = []
            for ex in a.examples:
                r = next(r for r in rows if r["n_kl"] == n_kl and r["k"] == k and r["example"] == ex)
                cells.append(f"{r['RE']:.3f} / {r['SSIM']:.3f}")
            lines.append(f"| {n_kl} | {k:g} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "representation_error.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(a.out, "representation_error.md"), "w") as f:
        f.write(table + "\n")
    print(table)
