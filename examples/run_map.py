# run_map.py
"""MAP estimate (5.10) with L-BFGS-B and adjoint gradients.  Usage: python run_map.py --example 1"""
import argparse
import os

import numpy as np
from scipy.optimize import minimize

import config as cfg
from diagnostics import mass_matrix, relative_l2_error, ssim
from io_utils import load_data, save_pickle, prior_settings, provenance, run_identity
from plotting import plot_field, plot_data_fit, TRUE_INTERFACE
from posterior import build_problem


def run_map(data_path, setting, k, out_dir, z0, maxiter, gtol, ftol, n_kl=None, tau=None, mean_a=None):
    data = load_data(data_path)
    post = build_problem(data, setting, k, n_kl, tau, mean_a)
    identity = run_identity(data, {"kind": "constant", "z0": float(z0)})       # captured at launch
    theta0 = np.zeros(post.dim)
    theta0[: post.n_kl] = z0

    last = {"it": 0}

    def fun(theta):
        # The plateaus are optimised through r only to enforce the bounds; the objective is the
        # negative log-posterior in the paper's coordinates (z, c_p, c_n), i.e. without the
        # Jacobian of c(r), so the minimiser is the MAP (5.10) and not the mode in r.
        last["J"], last["g"] = post.value_and_grad(theta, jacobian=False)
        return last["J"], last["g"]

    def callback(theta):
        last["it"] += 1
        print(f"[it {last['it']:3d}] J = {last['J']:.6e}  |grad J| = {np.linalg.norm(last['g']):.3e}")

    res = minimize(fun, theta0, jac=True, method="L-BFGS-B", callback=callback,
                   options=dict(maxiter=maxiter, gtol=gtol, ftol=ftol))
    print(res.message)

    theta = res.x
    c_map = post.field(theta).numpy()
    c_heaviside = post.prior.pushforward_heaviside(theta[: post.n_kl], *post.plateaus(theta)).numpy()
    cp, cn = (float(v) for v in post.plateaus(theta))
    y_map = post.model.forward(c_map, cp, cn)

    os.makedirs(out_dir, exist_ok=True)
    V = post.model.V
    re = relative_l2_error(mass_matrix(V), c_map, data["c_true"])
    ss = ssim(V, c_map, data["c_true"])
    print(f"MAP: c_p = {cp:.4f}, c_n = {cn:.4f}, RE = {re:.3f}, SSIM = {ss:.3f}")

    save_pickle(os.path.join(out_dir, "map.pickle"), dict(
        theta=theta, theta0=theta0, c_map=c_map, c_heaviside=c_heaviside, cp=cp, cn=cn, y_map=y_map,
        J=res.fun, RE=re, SSIM=ss, success=bool(res.success), message=str(res.message), n_iter=res.nit,
        grad_norm=float(np.linalg.norm(res.jac)), setting=setting, objective="J without Jacobian of c(r)",
        prior=prior_settings(post.prior), kl_basis=post.prior.basis.numpy(), latent_mean=post.prior.mean.numpy(),
        provenance=provenance(data_path, identity)))
    plot_field(V, c_map, os.path.join(out_dir, "map.png"), cbar_label=r"$c_{\mathrm{MAP}}(\boldsymbol{x})$",
               contours=[(data["c_true"], TRUE_INTERFACE)])
    plot_data_fit(post.model.electrode_x, data["y_obs"], y_map, os.path.join(out_dir, "map_data_fit.png"),
                  sigma=data["sigma"])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--example", type=int, default=1)
    ap.add_argument("--setting", type=int, default=1, help="1: known plateaus, 2: joint")
    ap.add_argument("--data", default=None, help="data file (default: obs/obs_Ex<example>.pickle)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=float, default=cfg.K_SIGMOID, help="sigmoid sharpness (the original joint run used 50)")
    ap.add_argument("--n_kl", type=int, default=None, help="KL truncation (default config.N_KL)")
    ap.add_argument("--tau", type=float, default=None, help="Matern tau (default config.TAU)")
    ap.add_argument("--mean_a", type=float, default=None, help="trend-mean slope a (default config.PRIOR_MEAN_A; 0: zero mean)")
    for key, val in cfg.MAP.items():
        ap.add_argument(f"--{key}", type=type(val), default=val)
    a = ap.parse_args()
    run_map(
        data_path=a.data or cfg.data_file(a.example),
        setting=a.setting,
        k=a.k,
        out_dir=a.out or os.path.join(cfg.RESULTS_DIR, f"map_Setting{a.setting}_Ex{a.example}"),
        z0=a.z0,
        maxiter=a.maxiter,
        gtol=a.gtol,
        ftol=a.ftol,
        n_kl=a.n_kl, tau=a.tau, mean_a=a.mean_a,
    )
