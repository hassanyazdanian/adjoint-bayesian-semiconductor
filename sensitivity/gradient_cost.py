# gradient_cost.py
"""
Cost of one gradient of the negative log-posterior J(theta) as a function of the KL truncation N_KL
(Reviewer 2: is the adjoint's dimension-independence actually tested?), Example 1, both settings
(d = N_KL with known plateaus, d = N_KL + 2 jointly):
  - adjoint: one recorded nonlinear forward solve and one adjoint solve per gradient, whatever d;
  - forward differences: d + 1 nonlinear forward solves per gradient.
For each N_KL: wall-clock time per gradient (median over repetitions at prior draws), the PDE solves
counted by the model, and the relative difference between the adjoint gradient and a central
finite-difference gradient as a correctness check. The finite-difference step is 1e-4: with 1e-6 the
perturbed residual is often already below the Newton tolerance, so the warm-started solver returns
without updating the state (0.38 Newton iterations per solve at N_KL = 96) and the differences are
wrong (relative error 1e-3). Run on an otherwise idle machine.
Usage: python gradient_cost.py [--n_kl 12 24 48 96 192] [--reps 5] [--out results/scaling_Ex1]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import config as cfg
from io_utils import load_data
from posterior import build_problem

FD_STEP = 1e-4


def value(post, theta):
    """J(theta) from a forward solve only (no adjoint tape)."""
    return post.neg_log_likelihood(theta) + float(post.neg_log_prior(torch.as_tensor(theta), jacobian=True))


def solves(model):
    return model.stats["forward"], model.stats["adjoint"]


def benchmark(n_kl, setting, reps, rng):
    post = build_problem(load_data(cfg.data_file(1)), setting, n_kl=n_kl)
    d, model = post.dim, post.model
    thetas = [post.initial_theta(rng) for _ in range(reps)]
    value(post, thetas[0]); post.value_and_grad(thetas[0])            # warm up (compilation, caches)

    t_adj, s0 = [], solves(model)
    for th in thetas:
        t = time.perf_counter(); post.value_and_grad(th); t_adj.append(time.perf_counter() - t)
    s1 = solves(model)

    t_fd = []
    for th in thetas[:2]:
        t = time.perf_counter()
        j0 = value(post, th)
        for i in range(d):
            e = np.zeros(d); e[i] = FD_STEP
            value(post, th + e)
        t_fd.append(time.perf_counter() - t)
    s2 = solves(model)

    th = thetas[0]
    g_adj = post.value_and_grad(th)[1]
    g_fd = np.array([(value(post, th + e) - value(post, th - e)) / (2 * FD_STEP) for e in np.eye(d) * FD_STEP])
    return dict(n_kl=n_kl, setting=setting, d=d,
                adjoint_time_s=float(np.median(t_adj)),
                adjoint_forward_solves=(s1[0] - s0[0]) / reps, adjoint_adjoint_solves=(s1[1] - s0[1]) / reps,
                fd_time_s=float(np.median(t_fd)), fd_forward_solves=(s2[0] - s1[0]) / 2,
                speedup=float(np.median(t_fd) / np.median(t_adj)),
                rel_diff_adjoint_vs_central_fd=float(np.linalg.norm(g_adj - g_fd) / np.linalg.norm(g_fd)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_kl", type=int, nargs="+", default=[12, 24, 48, 96, 192])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(cfg.RESULTS_DIR, "scaling_Ex1"))
    a = ap.parse_args()
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    rows = [benchmark(n, s, a.reps, rng) for s in (1, 2) for n in a.n_kl]
    lines = ["| Setting | N_KL | d | Adjoint gradient: time (PDE solves) | Forward differences: time (forward solves) | Speed-up | Adjoint vs central FD |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {'known plateaus' if r['setting'] == 1 else 'joint'} | {r['n_kl']} | {r['d']} "
                     f"| {1e3 * r['adjoint_time_s']:.0f} ms ({r['adjoint_forward_solves']:.0f} forward + {r['adjoint_adjoint_solves']:.0f} adjoint) "
                     f"| {1e3 * r['fd_time_s']:.0f} ms ({r['fd_forward_solves']:.0f}) | {r['speedup']:.0f}× | {r['rel_diff_adjoint_vs_central_fd']:.1e} |")
    table = "\n".join(lines)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "gradient_cost.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(a.out, "gradient_cost.md"), "w") as f:
        f.write(table + "\n")
    print(table)
