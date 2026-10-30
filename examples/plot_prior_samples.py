# plot_prior_samples.py
"""Prior samples of the sigmoid pushforward (Figure 3) and the KL variance retention (Section 6.3).
Usage: python plot_prior_samples.py [--out results/prior_samples.png]"""
import argparse
import os

import numpy as np
import torch
import dolfin as dl

import config as cfg
from plotting import plot_prior_samples
from prior import MaternKLPrior, kl_variance_fraction

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(cfg.RESULTS_DIR, "prior_samples.png"))
    a = ap.parse_args()
    V = dl.FunctionSpace(dl.UnitSquareMesh(dl.MPI.comm_self, cfg.MESH_N, cfg.MESH_N), "CG", 1)

    frac = kl_variance_fraction(V, cfg.TAU, cfg.ALPHA)
    print(f"N_KL = {cfg.N_KL} retains {frac[cfg.N_KL - 1]:.4%} of the prior variance")
    for level in (0.99, 0.998, 0.999):
        print(f"  {level:.1%} of the variance needs N_KL = {np.searchsorted(frac, level) + 1}")

    prior = MaternKLPrior(V, cfg.N_KL, cfg.TAU, cfg.ALPHA, cfg.K_SIGMOID, cfg.PRIOR_MEAN_A)
    torch.manual_seed(1368)
    fields = [prior.pushforward(torch.randn(cfg.N_KL), cfg.CP_TRUE, cfg.CN_TRUE).numpy() for _ in range(4)]
    path = a.out
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plot_prior_samples(V, fields, path)
    print("Saved:", path)
