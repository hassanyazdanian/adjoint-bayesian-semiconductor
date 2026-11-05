# verify_gradients.py
"""
Checks of the adjoint gradient of J(theta) (Section 5.1) against finite differences,
for both inference settings, on the Example 1 data of the experiments.
Usage: python verify_gradients.py [--inversion_mesh_n 96]   (default: the data's inversion mesh, 48)
"""
import argparse

import numpy as np
import torch

from generate_synthetic_data import make_data
from posterior import build_problem


def value(post, theta):
    """J(theta) without recording an adjoint tape."""
    return post.neg_log_likelihood(theta) + float(post.neg_log_prior(torch.from_numpy(theta)))


def taylor_test(post, theta, direction, hs=(1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4)):
    """|J(theta + h d) - J(theta) - h <grad J, d>| must decrease like h^2."""
    J0, g = post.value_and_grad(theta)
    dJ = g @ direction
    rem = np.array([abs(value(post, theta + h * direction) - J0 - h * dJ) for h in hs])
    rates = np.log2(rem[:-1] / rem[1:])
    print(f"  remainders {np.array2string(rem, precision=2)}  rates {np.array2string(rates, precision=2)}")
    return rates


CD_TOL = 1e-5     # relative error of an adjoint partial derivative against central differences (observed: below 1e-6)


def central_differences(post, theta, idxs, eps=1e-5):
    """Relative errors of the adjoint partial derivatives dJ/dtheta[i], i in idxs, against central differences."""
    _, g = post.value_and_grad(theta)
    errors = []
    for i in idxs:
        e = np.zeros_like(theta)
        e[i] = eps
        fd = (value(post, theta + e) - value(post, theta - e)) / (2 * eps)
        errors.append(abs(g[i] - fd) / abs(fd))
        print(f"  dJ/dtheta[{i:2d}]: adjoint {g[i]: .8e}  FD {fd: .8e}  rel.err {errors[-1]:.1e}")
    return np.array(errors)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inversion_mesh_n", type=int, default=None, help="inversion mesh n x n, a multiple of 48 (default 48)")
    a = ap.parse_args()
    data, _ = make_data(example=1)
    rng = np.random.default_rng(0)
    if a.inversion_mesh_n is not None:
        print(f"Inversion mesh {a.inversion_mesh_n} x {a.inversion_mesh_n}")
    for setting in (1, 2):
        post = build_problem(data, setting, inversion_mesh=a.inversion_mesh_n)
        theta = rng.standard_normal(post.dim)          # a prior draw, away from the MAP
        print(f"Setting {setting} (dim {post.dim}): J = {value(post, theta):.6e}")
        print(" Taylor test, random direction:")
        rates = taylor_test(post, theta, rng.standard_normal(post.dim))
        assert np.all(rates > 1.8), "Taylor test failed"
        if setting == 2:
            # the plateau values enter through the doping and through both Dirichlet boundary values
            print(" Taylor test, plateau values only:")
            direction = np.zeros(post.dim)
            direction[post.n_kl:] = rng.standard_normal(2)
            assert np.all(taylor_test(post, theta, direction) > 1.8), "Taylor test in the plateau directions failed"
        idxs = [0, 1, post.n_kl - 1] + ([post.n_kl, post.n_kl + 1] if setting == 2 else [])
        print(" Central differences:")
        errors = central_differences(post, theta, idxs)
        assert np.all(errors < CD_TOL), f"adjoint gradient and central differences differ by more than {CD_TOL:g}"
    print("Gradient checks passed.")
