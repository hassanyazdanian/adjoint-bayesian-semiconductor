# generate_synthetic_data.py
"""
Synthetic electrode data for the straight (Example 1) and curved (Example 2) junctions (Section 6.1),
and for two prior stress tests (Examples 3 and 4: the junction of Example 1 shifted towards the
bottom and the top contact, into the tails of the prior junction depth).
The data are computed on a finer mesh than the inversion mesh (no inverse crime).
Usage: python generate_synthetic_data.py [--examples 1 2] [--data_mesh 192]
"""
import argparse
import os

import numpy as np
import dolfin as dl

import config as cfg
from forward_model import PoissonBoltzmannModel
from io_utils import save_pickle
from plotting import plot_field, TRUE_INTERFACE


JUNCTIONS = {
    1: lambda x: 0.6 - 0.1 * x,                              # Example 1, straight
    2: lambda x: 0.6 - 0.1 * x + 0.2 * np.sin(np.pi * x),    # Example 2, curved
    3: lambda x: 0.3 - 0.1 * x,                              # stress test: Example 1 near the bottom contact
    4: lambda x: 0.9 - 0.1 * x,                              # stress test: Example 1 near the top (electrode) contact
}


def true_doping(V, example, cp=cfg.CP_TRUE, cn=cfg.CN_TRUE, width=cfg.INTERFACE_WIDTH):
    """c(x, y) = c_p + (c_n - c_p) (1 + tanh((y - y_J(x)) / width)) / 2 with the junction y_J of JUNCTIONS."""
    if example not in JUNCTIONS:
        raise ValueError(f"example must be one of {sorted(JUNCTIONS)}")
    x, y = V.tabulate_dof_coordinates().T
    return cp + (cn - cp) * 0.5 * (1.0 + np.tanh((y - JUNCTIONS[example](x)) / width))


def make_data(example, data_mesh_n=cfg.DATA_MESH_N, noise_level=cfg.NOISE_LEVEL, seed=cfg.DATA_SEED,
              width=cfg.INTERFACE_WIDTH):
    """
    y = G_fine(c_true) + eps,  eps ~ N(0, sigma^2 I),  sigma = rho ||G_fine(c_true)||_2 / sqrt(N_meas),
    where G_fine is the forward map on the data mesh. Also returns the inversion-mesh model.
    """
    fine = PoissonBoltzmannModel(data_mesh_n, cfg.EPS, cfg.DELTA, cfg.ELECTRODES)
    y_true = fine.forward(true_doping(fine.V, example, width=width), cfg.CP_TRUE, cfg.CN_TRUE)
    sigma = noise_level * np.linalg.norm(y_true) / np.sqrt(y_true.size)
    y_obs = y_true + sigma * np.random.default_rng(seed).standard_normal(y_true.size)

    # Truth on the inversion mesh (for RE/SSIM) and the inversion model's discretisation error
    model = PoissonBoltzmannModel(cfg.MESH_N, cfg.EPS, cfg.DELTA, cfg.ELECTRODES)
    c_true = true_doping(model.V, example, width=width)
    y_model = model.forward(c_true, cfg.CP_TRUE, cfg.CN_TRUE)

    data = {
        "example": example,
        "mesh_n": cfg.MESH_N,
        "data_mesh_n": data_mesh_n,
        "eps": cfg.EPS,
        "delta": cfg.DELTA,
        "electrodes": cfg.ELECTRODES,
        "cp_true": cfg.CP_TRUE,
        "cn_true": cfg.CN_TRUE,
        "c_true": c_true,
        "y_true": y_true,
        "y_obs": y_obs,
        "sigma": sigma,
        "noise_level": noise_level,
        "interface_width": width,
        "noise_definition": "sigma = rho * ||y_true||_2 / sqrt(N_meas)",
        "seed": seed,
        "model_error": y_model - y_true,      # inversion-mesh minus data-mesh noise-free data
    }
    return data, model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--data_mesh", type=int, default=cfg.DATA_MESH_N)
    ap.add_argument("--noise_level", type=float, default=cfg.NOISE_LEVEL, help="rho (default config.NOISE_LEVEL)")
    ap.add_argument("--width", type=float, default=cfg.INTERFACE_WIDTH, help="tanh width of the true junction")
    a = ap.parse_args()
    for example in a.examples:
        data, model = make_data(example, a.data_mesh, a.noise_level, width=a.width)
        e = data["model_error"]
        print(f"Example {example} (data mesh {a.data_mesh}): N_meas = {model.n_meas}, sigma = {data['sigma']:.4e} "
              f"({100 * data['sigma'] / np.sqrt(np.mean(data['y_true']**2)):.2f}% of RMS(y)); "
              f"inversion-mesh model error: RMS = {np.sqrt(np.mean(e**2)) / data['sigma']:.3f} sigma, "
              f"max = {np.abs(e).max() / data['sigma']:.3f} sigma")
        save_pickle(cfg.data_file(example, a.data_mesh, a.noise_level, a.width), data)
        if a.data_mesh == cfg.DATA_MESH_N:
            plot_field(model.V, data["c_true"], os.path.join(cfg.OBS_DIR, f"true_doping_Ex{example}.png"),
                       contours=[(data["c_true"], TRUE_INTERFACE)])
            c_fun = dl.Function(model.V, name="c")
            c_fun.vector().set_local(data["c_true"])
            dl.File(os.path.join(cfg.OBS_DIR, f"true_doping_Ex{example}.pvd")) << c_fun
