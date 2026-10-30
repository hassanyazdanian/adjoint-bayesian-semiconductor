# run_nuts.py
"""
Adjoint-based NUTS (Algorithm 5.1), one chain per call.
Usage: python run_nuts.py --example 1 --setting 1 --chain 0 [--full_mass True] [--tag _dense]
"""
import argparse
import time

import numpy as np
import torch
import pyro
from pyro.infer.mcmc import MCMC, NUTS

import config as cfg
from io_utils import load_data, run_identity, save_samples
from posterior import build_problem, starting_point


def _str2bool(s):
    return str(s).lower() in ("1", "true", "yes")


class DepthTrackingNUTS(NUTS):
    """Pyro NUTS that records the tree depth of each trajectory (number of doublings)."""

    depth = 0

    def _build_tree(self, z, r, z_grads, log_slice, direction, tree_depth, energy_current):
        self.depth = max(self.depth, tree_depth + 1)
        return super()._build_tree(z, r, z_grads, log_slice, direction, tree_depth, energy_current)


def run_nuts(data_path, setting, out_path, chain=0, k=None, n_kl=None, tau=None, mean_a=None, warmup_steps=500, num_samples=1000,
             target_accept_prob=0.8, max_tree_depth=10, full_mass=False, seed=0, init_file=None, manifest=None,
             cp_bounds=None, cn_bounds=None, inversion_mesh=None):
    if warmup_steps < 0 or num_samples < 1:
        raise ValueError("warmup_steps must be >= 0 and num_samples >= 1")
    t_launch = time.time()                          # epoch seconds, before the model is built
    torch.set_num_threads(1)
    data = load_data(data_path)
    post = build_problem(data, setting, k, n_kl, tau, mean_a, cp_bounds, cn_bounds, inversion_mesh)
    model = post.model
    pyro.set_rng_seed(seed + chain)
    theta0, init_policy = starting_point(post, np.random.default_rng(seed + chain), init_file, chain, seed + chain)
    theta0 = torch.from_numpy(theta0)
    identity = run_identity(data, init_policy, manifest)          # captured at launch, before sampling
    print(f"NUTS chain {chain}: setting {setting}, inversion mesh {model.n} x {model.n}, dim(theta) = {post.dim}, N_meas = {model.n_meas}, "
          f"sigma = {post.sigma:.4e}, k = {post.prior.k}, {'dense' if full_mass else 'diagonal'} mass matrix, "
          f"target acceptance {target_accept_prob}")

    # per-iteration log: stage, wall time, potential evaluations (forward solves), gradient evaluations
    # (adjoint solves) and forward-solve attempts so far, running mean acceptance, tree depth
    log, cost = [], {}

    def hook(kernel, samples, stage, i):
        log.append((stage, time.perf_counter() - t0, model.stats["forward"], model.stats["adjoint"],
                    model.stats["forward_attempts"], kernel._mean_accept_prob, kernel.depth))
        kernel.depth = 0
        if stage == "Warmup" and i == warmup_steps - 1:
            cost["warmup"] = dict(model.stats, wall_time=time.perf_counter() - t0)

    kernel = DepthTrackingNUTS(potential_fn=lambda params: post.potential(params["theta"]),
                               target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth,
                               full_mass=full_mass)
    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps,
                initial_params={"theta": theta0}, hook_fn=hook)
    t0, t0_epoch = time.perf_counter(), time.time()
    if warmup_steps == 0:
        cost["warmup"] = dict(model.stats, wall_time=0.0)
    mcmc.run()
    cost["total"] = dict(model.stats, wall_time=time.perf_counter() - t0)
    timing = dict(launch_epoch=t_launch, warmup_start_epoch=t0_epoch,
                  sampling_start_epoch=t0_epoch + cost["warmup"]["wall_time"], end_epoch=time.time())

    stage = np.array([s for s, *_ in log])
    wall, n_fwd, n_adj, n_att, mean_acc, depth = (np.array(col, dtype=float) for col in list(zip(*log))[1:])
    attempts = np.diff(np.concatenate([[0.0], n_att]))  # forward-solve attempts per iteration
    is_sample = stage == "Sample"
    evals = np.diff(np.concatenate([[0.0], n_fwd]))     # potential evaluations per iteration
    grads = np.diff(np.concatenate([[0.0], n_adj]))     # gradient evaluations (leapfrog steps) per iteration
    n_phase = np.where(is_sample, np.cumsum(is_sample), np.cumsum(~is_sample))
    prev = np.concatenate([[0.0], mean_acc[:-1]])
    accept = np.where(n_phase == 1, mean_acc, n_phase * mean_acc - (n_phase - 1) * prev)
    times = wall[is_sample] - cost["warmup"]["wall_time"]

    diag = mcmc.diagnostics()
    theta = mcmc.get_samples()["theta"].detach().numpy()
    print(f"NUTS finished in {cost['total']['wall_time'] / 3600:.2f} h; step size {float(kernel.step_size):.3e}; "
          f"mean accept prob {accept[is_sample].mean():.3f}; divergences {len(diag['divergences']['chain 0'])}; "
          f"gradient evaluations per iteration {grads[is_sample].mean():.0f} (max {grads[is_sample].max():.0f}); "
          f"tree depth max {depth[is_sample].max():.0f} of {max_tree_depth}")
    save_samples(out_path, post, theta, "nuts", times=times, cost=cost, identity=identity, setting=setting, data=data_path,
                 chain=chain, seed=seed + chain, warmup_steps=warmup_steps, target_accept_prob=target_accept_prob,
                 max_tree_depth=max_tree_depth, mass_matrix="dense" if full_mass else "diagonal",
                 inverse_mass_matrix={str(k): v.detach().numpy() for k, v in kernel.inverse_mass_matrix.items()},
                 step_size=float(kernel.step_size), divergences=np.asarray(diag["divergences"]["chain 0"]),
                 grads_per_iteration=grads[is_sample], evals_per_iteration=evals[is_sample],
                 tree_depth=depth[is_sample], accept_prob=accept[is_sample],
                 warmup_grads_per_iteration=grads[~is_sample], warmup_tree_depth=depth[~is_sample],
                 warmup_accept_prob=accept[~is_sample], forward_attempts_per_iteration=attempts[is_sample],
                 warmup_forward_attempts_per_iteration=attempts[~is_sample], timing=timing,
                 init_file=init_file, theta0=theta0.numpy(), inversion_mesh_n=model.n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", type=int, default=1)
    ap.add_argument("--setting", type=int, default=1, help="1: known plateaus, 2: joint")
    ap.add_argument("--chain", type=int, default=0, help="chain index; the seed is seed + chain")
    ap.add_argument("--data", default=None, help="data file (default: obs/obs_Ex<example>.pickle)")
    ap.add_argument("--tag", default="", help="suffix of the run folder, e.g. _dense or _k10")
    ap.add_argument("--k", type=float, default=None, help="sigmoid sharpness (default config.K_SIGMOID)")
    ap.add_argument("--n_kl", type=int, default=None, help="KL truncation (default config.N_KL)")
    ap.add_argument("--manifest", default=None, help="experiment manifest (JSON); its identity is stored with the chain")
    ap.add_argument("--init_file", default=None, help=".npy file with one starting point per chain (default: prior draw)")
    ap.add_argument("--tau", type=float, default=None, help="Matern tau (default config.TAU)")
    ap.add_argument("--mean_a", type=float, default=None, help="trend-mean slope a (default config.PRIOR_MEAN_A; 0: zero mean)")
    ap.add_argument("--cp_bounds", type=float, nargs=2, default=None, help="support of the uniform c_p prior, setting 2 (default config.CP_BOUNDS)")
    ap.add_argument("--cn_bounds", type=float, nargs=2, default=None, help="support of the uniform c_n prior, setting 2 (default config.CN_BOUNDS)")
    ap.add_argument("--inversion_mesh_n", type=int, default=None, help="inversion mesh n x n, a multiple of 48 (default: the data file's, 48)")
    for key, val in cfg.NUTS.items():
        ap.add_argument(f"--{key}", type=_str2bool if isinstance(val, bool) else type(val), default=val)
    a = ap.parse_args()
    run_nuts(
        data_path=a.data or cfg.data_file(a.example),
        setting=a.setting,
        out_path=cfg.chain_file("nuts", a.setting, a.example, a.chain, a.tag),
        chain=a.chain, k=a.k, n_kl=a.n_kl, tau=a.tau, mean_a=a.mean_a,
        warmup_steps=a.warmup_steps, num_samples=a.num_samples, target_accept_prob=a.target_accept_prob,
        max_tree_depth=a.max_tree_depth, full_mass=a.full_mass, seed=a.seed, init_file=a.init_file, manifest=a.manifest,
        cp_bounds=a.cp_bounds, cn_bounds=a.cn_bounds, inversion_mesh=a.inversion_mesh_n,
    )
