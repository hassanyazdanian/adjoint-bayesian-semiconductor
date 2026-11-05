# run_pcn.py
"""
pCN reference sampler (Section 6.3), one chain per call.
Usage: python run_pcn.py --example 1 --setting 1 --chain 0          (tuned beta)
       python run_pcn.py --example 1 --setting 1 --chain 0 --beta 0.01 --tag _beta0.01   (fixed beta)
"""
import argparse
import time

import numpy as np
import torch

import config as cfg
from io_utils import load_data, run_identity, save_samples
from posterior import build_problem, starting_point
from samplers import pcn, gibbs_pcn


def run_pcn(data_path, setting, out_path, chain=0, k=None, n_kl=None, tau=None, mean_a=None, beta=None, n_warmup=5000,
            n_samples=300000, n_burn=30000, beta0=0.1, target_acceptance=0.234, seed=0, init_file=None):
    torch.set_num_threads(1)
    t_launch = time.time()                          # epoch seconds, before the model is built
    if n_samples < 1 or n_warmup < 0 or n_burn < 0:
        raise ValueError("n_samples must be >= 1, n_warmup and n_burn >= 0")
    data = load_data(data_path)
    post = build_problem(data, setting, k, n_kl, tau, mean_a)
    model = post.model
    rng = np.random.default_rng(seed + chain)
    theta0, init_policy = starting_point(post, rng, init_file, chain, seed + chain)
    identity = run_identity(data, init_policy)          # captured at launch, before sampling
    adapt = beta is None
    print(f"pCN chain {chain}: setting {setting}, dim(theta) = {post.dim}, N_meas = {model.n_meas}, "
          f"sigma = {post.sigma:.4e}, k = {post.prior.k}, " + ("tuned beta" if adapt else f"fixed beta = {beta}"))

    cost, attempts_log = {}, []
    def snapshot():
        cost["warmup"] = dict(model.stats, wall_time=time.perf_counter() - t0)

    def counted(loglike):
        """loglike, recording the cumulative forward-solve attempts after every call."""
        def wrapped(*args):
            try:
                return loglike(*args)
            finally:
                attempts_log.append(model.stats["forward_attempts"])
        return wrapped

    t0, t0_epoch = time.perf_counter(), time.time()
    # tuning (n_warmup steps) and burn-in (n_burn steps) are both discarded warm-up; beta keeps being
    # tuned through both, so that it is not frozen while the chain is still moving towards the posterior
    kw = dict(n_warmup=n_warmup + n_burn, n_adapt=n_warmup + n_burn, beta0=beta0 if adapt else beta, adapt=adapt,
              target=target_acceptance, rng=rng, on_warmup_end=snapshot)
    if post.known_plateaus:
        out = pcn(counted(lambda z: -post.neg_log_likelihood(z)), theta0, n_samples, **kw)
        theta = out["z"]
        info = dict(beta=out["beta"], acceptance=out["acceptance"], accepted_fraction=out["accepted_fraction"],
                    beta_trace=out["beta_trace"])
        calls_per_step = 1
    else:
        out = gibbs_pcn(counted(lambda z, r: -post.neg_log_likelihood(np.concatenate([z, r]))),
                        post.neg_log_prior_plateaus, theta0[: post.n_kl], theta0[post.n_kl:],
                        n_samples, **kw)
        theta = np.hstack([out["z"], out["r"]])
        info = dict(beta=out["beta"], mh_scale=out["mh_scale"], acceptance=out["acceptance"])
        calls_per_step = 2
    cost["total"] = dict(model.stats, wall_time=time.perf_counter() - t0)
    timing = dict(launch_epoch=t_launch, warmup_start_epoch=t0_epoch,
                  sampling_start_epoch=t0_epoch + cost["warmup"]["wall_time"], end_epoch=time.time())
    # attempts per retained draw: the cumulative count after the last likelihood call of each step
    # (one initial call, then calls_per_step per step; the last n_samples steps are retained)
    cum = np.asarray(attempts_log, dtype=float)[::calls_per_step]       # after the initial call and after each step
    per_step = np.diff(cum)
    info.update(forward_attempts_per_draw=per_step[-n_samples:], acceptance_kind="mean acceptance probability",
                timing=timing)
    print(f"pCN finished in {cost['total']['wall_time'] / 3600:.2f} h; beta = {info['beta']:.4g}, "
          f"acceptance = {info['acceptance']}")

    save_samples(out_path, post, theta, "pcn", times=out["times"], cost=cost, identity=identity, setting=setting,
                 data=data_path, chain=chain, seed=seed + chain, adapted=adapt, n_warmup=n_warmup,
                 n_burn=n_burn, target_acceptance=target_acceptance,
                 tuning="Robbins-Monro, gain t^-0.6" if adapt else "none (fixed beta, selected by a step-size sweep)",
                 init_file=init_file, theta0=np.asarray(theta0), **info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", type=int, default=1)
    ap.add_argument("--setting", type=int, default=1, help="1: known plateaus, 2: joint (Gibbs)")
    ap.add_argument("--chain", type=int, default=0, help="chain index; the seed is seed + chain")
    ap.add_argument("--data", default=None, help="data file (default: obs/obs_Ex<example>.pickle)")
    ap.add_argument("--tag", default="", help="suffix of the run folder, e.g. _k10 or _beta0.01")
    ap.add_argument("--k", type=float, default=None, help="sigmoid sharpness (default config.K_SIGMOID)")
    ap.add_argument("--n_kl", type=int, default=None, help="KL truncation (default config.N_KL)")
    ap.add_argument("--init_file", default=None, help=".npy file with one starting point per chain (default: prior draw)")
    ap.add_argument("--tau", type=float, default=None, help="Matern tau (default config.TAU)")
    ap.add_argument("--mean_a", type=float, default=None, help="trend-mean slope a (default config.PRIOR_MEAN_A; 0: zero mean)")
    ap.add_argument("--beta", type=float, default=None, help="fixed pCN step size (no tuning)")
    for key, val in cfg.PCN.items():
        ap.add_argument(f"--{key}", type=type(val), default=val)
    a = ap.parse_args()
    run_pcn(
        data_path=a.data or cfg.data_file(a.example),
        setting=a.setting,
        out_path=cfg.chain_file("pcn", a.setting, a.example, a.chain, a.tag),
        chain=a.chain, k=a.k, n_kl=a.n_kl, tau=a.tau, mean_a=a.mean_a, beta=a.beta,
        n_warmup=a.n_warmup, n_samples=a.n_samples, n_burn=a.n_burn, beta0=a.beta0,
        target_acceptance=a.target_acceptance, seed=a.seed, init_file=a.init_file,
    )
