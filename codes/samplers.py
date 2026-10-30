# samplers.py
"""
Preconditioned Crank-Nicolson (pCN) for a N(0, I) prior on the KL coefficients,
and a Metropolis-within-Gibbs variant for the joint setting.

Step-size tuning (first n_adapt warm-up steps, then frozen): per-step Robbins-Monro update
    log s_{t+1} = log s_t + t^(-0.6) (a_t - a*),   a_t = min(1, exp(log-ratio_t)),
of the proposal scale s towards the target acceptance a* (Andrieu & Thoms 2008, Sec. 5).
"""
import time

import numpy as np


def _tune(scale, accept_prob, t, target, upper=np.inf):
    return min(upper, np.exp(np.log(scale) + (t + 1) ** -0.6 * (accept_prob - target)))


def _accept(log_ratio, rng):
    """
    Metropolis decision and acceptance probability for a log acceptance ratio.
    A non-finite ratio (failed or invalid proposal) is a rejection with probability 0.
    """
    if not np.isfinite(log_ratio):
        return False, 0.0
    a = float(np.exp(min(0.0, log_ratio)))
    return np.log(rng.random()) < log_ratio, a


def _check_scale(name, value, upper=None):
    if not (np.isfinite(value) and value > 0 and (upper is None or value <= upper)):
        raise ValueError(f"{name} must be in (0, {upper}]" if upper else f"{name} must be positive")


def _check_start(value):
    if not np.isfinite(value):
        raise ValueError("the log-target is not finite at the initial state")


def _progress(i, n, t0, every):
    if (i + 1) % every == 0:
        print(f"  {i + 1:>8d}/{n}  ({time.time() - t0:7.1f} s)", flush=True)


def pcn(loglike, z0, n_samples, n_warmup=0, n_adapt=None, beta0=0.1, adapt=True, target=0.234,
        rng=None, on_warmup_end=None):
    """
    pCN chain for pi(z) ∝ exp(loglike(z)) N(z; 0, I):
        z* = sqrt(1 - beta^2) z + beta xi,   accept w.p. min(1, exp(loglike(z*) - loglike(z))).
    The first n_warmup steps are discarded; beta is tuned during the first n_adapt <= n_warmup
    of them if adapt (default: all), otherwise kept at beta0. on_warmup_end() is called once
    when the warm-up is over (e.g. to snapshot the PDE work).

    Returns dict(z: (n_samples, d), times: seconds since the end of warm-up per draw, beta,
    acceptance: mean acceptance probability min(1, exp(log-ratio)) after warm-up, accepted_fraction:
    fraction of retained transitions actually accepted, beta_trace: beta while tuning).
    """
    rng = rng or np.random.default_rng()
    _check_scale("beta0", beta0, upper=1.0)
    n_adapt = n_warmup if n_adapt is None else min(n_adapt, n_warmup)
    z = np.array(z0, dtype=float)
    ll = loglike(z)
    _check_start(ll)
    beta = float(beta0)
    chain, times = np.empty((n_samples, z.size)), np.empty(n_samples)
    beta_trace, acc_sum, n_accepted = np.empty(n_adapt), 0.0, 0
    n_total, t0 = n_warmup + n_samples, time.time()

    for i in range(n_total):
        if i == n_warmup:
            t_start = time.perf_counter()
            if on_warmup_end is not None:
                on_warmup_end()
        z_new = np.sqrt(1.0 - beta**2) * z + beta * rng.standard_normal(z.size)
        ll_new = loglike(z_new)
        accepted, a = _accept(ll_new - ll, rng)
        if accepted:
            z, ll = z_new, ll_new

        if i < n_adapt:
            if adapt:
                beta = _tune(beta, a, i, target, upper=1.0)
            beta_trace[i] = beta
        if i >= n_warmup:
            j = i - n_warmup
            chain[j], times[j] = z, time.perf_counter() - t_start
            acc_sum += a
            n_accepted += bool(accepted)
        _progress(i, n_total, t0, every=max(1, n_total // 20))

    return dict(z=chain, times=times, beta=beta, acceptance=acc_sum / max(1, n_samples),
                accepted_fraction=n_accepted / max(1, n_samples), beta_trace=beta_trace)


def gibbs_pcn(loglike, neg_log_prior_r, z0, r0, n_samples, n_warmup=0, n_adapt=None, beta0=0.1,
              mh_scale0=0.2, adapt=True, target=0.234, rng=None, on_warmup_end=None):
    """
    Metropolis-within-Gibbs for pi(z, r) ∝ exp(loglike(z, r) - psi(r)) N(z; 0, I):
    a pCN step for z | r followed by a random-walk MH step for r | z, both tuned as in pcn().
    The log-target of the current joint state is carried across both blocks, so every
    acceptance ratio compares the same joint density.

    Returns dict(z, r, times, beta, mh_scale, acceptance: (z block, r block)).
    """
    rng = rng or np.random.default_rng()
    _check_scale("beta0", beta0, upper=1.0)
    _check_scale("mh_scale0", mh_scale0)
    n_adapt = n_warmup if n_adapt is None else min(n_adapt, n_warmup)
    z, r = np.array(z0, dtype=float), np.array(r0, dtype=float)
    ll, psi = loglike(z, r), neg_log_prior_r(r)
    _check_start(ll - psi)
    beta, mh_scale = float(beta0), float(mh_scale0)
    z_chain, r_chain, times = np.empty((n_samples, z.size)), np.empty((n_samples, r.size)), np.empty(n_samples)
    acc_sum = np.zeros(2)
    n_total, t0 = n_warmup + n_samples, time.time()

    for i in range(n_total):
        if i == n_warmup:
            t_start = time.perf_counter()
            if on_warmup_end is not None:
                on_warmup_end()
        # z | r : pCN (prior-reversible, so only the likelihood enters)
        z_new = np.sqrt(1.0 - beta**2) * z + beta * rng.standard_normal(z.size)
        ll_new = loglike(z_new, r)
        acc_z, a_z = _accept(ll_new - ll, rng)
        if acc_z:
            z, ll = z_new, ll_new

        # r | z : random-walk MH
        r_new = r + mh_scale * rng.standard_normal(r.size)
        ll_new, psi_new = loglike(z, r_new), neg_log_prior_r(r_new)
        acc_r, a_r = _accept((ll_new - psi_new) - (ll - psi), rng)
        if acc_r:
            r, ll, psi = r_new, ll_new, psi_new

        if i < n_adapt and adapt:
            beta = _tune(beta, a_z, i, target, upper=1.0)
            mh_scale = _tune(mh_scale, a_r, i, target)
        if i >= n_warmup:
            j = i - n_warmup
            z_chain[j], r_chain[j], times[j] = z, r, time.perf_counter() - t_start
            acc_sum += (a_z, a_r)
        _progress(i, n_total, t0, every=max(1, n_total // 20))

    return dict(z=z_chain, r=r_chain, times=times, beta=beta, mh_scale=mh_scale,
                acceptance=acc_sum / max(1, n_samples))
