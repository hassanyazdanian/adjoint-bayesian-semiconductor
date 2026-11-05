# postprocess.py
"""
Posterior summaries, convergence diagnostics and cost-normalised efficiency for one experiment
(Sections 5.3, 6.3, 6.4).
Usage: python postprocess.py results/nuts_Setting1_Ex1 --data obs/obs_Ex1.pickle
The argument is a run folder with chain*.pickle files, or one or more chain files.

Diagnostics follow Vehtari et al. (2021) via ArviZ: rank-normalised split R-hat, bulk and
tail ESS, and Monte Carlo standard errors, computed over all chains. Every monitored quantity
(all KL coefficients and the junction depth; c_p and c_n in the joint setting) must have finite
diagnostics: an undefined value makes the summary incomplete and the run "not passed", it is never
skipped. Divergences are assessed separately from R-hat and ESS.
The data file must be the one the chains were sampled with (SHA-256 check before any output).
"""
import argparse
import json
import os

import numpy as np
import torch
import arviz as az
import dolfin as dl

import config as cfg
from diagnostics import field_moments, interface_depth, mass_matrix, pointwise_hpd, relative_l2_error, ssim
from io_utils import chain_content_sha256, chain_prior, code_version, load_chains, load_data
from posterior import true_field
from plotting import plot_field, plot_histogram, plot_trace, TRUE_INTERFACE, MEAN_INTERFACE


def _cost(chains):
    """
    PDE work and time for the warm-up and the sampling phase, summed over the chains.
    - worker_time_s: wall-clock seconds of each chain's sampler, summed over chains (aggregate worker
      time; the chains run in parallel). Model construction and KL set-up are not included.
    - forward_solves / adjoint_solves: raw counts of successful forward and adjoint solves.
    - forward_equivalents: forward_solves + lambda * adjoint_solves, with lambda the measured ratio of
      the mean adjoint to the mean forward solve time over the whole run (stated as
      adjoint_in_forward_equivalents).
    - forward_attempts, forward_failures, forward_restarts, forward_failed_evaluations, adjoint_failures:
      only when every chain records them (the chains of the Example 1 step-size sweep do not).
    - batch: actual elapsed wall-clock time of the batch from the chains' epoch timestamps (first
      launch to last end, and from the first warm-up start), when every chain records them;
      elapsed_time_s_longest_chain is the longest single sampler run.
    """
    if any(c.get("cost") is None for c in chains):
        return None
    keys = ["forward", "adjoint", "newton_iterations", "wall_time", "forward_time", "adjoint_time"]
    extra = ["forward_attempts", "forward_failures", "forward_restarts", "forward_failed_evaluations", "adjoint_failures"]
    if all(k in c["cost"][phase] for c in chains for phase in ("total", "warmup") for k in extra):
        keys += extra
    tot = {k: sum(c["cost"]["total"][k] for c in chains) for k in keys}
    warm = {k: sum(c["cost"]["warmup"][k] for c in chains) for k in keys}
    # an adjoint solve expressed in units of an average nonlinear forward solve (measured times)
    adj_eq = (tot["adjoint_time"] / tot["adjoint"]) / (tot["forward_time"] / tot["forward"]) if tot["adjoint"] else 0.0
    out = {"adjoint_in_forward_equivalents": adj_eq,
           "newton_iterations_per_forward": tot["newton_iterations"] / tot["forward"],
           "elapsed_time_s_longest_chain": max(c["cost"]["total"]["wall_time"] for c in chains)}
    for phase, d in (("warmup", warm), ("sampling", {k: tot[k] - warm[k] for k in keys}), ("total", tot)):
        out[phase] = dict(worker_time_s=d["wall_time"], forward_solves=d["forward"], adjoint_solves=d["adjoint"],
                          newton_iterations=d["newton_iterations"],
                          forward_equivalents=d["forward"] + adj_eq * d["adjoint"],
                          **{k: d[k] for k in extra if k in d})
    timings = [c["info"].get("timing") for c in chains]
    if all(timings):
        out["batch"] = dict(
            elapsed_s_launch_to_end=max(t["end_epoch"] for t in timings) - min(t["launch_epoch"] for t in timings),
            elapsed_s_warmup_start_to_end=max(t["end_epoch"] for t in timings) - min(t["warmup_start_epoch"] for t in timings),
            setup_s_max=max(t["warmup_start_epoch"] - t["launch_epoch"] for t in timings),
            launch_spread_s=max(t["launch_epoch"] for t in timings) - min(t["launch_epoch"] for t in timings))
    return out


def _sampler_info(chains):
    """Per-chain sampler settings and statistics (step sizes, acceptance, divergences, tree depth)."""
    out = []
    for c in chains:
        i = c["info"]
        if c["method"] == "nuts" and "step_size" in i:
            g = np.asarray(i["grads_per_iteration"])
            row = dict(chain=i["chain"], mass_matrix=i["mass_matrix"], step_size=i["step_size"],
                       target_accept_prob=i["target_accept_prob"], mean_accept_prob=float(np.mean(i["accept_prob"])),
                       divergences=int(len(i["divergences"])),
                       grads_per_iteration_mean=float(g.mean()), grads_per_iteration_max=int(g.max()),
                       warmup_grads_per_iteration_mean=float(np.mean(i["warmup_grads_per_iteration"]))
                       if len(i["warmup_grads_per_iteration"]) else 0.0)
            d = np.asarray(i["tree_depth"])
            row.update(tree_depth_mean=float(d.mean()), tree_depth_max=int(d.max()),
                       fraction_at_max_tree_depth=float(np.mean(d >= i["max_tree_depth"])))
            out.append(row)
        elif c["method"] == "pcn" and "beta" in i:
            out.append({key: i[key] for key in ("chain", "beta", "acceptance", "adapted", "target_acceptance",
                                                "tuning", "n_warmup", "n_burn", "mh_scale") if key in i})
    return out


R_HAT_MAX, ESS_MIN, DIVERGENCE_FRACTION = 1.01, 400, 0.005     # convergence and divergence thresholds


def _divergences(chains, depth, z, out_dir):
    """
    Where the divergent NUTS transitions are (depth percentiles, Mahalanobis percentiles over all
    coefficients, pair plot of the leading coefficients) and whether they need follow-up: more than
    0.5 % of the draws, or at least 3 divergences of which 75 % or more are in one chain. Concentration
    in a region of the parameter space is judged from the locations reported here.
    """
    div = np.zeros(depth.shape, dtype=bool)
    for j, c in enumerate(chains):
        idx = np.asarray(c["info"].get("divergences", []), dtype=int)
        div[j, idx[idx < depth.shape[1]]] = True
    count, per_chain = int(div.sum()), div.sum(1).tolist()
    fraction = count / div.size
    concentrated = count >= 3 and max(per_chain) >= 0.75 * count
    out = {"count": count, "per_chain": per_chain, "fraction": fraction,
           "follow_up": bool(fraction > DIVERGENCE_FRACTION or concentrated),
           "follow_up_rule": "fraction > 0.005, or >= 3 divergences with >= 75 % in one chain"}
    if not div.any():
        return out
    pct = [float((depth < d).mean()) for d in depth[div]]
    zz = z.reshape(-1, z.shape[-1])
    prec = np.linalg.pinv(np.atleast_2d(np.cov(zz.T)))
    d2 = np.einsum("ij,jk,ik->i", zz - zz.mean(0), prec, zz - zz.mean(0))
    maha = [float(np.mean(d2 < v)) for v in d2[div.ravel()]]
    idata = az.from_dict(posterior={"z": z[:, :, :4], "depth": depth}, sample_stats={"diverging": div})
    axes = az.plot_pair(idata, var_names=["z", "depth"], divergences=True, kind="scatter",
                        scatter_kwargs={"alpha": 0.3, "s": 4})
    np.ravel(axes)[0].figure.savefig(os.path.join(out_dir, "divergences_pair.png"), dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close("all")
    out.update({"depth_percentiles_of_divergent_draws": [round(p, 3) for p in pct],
                "fraction_in_depth_tails_(<5% or >95%)": float(np.mean([(p < 0.05) or (p > 0.95) for p in pct])),
                "mahalanobis_percentiles_of_divergent_draws": [round(p, 3) for p in maha],
                "note": "the stored draw of a divergent iteration need not be where the integration failed"})
    return out


def _reduce(values, fn):
    """fn over all values if every value is finite, else None (an undefined diagnostic is never skipped)."""
    v = np.asarray(values, dtype=float)
    return float(fn(v)) if v.size and np.all(np.isfinite(v)) else None


def _finite_or_none(x):
    return float(x) if np.isfinite(x) else None


def postprocess(paths, data_path, out_dir, max_field_samples=5000, batch=5000, x_probe=0.5, cred_mass=0.95):
    data, chains = load_data(data_path), load_chains(paths)
    data_sha = data["_source"]["sha256"]
    used = {(c.get("provenance") or {}).get("data_sha256") for c in chains}
    if used != {data_sha}:
        raise ValueError(f"{data_path} (SHA-256 {data_sha[:12]}...) is not the data the chains were sampled with "
                         f"({', '.join(sorted(str(u)[:12] + '...' for u in used))}); nothing was written")
    os.makedirs(out_dir, exist_ok=True)
    n_chains, n = len(chains), len(chains[0]["z"])
    z = np.stack([c["z"] for c in chains])                  # (chains, draws, N_KL)
    cp, cn = np.stack([c["cp"] for c in chains]), np.stack([c["cn"] for c in chains])
    joint = chains[0]["r"] is not None

    # the inversion mesh the chains were sampled on (chains that do not record it used the data file's)
    n_inv = int(chains[0]["info"].get("inversion_mesh_n", data["mesh_n"]))
    mesh = dl.UnitSquareMesh(dl.MPI.comm_self, n_inv, n_inv)
    V = dl.FunctionSpace(mesh, "CG", 1)
    if chains[0].get("kl_basis") is not None and chains[0]["kl_basis"].shape[0] != V.dim():
        raise ValueError(f"the stored KL basis has {chains[0]['kl_basis'].shape[0]} rows, the {n_inv} x {n_inv} mesh {V.dim()} dofs")
    prior = chain_prior(chains[0], V)     # the prior the chains were sampled with

    zf, cpf, cnf = z.reshape(-1, z.shape[-1]), cp.ravel(), cn.ravel()

    def fields(idx):
        with torch.no_grad():
            return prior.pushforward(torch.from_numpy(zf[idx]), torch.from_numpy(cpf[idx])[:, None],
                                     torch.from_numpy(cnf[idx])[:, None]).numpy()

    # One streaming pass over all draws: field moments (Section 5.3) and interface depth
    depth, n_cross = [], []

    def field_batches():
        for i in range(0, len(zf), batch):
            C = fields(np.arange(i, min(i + batch, len(zf))))
            d, nc = interface_depth(V, C, x0=x_probe, level=0.0, clip=True)
            depth.append(d)
            n_cross.append(nc)
            yield C

    c_mean, c_std = field_moments(field_batches())
    depth = np.concatenate(depth).reshape(n_chains, n)
    n_cross = np.concatenate(n_cross).reshape(n_chains, n)
    sub = np.unique(np.linspace(0, len(zf) - 1, min(len(zf), max_field_samples)).astype(int))
    hpd_low, hpd_high = pointwise_hpd(fields(sub), cred_mass)

    # Convergence diagnostics (ArviZ)
    posterior = {"z": z, "depth": depth}
    if joint:
        posterior.update(cp=cp, cn=cn)
    diag = az.summary(az.convert_to_dataset(posterior), kind="diagnostics", round_to="none")
    diag.to_csv(os.path.join(out_dir, "diagnostics.csv"))
    zrows = diag.loc[[r for r in diag.index if r.startswith("z[")]]
    finite = np.isfinite(diag[["r_hat", "ess_bulk", "ess_tail"]].to_numpy(dtype=float)).all(axis=1)
    undefined = [r for r, ok in zip(diag.index, finite) if not ok]
    r_hat_all, bulk_all, tail_all = (_reduce(diag.r_hat, np.max), _reduce(diag.ess_bulk, np.min),
                                     _reduce(diag.ess_tail, np.min))
    convergence = {
        "monitored": list(diag.index), "undefined": undefined, "complete": not undefined,
        "thresholds": {"r_hat_below": R_HAT_MAX, "ess_bulk_at_least": ESS_MIN, "ess_tail_at_least": ESS_MIN},
        "r_hat_max": r_hat_all, "ess_bulk_min": bulk_all, "ess_tail_min": tail_all,
        "passes": bool(not undefined and r_hat_all < R_HAT_MAX and bulk_all >= ESS_MIN and tail_all >= ESS_MIN),
        "note": "divergences are assessed separately (see 'divergences')",
    }

    c_true = true_field(data, V)
    chain_hashes = [chain_content_sha256(c) for c in chains]
    summary = {
        "files": [c["file"] for c in chains],
        "chain_content_sha256": chain_hashes,
        "data_path": os.path.abspath(data_path),
        "data_sha256": data_sha,
        "chain_provenance": {
            "code": sorted({json.dumps((c.get("provenance") or {}).get("code"), sort_keys=True) for c in chains}),
            "init_policy": [(c.get("provenance") or {}).get("init_policy") for c in chains],
        },
        "postprocess_code": code_version(),
        "convergence": convergence,
        "method": chains[0]["method"],
        "inversion_mesh_n": n_inv,
        "n_chains": n_chains,
        "draws_per_chain": n,
        "RE": relative_l2_error(mass_matrix(V), c_mean, c_true),
        "SSIM": ssim(V, c_mean, c_true),
        "SSIM_gaussian": ssim(V, c_mean, c_true, gaussian=True),
        "z": dict(r_hat_max=_reduce(zrows.r_hat, np.max), ess_bulk_min=_reduce(zrows.ess_bulk, np.min),
                  ess_bulk_median=_reduce(zrows.ess_bulk, np.median), ess_tail_min=_reduce(zrows.ess_tail, np.min)),
        "depth_convention": (f"lowest crossing of c = 0 on x = {x_probe}; without a crossing d = 0 "
                             f"(n-type column) or 1 (p-type column)"),
        "depth_without_crossing": int((n_cross == 0).sum()),
        "depth_multiple_crossings": int((n_cross > 1).sum()),
        "depth_mean": float(depth.mean()),
        "depth_std": float(depth.std(ddof=1)),
    }
    for name in ["depth"] + (["cp", "cn"] if joint else []):
        row = diag.loc[name]
        summary[name] = dict(r_hat=_finite_or_none(row.r_hat), ess_bulk=_finite_or_none(row.ess_bulk),
                             ess_tail=_finite_or_none(row.ess_tail), mcse_mean=_finite_or_none(row.mcse_mean))
    if joint:
        ok = n_cross > 0
        for name, x in (("cp", cp), ("cn", cn)):
            lo, hi = az.hdi(x.ravel(), hdi_prob=cred_mass)
            summary[name].update(mean=float(x.mean()), std=float(x.std(ddof=1)), hdi=[float(lo), float(hi)],
                                 corr_with_depth=float(np.corrcoef(x[ok], depth[ok])[0, 1]))

    # Cost-normalised efficiency
    cost = _cost(chains)
    summary["sampler"] = _sampler_info(chains)
    if chains[0]["method"] == "nuts":
        summary["divergences"] = _divergences(chains, depth, z, out_dir)
    if cost is not None:
        summary["cost"] = cost
        ess = {"z_min_bulk": summary["z"]["ess_bulk_min"], "z_median_bulk": summary["z"]["ess_bulk_median"],
               "depth_bulk": summary["depth"]["ess_bulk"]}
        if joint:
            ess.update(cp_bulk=summary["cp"]["ess_bulk"], cn_bulk=summary["cn"]["ess_bulk"])
        summary["efficiency"] = {            # None where the ESS is undefined
            key: None if e is None else dict(
                ess=e,
                per_worker_second_sampling=e / cost["sampling"]["worker_time_s"],
                per_worker_second_total=e / cost["total"]["worker_time_s"],
                per_1000_forward_equivalents_sampling=1e3 * e / cost["sampling"]["forward_equivalents"],
                per_1000_forward_equivalents_total=1e3 * e / cost["total"]["forward_equivalents"])
            for key, e in ess.items()}

    np.savez(os.path.join(out_dir, "fields.npz"), c_mean=c_mean, c_std=c_std, hpd_low=hpd_low,
             hpd_high=hpd_high, depth=depth, depth_crossings=n_cross,
             chain_content_sha256=np.array(chain_hashes), data_sha256=np.array(data_sha))

    # Figures
    plot_field(V, c_mean, os.path.join(out_dir, "C_mean.png"), cbar_label=r"$\mathbb{E}[c(\boldsymbol{x}) \mid y]$",
               contours=[(c_true, TRUE_INTERFACE), (c_mean, MEAN_INTERFACE)])
    plot_field(V, c_std, os.path.join(out_dir, "C_std.png"), cmap="PuRd",
               cbar_label=r"$\mathrm{std}(c(\boldsymbol{x}) \mid y)$", contours=[(c_true, TRUE_INTERFACE)])
    thin = max(1, n // 1000)                    # at most ~1000 points per trace, for every sampler
    plot_trace(z[0, ::thin], os.path.join(out_dir, "trace.png"), x=np.arange(0, n, thin),
               xlabel=f"Iteration (every {thin}th draw shown)" if thin > 1 else "Iteration")
    if chains[0].get("times") is not None:
        plot_trace(z[0, ::thin], os.path.join(out_dir, "trace_time.png"), x=chains[0]["times"][::thin] / 3600,
                   xlabel="Wall-clock time after warm-up (h)")
    if joint:
        plot_histogram(cp.ravel(), os.path.join(out_dir, "hist_cp.png"), r"$c_{\mathrm{p}}$", data["cp_true"])
        plot_histogram(cn.ravel(), os.path.join(out_dir, "hist_cn.png"), r"$c_{\mathrm{n}}$", data["cn_true"])

    # summary.json last and atomically: a complete summary newer than fields.npz shows that this
    # invocation finished
    tmp = os.path.join(out_dir, "summary.json.tmp")
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    os.replace(tmp, os.path.join(out_dir, "summary.json"))

    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2, default=float))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run folder with chain*.pickle, or chain files")
    ap.add_argument("--data", required=True, help="data pickle used for the run (for c_true)")
    ap.add_argument("--out", default=None, help="output folder (default: results/<run name>_postprocess)")
    ap.add_argument("--max_field_samples", type=int, default=5000)
    a = ap.parse_args()
    name = os.path.basename(os.path.normpath(a.paths[0])).replace(".pickle", "")
    out = a.out or os.path.join(cfg.RESULTS_DIR, name + "_postprocess")
    postprocess(a.paths, a.data, out, max_field_samples=a.max_field_samples)
