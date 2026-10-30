# joint_report.py
"""
Report of the joint reconstruction (Example 1, Setting 2: KL coefficients and plateau values c_p, c_n, paper
Section 6.4) from two NUTS runs: Stage A (the paper's plateau intervals, config.CP_BOUNDS and config.CN_BOUNDS) and
Stage B (intervals twice as wide, same centres). For every stage the chains, the postprocessing output (summary.json,
fields.npz, diagnostics.csv) and the data must belong together, and all chains must have the same plateau bounds.

Outputs in --out:
- tables.json, tables.md: diagnostics over all monitored quantities (KL coefficients, c_p, c_n, d); divergences
  with chain and location; step sizes, tree depth, acceptance statistic; cost; RE and SSIM of the posterior mean;
  posterior of c_p, c_n and d (mean, sd, 5/50/95 % quantiles with MCSE; whether the 90 % interval contains the
  truth); for c_p and c_n the posterior/prior sd, the mass near each bound and, for a wider stage, the mass
  outside the paper's interval; corr(c_p, d), corr(c_n, d), corr(c_p, c_n); changes from the first to the second
  stage in combined MCSE units, and the relative L2 change of the mean and std fields; data fit: normalized
  misfit ||F(theta) - y||^2 / (sigma^2 N_meas) on the inversion mesh at 200 retained draws (evenly spaced, 50
  per chain) of each stage and of the known-plateau NUTS posterior (--known_plateau_run), and its value
  for the true field on the inversion mesh.
- joint_mean_A.png, joint_std_A.png: posterior mean and std of the first stage; true interface solid black,
  inferred interface (c = 0 of the mean) dashed red.
- joint_marginals.png: marginal posteriors of c_p, c_n and d of both stages, prior supports and true values.
- joint_pairs.png: draws of (c_p, d) and (c_n, d) of both stages.
- joint_traces.png: four-chain traces of d, c_p and c_n against the PDE solves after warm-up, divergences marked.
Definitions as in the paper (Section 6.3): rank-normalized split R-hat, bulk and tail ESS (ArviZ); minimum ESS =
smallest bulk or tail ESS over all monitored quantities; forward-equivalent solves N_F + lambda N_A with the run's
own lambda (ratio of mean adjoint to mean forward solve time); MCSE of means, sds, quantiles and masses (ArviZ).
Usage: python joint_report.py --run_a results/nuts_Setting2_Ex1_A --run_b results/nuts_Setting2_Ex1_B --out results/joint
       [--known_plateau_run results/nuts_Setting1_Ex1] [--data obs/obs_Ex1.pickle]
"""
import argparse
import json
import os

import arviz as az
import dolfin as dl
import numpy as np
import pandas as pd

from diagnostics import interface_depth, mass_matrix
from forward_model import PoissonBoltzmannModel
import config as cfg
from io_utils import chain_content_sha256, chain_prior, code_version, load_chains, load_data
from plotting import CHAIN_COLORS, MEAN_INTERFACE, TRUE_INTERFACE, plot_field, plot_trace_chains
from posterior import Posterior
import matplotlib.pyplot as plt                       # noqa: E402  (backend set by plotting)

DATA_FIT_DRAWS = 200           # retained draws per posterior for the data-fit check, evenly spaced over the chains
PROBS = (0.05, 0.5, 0.95)
STATS = ("mean", "sd") + tuple(f"q{round(100 * p):02d}" for p in PROBS)
NEAR = 0.05                    # "near a bound": within 5 % of the interval width
QUANT = ("cp", "cn", "depth")
TEX = {"cp": r"$c_{\mathrm{p}}$", "cn": r"$c_{\mathrm{n}}$", "depth": r"$d$"}
MD = {"cp": "c_p", "cn": "c_n", "depth": "d"}


def _q(q):
    """Quantity name of diagnostics.csv for text: z[3] -> z_4."""
    return f"z_{int(q[2:-1]) + 1}" if q.startswith("z[") else MD.get(q, q)


def load_stage(run, name, data_sha):
    pp = run + "_postprocess"
    with open(f"{pp}/summary.json") as f:
        s = json.load(f)
    fields = np.load(f"{pp}/fields.npz")
    chains = load_chains(run)
    if not ([chain_content_sha256(c) for c in chains] == s["chain_content_sha256"]
            == [str(h) for h in fields["chain_content_sha256"]] and s["data_sha256"] == str(fields["data_sha256"]) == data_sha):
        raise ValueError(f"{run}: chains, fields.npz, summary.json and the data do not belong together")
    if any(c["r"] is None for c in chains):
        raise ValueError(f"{run}: not a joint (Setting 2) run")
    b = chains[0]["plateau_bounds"]
    bounds = {"cp": tuple(map(float, b["cp"])), "cn": tuple(map(float, b["cn"]))}
    seed = chains[0]["info"]["seed"] - chains[0]["info"]["chain"]
    for c in chains:
        if c.get("plateau_bounds") != bounds:
            raise ValueError(f"{c['file']}: plateau bounds {c.get('plateau_bounds')} instead of {bounds}")
        if c["info"].get("seed") != seed + c["info"]["chain"]:
            raise ValueError(f"{c['file']}: seed {c['info'].get('seed')} instead of {seed + c['info']['chain']}")
    with open(run + "_batch.json") as f:
        batch = json.load(f)
    diag = pd.read_csv(f"{pp}/diagnostics.csv", index_col=0)
    draws = dict(cp=np.stack([c["cp"] for c in chains]), cn=np.stack([c["cn"] for c in chains]),
                 depth=np.asarray(fields["depth"], dtype=float))
    return dict(name=name, label=name.split("_")[-1], run=run, pp=pp, summary=s, fields=fields, chains=chains,
                diag=diag, batch=batch, bounds=bounds, seed=seed, draws=draws)


def _mcse(x, **kw):
    return float(az.mcse(az.convert_to_dataset({"x": x}), **kw)["x"])


def statistics(x):
    """Mean, sd and the 5/50/95 % quantiles of x (chains, draws) with their MCSE."""
    out = dict(mean=float(x.mean()), mcse_mean=_mcse(x, method="mean"), sd=float(x.std(ddof=1)), mcse_sd=_mcse(x, method="sd"))
    for p in PROBS:
        k = f"q{round(100 * p):02d}"
        out[k], out["mcse_" + k] = float(np.quantile(x, p)), _mcse(x, method="quantile", prob=p)
    return out


def _mass(ind):
    """Posterior mass of an indicator (chains, draws) with its MCSE (undefined if the indicator is constant)."""
    ind = ind.astype(float)
    return dict(value=float(ind.mean()), mcse=_mcse(ind, method="mean") if 0 < ind.mean() < 1 else None)


def diagnostics(st):
    d = st["diag"][["r_hat", "ess_bulk", "ess_tail"]].astype(float)
    b, t = d.ess_bulk, d.ess_tail
    kind = "bulk" if b.min() <= t.min() else "tail"
    conv = st["summary"]["convergence"]
    return dict(monitored=[_q(q) for q in d.index], complete=bool(np.isfinite(d.to_numpy()).all()),
                r_hat_max=float(d.r_hat.max()), r_hat_max_quantity=_q(d.r_hat.idxmax()),
                ess_bulk_min=float(b.min()), ess_bulk_min_quantity=_q(b.idxmin()),
                ess_tail_min=float(t.min()), ess_tail_min_quantity=_q(t.idxmin()),
                min_ess=float(min(b.min(), t.min())), min_ess_quantity=_q((b if kind == "bulk" else t).idxmin()), min_ess_kind=kind,
                per_quantity={_q(q): dict(r_hat=float(r.r_hat), ess_bulk=float(r.ess_bulk), ess_tail=float(r.ess_tail))
                              for q, r in d.iterrows()},
                passes=bool(conv["passes"]), thresholds=conv["thresholds"])


def divergences(st):
    s, dr = st["summary"]["divergences"], st["draws"]
    pooled = {q: np.sort(dr[q].ravel()) for q in QUANT}
    per_chain = []
    for i, c in enumerate(st["chains"]):
        idx = np.asarray(c["info"].get("divergences", []), dtype=int)
        idx = idx[idx < dr["depth"].shape[1]]
        loc = [dict(draw=int(j), **{MD[q]: float(dr[q][i, j]) for q in QUANT},
                    **{f"{MD[q]}_percentile": float(np.searchsorted(pooled[q], dr[q][i, j]) / pooled[q].size) for q in QUANT})
               for j in idx]
        per_chain.append(dict(chain=int(c["info"]["chain"]), count=int(idx.size), locations=loc))
    near = {q: [] for q in ("cp", "cn")}
    for ch in per_chain:
        for l in ch["locations"]:
            for q in near:
                a, b = st["bounds"][q]
                near[q].append(bool(l[MD[q]] < a + NEAR * (b - a) or l[MD[q]] > b - NEAR * (b - a)))
    return dict(count=s["count"], fraction=s["fraction"], per_chain_count=s["per_chain"], follow_up=s["follow_up"],
                follow_up_rule=s["follow_up_rule"], per_chain=per_chain,
                fraction_near_a_plateau_bound={MD[q]: (float(np.mean(v)) if v else None) for q, v in near.items()},
                depth_percentiles=s.get("depth_percentiles_of_divergent_draws", []),
                note=s.get("note"))


def sampler(st):
    rows = st["summary"]["sampler"]
    keys = ("chain", "step_size", "mean_accept_prob", "tree_depth_mean", "tree_depth_max", "fraction_at_max_tree_depth",
            "grads_per_iteration_mean", "warmup_grads_per_iteration_mean")
    return [{k: r.get(k) for k in keys} for r in rows]


def cost(st, min_ess):
    c, wall = st["summary"]["cost"], st["batch"]["elapsed_s"]
    pde = {ph: c[ph]["forward_solves"] + c[ph]["adjoint_solves"] for ph in ("warmup", "sampling", "total")}
    fe = {ph: c[ph]["forward_equivalents"] for ph in ("sampling", "total")}
    return dict(lambda_adjoint_over_forward=c["adjoint_in_forward_equivalents"],
                gradients={ph: c[ph]["adjoint_solves"] for ph in ("warmup", "sampling", "total")},
                forward_solves={ph: c[ph]["forward_solves"] for ph in ("warmup", "sampling", "total")},
                pde_solves=pde, forward_equivalents=fe, warmup_share_of_pde_solves=pde["warmup"] / pde["total"],
                wall_clock_s=wall, batch=dict(launch_epoch=st["batch"]["launch_epoch"], end_epoch=st["batch"]["end_epoch"],
                                              load_start=st["batch"].get("load_start"), load_end=st["batch"].get("load_end")),
                min_ess_per_1000_forward_equivalents=dict(sampling=1e3 * min_ess / fe["sampling"], total=1e3 * min_ess / fe["total"]),
                min_ess_per_wall_clock_s=min_ess / wall)


def plateaus(st, paper):
    out = {}
    for q in ("cp", "cn"):
        x, (a, b) = st["draws"][q], st["bounds"][q]
        w = b - a
        prior_sd = w / np.sqrt(12.0)
        r = dict(bounds=[a, b], prior_sd=prior_sd, posterior_sd=float(x.std(ddof=1)),
                 sd_ratio=float(x.std(ddof=1)) / prior_sd, near_band_width=NEAR * w,
                 mass_near_lower=_mass(x < a + NEAR * w), mass_near_upper=_mass(x > b - NEAR * w), prior_mass_near_each_bound=NEAR)
        pa, pb = paper[q]
        if (a, b) != (pa, pb):
            r.update(paper_interval=[pa, pb], mass_outside_paper_interval=_mass((x < pa) | (x > pb)),
                     mass_below_paper_interval=_mass(x < pa), mass_above_paper_interval=_mass(x > pb),
                     prior_mass_outside_paper_interval=1.0 - (pb - pa) / w)
        out[MD[q]] = r
    return out


def stage_record(st, truth, paper):
    dg = diagnostics(st)
    dr = st["draws"]
    post = {}
    for q in QUANT:
        s = statistics(dr[q])
        s.update(truth=truth[q], interval90=[s["q05"], s["q95"]], interval90_contains_truth=bool(s["q05"] <= truth[q] <= s["q95"]))
        post[MD[q]] = s
    flat = {q: dr[q].ravel() for q in QUANT}
    corr = {"c_p,d": float(np.corrcoef(flat["cp"], flat["depth"])[0, 1]), "c_n,d": float(np.corrcoef(flat["cn"], flat["depth"])[0, 1]),
            "c_p,c_n": float(np.corrcoef(flat["cp"], flat["cn"])[0, 1])}
    s = st["summary"]
    # draws without a crossing get d = 0 (n-type column) or d = 1 (p-type column), see summary.json "depth_convention"
    d = dr["depth"]
    split = dict(d0=int((d == 0).sum()), d1=int((d == 1).sum()), per_chain=[int(((x == 0) | (x == 1)).sum()) for x in d])
    if split["d0"] + split["d1"] != s["depth_without_crossing"]:
        raise ValueError(f"{st['run']}: {split['d0']} + {split['d1']} draws at d = 0 or 1, but {s['depth_without_crossing']} without a crossing")
    return dict(stage=st["name"], run=st["run"], seed=st["seed"], plateau_bounds={MD[k]: list(v) for k, v in st["bounds"].items()},
                chains=s["n_chains"], draws_per_chain=s["draws_per_chain"], chain_content_sha256=s["chain_content_sha256"],
                chain_code=s["chain_provenance"]["code"], diagnostics=dg, divergences=divergences(st), sampler=sampler(st),
                cost=cost(st, dg["min_ess"]), RE=s["RE"], SSIM=s["SSIM"], posterior=post, plateaus=plateaus(st, paper),
                correlations=corr, depth_without_crossing=s["depth_without_crossing"], depth_without_crossing_split=split,
                depth_convention=s["depth_convention"], depth_multiple_crossings=s["depth_multiple_crossings"])


def changes(ra, rb, fa, fb, M):
    out = {}
    for q in QUANT:
        a, b = ra["posterior"][MD[q]], rb["posterior"][MD[q]]
        out[MD[q]] = {k: dict(difference=b[k] - a[k], z=(b[k] - a[k]) / float(np.hypot(a["mcse_" + k], b["mcse_" + k])))
                      for k in STATS}
    l2 = lambda u: float(np.sqrt(u @ (M @ u)))
    out["fields"] = dict(mean_relative_L2=l2(fb["c_mean"] - fa["c_mean"]) / l2(fa["c_mean"]),
                         std_relative_L2=l2(fb["c_std"] - fa["c_std"]) / l2(fa["c_std"]))
    return out


# ---- data fit -------------------------------------------------------------------------------------------

def _model(data):
    return PoissonBoltzmannModel(data["mesh_n"], data["eps"], data["delta"], data["electrodes"])


def _normalized_misfit(y, data):
    r = np.asarray(y) - data["y_obs"]
    return float(r @ r) / (data["sigma"] ** 2 * r.size)


def data_fit(chains, data, n=DATA_FIT_DRAWS):
    """
    Normalized misfit ||F(theta) - y||^2 / (sigma^2 N_meas) on the inversion mesh at n retained draws, n / (number
    of chains) evenly spaced draws of each chain (first and last included), with the prior (stored KL basis and
    mean) and plateau setting the chains were sampled with. A draw whose forward solve fails counts as failed.
    """
    if n % len(chains):
        raise ValueError(f"{n} draws cannot be split evenly over {len(chains)} chains")
    model = _model(data)
    c0 = chains[0]
    if c0["r"] is None:
        kw = dict(plateaus=(data["cp_true"], data["cn_true"]))
    else:
        kw = dict(cp_bounds=c0["plateau_bounds"]["cp"], cn_bounds=c0["plateau_bounds"]["cn"])
    post = Posterior(model, chain_prior(c0, model.V), data["y_obs"], data["sigma"], **kw)
    values, picks = [], []
    for c in chains:
        idx = np.linspace(0, len(c["z"]) - 1, n // len(chains)).round().astype(int)
        for j in idx:
            theta = c["z"][j] if c["r"] is None else np.concatenate([c["z"][j], c["r"][j]])
            cp, cn = (float(v) for v in post.plateaus(theta))
            if abs(cp - c["cp"][j]) > 1e-12 or abs(cn - c["cn"][j]) > 1e-12:
                raise ValueError(f"{c['file']}, draw {j}: plateaus ({cp}, {cn}) instead of the stored ({c['cp'][j]}, {c['cn'][j]})")
            values.append(2.0 * post.neg_log_likelihood(theta) / model.n_meas)
        picks.append(dict(chain=int(c["info"]["chain"]), draws=idx.tolist()))
    v = np.asarray(values)
    ok = v[np.isfinite(v)]
    return dict(n_draws=int(v.size), failed_solves=int(v.size - ok.size), median=float(np.median(ok)),
                q05=float(np.quantile(ok, 0.05)), q95=float(np.quantile(ok, 0.95)), mean=float(ok.mean()),
                min=float(ok.min()), max=float(ok.max()), selection=picks, values=v.tolist())


def known_plateau_run(run, data_sha):
    """Chains of the known-plateau posterior, checked against their postprocessing output and the data."""
    chains = load_chains(run)
    with open(run + "_postprocess/summary.json") as f:
        s = json.load(f)
    if [chain_content_sha256(c) for c in chains] != s["chain_content_sha256"] or s["data_sha256"] != data_sha:
        raise ValueError(f"{run}: chains, summary.json and the data do not belong together")
    if any(c["r"] is not None for c in chains):
        raise ValueError(f"{run}: not a known-plateau (Setting 1) run")
    return chains, s


def data_fit_truth(data):
    """The normalized misfit of the true field on the inversion mesh (forward map of the inversion mesh), and, for
    reference, of the noise-free data on the data mesh (the noise realisation alone) and the model error."""
    y_model = _model(data).forward(data["c_true"], data["cp_true"], data["cn_true"])
    stored = data["y_true"] + data["model_error"]           # the same map, evaluated when the data were generated
    return dict(true_field_inversion_mesh=_normalized_misfit(y_model, data),
                noise_free_data_mesh=_normalized_misfit(data["y_true"], data),
                model_error_rms_sigma=float(np.sqrt(np.mean(data["model_error"] ** 2)) / data["sigma"]),
                check_max_abs_difference_from_data_file_sigma=float(np.abs(y_model - stored).max() / data["sigma"]),
                inversion_mesh_n=int(data["mesh_n"]), data_mesh_n=int(data["data_mesh_n"]), n_meas=int(y_model.size))


# ---- figures ------------------------------------------------------------------------------------------

def _solves(ch):
    return np.cumsum(np.asarray(ch["info"]["evals_per_iteration"]) + np.asarray(ch["info"]["grads_per_iteration"]))


def figures(stages, recs, V, truth, c_true, out):
    A = stages[0]
    fa = A["fields"]
    c_mean = fa["c_mean"]
    vlim = (min(truth["cp"], float(c_mean.min())), max(truth["cn"], float(c_mean.max())))
    plot_field(V, c_mean, os.path.join(out, f"joint_mean_{A['label']}.png"), cbar_label=r"$\mathbb{E}[c(\boldsymbol{x}) \mid y]$",
               vlim=vlim, contours=[(c_true, TRUE_INTERFACE), (c_mean, MEAN_INTERFACE)])
    plot_field(V, fa["c_std"], os.path.join(out, f"joint_std_{A['label']}.png"), cmap="PuRd",
               cbar_label=r"$\mathrm{std}(c(\boldsymbol{x}) \mid y)$", contours=[(c_true, TRUE_INTERFACE), (c_mean, MEAN_INTERFACE)])

    colors = {st["name"]: CHAIN_COLORS[i] for i, st in enumerate(stages)}
    # marginals
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.6))
    for ax, q in zip(axs, QUANT):
        lo = min(float(st["draws"][q].min()) for st in stages)
        hi = max(float(st["draws"][q].max()) for st in stages)
        bins = np.linspace(lo, hi, 61)
        for st in stages:
            lab = f"Stage {st['label']}"
            if q != "depth":
                a, b = st["bounds"][q]
                lab += rf": $\mathcal{{U}}({a:g}, {b:g})$"
                for v in (a, b):
                    ax.axvline(v, color=colors[st["name"]], linestyle=":", linewidth=1.2)
            ax.hist(st["draws"][q].ravel(), bins=bins, density=True, histtype="step", linewidth=1.6, color=colors[st["name"]], label=lab)
        ax.axvline(truth[q], color="black", linestyle="--", linewidth=1.0, label="true value")
        ax.set_xlabel(TEX[q], fontsize=13)
        ax.set_ylabel("Density", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "joint_marginals.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # pair plots (c_p, d) and (c_n, d)
    fig, axs = plt.subplots(len(stages), 2, figsize=(9.0, 3.4 * len(stages)), sharex="col", sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for i, (st, rec) in enumerate(zip(stages, recs)):
        n = st["draws"]["depth"].size
        sel = rng.choice(n, size=min(n, 4000), replace=False)
        for j, q in enumerate(("cp", "cn")):
            ax = axs[i, j]
            ax.scatter(st["draws"][q].ravel()[sel], st["draws"]["depth"].ravel()[sel], s=3, alpha=0.3, color=colors[st["name"]],
                       rasterized=True)
            for v in st["bounds"][q]:
                ax.axvline(v, color=colors[st["name"]], linestyle=":", linewidth=1.2)
            ax.axvline(truth[q], color="black", linestyle="--", linewidth=0.8)
            ax.axhline(truth["depth"], color="black", linestyle="--", linewidth=0.8)
            ax.set_title(f"Stage {st['label']}: corr({TEX[q]}, $d$) $= {rec['correlations'][MD[q] + ',d']:.2f}$", fontsize=11)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            if i == len(stages) - 1:
                ax.set_xlabel(TEX[q], fontsize=13)
            if j == 0:
                ax.set_ylabel(r"junction depth $d$", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "joint_pairs.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # traces of d, c_p, c_n against the PDE solves after warm-up (rows in this order)
    rows = ("depth", "cp", "cn")
    cols = []
    for st in stages:
        xs, ys, marks = [], [], []
        for i, ch in enumerate(st["chains"]):
            x = _solves(ch) / 1e3
            y = np.column_stack([st["draws"][q][i] for q in rows])
            div = np.asarray(ch["info"].get("divergences", []), dtype=int)
            div = div[div < len(x)]
            xs.append(x)
            ys.append(y)
            marks.append((x[div], y[div]))
        a_p, b_p = st["bounds"]["cp"]
        a_n, b_n = st["bounds"]["cn"]
        cols.append(dict(title=rf"Stage {st['label']}: $c_{{\mathrm{{p}}}}\sim\mathcal{{U}}({a_p:g}, {b_p:g})$, "
                               rf"$c_{{\mathrm{{n}}}}\sim\mathcal{{U}}({a_n:g}, {b_n:g})$", x=xs, y=ys, marks=marks))
    plot_trace_chains(cols, os.path.join(out, "joint_traces.png"), xlabel=r"PDE solves per chain after warm-up ($\times 10^3$)",
                      row_labels=[r"junction depth $d$", TEX["cp"], TEX["cn"]], true_values=[truth[q] for q in rows])


# ---- tables ---------------------------------------------------------------------------------------------

def md_report(recs, ch, fit):
    names = [f"Stage {r['stage'].split('_')[-1]}" for r in recs]
    L = ["# Joint reconstruction, Example 1", "",
         "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(recs)]

    def row(name, f):
        L.append(f"| {name} | " + " | ".join(f(r) for r in recs) + " |")

    row("Run", lambda r: f"`{r['run']}` (seeds {r['seed']}-{r['seed'] + r['chains'] - 1})")
    row("Prior c_p / c_n", lambda r: f"U({r['plateau_bounds']['c_p'][0]:g}, {r['plateau_bounds']['c_p'][1]:g}) / "
                                     f"U({r['plateau_bounds']['c_n'][0]:g}, {r['plateau_bounds']['c_n'][1]:g})")
    row("Meets the criteria", lambda r: "yes" if r["diagnostics"]["passes"] else "**no**")
    row("max R̂ (quantity)", lambda r: f"{r['diagnostics']['r_hat_max']:.4f} ({r['diagnostics']['r_hat_max_quantity']})")
    row("min bulk ESS (quantity)", lambda r: f"{r['diagnostics']['ess_bulk_min']:.0f} ({r['diagnostics']['ess_bulk_min_quantity']})")
    row("min tail ESS (quantity)", lambda r: f"{r['diagnostics']['ess_tail_min']:.0f} ({r['diagnostics']['ess_tail_min_quantity']})")
    row("ESS of c_p, c_n, d (bulk / tail)", lambda r: "; ".join(
        f"{q} {r['diagnostics']['per_quantity'][q]['ess_bulk']:.0f} / {r['diagnostics']['per_quantity'][q]['ess_tail']:.0f}" for q in ("c_p", "c_n", "d")))
    row("Divergences (per chain; follow-up rule)", lambda r: f"{r['divergences']['count']} ({100 * r['divergences']['fraction']:.2f} %), "
        f"{r['divergences']['per_chain_count']}; {'triggered' if r['divergences']['follow_up'] else 'not triggered'}")
    row("Step sizes", lambda r: ", ".join(f"{s['step_size']:.3f}" for s in r["sampler"]))
    row("Acceptance statistic (mean per chain)", lambda r: ", ".join(f"{s['mean_accept_prob']:.3f}" for s in r["sampler"]))
    row("Tree depth mean (max)", lambda r: ", ".join(f"{s['tree_depth_mean']:.2f}" for s in r["sampler"])
        + f" ({max(s['tree_depth_max'] for s in r['sampler'])})")
    row("Fraction of draws at the maximum tree depth", lambda r: ", ".join(f"{s['fraction_at_max_tree_depth']:.3f}" for s in r["sampler"]))
    row("Gradients per iteration, sampling / warm-up", lambda r: ", ".join(
        f"{s['grads_per_iteration_mean']:.1f} / {s['warmup_grads_per_iteration_mean']:.1f}" for s in r["sampler"]))
    row("λ", lambda r: f"{r['cost']['lambda_adjoint_over_forward']:.3f}")
    row("Gradients, retained / total", lambda r: f"{r['cost']['gradients']['sampling']} / {r['cost']['gradients']['total']}")
    row("PDE solves, retained / total", lambda r: f"{r['cost']['pde_solves']['sampling']} / {r['cost']['pde_solves']['total']} "
        f"(warm-up {100 * r['cost']['warmup_share_of_pde_solves']:.0f} %)")
    row("Forward equivalents, retained / total", lambda r: f"{r['cost']['forward_equivalents']['sampling']:.0f} / {r['cost']['forward_equivalents']['total']:.0f}")
    row("Wall-clock, batch", lambda r: f"{r['cost']['wall_clock_s']:.0f} s ({r['cost']['wall_clock_s'] / 60:.1f} min)")
    row("Min ESS per 10^3 FE, retained / total", lambda r: f"{r['cost']['min_ess_per_1000_forward_equivalents']['sampling']:.2f} / "
        f"{r['cost']['min_ess_per_1000_forward_equivalents']['total']:.2f}")
    row("Min ESS per wall-clock second", lambda r: f"{r['cost']['min_ess_per_wall_clock_s']:.3f}")
    row("Load average at start / end (1/5/15 min)", lambda r: " / ".join(
        ", ".join(f"{x:.1f}" for x in (r["cost"]["batch"][k] or {}).get("loadavg_1_5_15_min", [])) for k in ("load_start", "load_end")))
    row("RE / SSIM", lambda r: f"{r['RE']:.4f} / {r['SSIM']:.4f}")
    row("Draws without a crossing on x = 0.5 (d set to 0 / to 1; per chain)", lambda r: f"{r['depth_without_crossing']} "
        f"({r['depth_without_crossing_split']['d0']} / {r['depth_without_crossing_split']['d1']}; {r['depth_without_crossing_split']['per_chain']})")
    L += ["", "## Posterior of c_p, c_n and d (value ± MCSE)", ""]
    L += ["| Stage | Quantity | Truth | Mean | sd | 5 % | 50 % | 95 % | 90 % interval contains truth |", "|---|---|---|---|---|---|---|---|---|"]
    for n, r in zip(names, recs):
        for q in ("c_p", "c_n", "d"):
            s = r["posterior"][q]
            L.append(f"| {n} | {q} | {s['truth']:.4f} | " + " | ".join(f"{s[k]:.4f} ± {s['mcse_' + k]:.4f}" for k in STATS)
                     + f" | {'yes' if s['interval90_contains_truth'] else '**no**'} |")
    L += ["", "## Plateaus: spread and mass near the bounds", "",
          f"Near a bound: within {100 * NEAR:.0f} % of the interval width (prior mass {100 * NEAR:.0f} % each). Masses ± MCSE.", "",
          "| Stage | Quantity | Prior sd | Posterior sd | Post./prior sd | Mass near lower bound | Mass near upper bound | Mass outside the paper's interval (prior) |",
          "|---|---|---|---|---|---|---|---|"]
    fm = lambda x: "–" if x is None else (f"{100 * x['value']:.2f} %" + ("" if x["mcse"] is None else f" ± {100 * x['mcse']:.2f}"))
    for n, r in zip(names, recs):
        for q in ("c_p", "c_n"):
            p = r["plateaus"][q]
            out = ("–" if "mass_outside_paper_interval" not in p else
                   f"{fm(p['mass_outside_paper_interval'])} (below {fm(p['mass_below_paper_interval'])}, above {fm(p['mass_above_paper_interval'])}; "
                   f"prior {100 * p['prior_mass_outside_paper_interval']:.0f} %)")
            L.append(f"| {n} | {q} | {p['prior_sd']:.4f} | {p['posterior_sd']:.4f} | {p['sd_ratio']:.3f} | {fm(p['mass_near_lower'])} "
                     f"| {fm(p['mass_near_upper'])} | {out} |")
    L += ["", "## Correlations (all draws)", "", "| Stage | corr(c_p, d) | corr(c_n, d) | corr(c_p, c_n) |", "|---|---|---|---|"]
    for n, r in zip(names, recs):
        L.append(f"| {n} | " + " | ".join(f"{r['correlations'][k]:.3f}" for k in ("c_p,d", "c_n,d", "c_p,c_n")) + " |")
    L += ["", "## Divergences: location", ""]
    for n, r in zip(names, recs):
        dv = r["divergences"]
        L.append(f"- {n}: {dv['count']} divergences; fraction with c_p / c_n within {100 * NEAR:.0f} % of a bound: "
                 + " / ".join("–" if v is None else f"{100 * v:.0f} %" for v in dv["fraction_near_a_plateau_bound"].values()))
        for c in dv["per_chain"]:
            if c["count"]:
                L.append(f"  - chain {c['chain']}: draws " + ", ".join(
                    f"{l['draw']} (d {l['d']:.3f}, {100 * l['d_percentile']:.0f}th pct; c_p {l['c_p']:.3f}; c_n {l['c_n']:.3f})" for l in c["locations"]))
    L += ["", f"## Changes from {names[0]} to {names[1]}", "",
          "Difference and, in parentheses, the difference in units of the combined MCSE.", "",
          "| Quantity | " + " | ".join(STATS) + " |", "|---|" + "---|" * len(STATS)]
    for q in ("c_p", "c_n", "d"):
        L.append(f"| {q} | " + " | ".join(f"{ch[q][k]['difference']:+.4f} ({ch[q][k]['z']:+.1f})" for k in STATS) + " |")
    L += ["", f"Relative L2 change of the posterior mean field: {100 * ch['fields']['mean_relative_L2']:.2f} %; "
              f"of the std field: {100 * ch['fields']['std_relative_L2']:.1f} %."]
    kp = fit["known_plateaus"]
    L += ["", "## Data fit", "",
          f"Normalized misfit ||F(θ) − y||² / (σ² N_meas), N_meas = {fit['n_meas']}, with the forward map F of the "
          f"{fit['inversion_mesh_n']}×{fit['inversion_mesh_n']} inversion mesh (data generated on the "
          f"{fit['data_mesh_n']}×{fit['data_mesh_n']} mesh), at {fit['draws_per_posterior']} retained draws of each posterior "
          f"({fit['draws_per_posterior'] // len(kp['selection'])} evenly spaced draws per chain, first and last included).", "",
          "| Posterior | Median | 5 % | 95 % | Failed solves |", "|---|---|---|---|---|"]
    fr = lambda name, f: L.append(f"| {name} | {f['median']:.3f} | {f['q05']:.3f} | {f['q95']:.3f} | {f['failed_solves']} of {f['n_draws']} |")
    for n, r in zip(names, recs):
        fr(f"{n} (joint)", r["data_fit"])
    fr(f"Known plateaus (`{kp['run']}`)", kp)
    L += ["", f"- True field on the inversion mesh: {fit['true_field_inversion_mesh']:.3f}.",
          f"- For reference, noise-free data of the data mesh (the noise realisation alone): {fit['noise_free_data_mesh']:.3f}; "
          f"model error of the inversion mesh: RMS {fit['model_error_rms_sigma']:.3f} σ. Check: the inversion-mesh forward "
          f"map of the true field differs from the one stored in the data file by at most "
          f"{fit['check_max_abs_difference_from_data_file_sigma']:.1e} σ."]
    return L


def main(a):
    data = load_data(a.data)
    data_sha = data["_source"]["sha256"]
    stages = [load_stage(a.run_a, "joint_A", data_sha), load_stage(a.run_b, "joint_B", data_sha)]
    V = dl.FunctionSpace(dl.UnitSquareMesh(dl.MPI.comm_self, data["mesh_n"], data["mesh_n"]), "CG", 1)
    M = mass_matrix(V)
    c_true = data["c_true"]
    truth = dict(cp=float(data["cp_true"]), cn=float(data["cn_true"]),
                 depth=float(interface_depth(V, c_true[None, :], x0=0.5, level=0.0, clip=True)[0][0]))
    paper = {"cp": tuple(map(float, cfg.CP_BOUNDS)), "cn": tuple(map(float, cfg.CN_BOUNDS))}
    recs = [stage_record(st, truth, paper) for st in stages]
    ch = changes(recs[0], recs[1], stages[0]["fields"], stages[1]["fields"], M)
    for st, r in zip(stages, recs):
        r["data_fit"] = data_fit(st["chains"], data)
    kp_chains, kp = known_plateau_run(a.known_plateau_run, data_sha)
    fit_ref = dict(definition="||F(theta) - y||^2 / (sigma^2 N_meas), forward map F of the inversion mesh",
                   draws_per_posterior=DATA_FIT_DRAWS,
                   known_plateaus=dict(run=a.known_plateau_run, chain_content_sha256=kp["chain_content_sha256"],
                                                 chain_code=(kp.get("chain_provenance") or {}).get("code"), **data_fit(kp_chains, data)),
                   **data_fit_truth(data))
    os.makedirs(a.out, exist_ok=True)
    record = dict(runs=dict(A=a.run_a, B=a.run_b), truth=truth,
                  paper_intervals={MD[k]: list(v) for k, v in paper.items()}, near_bound_fraction_of_width=NEAR,
                  stages=recs, changes_first_to_second=ch, data_fit_reference=fit_ref, code=code_version())
    with open(os.path.join(a.out, "tables.json"), "w") as f:
        json.dump(record, f, indent=2, default=float)
    with open(os.path.join(a.out, "tables.json")) as f:          # tables are generated from the JSON as written
        record = json.load(f)
    recs, ch = record["stages"], record["changes_first_to_second"]
    md = md_report(recs, ch, record["data_fit_reference"])
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    figures(stages, recs, V, truth, c_true, a.out)
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_a", default="results/nuts_Setting2_Ex1_A", help="Stage A run folder (the paper's plateau intervals)")
    ap.add_argument("--run_b", default="results/nuts_Setting2_Ex1_B", help="Stage B run folder (intervals twice as wide)")
    ap.add_argument("--data", default="obs/obs_Ex1.pickle")
    ap.add_argument("--out", default="results/joint")
    ap.add_argument("--known_plateau_run", default="results/nuts_Setting1_Ex1",
                    help="known-plateau NUTS posterior on the same data, for the data-fit comparison")
    main(ap.parse_args())
