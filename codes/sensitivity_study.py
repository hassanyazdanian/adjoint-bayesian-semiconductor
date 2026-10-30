# sensitivity_study.py
"""
Sensitivity and scaling of the Example 1 posterior with known plateau values (paper Section 6.3.3, Table 3),
from the runs of the manifests below, made with production.sh. Subcommands:

  rows  Table 3: NUTS runs that change one setting of the reference run (data-generation mesh, KL truncation
        N_KL, sigmoid steepness k), each one four-chain batch run alone on the machine:
          Ref.              experiments/study_k_v1.json, stage k14
          Data 96, 384      experiments/study_timing_v1.json, stages data96, data384
          N_KL = 24, 48, 96 experiments/study_timing_v1.json, stages nkl24, nkl48, nkl96
          k = 10            experiments/study_k_v1_k10r.json, stage k10
          k = 20, 28        experiments/study_k_v1.json, stages k20, k28
        -> <out>/sensitivity.{json,md}; make_paper_tables.py writes the LaTeX table from sensitivity.json.
  pcn   pCN at the fixed production beta and lengths for N_KL = 24, 48, 96 (experiments/study_pcn_nkl_v1.json),
        next to the N_KL = 12 production pCN run (experiments/production_v1.json, stage ex1_pcn).
        -> <out>/pcn_nkl.{json,md}

Definitions (paper Section 6.3): rank-normalized split R-hat, bulk and tail ESS (ArviZ); minimum ESS = smallest
bulk or tail ESS over all KL coefficients and the junction depth d (lowest crossing of c = 0 on x = 0.5), and also
over the first 12 KL coefficients and d; efficiency = minimum ESS per 10^3 forward-equivalent solves N_F + lambda N_A,
lambda the run's own ratio of mean adjoint to mean forward solve time, for the retained draws and including
warm-up/burn-in; ESS per wall-clock second of the four-chain batch (<run>_batch.json), including warm-up.
Model error: RMS difference between the noise-free data of the data-generation and the inversion forward maps, in
units of sigma_noise (stored in the data file).
Usage: python sensitivity_study.py rows | pcn [--out results/sensitivity_ex1]
"""
import argparse
import hashlib
import json
import os

import numpy as np

ROWS = [("Ref.", "experiments/study_k_v1.json", "k14"),
        ("Data 96", "experiments/study_timing_v1.json", "data96"),
        ("Data 384", "experiments/study_timing_v1.json", "data384"),
        ("N_KL 24", "experiments/study_timing_v1.json", "nkl24"),
        ("N_KL 48", "experiments/study_timing_v1.json", "nkl48"),
        ("N_KL 96", "experiments/study_timing_v1.json", "nkl96"),
        ("k 10", "experiments/study_k_v1_k10r.json", "k10"),
        ("k 20", "experiments/study_k_v1.json", "k20"),
        ("k 28", "experiments/study_k_v1.json", "k28")]
PCN_ROWS = [("N_KL 12", "experiments/production_v1.json", "ex1_pcn")] + [
    (f"N_KL {n}", "experiments/study_pcn_nkl_v1.json", f"pcn_nkl{n}") for n in (24, 48, 96)]
DEPTH_PROBS = (0.05, 0.5, 0.95, 0.99)


def run_dir(manifest, stage):
    """Run folder of a manifest stage, as named by production.sh; the chains must have been run with this manifest."""
    from io_utils import load_chains
    with open(manifest, "rb") as f:
        raw = f.read()
    m, sha = json.loads(raw), hashlib.sha256(raw).hexdigest()
    s = m["stages"][stage]
    run = f"results/{s['method']}_Setting{s['setting']}_Ex{s['example']}{m['output_tag']}{s.get('tag', '')}"
    for c in load_chains(run):
        if c["provenance"]["manifest"]["sha256"] != sha:
            raise ValueError(f"{run}: chain {c['info']['chain']} was not run with {manifest} as it is now")
    return run


def _diag_block(diag, rows):
    """R-hat and ESS over the given quantities; the minimum ESS and the quantity and kind that set it."""
    sub = diag.loc[rows, ["r_hat", "ess_bulk", "ess_tail"]].astype(float)
    finite = bool(np.isfinite(sub.to_numpy()).all())
    b, t = sub.ess_bulk, sub.ess_tail
    kind = "bulk" if b.min() <= t.min() else "tail"
    q = (b if kind == "bulk" else t).idxmin()
    m = float(min(b.min(), t.min()))
    return dict(quantities=len(rows), r_hat_max=float(sub.r_hat.max()), r_hat_max_quantity=sub.r_hat.idxmax(),
                ess_bulk_min=float(b.min()), ess_tail_min=float(t.min()), min_ess=m, min_ess_quantity=q,
                min_ess_kind=kind, complete=finite, passes=bool(finite and sub.r_hat.max() < 1.01 and m >= 400))


def _depth(depth):
    import arviz as az
    ds = az.convert_to_dataset({"d": depth})
    mc = lambda **kw: float(az.mcse(ds, **kw)["d"])
    out = dict(mean=float(depth.mean()), mcse_mean=mc(method="mean"), sd=float(depth.std(ddof=1)), mcse_sd=mc(method="sd"))
    for p in DEPTH_PROBS:
        out[f"q{round(100 * p):02d}"] = dict(value=float(np.quantile(depth, p)), mcse=mc(method="quantile", prob=p))
    return out


def row_metrics(run, M, w):
    """Diagnostics, costs and field metrics of one run from its postprocessing output (summary.json, fields.npz,
    diagnostics.csv), which must belong to the chains of the run. M and w: mass matrix and lumped weights of the
    48 x 48 inversion mesh."""
    import pandas as pd
    from io_utils import chain_content_sha256, load_chains, load_data
    pp = run + "_postprocess"
    with open(os.path.join(pp, "summary.json")) as f:
        s = json.load(f)
    fields = np.load(os.path.join(pp, "fields.npz"))
    chains = load_chains(run)
    hashes = [chain_content_sha256(c) for c in chains]
    if not hashes == s["chain_content_sha256"] == [str(h) for h in fields["chain_content_sha256"]]:
        raise ValueError(f"{pp} does not belong to the chains of {run}")
    diag = pd.read_csv(os.path.join(pp, "diagnostics.csv"), index_col=0)
    c0 = chains[0]
    n_kl = int(c0["prior"]["n_kl"])
    data = load_data(c0["info"]["data"])
    if int(s.get("inversion_mesh_n", data["mesh_n"])) != int(data["mesh_n"]):
        raise ValueError(f"{run}: inversion mesh other than {data['mesh_n']} x {data['mesh_n']}")
    fl = {key: np.asarray(fields[key], dtype=float) for key in ("c_std", "hpd_low", "hpd_high")}
    d_all = _diag_block(diag, [f"z[{j}]" for j in range(n_kl)] + ["depth"])
    d12 = _diag_block(diag, [f"z[{j}]" for j in range(min(12, n_kl))] + ["depth"])
    cost = s["cost"]
    n_draws = s["n_chains"] * s["draws_per_chain"]
    fe = {ph: cost[ph]["forward_equivalents"] for ph in ("sampling", "total")}
    row = dict(
        run=run, method=s["method"], chains=s["n_chains"], draws_per_chain=s["draws_per_chain"],
        prior={k: c0["prior"][k] for k in ("tau", "alpha", "n_kl", "k", "mean_a")},
        data=dict(path=s["data_path"], sha256=s["data_sha256"], data_mesh_n=int(data["data_mesh_n"]),
                  model_error_rms_sigma=float(np.sqrt(np.mean(np.asarray(data["model_error"]) ** 2)) / data["sigma"])),
        diagnostics_all=d_all, diagnostics_first12_and_depth=d12,
        cost=dict(lambda_adjoint_over_forward=cost["adjoint_in_forward_equivalents"],
                  forward_solves={ph: cost[ph]["forward_solves"] for ph in ("warmup", "sampling", "total")},
                  adjoint_solves={ph: cost[ph]["adjoint_solves"] for ph in ("warmup", "sampling", "total")},
                  forward_equivalents=fe, newton_iterations_per_forward=cost.get("newton_iterations_per_forward"),
                  pde_solves_per_draw=(cost["sampling"]["forward_solves"] + cost["sampling"]["adjoint_solves"]) / n_draws),
        efficiency_per_1000_forward_equivalents={ph: 1e3 * d_all["min_ess"] / fe[ph] for ph in fe},
        efficiency_first12_and_depth_per_1000_forward_equivalents={ph: 1e3 * d12["min_ess"] / fe[ph] for ph in fe},
        RE=s["RE"], SSIM=s["SSIM"], depth=_depth(np.asarray(fields["depth"], dtype=float)),
        depth_without_crossing=s["depth_without_crossing"],
        spread=dict(rms_pointwise_std=float(np.sqrt(fl["c_std"] @ (M @ fl["c_std"]) / w.sum())),
                    mean_hpd95_width=float(w @ (fl["hpd_high"] - fl["hpd_low"]) / w.sum())),
    )
    if s["method"] == "nuts":
        sm = s["sampler"]
        row["sampler"] = dict(step_size=[c["step_size"] for c in sm], tree_depth_mean=[c.get("tree_depth_mean") for c in sm],
                              tree_depth_max=[c.get("tree_depth_max") for c in sm],
                              divergences=s["divergences"]["count"], divergences_per_chain=s["divergences"]["per_chain"],
                              divergence_fraction=s["divergences"]["fraction"])
    else:
        row["sampler"] = dict(beta=[c["info"]["beta"] for c in chains],
                              mean_acceptance_probability=[c["info"]["acceptance"] for c in chains],
                              accepted_fraction=[c["info"].get("accepted_fraction") for c in chains])
    with open(run + "_batch.json") as f:
        b = json.load(f)
    row["timing"] = dict(batch=run + "_batch.json", wall_clock_s=b["elapsed_s"],
                         min_ess_per_wall_clock_s=d_all["min_ess"] / b["elapsed_s"],
                         newton_iterations_per_forward=cost.get("newton_iterations_per_forward"),
                         worker_time_per_pde_solve_ms=1e3 * cost["total"]["worker_time_s"]
                         / (cost["total"]["forward_solves"] + cost["total"]["adjoint_solves"]),
                         load_start=b.get("load_start"), load_end=b.get("load_end"), threads=b.get("threads"))
    return row


def _mass():
    from diagnostics import mass_matrix
    from forward_model import PoissonBoltzmannModel
    import config as cfg
    M = mass_matrix(PoissonBoltzmannModel(cfg.MESH_N, cfg.EPS, cfg.DELTA).V)
    return M, M @ np.ones(M.shape[0])


def _dump(out, name, obj, md):
    from io_utils import code_version
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name + ".json"), "w") as f:
        json.dump(dict(definitions=__doc__.split("Definitions (paper Section 6.3): ")[1].split("Usage:")[0].strip(),
                       code=code_version(), **obj), f, indent=2, default=float)
    with open(os.path.join(out, name + ".md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


def rows(a):
    M, w = _mass()
    out = {label: row_metrics(run_dir(man, stage), M, w) for label, man, stage in ROWS}
    md = ["# Example 1, known plateaus: sensitivity (Table 3)", "",
          "Minimum ESS over all KL coefficients and d; efficiency per 10^3 forward-equivalent solves, retained draws / "
          "including warm-up; in the last column over the first 12 KL coefficients and d.", "",
          "| Row | Run | Data mesh (model error σ) | N_KL | k | RE / SSIM | Depth mean ± sd | RMS std | max R̂ "
          "| min bulk / tail ESS | Efficiency | Div. | Batch (s) | ESS/s | Efficiency, first 12 and d |", "|" + "---|" * 15]
    for label, r in out.items():
        da, e, e12, dep, t = (r["diagnostics_all"], r["efficiency_per_1000_forward_equivalents"],
                              r["efficiency_first12_and_depth_per_1000_forward_equivalents"], r["depth"], r["timing"])
        md.append(f"| {label} | `{r['run'].split('/')[-1]}` | {r['data']['data_mesh_n']} ({r['data']['model_error_rms_sigma']:.3f}) "
                  f"| {r['prior']['n_kl']} | {r['prior']['k']:g} | {r['RE']:.3f} / {r['SSIM']:.3f} | {dep['mean']:.4f} ± {dep['sd']:.4f} "
                  f"| {r['spread']['rms_pointwise_std']:.4f} | {da['r_hat_max']:.4f} | {da['ess_bulk_min']:.0f} / {da['ess_tail_min']:.0f} "
                  f"| {e['sampling']:.2f} / {e['total']:.2f} | {r['sampler']['divergences']} | {t['wall_clock_s']:.0f} "
                  f"| {t['min_ess_per_wall_clock_s']:.3f} | {e12['sampling']:.2f} / {e12['total']:.3f} |")
    _dump(a.out, "sensitivity", dict(manifests=sorted({m for _, m, _ in ROWS}), rows=out), md)


def pcn(a):
    M, w = _mass()
    out = {label: row_metrics(run_dir(man, stage), M, w) for label, man, stage in PCN_ROWS}
    betas = {round(r["sampler"]["beta"][0], 12) for r in out.values()}
    if len(betas) > 1:
        raise ValueError(f"the runs use different beta: {betas}")
    md = ["# pCN at fixed beta against N_KL, production lengths (Example 1, known plateaus)", "",
          "Minimum ESS per 10^3 PDE solves, retained draws / including the discarded steps.", "",
          "| Row | Run | β | Chains × draws | Mean acceptance probability | max R̂ (quantity) | Min ESS | Meets the criteria "
          "| Efficiency | max R̂, first 12 and d | Min ESS, first 12 and d | Meets the criteria, first 12 and d | Efficiency, first 12 and d |",
          "|" + "---|" * 13]
    for label, r in out.items():
        da, d12, e, e12 = (r["diagnostics_all"], r["diagnostics_first12_and_depth"], r["efficiency_per_1000_forward_equivalents"],
                           r["efficiency_first12_and_depth_per_1000_forward_equivalents"])
        md.append(f"| {label} | `{r['run'].split('/')[-1]}` | {r['sampler']['beta'][0]:.4f} | {r['chains']} × {r['draws_per_chain']} "
                  f"| {np.mean(r['sampler']['mean_acceptance_probability']):.4f} | {da['r_hat_max']:.4f} ({da['r_hat_max_quantity']}) "
                  f"| {da['min_ess']:.0f} | {'yes' if da['passes'] else 'no'} | {e['sampling']:.3f} / {e['total']:.3f} "
                  f"| {d12['r_hat_max']:.4f} | {d12['min_ess']:.0f} | {'yes' if d12['passes'] else 'no'} | {e12['sampling']:.3f} / {e12['total']:.3f} |")
    _dump(a.out, "pcn_nkl", dict(manifests=sorted({m for _, m, _ in PCN_ROWS}), rows=out), md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["rows", "pcn"])
    ap.add_argument("--out", default="results/sensitivity_ex1")
    a = ap.parse_args()
    {"rows": rows, "pcn": pcn}[a.cmd](a)
