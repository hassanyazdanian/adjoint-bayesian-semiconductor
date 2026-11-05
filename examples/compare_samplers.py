# compare_samplers.py
"""
Agreement of the posteriors of completed runs of the same target (Example 2, Section 6.3.2: pCN and NUTS
with target acceptance 0.95 and 0.99): for every monitored quantity (all KL coefficients and the junction depth d), the
mean, sd and the 1, 5, 50, 95 and 99 % quantiles of each run with their Monte Carlo standard errors
(ArviZ, Vehtari et al. 2021), and the difference to the reference run in units of the combined MCSE,
z = (stat_run - stat_ref) / sqrt(mcse_run^2 + mcse_ref^2). For d also the per-chain 95 and 99 %
quantiles and the fraction of draws above the reference's 95 and 99 % quantiles.

The chains, fields.npz (per-draw depths) and summary.json of every run must belong together, and all
runs must have the same posterior: the same data, setting, prior (including k), plateau bounds, inversion
mesh, KL basis and latent mean, so that the KL coefficients are comparable. Sampler settings may differ.
The first run is the reference.

Usage: python compare_samplers.py results/pcn_Setting1_Ex2 results/nuts_Setting1_Ex2 results/nuts_Setting1_Ex2_t099 \
         --example 2 --out results/ex2_depth_quantiles
Writes comparison.json, comparison.md and depth_agreement.tex (the table of d for the paper).
"""
import argparse
import json
import os

import arviz as az
import numpy as np

from io_utils import chain_content_sha256, code_version, load_chains

PROBS = (0.01, 0.05, 0.5, 0.95, 0.99)
STATS = ("mean", "sd") + tuple(f"q{round(100 * p):02d}" for p in PROBS)


def _label(q):
    return "d" if q == "depth" else f"z_{int(q[2:-1]) + 1}"


def load_run(run):
    pp = run + "_postprocess"
    with open(f"{pp}/summary.json") as f:
        s = json.load(f)
    fields = np.load(f"{pp}/fields.npz")
    chains = load_chains(run)
    if not ([chain_content_sha256(c) for c in chains] == s["chain_content_sha256"]
            == [str(h) for h in fields["chain_content_sha256"]] and s["data_sha256"] == str(fields["data_sha256"])):
        raise ValueError(f"{run}: chains, fields.npz and summary.json do not belong together")
    draws = {f"z[{j}]": np.stack([c["z"][:, j] for c in chains]) for j in range(chains[0]["z"].shape[1])}
    draws["depth"] = np.asarray(fields["depth"], dtype=float)
    c0 = chains[0]
    target = dict(setting=1 if c0["r"] is None else 2, prior={k: c0["prior"][k] for k in ("n_kl", "tau", "alpha", "k", "mean_a")},
                  plateau_bounds=c0.get("plateau_bounds"),
                  inversion_mesh_n=s.get("inversion_mesh_n", c0["info"].get("inversion_mesh_n")))
    return dict(run=run, method=s["method"], data_sha256=s["data_sha256"], summary=s, draws=draws, target=target,
                kl_basis=c0.get("kl_basis"), latent_mean=c0.get("latent_mean"),
                target_accept_prob=(s.get("sampler") or [{}])[0].get("target_accept_prob"))


def _same_array(a, b):
    """Equal up to round-off (None only matches None)."""
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=0, atol=1e-10 * max(1.0, float(np.abs(b).max()))))


def check_same_target(runs):
    """Raise unless every run samples the posterior of the first in the same coordinates."""
    ref = runs[0]
    for r in runs[1:]:
        if r["data_sha256"] != ref["data_sha256"]:
            raise ValueError(f"{r['run']} and {ref['run']} use different data")
        if r["target"] != ref["target"]:
            raise ValueError(f"{r['run']} and {ref['run']} have different posteriors: {r['target']} and {ref['target']}")
        for key in ("kl_basis", "latent_mean"):
            if not _same_array(r[key], ref[key]):
                raise ValueError(f"{r['run']} and {ref['run']} have different {key.replace('_', ' ')}s, so their KL "
                                 "coefficients are not comparable")


def statistics(x):
    """Mean, sd and quantiles of x (chains, draws) with their MCSE."""
    ds = az.convert_to_dataset({"x": x})
    out = dict(mean=float(x.mean()), sd=float(x.std()),
               mcse_mean=float(az.mcse(ds, method="mean")["x"]), mcse_sd=float(az.mcse(ds, method="sd")["x"]))
    for p in PROBS:
        k = f"q{round(100 * p):02d}"
        out[k] = float(np.quantile(x, p))
        out["mcse_" + k] = float(az.mcse(ds, method="quantile", prob=p)["x"])
    return out


def main(a):
    runs = [load_run(r) for r in a.runs]
    check_same_target(runs)
    ref = runs[0]
    quantities = list(ref["draws"])
    stats = {r["run"]: {q: statistics(r["draws"][q]) for q in quantities} for r in runs}
    for r in runs[1:]:
        for q in quantities:
            s, t = stats[r["run"]][q], stats[ref["run"]][q]
            s["z_vs_reference"] = {k: (s[k] - t[k]) / float(np.hypot(s["mcse_" + k], t["mcse_" + k])) for k in STATS}
    dref = ref["draws"]["depth"]
    thresholds = {"q95": float(np.quantile(dref, 0.95)), "q99": float(np.quantile(dref, 0.99))}
    depth_chains = {r["run"]: dict(q95=np.quantile(r["draws"]["depth"], 0.95, axis=1).tolist(),
                                   q99=np.quantile(r["draws"]["depth"], 0.99, axis=1).tolist(),
                                   above_reference_q95=float((r["draws"]["depth"] > thresholds["q95"]).mean()),
                                   above_reference_q99=float((r["draws"]["depth"] > thresholds["q99"]).mean()))
                    for r in runs}

    names = [{"nuts": "NUTS", "pcn": "pCN"}[r["method"]] + (f" ({r['target_accept_prob']})" if r["method"] == "nuts" else "") for r in runs]
    md = [f"# Posterior agreement (reference: {names[0]}, `{ref['run']}`)", "",
          "z = difference to the reference in units of the combined MCSE.", ""]
    for q in quantities:
        md += [f"## {_label(q)}", "", "| Statistic | " + " | ".join(names) + " | " + " | ".join(f"z {n}" for n in names[1:]) + " |",
               "|---|" + "---|" * (2 * len(runs) - 1)]
        for k in STATS:
            cells = [f"{stats[r['run']][q][k]:.4f} ({stats[r['run']][q]['mcse_' + k]:.4f})" for r in runs]
            zs = [f"{stats[r['run']][q]['z_vs_reference'][k]:+.2f}" for r in runs[1:]]
            md.append(f"| {k} | " + " | ".join(cells) + " | " + " | ".join(zs) + " |")
        md.append("")
    md += ["## d per chain", "", "| Run | q95 per chain | q99 per chain | above reference q95 / q99 |", "|---|---|---|---|"]
    for n, r in zip(names, runs):
        c = depth_chains[r["run"]]
        md.append(f"| {n} | " + ", ".join(f"{v:.4f}" for v in c["q95"]) + " | " + ", ".join(f"{v:.4f}" for v in c["q99"])
                  + f" | {100 * c['above_reference_q95']:.2f} % / {100 * c['above_reference_q99']:.2f} % |")

    # LaTeX table of the junction depth (booktabs), for the paper
    rows = []
    for k, lab in zip(STATS, ["mean", "sd"] + [f"{round(100 * p)}\\,\\% quantile" for p in PROBS]):
        cells = [lab]
        for i, r in enumerate(runs):
            s = stats[r["run"]]["depth"]
            cells.append(f"{s[k]:.4f} ({s['mcse_' + k]:.4f})")
            if i:
                cells.append(f"${s['z_vs_reference'][k]:+.1f}$")
        rows.append(" & ".join(cells) + " \\\\")
    header = " & ".join(["Statistic", names[0]] + [c for n in names[1:] for c in (n, "$z$")]) + " \\\\"
    tex = ["\\begin{table}[htbp]", "\\centering", "\\small",
           f"\\caption{{Posterior of the junction depth $d$ at $x=0.5$, Example~{a.example}: estimates with their Monte Carlo "
           f"standard errors (MCSE) in parentheses, and the difference to {names[0]} in units of the combined MCSE ($z$).}}",
           f"\\label{{tab:ex{a.example}_depth_agreement}}",
           "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
           "\\begin{tabular}{l" + "c" * (2 * len(runs) - 1) + "}", "\\toprule", header, "\\midrule"] + rows + [
           "\\bottomrule", "\\end{tabular}}", "", "\\smallskip",
           "\\parbox{0.97\\linewidth}{\\footnotesize Per-chain 99\\,\\% quantiles: " + "; ".join(
               f"{n} " + ", ".join(f"{v:.4f}" for v in depth_chains[r['run']]['q99']) for n, r in zip(names, runs))
           + f". MCSE of means, standard deviations and quantiles as in ArviZ~\\cite{{vehtari2021rank}}.}}", "\\end{table}", ""]

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "depth_agreement.tex"), "w") as f:
        f.write("\n".join(tex))
    record = dict(reference=ref["run"], runs=[r["run"] for r in runs], data_sha256=ref["data_sha256"],
                  chain_content_sha256={r["run"]: r["summary"]["chain_content_sha256"] for r in runs},
                  statistics=stats, depth_reference_quantiles=thresholds, depth_per_chain=depth_chains, code=code_version())
    with open(os.path.join(a.out, "comparison.json"), "w") as f:
        json.dump(record, f, indent=2)
    with open(os.path.join(a.out, "comparison.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders (results/<run>); the first is the reference")
    ap.add_argument("--example", type=int, default=2, help="example number, for the label and caption of depth_agreement.tex")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
