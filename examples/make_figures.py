# make_figures.py
"""
Figures of one example with known plateau values, from a pCN run, a NUTS run and a MAP run (Example 1: Figure 4
and Figure 5 (a, b); Example 2: Figure 6 and Figure 5 (c, d); Figure 3, the prior samples, is plot_prior_samples.py).
Inputs are checked before use: the chain files, fields.npz and summary.json of each run must belong together and to
the data file, and the MAP must have the chains' prior and data and must have converged.
File names start with ex<example>.
- ex<e>_true.png, ex<e>_mean_pcn.png, ex<e>_mean_nuts.png, ex<e>_map.png: one color scale (c_p, c_n);
  true interface solid black, inferred interface (c = 0 of the field shown) dashed red.
- ex<e>_std_pcn.png, ex<e>_std_nuts.png: posterior std on one color scale, with both interfaces.
- ex<e>_traces.png: the four chains of each sampler for the junction depth and the KL coefficients
  with the smallest ESS, against the PDE solves of each chain after warm-up (forward + adjoint), over
  an equal-cost window: the PDE solves of the longest NUTS chain. pCN is shown at the same density
  per PDE solve as NUTS (every n-th draw, n = mean NUTS solves per draw; display only). NUTS divergent
  transitions are marked. --panel_letters a b prefixes the two panel titles with "(a) " and "(b) ";
  --no_legend omits the legend row below the panels (both also for ex<e>_traces_full.png).
- ex<e>_traces_full.png: the same over each sampler's whole run (pCN at every n-th draw, display only).
Usage: python make_figures.py --example 1 --pcn results/pcn_Setting1_Ex1 --nuts results/nuts_Setting1_Ex1 \
           --map results/map_Setting1_Ex1 --out results/figures/ex1
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import dolfin as dl

from diagnostics import interface_depth
from io_utils import chain_content_sha256, code_version, load_chains, load_data
from plotting import plot_field, plot_trace_chains, TRUE_INTERFACE, MEAN_INTERFACE

NAMES = {"nuts": "NUTS", "pcn": "pCN"}
MAX_POINTS = 2000          # trace points per chain shown


def _ordinal(n):
    """1st, 2nd, 3rd, 4th, ..., 11th-13th, 21st, ..., 71st, 101st, 111th."""
    suffix = "th" if n % 100 in (11, 12, 13) else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _label(q):
    return r"junction depth $d$" if q == "depth" else rf"$z_{{{int(q[2:-1]) + 1}}}$"


def load_stage(run, method, data_sha):
    pp = run + "_postprocess"
    with open(f"{pp}/summary.json") as f:
        s = json.load(f)
    fields = np.load(f"{pp}/fields.npz")
    chains = load_chains(run)
    hashes = [chain_content_sha256(c) for c in chains]
    if not (hashes == s["chain_content_sha256"] == [str(h) for h in fields["chain_content_sha256"]]
            and s["data_sha256"] == str(fields["data_sha256"]) == data_sha and s["method"] == method):
        raise ValueError(f"{run}: chains, fields.npz, summary.json and the data do not belong together")
    diag = pd.read_csv(f"{pp}/diagnostics.csv", index_col=0)
    return dict(method=method, run=run, summary=s, fields=fields, chains=chains, diag=diag)


def _solves(st, ch):
    """Cumulative PDE solves (forward + adjoint) of a chain after warm-up, per retained draw."""
    if st["method"] == "nuts":
        return np.cumsum(np.asarray(ch["info"]["evals_per_iteration"]) + np.asarray(ch["info"]["grads_per_iteration"]))
    return np.arange(1, len(ch["z"]) + 1)          # one forward solve per pCN step


def trace_column(st, quantities, thin=None, x_max=None):
    """Per-chain x (cumulative PDE solves after warm-up, in thousands), y (n_draws, n_quantities) and
    NUTS divergences; draws up to x_max solves, every `thin`-th (default: at most MAX_POINTS per chain)."""
    xs, ys, marks = [], [], []
    n = len(st["chains"][0]["z"])
    thin = thin or max(1, n // MAX_POINTS)
    depth = st["fields"]["depth"]
    for i, ch in enumerate(st["chains"]):
        info = ch["info"]
        x = _solves(st, ch)
        y = np.column_stack([depth[i] if q == "depth" else ch["z"][:, int(q[2:-1])] for q in quantities])
        keep = np.arange(0, n if x_max is None else int(np.searchsorted(x, x_max, side="right")), thin)
        xs.append(x[keep] / 1e3)
        ys.append(y[keep])
        div = np.asarray(info.get("divergences", []), dtype=int) if st["method"] == "nuts" else np.array([], int)
        div = div[x[div] <= x_max] if x_max is not None else div
        marks.append((x[div] / 1e3, y[div]))
    title = NAMES[st["method"]] + (f" (every {_ordinal(thin)} draw shown)" if thin > 1 else " (every draw shown)")
    return dict(title=title, x=xs, y=ys, marks=marks if st["method"] == "nuts" else None)


def main(a):
    data = load_data(a.data or f"obs/obs_Ex{a.example}.pickle")
    data_sha = data["_source"]["sha256"]
    stages = {meth: load_stage(run, meth, data_sha) for meth, run in (("pcn", a.pcn), ("nuts", a.nuts))}
    keys = ("tau", "alpha", "n_kl", "k", "mean_a")
    p = stages["nuts"]["chains"][0]["prior"]
    if [stages["pcn"]["chains"][0]["prior"].get(k) for k in keys] != [p.get(k) for k in keys]:
        raise ValueError(f"{a.pcn} and {a.nuts} were sampled with different priors")

    with open(os.path.join(a.map, "map.pickle"), "rb") as f:
        mp = pickle.load(f)
    if ((mp.get("provenance") or {}).get("data_sha256") != data_sha
            or [mp["prior"].get(k) for k in keys] != [p.get(k) for k in keys]):
        raise ValueError(f"{a.map}: the MAP was not computed with the chains' prior and data")
    if mp.get("success") is not True:
        raise ValueError(f"{a.map}: the MAP optimisation did not converge ({mp.get('message')}); "
                         "run run_map.py again, for example with a larger --maxiter")

    os.makedirs(a.out, exist_ok=True)
    V = dl.FunctionSpace(dl.UnitSquareMesh(dl.MPI.comm_self, data["mesh_n"], data["mesh_n"]), "CG", 1)
    c_true = data["c_true"]
    cp, cn = float(data["cp_true"]), float(data["cn_true"])
    lab = r"$c(\boldsymbol{x})$"
    mean_lab, std_lab = r"$\mathbb{E}[c(\boldsymbol{x}) \mid y]$", r"$\mathrm{std}(c(\boldsymbol{x}) \mid y)$"
    pre = os.path.join(a.out, f"ex{a.example}_")
    plot_field(V, c_true, pre + "true.png", cbar_label=lab, vlim=(cp, cn),
               contours=[(c_true, TRUE_INTERFACE)])
    for meth in ("pcn", "nuts"):
        c_mean = stages[meth]["fields"]["c_mean"]
        plot_field(V, c_mean, pre + f"mean_{meth}.png", cbar_label=mean_lab, vlim=(cp, cn),
                   contours=[(c_true, TRUE_INTERFACE), (c_mean, MEAN_INTERFACE)])
    plot_field(V, mp["c_map"], pre + "map.png", cbar_label=r"$c_{\mathrm{MAP}}(\boldsymbol{x})$",
               vlim=(cp, cn), contours=[(c_true, TRUE_INTERFACE), (mp["c_map"], MEAN_INTERFACE)])
    smax = max(float(stages[meth]["fields"]["c_std"].max()) for meth in stages)
    for meth in ("pcn", "nuts"):
        f_ = stages[meth]["fields"]
        plot_field(V, f_["c_std"], pre + f"std_{meth}.png", cmap="PuRd", cbar_label=std_lab,
                   vlim=(0.0, smax), contours=[(c_true, TRUE_INTERFACE), (f_["c_mean"], MEAN_INTERFACE)])

    # Figure 5: depth and the bottleneck KL coefficients of both samplers
    if a.quantities:
        quantities = a.quantities
    else:
        quantities = ["depth"]
        for st in stages.values():
            d = st["diag"].drop(index="depth")
            q = min(d.index, key=lambda r: min(d.loc[r, "ess_bulk"], d.loc[r, "ess_tail"]))
            quantities += [q] if q not in quantities else []
    d_true = float(interface_depth(V, c_true[None, :], x0=0.5, level=0.0, clip=True)[0][0])
    nuts = stages["nuts"]
    budget = max(float(_solves(nuts, ch)[-1]) for ch in nuts["chains"])            # longest NUTS chain
    per_draw = np.mean([_solves(nuts, ch)[-1] / len(ch["z"]) for ch in nuts["chains"]])
    pcn_thin = max(1, int(round(per_draw)))
    kw = dict(xlabel=r"PDE solves per chain after warm-up ($\times 10^3$)", row_labels=[_label(q) for q in quantities],
              true_values=[d_true if q == "depth" else None for q in quantities], legend=not a.no_legend)

    def lettered(columns):
        for col, letter in zip(columns, a.panel_letters or []):
            col["title"] = f"({letter}) " + col["title"]
        return columns

    plot_trace_chains(lettered([trace_column(stages["pcn"], quantities, thin=pcn_thin, x_max=budget), trace_column(nuts, quantities)]),
                      pre + "traces.png", xlim=(0, budget / 1e3 * 1.01), **kw)
    plot_trace_chains(lettered([trace_column(stages[meth], quantities) for meth in ("pcn", "nuts")]),
                      pre + "traces_full.png", **kw)
    record = dict(example=a.example, runs={k: v["run"] for k, v in stages.items()},
                  chain_content_sha256={k: v["summary"]["chain_content_sha256"] for k, v in stages.items()},
                  map=a.map, map_RE=mp["RE"], map_SSIM=mp["SSIM"], trace_quantities=quantities, true_depth_x05=d_true,
                  std_color_max=smax, equal_cost_window_solves=budget, nuts_solves_per_draw=per_draw,
                  pcn_display_every=pcn_thin, panel_letters=a.panel_letters, legend=not a.no_legend, pcn_fraction_of_run_in_window=budget / len(stages["pcn"]["chains"][0]["z"]),
                  map_code=mp["provenance"].get("code"), figure_code=code_version())
    with open(pre + "figures.json", "w") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", type=int, required=True)
    ap.add_argument("--pcn", required=True, help="pCN run folder")
    ap.add_argument("--nuts", required=True, help="NUTS run folder")
    ap.add_argument("--map", required=True, help="run_map.py output folder (map.pickle) with the chains' prior")
    ap.add_argument("--data", default=None, help="data file (default obs/obs_Ex<example>.pickle)")
    ap.add_argument("--quantities", nargs="*", default=None,
                    help="trace rows, e.g. depth 'z[3]' (default: depth and each sampler's bottleneck coefficient)")
    ap.add_argument("--panel_letters", nargs=2, default=None, metavar=("L1", "L2"),
                    help="prefix the two trace-panel titles with (L1) and (L2) (default: no letters)")
    ap.add_argument("--no_legend", action="store_true", help="omit the legend row below the trace panels")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
