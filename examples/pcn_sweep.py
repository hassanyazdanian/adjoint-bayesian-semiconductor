# pcn_sweep.py
"""
pCN step-size sweep for Example 1 or 2 with known plateau values (paper Sections 6.3.1 and 6.3.2). Four chains of
5000 tuning and 10 000 burn-in steps (discarded) and 100 000 retained steps: first with the step tuned to the
acceptance rate 0.234 (beta_0 = mean tuned step of the four chains), then with the fixed steps beta = f beta_0 for
the given factors f. Each run is made with run_chains.py and postprocessed with postprocess.py; a run whose
postprocessing summary exists is not run again. For every run the table gives the mean acceptance rate, max R-hat,
the minimum bulk or tail ESS over all KL coefficients and the junction depth, and that ESS per 10^3 PDE solves of
the retained draws. The paper's pCN runs use the f with the largest ESS per 10^3 PDE solves (f = 8 in both
examples).
Usage: python pcn_sweep.py --example 1 --factors 0.5 2 3 4 6 8 12
       python pcn_sweep.py --example 2 --factors 6 8 12
       (--table_only: only the table, from existing runs; --out: output path without extension)
Runs: results/pcn_Setting1_Ex<e><tag> (tuned) and results/pcn_Setting1_Ex<e><tag>_beta_x<f>; output
results/pcn_sweep_Ex<e>.{json,md}.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from io_utils import chain_content_sha256, load_chains

HERE = os.path.dirname(os.path.abspath(__file__))
PRIOR = ["--tau", "16", "--n_kl", "12", "--k", "14", "--mean_a", "2"]
LENGTHS = ["--n_warmup", "5000", "--n_burn", "10000", "--n_samples", "100000", "--seed", "0"]


def ensure_run(example, data, tag, beta, table_only):
    """The run folder, made and postprocessed unless its postprocessing summary exists."""
    run = os.path.join("results", f"pcn_Setting1_Ex{example}{tag}")
    if not os.path.exists(run + "_postprocess/summary.json"):
        if table_only:
            sys.exit(f"{run}: no postprocessing summary")
        args = ["pcn", "--example", str(example), "--setting", "1", "--data", data, "--tag", tag, *PRIOR, *LENGTHS]
        subprocess.run([sys.executable, os.path.join(HERE, "run_chains.py"), *args, *([] if beta is None else ["--beta", repr(beta)])],
                       check=True)
        subprocess.run([sys.executable, os.path.join(HERE, "postprocess.py"), run, "--data", data])
        if not os.path.exists(run + "_postprocess/summary.json"):
            sys.exit(f"{run}: postprocessing failed")
    return run


def record(label, run, factor):
    """Acceptance, convergence and efficiency of one run, from its chains and postprocessing output."""
    pp = run + "_postprocess"
    with open(f"{pp}/summary.json") as f:
        s = json.load(f)
    chains = load_chains(run)
    hashes = s.get("chain_content_sha256")             # not recorded in the summaries of the paper's Example 1 sweep
    if ((hashes is not None and hashes != [chain_content_sha256(c) for c in chains])
            or s["n_chains"] != len(chains) or s["draws_per_chain"] != len(chains[0]["z"])):
        raise ValueError(f"{pp} does not belong to the chains of {run}")
    d = pd.read_csv(f"{pp}/diagnostics.csv", index_col=0)[["r_hat", "ess_bulk", "ess_tail"]].astype(float)
    if not np.isfinite(d.to_numpy()).all():
        raise ValueError(f"{pp}: undefined R-hat or ESS")
    ess = float(min(d.ess_bulk.min(), d.ess_tail.min()))
    c = s["cost"]
    solves = c["sampling"]["forward_solves"] + c["sampling"]["adjoint_solves"]
    beta = [float(ch["info"]["beta"]) for ch in chains]
    return dict(label=label, run=run, factor=factor, adapted=bool(chains[0]["info"]["adapted"]), beta=float(np.mean(beta)),
                beta_per_chain=beta, acceptance=float(np.mean([ch["info"]["acceptance"] for ch in chains])),
                chains=s["n_chains"], draws_per_chain=s["draws_per_chain"], r_hat_max=float(d.r_hat.max()),
                ess_bulk_min=float(d.ess_bulk.min()), ess_tail_min=float(d.ess_tail.min()), min_ess=ess,
                passes=bool(d.r_hat.max() < 1.01 and ess >= 400), pde_solves_sampling=solves, min_ess_per_1000_pde_solves=1e3 * ess / solves)


def main(a):
    data = a.data or f"obs/obs_Ex{a.example}.pickle"
    tuned = ensure_run(a.example, data, a.tag, None, a.table_only)
    rows = [record("tuned (acceptance 0.234)", tuned, None)]
    beta0 = float(np.mean(rows[0]["beta_per_chain"]))
    for f in a.factors:
        run = ensure_run(a.example, data, f"{a.tag}_beta_x{f:g}", beta0 * f, a.table_only)
        r = record(f"{f:g} x beta_0", run, f)
        if r["adapted"] or any(abs(b - beta0 * f) > 1e-12 * beta0 * f for b in r["beta_per_chain"]):
            raise ValueError(f"{run}: step {r['beta_per_chain']}, not {f:g} x beta_0 = {beta0 * f!r}")
        rows.append(r)
    fixed = [r for r in rows if r["factor"] is not None]
    best = max(fixed, key=lambda r: r["min_ess_per_1000_pde_solves"]) if fixed else None
    md = [f"# pCN step-size sweep, Example {a.example} (known plateaus)", "",
          f"beta_0 = {beta0:.6g} (mean tuned step of the four chains). Each run: {rows[0]['chains']} chains of "
          f"{rows[0]['draws_per_chain']} retained draws. Min ESS: smallest bulk or tail ESS over all KL coefficients and d.", "",
          "| Run | beta | Acceptance | max R-hat | Min ESS | Min ESS per 10^3 PDE solves |", "|---|---|---|---|---|---|"]
    md += [f"| {r['label']} | {r['beta']:.4f} | {r['acceptance']:.3f} | {r['r_hat_max']:.3f} | {r['min_ess']:.0f} "
           f"| {r['min_ess_per_1000_pde_solves']:.3f} |" for r in rows]
    if best:
        md += ["", f"Largest ESS per 10^3 PDE solves: f = {best['factor']:g} (beta = {best['beta']!r})."]
    out = a.out or os.path.join("results", f"pcn_sweep_Ex{a.example}")
    with open(out + ".json", "w") as f:
        json.dump(dict(example=a.example, beta0=beta0, rows=rows, best_factor=best and best["factor"]), f, indent=2)
    with open(out + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", type=int, required=True)
    ap.add_argument("--factors", type=float, nargs="+", required=True)
    ap.add_argument("--data", default=None, help="data file (default obs/obs_Ex<example>.pickle)")
    ap.add_argument("--tag", default="_sweep", help="run-folder suffix of the tuned run; fixed runs add _beta_x<f>")
    ap.add_argument("--table_only", action="store_true", help="only the table, from existing runs")
    ap.add_argument("--out", default=None, help="output path without extension (default results/pcn_sweep_Ex<example>)")
    main(ap.parse_args())
