# make_tables.py
"""
Tables 1 and 2 of the paper (LaTeX) and the sampler comparisons quoted in the text of Sections 6.3.1, 6.3.2 and 6.4,
from the postprocessing output of the runs (summary.json, diagnostics.csv) and their batch records
(<run>_batch.json, written by run_chains.py).
Table 1: Example 1, pCN and NUTS with known plateau values, and NUTS with the plateau values inferred (Stage A of the
joint reconstruction). Table 2: Example 2, pCN and NUTS. The NUTS runs with target acceptance probability 0.99 of
Examples 1 and 2 are compared in tables.md only.
Minimum ESS: smallest bulk or tail ESS over all monitored quantities (KL coefficients, junction depth d and, with the
plateau values inferred, c_p and c_n). Forward-equivalent solves N_F + lambda N_A, lambda the run's own ratio of mean
adjoint to mean forward solve time. Pairs a / b: retained draws / including warm-up and burn-in. Wall-clock: the
four-chain batch, including warm-up.
Usage: python make_tables.py [--out results/tables] [--ex1_pcn RUN] [--ex1_nuts RUN] ... (defaults: the run folders of
the README commands)
Output: table1_ex1.tex, table2_ex2.tex, tables.json, tables.md
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from io_utils import chain_content_sha256, load_chains

RUNS = dict(ex1_pcn="results/pcn_Setting1_Ex1", ex1_nuts="results/nuts_Setting1_Ex1", ex1_nuts099="results/nuts_Setting1_Ex1_t099",
            joint="results/nuts_Setting2_Ex1_A", ex2_pcn="results/pcn_Setting1_Ex2", ex2_nuts="results/nuts_Setting1_Ex2",
            ex2_nuts099="results/nuts_Setting1_Ex2_t099")


def record(run):
    """The numbers of one run, after checking that its postprocessing output belongs to its chains."""
    pp = run + "_postprocess"
    with open(f"{pp}/summary.json") as f:
        s = json.load(f)
    with open(run + "_batch.json") as f:
        batch = json.load(f)
    chains = load_chains(run)
    if [chain_content_sha256(c) for c in chains] != s["chain_content_sha256"]:
        raise ValueError(f"{pp} does not belong to the chains of {run}")
    d = pd.read_csv(f"{pp}/diagnostics.csv", index_col=0)[["r_hat", "ess_bulk", "ess_tail"]].astype(float)
    if not np.isfinite(d.to_numpy()).all():
        raise ValueError(f"{pp}: undefined R-hat or ESS")
    b, t = float(d.ess_bulk.min()), float(d.ess_tail.min())
    ess, c, wall = min(b, t), s["cost"], batch["elapsed_s"]
    phases = ("warmup", "sampling", "total")
    fe = {ph: c[ph]["forward_equivalents"] for ph in phases}
    info = chains[0]["info"]
    rec = dict(run=run, method=s["method"], setting=1 if chains[0]["r"] is None else 2, chains=s["n_chains"],
               draws_per_chain=s["draws_per_chain"], target_accept_prob=info.get("target_accept_prob"),
               r_hat_max=float(d.r_hat.max()), ess_bulk_min=b, ess_tail_min=t, min_ess=ess, min_ess_kind="bulk" if b <= t else "tail",
               passes=bool(s["convergence"]["passes"]), RE=s["RE"], SSIM=s["SSIM"], depth_mean=s["depth_mean"], depth_sd=s["depth_std"],
               gradients={ph: c[ph]["adjoint_solves"] for ph in phases},
               pde_solves={ph: c[ph]["forward_solves"] + c[ph]["adjoint_solves"] for ph in phases},
               forward_equivalents=fe, lambda_adjoint_over_forward=c["adjoint_in_forward_equivalents"], wall_clock_s=wall,
               min_ess_per_1000_forward_equivalents={ph: 1e3 * ess / fe[ph] for ph in ("sampling", "total")},
               min_ess_per_wall_clock_s=ess / wall, sampler=s["sampler"])
    if s["method"] == "nuts":
        rec["divergences"] = {k: s["divergences"][k] for k in ("count", "per_chain", "fraction")}
    return rec


# ---- LaTeX tables (formats of the paper) ------------------------------------------------------------------

def i3(x): return f"{x:,.0f}".replace(",", "{,}")
def rng(v, f):
    a, b = f.format(min(v)), f.format(max(v)); return a if a == b else a + "--" + b
def pair(a, b, f): return f.format(a) + " / " + f.format(b)
def bold_ess(bulk, tail, kind):
    bu, ta = i3(bulk), i3(tail)
    return (r"\textbf{" + bu + "} / " + ta) if kind == 'bulk' else (bu + r" / \textbf{" + ta + "}")


def column(r):
    sm, nuts, e = r["sampler"], r["method"] == "nuts", r["min_ess_per_1000_forward_equivalents"]
    return {
        'draws': f"{r['chains']} $\\times$ {i3(r['draws_per_chain'])}",
        'step': rng([x['step_size'] for x in sm], '{:.2f}' if r['setting'] == 1 else '{:.3f}') if nuts
                else f"$\\beta={sm[0]['beta']:.3f}$ (fixed)",
        'acc': rng([x['mean_accept_prob'] for x in sm], '{:.2f}') if nuts else f"{100 * sum(x['acceptance'] for x in sm) / len(sm):.1f}\\,\\%",
        'depth': (rng([x['tree_depth_mean'] for x in sm], '{:.1f}') + f" ({max(x['tree_depth_max'] for x in sm)})") if nuts else "--",
        'div': f"{r['divergences']['count']} ({100 * r['divergences']['fraction']:.2f}\\,\\%)" if nuts else "--",
        'rhat': f"{r['r_hat_max']:.3f}",
        'ess': bold_ess(r['ess_bulk_min'], r['ess_tail_min'], r['min_ess_kind']),
        'grad': pair(r['gradients']['sampling'] / 1e6, r['gradients']['total'] / 1e6, '{:.2f}') if nuts else "--",
        'pde': pair(r['pde_solves']['sampling'] / 1e6, r['pde_solves']['total'] / 1e6, '{:.2f}'),
        'wall': f"{r['wall_clock_s'] / 60:.0f}",
        'eff': pair(e['sampling'], e['total'], '{:.2f}'),
        'effw': f"{r['min_ess_per_wall_clock_s']:.3f}",
    }


ROWS = [('draws', 'Retained draws'), ('step', 'Step size'), ('acc', 'Acceptance'), ('depth', 'Tree depth, mean (max.)'),
        ('div', 'Divergent transitions'), ('rhat', '$\\max\\hat R$'), ('ess', 'Min.\\ bulk / tail ESS'),
        ('grad', 'Gradients ($10^6$)'), ('pde', 'PDE solves ($10^6$)'),
        ('wall', 'Wall-clock time (min)'), ('eff', 'ESS per $10^3$ forward-eq.\\ solves'),
        ('effw', 'ESS per wall-clock second')]


def table(cols, head, caption, label, foot, src):
    spec = 'l' + 'c' * len(cols)
    body = "\n".join(lab + " & " + " & ".join(c[k] for c in cols) + r" \\" for k, lab in ROWS)
    t = (f"% generated by make_tables.py from {src}; do not edit by hand\n"
         "\\begin{table}[t]\n\\centering\n\\color{blue}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
         f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}{{{spec}}}\n\\hline\n"
         f"{head}\n\\hline\n{body}\n\\hline\n\\end{{tabular}}\n")
    if foot:
        t += f"\n\\smallskip\n\\parbox{{0.97\\linewidth}}{{\\footnotesize {foot}}}\n"
    return t + "\\end{table}\n"


FOOT1 = ("NUTS: dense mass matrix, maximum tree depth 10, target acceptance probability 0.95 (known plateaus) "
         "and 0.99 (plateau values inferred). Acceptance: mean acceptance probability (pCN) and mean acceptance "
         "statistic (NUTS). Bold: minimum ESS. Pairs $a$\\,/\\,$b$: retained draws / including warm-up and burn-in, "
         "which the ESS per wall-clock second always includes.")


def md_summary(recs):
    """Every run and, for each NUTS run with known plateaus, its efficiency relative to pCN of the same example."""
    L = ["# Sampler comparison (Examples 1 and 2, joint reconstruction Stage A)", "",
         "| Run | Folder | RE / SSIM | d mean / sd | max R-hat | Min bulk / tail ESS | Divergences (per chain) "
         "| ESS per 10^3 FE, retained / total | Wall-clock (min) | ESS per s | Relative to pCN: retained / total / per s |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in recs.items():
        e, rel = r["min_ess_per_1000_forward_equivalents"], "–"
        pcn = recs.get(k[:3] + "_pcn")
        if r["method"] == "nuts" and r["setting"] == 1 and pcn:
            ep = pcn["min_ess_per_1000_forward_equivalents"]
            rel = (f"{e['sampling'] / ep['sampling']:.1f} / {e['total'] / ep['total']:.1f} / "
                   f"{r['min_ess_per_wall_clock_s'] / pcn['min_ess_per_wall_clock_s']:.1f}")
        div = f"{r['divergences']['count']} ({r['divergences']['per_chain']})" if "divergences" in r else "–"
        L.append(f"| {k} | `{r['run']}` | {r['RE']:.3f} / {r['SSIM']:.3f} | {r['depth_mean']:.3f} / {r['depth_sd']:.3f} "
                 f"| {r['r_hat_max']:.4f} | {r['ess_bulk_min']:.0f} / {r['ess_tail_min']:.0f} | {div} "
                 f"| {e['sampling']:.2f} / {e['total']:.2f} | {r['wall_clock_s'] / 60:.0f} | {r['min_ess_per_wall_clock_s']:.3f} | {rel} |")
    return L


def main(a):
    runs = {k: getattr(a, k) for k in RUNS}
    recs = {k: record(run) for k, run in runs.items() if run}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "tables.json"), "w") as f:
        json.dump(recs, f, indent=2, default=float)
    t1 = table([column(recs[k]) for k in ("ex1_pcn", "ex1_nuts", "joint")],
               r" & \multicolumn{2}{c}{Known plateaus} & Plateaus inferred \\" + "\n" + r" & pCN & NUTS & NUTS \\",
               r"Sampler comparison for Example~1 (four chains each): pCN and NUTS with known plateau values, and NUTS with the plateau values inferred (Section~\ref{subsec:joint_reconstruction}).",
               "tab:ex1_sampler_comparison", FOOT1, ", ".join(runs[k] for k in ("ex1_pcn", "ex1_nuts", "joint")))
    t2 = table([column(recs[k]) for k in ("ex2_pcn", "ex2_nuts")], r" & pCN & NUTS \\",
               r"Sampler comparison for Example~2 with known plateau values (four chains each); entries as in Table~\ref{tab:ex1_sampler_comparison}.",
               "tab:ex2_sampler_comparison", None, ", ".join(runs[k] for k in ("ex2_pcn", "ex2_nuts")))
    for name, t in (("table1_ex1.tex", t1), ("table2_ex2.tex", t2)):
        with open(os.path.join(a.out, name), "w") as f:
            f.write(t)
    md = md_summary(recs)
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(t1 + "\n" + t2 + "\n" + "\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in RUNS.items():
        ap.add_argument(f"--{k}", default=v, help=f"run folder (default {v}; an empty string leaves an optional run out)")
    ap.add_argument("--out", default="results/tables")
    main(ap.parse_args())
