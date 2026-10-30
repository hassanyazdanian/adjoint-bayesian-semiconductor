# production_table.py
"""
Tables of the production runs of one manifest for the revised paper (Reviewer 1, comment 1).
Everything is read from the outputs of production.sh and checked before use:
<run>_postprocess/{summary.json, diagnostics.csv, checkpoints.json, fields.npz}, <run>_batch.json and
the chain files must match the manifest (data hash, manifest identity, draws, chains) and each other
(chain contents, data hash). Optional tuning-pilot runs (pCN step-size sweep, NUTS target acceptance)
must have the manifest's prior and data.

Efficiency is measured with the bottleneck ESS: the smallest bulk or tail ESS over all monitored
quantities (every KL coefficient and the junction depth), which sets the run length under the
stopping rule (R-hat < 1.01, bulk and tail ESS >= 400). Costs:
- PDE solves: raw counts of forward plus adjoint solves. A NUTS gradient is one forward and one adjoint
  solve; pCN makes one forward solve per step and no adjoint solve.
- Chain time: the elapsed (wall-clock) sampler time of each chain, summed over the chains (model set-up
  excluded). It is not CPU time: each process also ran two idle-polling MPI (libfabric) threads.
- Wall-clock time: elapsed time of the four-chain batch (one process per chain, from the launch to the
  end of the last process, including start-up and model set-up).
"Sampling" counts the retained draws only; "incl. warm-up" adds the NUTS adaptation or the pCN
warm-up and burn-in.

Usage (stages of another manifest as <manifest file>:<stage>):
  python production_table.py --stages ex1_pcn ex1_nuts experiments/production_v1_t095.json:ex1_nuts --out results/production_v1_report/ex1 \
      --pilot "pCN tuned=results/pcn_Setting1_Ex1_pilot" --pilot "pCN 8x=results/pcn_Setting1_Ex1_pilot_beta_x8" ...
Writes tables.json, tables.md and table1.tex, tableA1.tex ... tableA5.tex (booktabs).
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from io_utils import chain_content_sha256, code_version, file_sha256, load_chains

FAILURE_KEYS = ("forward_failures", "forward_restarts", "forward_failed_evaluations", "adjoint_failures")
NAMES = {"nuts": "NUTS", "pcn": "pCN"}


def _json(path):
    with open(path) as f:
        return json.load(f)


def _label(q, tex=False):
    """Quantity label as in the paper: z[j] -> z_{j+1}, depth -> d."""
    if q == "depth":
        return "$d$" if tex else "d"
    j = int(q[2:-1]) + 1
    return f"$z_{{{j}}}$" if tex else f"z_{j}"


def _n(x, tex=False):
    s = f"{x:,.0f}"
    return s.replace(",", "{,}") if tex else s


def _tex_run_label(label):
    """'pCN 8x' -> 'pCN, $8\\beta_0$'; 'pCN 0.5x' -> 'pCN, $\\tfrac12\\beta_0$'; other labels unchanged."""
    m = re.fullmatch(r"(\S+) (\d*\.?\d+)x", label)
    if m is None:
        return label
    f = "\\tfrac12" if m.group(2) == "0.5" else m.group(2)
    return f"{m.group(1)}, ${f}\\beta_0$"


def _diagnostics(path):
    """Smallest bulk and tail ESS and largest R-hat over the monitored quantities (diagnostics.csv)."""
    diag = pd.read_csv(path, index_col=0)
    if not np.isfinite(diag[["r_hat", "ess_bulk", "ess_tail"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: undefined R-hat or ESS; the run has no bottleneck ESS")
    jb, jt, jr = diag.ess_bulk.idxmin(), diag.ess_tail.idxmin(), diag.r_hat.idxmax()
    b, t = float(diag.ess_bulk[jb]), float(diag.ess_tail[jt])
    out = dict(r_hat_max=float(diag.r_hat[jr]), r_hat_max_quantity=jr, ess_bulk_min=b, ess_bulk_min_quantity=jb,
               ess_tail_min=t, ess_tail_min_quantity=jt, ess=min(b, t), quantity=jb if b <= t else jt,
               kind="bulk" if b <= t else "tail")
    per_quantity = {q: dict(r_hat=float(r.r_hat), ess_bulk=float(r.ess_bulk), ess_tail=float(r.ess_tail))
                    for q, r in diag.iterrows()}
    return out, per_quantity


def _efficiency(ess, solves, chain_time, wall=None):
    out = dict(per_1000_solves_sampling=1e3 * ess / solves["sampling"], per_1000_solves_total=1e3 * ess / solves["total"],
               per_chain_s_sampling=ess / chain_time["sampling"], per_chain_s_total=ess / chain_time["total"])
    if wall is not None:
        out["per_wall_clock_s"] = ess / wall
    return out


def _clusters(idx, depth, gap=30):
    """Divergent retained draws (0-based) grouped into runs with gaps <= `gap` draws; 1-based in the output."""
    out = []
    for i in np.sort(np.asarray(idx, dtype=int)):
        if out and i - out[-1]["last"] + 1 <= gap:
            out[-1]["last"] = int(i) + 1
            out[-1]["count"] += 1
            out[-1]["depth"] = [min(out[-1]["depth"][0], float(depth[i])), max(out[-1]["depth"][1], float(depth[i]))]
        else:
            out.append(dict(first=int(i) + 1, last=int(i) + 1, count=1, depth=[float(depth[i])] * 2))
    return out


def stage_record(m, manifest_path, name):
    """All numbers of one production stage, after checking that its outputs belong together."""
    st = m["stages"][name]
    method, example = st["method"], st["example"]
    run = f"results/{method}_Setting{st['setting']}_Ex{example}{m['output_tag']}"
    pp = run + "_postprocess"
    s, ck, batch = _json(f"{pp}/summary.json"), _json(f"{pp}/checkpoints.json"), _json(f"{run}_batch.json")
    fields = np.load(f"{pp}/fields.npz")
    chains = load_chains(run)
    hashes = [chain_content_sha256(c) for c in chains]
    data_sha = m["data"][f"ex{example}"]["sha256"]
    manifests = [json.loads(x) for x in s["chain_provenance"]["manifest"]]
    cap = m["nuts"]["num_samples_cap"] if method == "nuts" else m["pcn"]["n_samples_cap"]
    checks = {
        "summary data hash = manifest": s["data_sha256"] == data_sha,
        "checkpoint data hash = manifest": ck["data_sha256"] == data_sha,
        "fields data hash = manifest": str(fields["data_sha256"]) == data_sha,
        "chain data hashes = manifest": all(c["provenance"]["data_sha256"] == data_sha for c in chains),
        "chain contents = summary": hashes == s["chain_content_sha256"],
        "chain contents = checkpoints": hashes == ck["chain_content_sha256"],
        "chain contents = fields.npz": hashes == [str(h) for h in fields["chain_content_sha256"]],
        "manifest identity": len(manifests) == 1 and manifests[0]["id"] == m["id"]
                             and manifests[0]["sha256"] == file_sha256(manifest_path),
        "method": s["method"] == method,
        "chains and draws": s["n_chains"] == m["chains"]["count"] and s["draws_per_chain"] == cap,
        "diagnostics complete": s["convergence"]["complete"],
        "batch record": batch["stage"] == name and os.path.basename(batch["manifest"]) == os.path.basename(manifest_path),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{run}: outputs do not belong together ({', '.join(failed)}); no table written")

    diag, per_quantity = _diagnostics(f"{pp}/diagnostics.csv")
    c = s["cost"]
    solves = {ph: c[ph]["forward_solves"] + c[ph]["adjoint_solves"] for ph in ("warmup", "sampling", "total")}
    chain_time = {ph: c[ph]["worker_time_s"] for ph in ("warmup", "sampling", "total")}
    rec = dict(
        stage=name, run=run, method=method, example=example, setting=st["setting"],
        code=sorted({json.loads(x)["git_commit"] for x in s["chain_provenance"]["code"]}),
        chains=s["n_chains"], draws_per_chain=s["draws_per_chain"], passes=s["convergence"]["passes"],
        diagnostics=diag, per_quantity=per_quantity,
        RE=s["RE"], SSIM=s["SSIM"], depth_mean=s["depth_mean"], depth_std=s["depth_std"],
        depth_mcse=s["depth"]["mcse_mean"],
        cost=dict(solves=solves, chain_s=chain_time,
                  forward={ph: c[ph]["forward_solves"] for ph in ("warmup", "sampling", "total")},
                  adjoint={ph: c[ph]["adjoint_solves"] for ph in ("warmup", "sampling", "total")},
                  newton={ph: c[ph]["newton_iterations"] for ph in ("warmup", "sampling", "total")},
                  forward_equivalents={ph: c[ph]["forward_equivalents"] for ph in ("warmup", "sampling", "total")},
                  adjoint_in_forward_equivalents=c["adjoint_in_forward_equivalents"],
                  failures={k: c["total"][k] for k in FAILURE_KEYS},
                  wall_clock_s=batch["elapsed_s"], wall_clock_s_chain_timestamps=c["batch"]["elapsed_s_launch_to_end"],
                  longest_chain_s=c["elapsed_time_s_longest_chain"], process_exit_status=batch["exit_status"]),
        checkpoints=ck["checkpoints"], milestone=ck["milestone"], checkpoint_every=ck["every"])
    rec["efficiency"] = _efficiency(diag["ess"], solves, chain_time, batch["elapsed_s"])
    rec["manifest"], rec["data_sha256"] = m["id"], data_sha
    if method == "nuts":
        depth = fields["depth"]
        rec["sampler"] = [dict(x, divergent_draws=_clusters(ch["info"]["divergences"], depth[i]))
                          for i, (x, ch) in enumerate(zip(s["sampler"], chains))]
        rec["divergences"] = {k: s["divergences"][k] for k in ("count", "per_chain", "fraction", "follow_up", "follow_up_rule")}
        rec["mass_matrix"], rec["target_accept_prob"] = chains[0]["info"]["mass_matrix"], chains[0]["info"]["target_accept_prob"]
        rec["warmup_steps"] = chains[0]["info"]["warmup_steps"]
    else:
        rec["sampler"] = [dict(x, accepted_fraction=ch["info"].get("accepted_fraction")) for x, ch in zip(s["sampler"], chains)]
        rec["beta_rule"] = m["pcn"]["beta_rule"]
        rule = re.match(r"(\d+(?:\.\d+)?) x the mean tuned beta \(([\d.]+)\)", rec["beta_rule"])
        rec["beta_multiple"], rec["beta_tuned"] = (float(rule.group(1)), float(rule.group(2))) if rule else (None, None)
    return rec


def _labels(recs):
    """Row labels: the sampler, and for several NUTS runs also the target acceptance probability."""
    n_nuts = sum(r["method"] == "nuts" for r in recs)
    for r in recs:
        r["label"] = NAMES[r["method"]] + (f" ({r['target_accept_prob']})" if r["method"] == "nuts" and n_nuts > 1 else "")
        r["key"] = r["method"] + (f"_t{str(r['target_accept_prob']).replace('.', '')}" if r["method"] == "nuts" and n_nuts > 1 else "")


def pilot_record(m, label, run):
    """A tuning-pilot run (earlier, shorter): same prior and data as the manifest."""
    pp = run.rstrip("/") + "_postprocess"
    s = _json(f"{pp}/summary.json")
    chains = load_chains(run)
    p = m["model"]["prior"]
    shas = {v["sha256"] for k, v in m["data"].items() if k.startswith("ex")}
    for ch in chains:
        pr = ch["prior"]
        if [pr.get(k) for k in ("tau", "alpha", "n_kl", "k", "mean_a")] != [p[k] for k in ("tau", "alpha", "n_kl", "k", "mean_a")]:
            raise ValueError(f"{run}: prior {pr} is not the manifest's")
        if ch["provenance"]["data_sha256"] not in shas:
            raise ValueError(f"{run}: sampled for data {ch['provenance']['data_sha256'][:12]}..., not a manifest data file")
    if s["draws_per_chain"] != len(chains[0]["z"]) or s["n_chains"] != len(chains):
        raise ValueError(f"{run}: summary does not match the chain files")
    diag, _ = _diagnostics(f"{pp}/diagnostics.csv")
    c = s["cost"]
    solves = {ph: c[ph]["forward_solves"] + c[ph]["adjoint_solves"] for ph in ("warmup", "sampling", "total")}
    chain_time = {ph: c[ph]["worker_time_s"] for ph in ("warmup", "sampling", "total")}
    rec = dict(label=label, run=run, method=s["method"], chains=s["n_chains"], draws_per_chain=s["draws_per_chain"],
               diagnostics=diag, solves=solves, chain_s=chain_time, efficiency=_efficiency(diag["ess"], solves, chain_time))
    info = [ch["info"] for ch in chains]
    if s["method"] == "nuts":
        st = [x["step_size"] for x in s["sampler"]]
        rec.update(warmup=info[0]["warmup_steps"], target_accept_prob=info[0]["target_accept_prob"],
                   mass_matrix=info[0]["mass_matrix"], step_size=[min(st), max(st)],
                   acceptance=float(np.mean([x["mean_accept_prob"] for x in s["sampler"]])),
                   divergences=int(sum(x["divergences"] for x in s["sampler"])))
    else:
        beta = [float(i["beta"]) for i in info]
        rec.update(warmup=info[0]["n_warmup"] + info[0]["n_burn"], beta=float(np.mean(beta)), beta_range=[min(beta), max(beta)],
                   adapted=bool(info[0]["adapted"]), acceptance=float(np.mean([i["acceptance"] for i in info])))
    return rec


# ---- output -----------------------------------------------------------------------------------------

def _tex_table(caption, label, colspec, header, rows, note=None):
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    lines += [" & ".join(h) + " \\\\" for h in header] + ["\\midrule"]
    lines += [r if isinstance(r, str) else " & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}}"]
    if note:
        lines += ["", "\\smallskip", f"\\parbox{{0.97\\linewidth}}{{\\footnotesize {note}}}"]
    return "\n".join(lines + ["\\end{table}", ""])


def _md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def _ess_cell(d, kind, tex):
    v = d[f"ess_{kind}_min"]
    s = _n(v, tex)
    return (f"\\textbf{{{s}}}" if tex else f"**{s}**") if d["kind"] == kind else s


def table1(recs, tex):
    """Compact comparison for the main text."""
    rows = []
    for r in recs:
        d, e, c = r["diagnostics"], r["efficiency"], r["cost"]
        rows.append([r["label"], f"{r['chains']} $\\times$ {_n(r['draws_per_chain'], True)}" if tex else f"{r['chains']} × {_n(r['draws_per_chain'])}",
                     f"{d['r_hat_max']:.3f}", f"{_ess_cell(d, 'bulk', tex)} / {_ess_cell(d, 'tail', tex)}",
                     f"{c['solves']['sampling'] / 1e6:.2f} / {c['solves']['total'] / 1e6:.2f}",
                     f"{c['wall_clock_s'] / 60:.0f}",
                     f"{e['per_1000_solves_sampling']:.2f} / {e['per_1000_solves_total']:.2f}",
                     f"{e['per_chain_s_sampling']:.3f} / {e['per_chain_s_total']:.3f}",
                     f"{r['RE']:.3f}", f"{r['SSIM']:.3f}"])
    if tex:
        header = [["Method", "Draws", "$\\max \\hat R$", "Min.\\ ESS", "PDE solves ($10^6$)", "Wall-clock",
                   "ESS per $10^3$", "ESS per", "RE", "SSIM"],
                  ["", "", "", "bulk / tail", "sampl.\\ / total", "(min)", "PDE solves", "chain-second", "", ""]]
        return header, rows
    header = ["Method", "Draws", "max R̂", "min ESS bulk / tail", "PDE solves (10⁶), sampling / total", "Wall-clock (min)",
              "ESS per 1000 PDE solves, sampling / total", "ESS per chain-second, sampling / total", "RE", "SSIM"]
    return header, rows


def write(recs, pilots, out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "tables.json"), "w") as f:
        json.dump(dict(stages=recs, pilots=pilots, table_code=code_version()), f, indent=2, default=float)
    _labels(recs)
    nuts_recs = [r for r in recs if r["method"] == "nuts"]
    pcn = next((r for r in recs if r["method"] == "pcn"), None)
    md, tex = [], {}
    ex = recs[0]["example"]

    # Table 1
    h, rows = table1(recs, False)
    md += [f"## Table 1: Example {ex}, known plateaus", "", _md_table(h, rows), ""]
    for r in nuts_recs if pcn is not None else []:
        en, ep = r["efficiency"], pcn["efficiency"]
        md += [f"{r['label']} / pCN, bottleneck ESS: " + ", ".join(
            f"{k.replace('_', ' ')} {en[k] / ep[k]:.1f}×" for k in en), ""]
    for r in recs:
        d = r["diagnostics"]
        md.append(f"- {r['label']} ({r['manifest']}): bottleneck ESS {d['ess']:.0f} ({d['kind']} ESS of {_label(d['quantity'])}); "
                  f"max R̂ {d['r_hat_max']:.4f} ({_label(d['r_hat_max_quantity'])}); passes: {r['passes']}; "
                  f"depth {r['depth_mean']:.4f} ± {r['depth_std']:.4f} (MCSE {r['depth_mcse']:.5f}); code {', '.join(r['code'])}")
    h, rows = table1(recs, True)
    tex["table1"] = _tex_table(
        f"Comparison of pCN and NUTS for Example~{ex} with known plateaus (four chains each).", f"tab:ex{ex}_comparison",
        "lcccccccc" + "c", h, rows,
        "Draws: chains $\\times$ retained draws per chain. $\\hat R$ and ESS are the rank-normalized split $\\hat R$ "
        "and the bulk and tail ESS of Vehtari et al., over all KL coefficients and the junction depth; bold marks the "
        "bottleneck ESS (the smallest bulk or tail ESS), which is used for all efficiencies. PDE solves are forward plus "
        "adjoint solves, for the retained draws (sampl.) and including warm-up and burn-in (total). Wall-clock is the "
        "elapsed time of the four-chain batch (one process per chain); chain time is the elapsed sampler time of each chain, "
        "summed over the four chains. "
        "Efficiencies are given for the retained draws and including warm-up."
        + (" NUTS (0.95), NUTS (0.99): target acceptance probability." if len(nuts_recs) > 1 else ""))

    # A1: sampler settings and per-chain statistics
    for r in recs:
        if r["method"] == "nuts":
            h = ["Chain", "Step size", "Accept. prob.", "Tree depth mean / max", "Gradients / iter. mean / max",
                 "Warm-up gradients / iter.", "Divergences (retained draws)"]
            rows = [[str(x["chain"] + 1), f"{x['step_size']:.3f}", f"{x['mean_accept_prob']:.3f}",
                     f"{x['tree_depth_mean']:.2f} / {x['tree_depth_max']}",
                     f"{x['grads_per_iteration_mean']:.1f} / {x['grads_per_iteration_max']}",
                     f"{x['warmup_grads_per_iteration_mean']:.1f}",
                     f"{x['divergences']}" + ("" if not x["divergent_draws"] else " (" + "; ".join(
                         f"{g['first']}" if g["count"] == 1 else f"{g['first']}–{g['last']}: {g['count']}"
                         for g in x["divergent_draws"]) + ")")] for x in r["sampler"]]
            dv = r["divergences"]
            sfx = r["key"][4:]
            md += ["", f"## Table A1 ({r['label']}): {r['mass_matrix']} mass matrix, target acceptance {r['target_accept_prob']}, "
                       f"maximum tree depth 10, {r['warmup_steps']} warm-up iterations; divergences {dv['count']} "
                       f"({100 * dv['fraction']:.2f} %), follow-up rule triggered: {dv['follow_up']}", "", _md_table(h, rows)]
            tex[f"tableA1_nuts{sfx}"] = _tex_table(
                f"NUTS for Example~{ex}: per-chain statistics ({r['mass_matrix']} mass matrix, target acceptance probability "
                f"{r['target_accept_prob']}, maximum tree depth 10, {r['warmup_steps']} warm-up iterations, "
                f"{_n(r['draws_per_chain'], True)} retained draws per chain).", f"tab:ex{ex}_nuts_chains{sfx}", "ccccccl",
                [["Chain", "Step size", "Accept.", "Tree depth", "Gradients / iter.", "Warm-up", "Divergences"],
                 ["", "", "prob.", "mean / max", "mean / max", "grad.\\ / iter.", "(retained draws)"]],
                [[c.replace("–", "--") for c in row] for row in rows],
                f"Divergences: {dv['count']} of {_n(r['chains'] * r['draws_per_chain'], True)} retained draws "
                f"({100 * dv['fraction']:.2f}\\,\\%); in parentheses the retained draws (or ranges: count) with a "
                f"divergent transition. Accept.\\ prob.\\ is the mean acceptance statistic of the retained iterations.")
        else:
            h = ["Chain", "β", "Mean acceptance probability", "Accepted fraction", "Warm-up + burn-in steps", "Tuning"]
            rows = [[str(x["chain"] + 1), f"{x['beta']:.4f}", f"{x['acceptance']:.4f}",
                     "–" if x["accepted_fraction"] is None else f"{x['accepted_fraction']:.4f}",
                     f"{_n(x['n_warmup'])} + {_n(x['n_burn'])}", x["tuning"]] for x in r["sampler"]]
            md += ["", f"## Table A1 (pCN): fixed β; rule: {r['beta_rule']}", "", _md_table(h, rows)]
            tex["tableA1_pcn"] = _tex_table(
                f"pCN for Example~{ex}: per-chain statistics (fixed step $\\beta$, no adaptation during the run, "
                f"{_n(r['draws_per_chain'], True)} retained draws per chain).", f"tab:ex{ex}_pcn_chains", "cccc",
                [["Chain", "$\\beta$", "Mean acceptance probability", "Accepted fraction"]],
                [[row[0], row[1], row[2], row[3]] for row in rows],
                f"$\\beta$ is {r['beta_multiple']:g} times the mean step size $\\beta_0={r['beta_tuned']:.4f}$ of four chains tuned "
                f"to an acceptance rate of 0.234; the multiple was selected by the step-size sweep of "
                f"Table~\\ref{{tab:ex{ex}_tuning}}. Warm-up: "
                f"{_n(r['sampler'][0]['n_warmup'], True)} + {_n(r['sampler'][0]['n_burn'], True)} discarded steps per chain.")

    # A2: cost
    h = ["Sampler", "Phase", "Forward solves", "Adjoint solves (= gradients)", "Newton iterations", "Chain time (s)"]
    rows, trows = [], []
    for r in recs:
        c = r["cost"]
        trows += ["\\midrule"] if trows else []
        for ph, name in (("warmup", "warm-up"), ("sampling", "sampling"), ("total", "total")):
            rows.append([r["label"], name, _n(c["forward"][ph]), _n(c["adjoint"][ph]), _n(c["newton"][ph]), _n(c["chain_s"][ph])])
            trows.append([r["label"] if ph == "warmup" else "", name, _n(c["forward"][ph], True), _n(c["adjoint"][ph], True),
                          _n(c["newton"][ph], True), _n(c["chain_s"][ph], True)])
    md += ["", "## Table A2: PDE work and time (sums over the four chains)", "", _md_table(h, rows), ""]
    for r in recs:
        c = r["cost"]
        md.append(f"- {r['label']}: batch wall-clock {c['wall_clock_s']:.0f} s (chains' own timestamps "
                  f"{c['wall_clock_s_chain_timestamps']:.0f} s), longest chain {c['longest_chain_s']:.0f} s; failed or restarted "
                  f"solves: {sum(c['failures'].values())}; an adjoint solve takes {c['adjoint_in_forward_equivalents']:.2f} "
                  f"forward-solve times; process exit statuses {c['process_exit_status']} (MPI finalisation, outputs validated)")
    note = "; ".join(f"{r['label']}: wall-clock {r['cost']['wall_clock_s'] / 60:.0f}\\,min for the batch, "
                     f"longest chain {r['cost']['longest_chain_s'] / 60:.0f}\\,min" for r in recs)
    failures = sum(sum(r["cost"]["failures"].values()) for r in recs)
    note += ". " + ("No forward or adjoint solve failed or was restarted." if failures == 0 else
                    f"{failures} solves failed or were restarted (see tables.json).")
    if nuts_recs:
        lam = sorted({f"{r['cost']['adjoint_in_forward_equivalents']:.2f}" for r in nuts_recs})
        note += (" A NUTS gradient costs one forward and one adjoint solve; an adjoint solve took on average "
                 f"{' and '.join(lam)} times as long as a nonlinear forward solve.")
    tex["tableA2"] = _tex_table(
        f"PDE work and chain time (elapsed sampler time per chain) of the Example~{ex} runs, summed over the four chains.", f"tab:ex{ex}_cost",
        "llrrrr", [["Sampler", "Phase", "Forward", "Adjoint solves", "Newton", "Chain time (s)"],
                   ["", "", "solves", "(= gradients)", "iterations", ""]], trows,
        note + " Warm-up: NUTS adaptation; pCN warm-up and burn-in.")

    # A3: per-quantity diagnostics
    qs = list(recs[0]["per_quantity"])
    h = ["Quantity"] + [f"{r['label']} {k}" for r in recs for k in ("R̂", "bulk ESS", "tail ESS")]
    rows = [[_label(q)] + [f"{r['per_quantity'][q][k]:.4f}" if k == "r_hat" else _n(r["per_quantity"][q][k])
                           for r in recs for k in ("r_hat", "ess_bulk", "ess_tail")] for q in qs]
    md += ["", "## Table A3: diagnostics of every monitored quantity", "", _md_table(h, rows)]
    trows = [[_label(q, True)] + [f"{r['per_quantity'][q][k]:.4f}" if k == "r_hat" else _n(r["per_quantity"][q][k], True)
                                  for r in recs for k in ("r_hat", "ess_bulk", "ess_tail")] for q in qs]
    tex["tableA3"] = _tex_table(
        f"Rank-normalized split $\\hat R$ and bulk and tail ESS of every monitored quantity, Example~{ex} "
        f"($z_j$: KL coefficients, $d$: junction depth at $x=0.5$).", f"tab:ex{ex}_diagnostics",
        "c" + "ccc" * len(recs),
        [[""] + [f"\\multicolumn{{3}}{{c}}{{{r['label']}}}" for r in recs],
         ["Quantity"] + ["$\\hat R$", "bulk ESS", "tail ESS"] * len(recs)], trows)

    # A4: checkpoints
    h = ["Sampler", "Draws per chain", "max R̂", "min bulk / tail ESS", "Passes", "Divergences",
         "PDE solves incl. warm-up (10⁶)", "Elapsed from launch (min)"]
    rows, trows, milestones = [], [], []
    for r in recs:
        for k, p in enumerate(r["checkpoints"]):
            cst = p["cost"]
            total = cst["forward"] + cst["adjoint"] + cst["warmup_forward"] + cst["warmup_adjoint"]
            div = "–" if r["method"] != "nuts" else str(p["divergences"])
            cells = [_n(p["draws_per_chain"]), f"{p['r_hat_max']:.4f}", f"{_n(p['ess_bulk_min'])} / {_n(p['ess_tail_min'])}",
                     "yes" if p["passes"] else "no", div, f"{total / 1e6:.2f}", f"{cst['elapsed_s_from_launch'] / 60:.0f}"]
            rows.append([r["label"]] + cells)
            trows.append([r["label"] if k == 0 else "", _n(p["draws_per_chain"], True), cells[1]]
                         + [f"{_n(p['ess_bulk_min'], True)} / {_n(p['ess_tail_min'], True)}"] + [x.replace("–", "--") for x in cells[3:]])
        trows.append("\\midrule")
        ms = r["milestone"]
        md_ms = (f"first pass at {_n(ms['first_pass_draws'])} draws per chain, persists: {ms['persists']}"
                 if ms["first_pass"] is not None else "not reached within the cap")
        milestones.append(f"- {r['label']} milestone ({ms['statement']}): {md_ms}")
    trows = trows[:-1]
    md += ["", "## Table A4: diagnostics on prefixes of the chains (retrospective)", "", _md_table(h, rows), ""] + milestones
    tex["tableA4"] = _tex_table(
        f"Diagnostics on prefixes of the four chains, Example~{ex} (retrospective: every run was executed to its cap).",
        f"tab:ex{ex}_checkpoints", "lccccccc",
        [["Sampler", "Draws", "$\\max\\hat R$", "Min.\\ ESS", "Passes", "Div.", "PDE solves", "Elapsed"],
         ["", "per chain", "", "bulk / tail", "", "", "total ($10^6$)", "(min)"]], trows,
        "Passes: $\\hat R<1.01$ and bulk and tail ESS $\\ge 400$ for all KL coefficients and the junction depth. "
        "PDE solves include the warm-up; elapsed time is measured from the launch of the batch.")

    # A5: tuning pilot
    if pilots:
        h = ["Run", "Setting", "Warm-up", "Draws", "Acceptance", "Divergences", "max R̂", "min bulk / tail ESS (quantity)",
             "Bottleneck ESS per 1000 PDE solves, sampling / incl. warm-up"]
        rows, trows = [], []
        for p in pilots:
            d, e = p["diagnostics"], p["efficiency"]
            setting = (f"β = {p['beta']:.3f}" + (f" ({p['beta_range'][0]:.3f}–{p['beta_range'][1]:.3f}, tuned)" if p["adapted"] else "")
                       if p["method"] == "pcn" else f"target {p['target_accept_prob']}, {p['mass_matrix']}")
            cells = [setting, _n(p["warmup"]), f"{p['chains']} × {_n(p['draws_per_chain'])}", f"{p['acceptance']:.3f}",
                     "–" if p["method"] == "pcn" else str(p["divergences"]), f"{d['r_hat_max']:.3f}",
                     f"{_n(d['ess_bulk_min'])} / {_n(d['ess_tail_min'])} ({_label(d['quantity'])}, {d['kind']})",
                     f"{e['per_1000_solves_sampling']:.2f} / {e['per_1000_solves_total']:.2f}"]
            rows.append([p["label"]] + cells)
            tset = (f"$\\beta={p['beta']:.3f}$" + (" (tuned)" if p["adapted"] else "")
                    if p["method"] == "pcn" else f"target {p['target_accept_prob']}")
            trows.append([_tex_run_label(p["label"]), tset, _n(p["warmup"], True), f"{p['chains']} $\\times$ {_n(p['draws_per_chain'], True)}",
                          f"{p['acceptance']:.3f}",
                          "--" if p["method"] == "pcn" else str(p["divergences"]), f"{d['r_hat_max']:.3f}",
                          f"{_n(d['ess'], True)} ({_label(d['quantity'], True)}, {d['kind']})",
                          f"{e['per_1000_solves_sampling']:.2f} / {e['per_1000_solves_total']:.2f}"])
        md += ["", "## Table A5: tuning pilot (shorter runs, same prior and data)", "", _md_table(h, rows)]
        notes = []
        tuned = [p for p in pilots if p["method"] == "pcn" and p["adapted"]]
        if tuned:
            notes.append("pCN tuned: per-chain Robbins--Monro adaptation to an acceptance rate of 0.234 during the warm-up; "
                         f"the other pCN rows use fixed multiples of the mean tuned step $\\beta_0={tuned[0]['beta']:.4f}$.")
        mass = sorted({p["mass_matrix"] for p in pilots if p["method"] == "nuts"})
        if mass:
            notes.append(f"NUTS: {', '.join(mass)} mass matrix.")
        notes.append("Warm-up: discarded steps (pCN) or adaptation iterations (NUTS) per chain.")
        notes.append("Bottleneck ESS: the smallest bulk or tail ESS over all KL coefficients and the junction depth. "
                     "PDE solves: forward plus adjoint solves.")
        tex["tableA5"] = _tex_table(
            f"Tuning pilot for Example~{ex} (four chains per row, shorter than the production runs, same prior and data).",
            f"tab:ex{ex}_tuning", "llccccccc",
            [["Run", "Setting", "Warm-up", "Draws", "Accept.", "Div.", "$\\max\\hat R$", "Bottleneck ESS", "ESS per $10^3$ solves"],
             ["", "", "", "", "", "", "", "(quantity, kind)", "sampl.\\ / total"]], trows, " ".join(notes))

    with open(os.path.join(out, "tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    for old in os.listdir(out):            # tables of an earlier invocation that this one does not write
        if re.fullmatch(r"table(1|A\d\w*)\.tex", old) and old[:-4] not in tex:
            os.remove(os.path.join(out, old))
    for name, body in tex.items():
        with open(os.path.join(out, f"{name}.tex"), "w") as f:
            f.write(body)
    return md


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="experiments/production_v1.json")
    ap.add_argument("--stages", nargs="+", required=True,
                    help="stages of one example, in table order: <stage> (of --manifest) or <manifest file>:<stage>")
    ap.add_argument("--pilot", action="append", default=[], help='"label=run folder" of a tuning-pilot run')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = _json(a.manifest)
    recs = []
    for spec in a.stages:
        path, stage = spec.rsplit(":", 1) if ":" in spec else (a.manifest, spec)
        recs.append(stage_record(_json(path), path, stage))
    if len({r["data_sha256"] for r in recs}) != 1:
        raise SystemExit("the stages must use the same data file")
    pilots = [pilot_record(m, *p.rsplit("=", 1)) for p in a.pilot]
    print("\n".join(write(recs, pilots, a.out)))
