# checkpoint_diagnostics.py
"""
Retrospective checkpoint diagnostics of a completed (capped) run: R-hat and bulk/tail ESS
(ArviZ, Vehtari et al. 2021) of every monitored quantity on the first n draws of every chain, at
n = every, 2 every, ..., and the PDE work and time spent to reach n.

- Monitored: all KL coefficients and the junction depth (per-draw depths from the run's
  fields.npz, which must belong to exactly these chains); c_p and c_n in the joint setting.
- Pass at a checkpoint: every diagnostic defined, R-hat < 1.01, bulk and tail ESS >= 400.
  Divergences up to the checkpoint are reported separately (follow-up rule: more than 0.5 % of
  the draws, or >= 3 divergences with >= 75 % in one chain).
- Costs at a checkpoint come from prefix sums of the per-draw records, never from full-run totals:
  NUTS: potential evaluations (forward solves), gradients (adjoint solves) and forward attempts
  per draw; pCN: one logical forward evaluation per draw, and the recorded attempts per draw. Times
  are each chain's own sampler time at its n-th draw (summed over chains: worker time); with epoch
  timestamps the actual elapsed time of the batch from launch is also given. "incl. warm-up" adds
  each chain's complete warm-up. At the full length the prefix sums must reproduce the run totals
  exactly, otherwise the script stops with an error.
- The first passing checkpoint is a retrospective milestone of a run that was executed to its cap,
  not runtime saved by stopping early. It is reported together with whether the pass persists at
  every later checkpoint; if no checkpoint passes, the result is "not reached within the cap".

Usage: python checkpoint_diagnostics.py results/<run> --data obs/obs_Ex1.pickle --every 250
Writes <run>_postprocess/checkpoints.json and checkpoints.md.
"""
import argparse
import json
import os

import arviz as az
import numpy as np

from io_utils import chain_content_sha256, load_chains, load_data

R_HAT_MAX, ESS_MIN, DIVERGENCE_FRACTION = 1.01, 400, 0.005


def milestone(passes):
    """First passing checkpoint index, whether every later checkpoint also passes, and the first
    checkpoint from which all checkpoints pass (None if never)."""
    first = next((i for i, p in enumerate(passes) if p), None)
    persistent_from = next((i for i in range(len(passes)) if all(passes[i:])), None)
    return {"first_pass": first, "persists": first is not None and all(passes[first:]),
            "passes_from": persistent_from}


def _prefix_costs(chains, n, adj_eq):
    """PDE work and times of the first n draws of every chain (sampling phase), and with warm-up."""
    out = dict(forward=0, adjoint=0, attempts=0, worker_time_s=0.0, longest_chain_time_s=0.0,
               warmup_forward=0, warmup_adjoint=0, warmup_attempts=0, warmup_worker_time_s=0.0)
    attempts_known, ends = True, []
    for c in chains:
        info, w = c["info"], c["cost"]["warmup"]
        if c["method"] == "nuts":
            out["forward"] += int(np.sum(info["evals_per_iteration"][:n]))
            out["adjoint"] += int(np.sum(info["grads_per_iteration"][:n]))
            att = info.get("forward_attempts_per_iteration")
        else:
            out["forward"] += n * (1 if c.get("r") is None else 2)
            att = info.get("forward_attempts_per_draw")
        if att is None:
            attempts_known = False
        else:
            out["attempts"] += int(np.sum(att[:n]))
        t = float(c["times"][n - 1])
        out["worker_time_s"] += t
        out["longest_chain_time_s"] = max(out["longest_chain_time_s"], t)
        out["warmup_forward"] += int(w["forward"])
        out["warmup_adjoint"] += int(w["adjoint"])
        out["warmup_attempts"] += int(w.get("forward_attempts", 0))
        out["warmup_worker_time_s"] += float(w["wall_time"])
        timing = info.get("timing")
        ends.append(None if not timing else timing["sampling_start_epoch"] + t)
    if not attempts_known:
        out["attempts"] = out["warmup_attempts"] = None
    out["forward_equivalents"] = out["forward"] + adj_eq * out["adjoint"]
    out["forward_equivalents_incl_warmup"] = (out["forward"] + out["warmup_forward"]
                                              + adj_eq * (out["adjoint"] + out["warmup_adjoint"]))
    timings = [c["info"].get("timing") for c in chains]
    out["elapsed_s_from_launch"] = (max(ends) - min(t["launch_epoch"] for t in timings)) if all(timings) else None
    return out


def _check_full_length(chains, full):
    """At the full length, prefix sums must equal the run totals of the sampling phase."""
    t = {k: sum(c["cost"]["total"][k] - c["cost"]["warmup"][k] for c in chains) for k in ("forward", "adjoint")}
    problems = []
    if chains[0]["method"] == "nuts":
        if full["forward"] != t["forward"] or full["adjoint"] != t["adjoint"]:
            problems.append(f"prefix sums {full['forward']}/{full['adjoint']} != run totals {t['forward']}/{t['adjoint']}")
    else:
        failed = sum(c["cost"]["total"].get("forward_failed_evaluations", 0) - c["cost"]["warmup"].get("forward_failed_evaluations", 0)
                     for c in chains)
        if full["forward"] != t["forward"] + failed:
            problems.append(f"{full['forward']} draws but {t['forward']} + {failed} forward evaluations in the run")
    if full["attempts"] is not None:
        ta = sum(c["cost"]["total"]["forward_attempts"] - c["cost"]["warmup"]["forward_attempts"] for c in chains)
        if full["attempts"] != ta:
            problems.append(f"prefix attempts {full['attempts']} != run attempts {ta}")
    if problems:
        raise ValueError("checkpoint accounting does not reproduce the run totals: " + "; ".join(problems))


def checkpoints(run_dir, data_path, every):
    chains = load_chains(run_dir)
    data = load_data(data_path)
    used = {(c.get("provenance") or {}).get("data_sha256") for c in chains}
    if used != {data["_source"]["sha256"]}:
        raise ValueError(f"{data_path} is not the data the chains were sampled with")
    pp = run_dir.rstrip("/") + "_postprocess"
    with open(os.path.join(pp, "summary.json")) as f:
        summary = json.load(f)
    fields = np.load(os.path.join(pp, "fields.npz"))
    hashes = [chain_content_sha256(c) for c in chains]
    if "chain_content_sha256" not in fields.files or list(fields["chain_content_sha256"]) != hashes:
        raise ValueError(f"{pp}/fields.npz does not belong to these chains; rerun postprocess.py")
    depth = np.asarray(fields["depth"]).reshape(len(chains), -1)
    z = np.stack([c["z"] for c in chains])
    joint = chains[0].get("r") is not None
    N = z.shape[1]
    adj_eq = summary["cost"]["adjoint_in_forward_equivalents"]
    div = np.zeros((len(chains), N), dtype=bool)
    for j, c in enumerate(chains):
        idx = np.asarray(c["info"].get("divergences", []), dtype=int)
        div[j, idx[idx < N]] = True

    ns = sorted(set(list(range(every, N + 1, every)) + [N]))
    rows = []
    for n in ns:
        posterior = {"z": z[:, :n], "depth": depth[:, :n]}
        if joint:
            posterior.update(cp=np.stack([c["cp"][:n] for c in chains]), cn=np.stack([c["cn"][:n] for c in chains]))
        d = az.summary(az.convert_to_dataset(posterior), kind="diagnostics", round_to="none")
        vals = d[["r_hat", "ess_bulk", "ess_tail"]].to_numpy(dtype=float)
        undefined = [r for r, ok in zip(d.index, np.isfinite(vals).all(axis=1)) if not ok]
        complete = not undefined
        # each extreme is reported unless that statistic is undefined for some quantity (as in
        # postprocess.py); the pass requires every statistic of every quantity to be defined
        r_hat, bulk, tail = (float(fn(col)) if np.all(np.isfinite(col)) else None
                             for fn, col in ((np.max, vals[:, 0]), (np.min, vals[:, 1]), (np.min, vals[:, 2])))
        passes = bool(complete and r_hat < R_HAT_MAX and bulk >= ESS_MIN and tail >= ESS_MIN)
        dv = div[:, :n]
        count, per_chain = int(dv.sum()), dv.sum(1).tolist()
        cost = _prefix_costs(chains, n, adj_eq)
        bottleneck = min(bulk, tail) if bulk is not None and tail is not None else None
        rows.append(dict(
            draws_per_chain=n, r_hat_max=r_hat, ess_bulk_min=bulk, ess_tail_min=tail, undefined=undefined,
            passes=passes, divergences=count, divergences_per_chain=per_chain,
            divergence_follow_up=bool(count / dv.size > DIVERGENCE_FRACTION
                                      or (count >= 3 and max(per_chain) >= 0.75 * count)),
            cost=cost,
            bottleneck_ess=bottleneck,
            bottleneck_ess_per_1000_raw_solves=None if bottleneck is None else 1e3 * bottleneck / (cost["forward"] + cost["adjoint"]),
            bottleneck_ess_per_1000_raw_solves_incl_warmup=None if bottleneck is None else
                1e3 * bottleneck / (cost["forward"] + cost["adjoint"] + cost["warmup_forward"] + cost["warmup_adjoint"]),
            bottleneck_ess_per_worker_second=None if bottleneck is None else bottleneck / cost["worker_time_s"],
        ))
    _check_full_length(chains, rows[-1]["cost"])
    m = milestone([r["passes"] for r in rows])
    result = dict(run=run_dir, data_sha256=data["_source"]["sha256"], chain_content_sha256=hashes, every=every,
                  thresholds={"r_hat_below": R_HAT_MAX, "ess_bulk_at_least": ESS_MIN, "ess_tail_at_least": ESS_MIN},
                  monitored=list(d.index), checkpoints=rows,
                  milestone=dict(m, first_pass_draws=None if m["first_pass"] is None else rows[m["first_pass"]]["draws_per_chain"],
                                 passes_from_draws=None if m["passes_from"] is None else rows[m["passes_from"]]["draws_per_chain"],
                                 statement="not reached within the cap" if m["first_pass"] is None else
                                 "retrospective milestone of a run executed to its cap; not runtime saved"),
                  adjoint_in_forward_equivalents=adj_eq)
    return result


def to_markdown(r):
    def f(v, fmt):
        return "undefined" if v is None else format(v, fmt)
    lines = [f"Run `{r['run']}`; monitored: {len(r['monitored'])} quantities; pass = R̂ < 1.01 and bulk, tail ESS ≥ 400.",
             f"Milestone: {r['milestone']['statement']}"
             + ("" if r["milestone"]["first_pass"] is None else
                f"; first pass at {r['milestone']['first_pass_draws']} draws per chain, persists: {r['milestone']['persists']}"),
             "",
             "| Draws per chain | max R̂ | min bulk ESS | min tail ESS | Pass | Divergences (follow-up) | Forward / adjoint solves (sampling) "
             "| Attempts | Worker time (s) | Elapsed from launch (s) | Bottleneck ESS per 1000 raw solves (sampling / incl. warm-up) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in r["checkpoints"]:
        k = c["cost"]
        lines.append(f"| {c['draws_per_chain']} | {f(c['r_hat_max'], '.4f')} | {f(c['ess_bulk_min'], '.0f')} | {f(c['ess_tail_min'], '.0f')} "
                     f"| {'yes' if c['passes'] else 'no'} | {c['divergences']} ({'yes' if c['divergence_follow_up'] else 'no'}) "
                     f"| {k['forward']} / {k['adjoint']} | {'—' if k['attempts'] is None else k['attempts']} "
                     f"| {k['worker_time_s']:.0f} | {f(k['elapsed_s_from_launch'], '.0f')} "
                     f"| {f(c['bottleneck_ess_per_1000_raw_solves'], '.3f')} / {f(c['bottleneck_ess_per_1000_raw_solves_incl_warmup'], '.3f')} |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--data", required=True)
    ap.add_argument("--every", type=int, required=True, help="checkpoint interval in draws per chain")
    a = ap.parse_args()
    res = checkpoints(a.run_dir, a.data, a.every)
    pp = a.run_dir.rstrip("/") + "_postprocess"
    with open(os.path.join(pp, "checkpoints.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    table = to_markdown(res)
    with open(os.path.join(pp, "checkpoints.md"), "w") as fh:
        fh.write(table + "\n")
    print(table)
