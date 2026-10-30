# beta_sweep.py
"""
Selection of the fixed pCN step from a step-size sweep (Reviewer 1, item vi). For each run (four
chains with a fixed beta = f x the mean tuned beta) the criterion is the bottleneck ESS (smallest
bulk or tail ESS over all KL coefficients and the junction depth) per 1000 PDE solves of the retained
draws. The selected multiple is the one with the largest criterion; if it is the smallest or the
largest tested multiple, the sweep is extended on that side (status "extend") instead of declaring
the boundary value near-optimal.
Usage: python beta_sweep.py --run 6=results/pcn_Setting1_Ex2_pilot_beta_x6 --run 8=... --run 12=... [--json out.json]
Prints one line per run and a last line "selected <f>" or "extend <f>".
"""
import argparse
import json

from production_table import _diagnostics, _json


def criterion(run):
    s = _json(f"{run}_postprocess/summary.json")
    d, _ = _diagnostics(f"{run}_postprocess/diagnostics.csv")
    c = s["cost"]["sampling"]
    solves = c["forward_solves"] + c["adjoint_solves"]
    beta = sorted({x["beta"] for x in s["sampler"]})
    return dict(run=run, beta=beta, acceptance=[x["acceptance"] for x in s["sampler"]], r_hat_max=d["r_hat_max"],
                bottleneck_ess=d["ess"], quantity=d["quantity"], kind=d["kind"], solves=solves,
                per_1000_solves=1e3 * d["ess"] / solves)


def select(results):
    """results: {multiple: criterion dict}; returns (status, multiple)."""
    fs = sorted(results)
    best = max(fs, key=lambda f: results[f]["per_1000_solves"])
    if len(fs) > 1 and best in (fs[0], fs[-1]):
        return "extend", (fs[0] if best == fs[0] else fs[-1])
    return "selected", best


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="multiple=run folder")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    results = {float(f): criterion(r) for f, r in (x.split("=", 1) for x in a.run)}
    for f in sorted(results):
        r = results[f]
        print(f"{f:g} x: beta {r['beta'][0]:.4f}, acceptance {sum(r['acceptance']) / len(r['acceptance']):.4f}, "
              f"max R-hat {r['r_hat_max']:.3f}, bottleneck ESS {r['bottleneck_ess']:.0f} ({r['kind']} of {r['quantity']}), "
              f"{r['per_1000_solves']:.3f} per 1000 PDE solves")
    status, f = select(results)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dict(results={str(k): v for k, v in results.items()}, status=status, multiple=f), fh, indent=2)
    print(f"{status} {f:g}")
