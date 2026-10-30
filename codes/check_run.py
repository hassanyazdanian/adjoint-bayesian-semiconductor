# check_run.py
"""
Validate a run folder: it holds the expected chains (IDs 0..n-1, seeds seed + chain, the
requested number of draws), each with the stored schema of save_samples (array shapes, finite
values, sampler statistics and PDE cost), which are mutually consistent (see io_utils.load_chains)
and were sampled with the requested method and sampler settings, for the given data file, the
prior and KL basis of the current configuration and, in the joint setting, its plateau bounds.
With --summary, validate the folder's postprocessing output instead: <run>_postprocess/fields.npz
and summary.json must exist, be newer than every chain file (summary.json is written last), hold
finite values for all reported quantities, and describe exactly the chain files of the folder.
--production applies the stricter checks for production runs (see _production_schema and
check_summary): identity captured at launch from committed code, complete per-draw records,
counts reconciled with the cumulative PDE counters, and summaries bound to the data file and to the
exact chain contents. Without it (legacy mode) the earlier pilot runs are validated as before.
Output validity is separate from statistical adequacy: a valid run may still fail R-hat/ESS or
need divergence follow-up; that is recorded in the summary, not decided here.
Exit code 0 if valid, 1 otherwise.
Usage: python check_run.py results/pcn_Setting1_Ex1_pilot --data obs/obs_Ex1.pickle --setting 1 --chains 4 \
           --method pcn --draws 100000 --expect n_warmup=5000 --expect n_burn=10000
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

from io_utils import chain_content_sha256, file_sha256, load_chains, load_data, plateau_bounds, prior_settings
from posterior import build_problem

TOL = 1e-9         # rounding tolerance for probabilities (Pyro averages them, e.g. 1 + 2e-16)
ATTEMPT_KEYS = ("forward_attempts", "forward_failures", "forward_restarts", "forward_failed_evaluations",
                "adjoint_failures")
COST_KEYS = ("forward", "adjoint", "newton_iterations", "forward_time", "adjoint_time", "wall_time")


def _matches(value, expected):
    try:
        return bool(np.isclose(float(value), float(expected), rtol=1e-9, atol=0.0))
    except (TypeError, ValueError):
        return str(value) == str(expected)


def _chain_files(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, "chain*.pickle")))


def _all_finite(obj):
    """True if every number in a nested structure of dicts, lists, tuples and arrays is finite."""
    if isinstance(obj, dict):
        return all(_all_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_all_finite(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.dtype.kind not in "fc" or bool(np.all(np.isfinite(obj)))
    if isinstance(obj, (float, np.floating)):
        return bool(np.isfinite(obj))
    return True


def _shape(value):
    return None if value is None else np.shape(value)


def _schema(chain):
    """Problems with the stored arrays, sampler statistics and cost of one chain (empty if none)."""
    bad = []
    z, info = chain.get("z"), chain["info"]
    if z is None or np.ndim(z) != 2:
        return ["z is missing or not a 2-D array"]
    n, d = np.shape(z)
    if d != chain["prior"].get("n_kl"):
        bad.append(f"z has {d} columns but N_KL = {chain['prior'].get('n_kl')}")
    basis = chain.get("kl_basis")
    if basis is None or np.shape(basis)[1:] != (d,):
        bad.append(f"kl_basis has shape {_shape(basis)} for {d} coefficients")
    elif chain.get("latent_mean") is not None and _shape(chain["latent_mean"]) != (np.shape(basis)[0],):
        bad.append(f"latent_mean has shape {_shape(chain['latent_mean'])}")
    for key in ("times", "cp", "cn"):
        if _shape(chain.get(key)) != (n,):
            bad.append(f"{key} has shape {_shape(chain.get(key))} instead of ({n},)")
    r = chain.get("r")
    if info.get("setting") == 1 and r is not None:
        bad.append("r is stored although the plateaus are known")
    if info.get("setting") != 1 and _shape(r) != (n, 2):
        bad.append(f"r has shape {_shape(r)} instead of ({n}, 2)")
    for key in ("z", "times", "cp", "cn", "r", "kl_basis", "latent_mean"):
        if chain.get(key) is not None and not _all_finite(np.asarray(chain[key], dtype=float)):
            bad.append(f"{key} is not finite")

    cost = chain.get("cost")
    if not isinstance(cost, dict) or any(not isinstance(cost.get(p), dict) for p in ("warmup", "total")):
        bad.append("PDE cost is missing")
    else:
        for phase in ("warmup", "total"):
            values = [cost[phase].get(k) for k in COST_KEYS]
            if any(v is None or not np.isfinite(v) or v < 0 for v in values):
                bad.append(f"PDE cost of the {phase} phase is incomplete, negative or not finite")
        if not bad and any(cost["total"][k] < cost["warmup"][k] for k in COST_KEYS):
            bad.append("PDE cost of the whole run is smaller than that of its warm-up")

    if chain.get("method") == "nuts":
        for key in ("grads_per_iteration", "accept_prob", "evals_per_iteration", "tree_depth"):
            if key in info and _shape(info[key]) != (n,):
                bad.append(f"{key} has shape {_shape(info[key])} instead of ({n},)")
        w = info.get("warmup_steps")
        for key in ("warmup_grads_per_iteration", "warmup_accept_prob", "warmup_tree_depth"):
            if key in info and _shape(info[key]) != (w,):
                bad.append(f"{key} has shape {_shape(info[key])} instead of ({w},)")
        for key in ("grads_per_iteration", "accept_prob", "step_size", "divergences"):
            if key not in info:
                bad.append(f"NUTS statistic {key} is missing")
        if "step_size" in info and not (np.isfinite(info["step_size"]) and info["step_size"] > 0):
            bad.append(f"step size {info['step_size']}")
        acc = np.asarray(info.get("accept_prob", []), dtype=float)
        if acc.size and not np.all((acc >= -TOL) & (acc <= 1 + TOL)):
            bad.append("acceptance probabilities outside [0, 1]")
        div = np.asarray(info.get("divergences", []))
        if div.size and (div.dtype.kind not in "iu" or div.min() < 0 or div.max() >= n):
            bad.append("divergence indices are not draws of this chain")
    elif chain.get("method") in ("pcn", "gibbs_pcn"):
        beta = info.get("beta")
        if beta is None or not (np.isfinite(beta) and 0 < beta <= 1):
            bad.append(f"pCN step size beta = {beta}")
        acc = np.atleast_1d(np.asarray(info.get("acceptance", np.nan), dtype=float))
        if not np.all((acc >= -TOL) & (acc <= 1 + TOL)):
            bad.append(f"acceptance {info.get('acceptance')}")
        trace = info.get("beta_trace")
        if trace is not None and "n_warmup" in info and _shape(trace) != (info["n_warmup"] + info.get("n_burn", 0),):
            bad.append(f"beta_trace has shape {_shape(trace)}")
    if not _all_finite(info):
        bad.append("sampler statistics are not finite")
    return bad


def _is_count(values):
    """Non-negative integer values (stored as int or float)."""
    v = np.asarray(values, dtype=float)
    return bool(np.all(np.isfinite(v)) and np.all(v >= 0) and np.all(np.mod(v, 1) == 0))


def _production_schema(chain, manifest=None, seed=0):
    """Additional requirements for production chains; returns the list of problems (empty if none)."""
    bad = []
    info, prov, n = chain["info"], chain.get("provenance") or {}, len(chain["z"])
    method = chain.get("method")
    # identity captured at launch
    if prov.get("data_hashed_at") != "load":
        bad.append("the data hash was not captured when the data were loaded")
    elif prov.get("data_sha256_at_save") != prov.get("data_sha256"):
        bad.append("the data file changed between loading and saving")
    code = prov.get("code") or {}
    if not code.get("git_commit") or code.get("git_dirty") is not False or not code.get("source_sha256"):
        bad.append(f"not run from committed code ({code})")
    policy = prov.get("init_policy")
    if not policy:
        bad.append("no initialisation policy")
    elif policy.get("kind") == "prior_draw" and policy.get("rng_seed") != seed + info.get("chain", -1):
        bad.append(f"initialisation seed {policy.get('rng_seed')} instead of {seed + info.get('chain', -1)}")
    if manifest is not None and (prov.get("manifest") or {}).get("sha256") != file_sha256(manifest):
        bad.append(f"not run with the manifest {manifest}")
    # times and cumulative counters
    times = np.asarray(chain["times"], dtype=float)
    if np.any(times < 0) or np.any(np.diff(times) < 0):
        bad.append("times are negative or decrease")
    t, w = chain["cost"]["total"], chain["cost"]["warmup"]
    counters = ("forward", "adjoint", "newton_iterations") + ATTEMPT_KEYS
    for phase in (t, w):
        if any(k not in phase for k in counters) or not _is_count([phase[k] for k in counters]):
            bad.append("PDE counters (including attempts and failures) missing or not non-negative integers")
            return bad
    if t["forward_attempts"] - w["forward_attempts"] < t["forward"] - w["forward"]:
        bad.append("fewer forward-solve attempts than successful solves")
    timing = info.get("timing") or {}
    order = [timing.get(k) for k in ("launch_epoch", "warmup_start_epoch", "sampling_start_epoch", "end_epoch")]
    if None in order or any(b < a for a, b in zip(order, order[1:])):
        bad.append("epoch timestamps missing or out of order")
    d = np.shape(chain["z"])[1] + (0 if chain.get("r") is None else np.shape(chain["r"])[1])
    if method == "nuts":
        w_steps = info.get("warmup_steps")
        for key, length in (("grads_per_iteration", n), ("evals_per_iteration", n), ("tree_depth", n),
                            ("accept_prob", n), ("forward_attempts_per_iteration", n),
                            ("warmup_grads_per_iteration", w_steps), ("warmup_tree_depth", w_steps),
                            ("warmup_accept_prob", w_steps), ("warmup_forward_attempts_per_iteration", w_steps)):
            if _shape(info.get(key)) != (length,):
                bad.append(f"{key} has shape {_shape(info.get(key))} instead of ({length},)")
        if bad:
            return bad
        for key in ("grads_per_iteration", "evals_per_iteration", "tree_depth", "warmup_grads_per_iteration",
                    "warmup_tree_depth", "forward_attempts_per_iteration", "warmup_forward_attempts_per_iteration"):
            if not _is_count(info[key]):
                bad.append(f"{key} is not a non-negative integer count")
        if np.max(info["tree_depth"]) > info.get("max_tree_depth", np.inf):
            bad.append("tree depth above the maximum")
        if np.sum(info["evals_per_iteration"]) != t["forward"] - w["forward"]:
            bad.append("per-draw potential evaluations do not add up to the forward solves of the sampling phase")
        if np.sum(info["grads_per_iteration"]) != t["adjoint"] - w["adjoint"]:
            bad.append("per-draw gradients do not add up to the adjoint solves of the sampling phase")
        if np.sum(info["warmup_grads_per_iteration"]) != w["adjoint"]:
            bad.append("warm-up gradients do not add up to the adjoint solves of the warm-up")
        if np.sum(info["forward_attempts_per_iteration"]) != t["forward_attempts"] - w["forward_attempts"]:
            bad.append("per-draw forward attempts do not add up to the attempts of the sampling phase")
        if np.sum(info["warmup_forward_attempts_per_iteration"]) != w["forward_attempts"]:
            bad.append("warm-up forward attempts do not add up to the attempts of the warm-up")
        if np.any(np.asarray(info["forward_attempts_per_iteration"]) < np.asarray(info["evals_per_iteration"])):
            bad.append("an iteration has fewer forward attempts than potential evaluations")
        if not 0 <= w["forward"] - w["adjoint"] <= 10:
            bad.append("warm-up forward solves exceed the gradients by more than the initial evaluations")
        mass = info.get("inverse_mass_matrix")
        values = list(mass.values()) if isinstance(mass, dict) else []
        if len(values) != 1:
            bad.append("inverse mass matrix missing")
        else:
            m = np.asarray(values[0], dtype=float)
            if info.get("mass_matrix") == "dense":
                if m.shape != (d, d) or not np.all(np.isfinite(m)) or not np.allclose(m, m.T, rtol=1e-10, atol=1e-14) \
                        or np.min(np.linalg.eigvalsh(0.5 * (m + m.T))) <= 0:
                    bad.append(f"dense inverse mass matrix of shape {m.shape} is not a finite SPD ({d} x {d}) matrix")
            elif m.shape != (d,) or not np.all(np.isfinite(m)) or np.any(m <= 0):
                bad.append(f"diagonal inverse mass matrix of shape {m.shape} is not finite and positive")
    elif method in ("pcn", "gibbs_pcn"):
        calls = 1 if chain.get("r") is None else 2               # likelihood evaluations per step
        logical = (t["forward"] + t["forward_failed_evaluations"]) - (w["forward"] + w["forward_failed_evaluations"])
        if logical != n * calls:
            bad.append(f"{logical} forward evaluations for {n} retained draws")
        per_draw = info.get("forward_attempts_per_draw")
        if _shape(per_draw) != (n,) or not _is_count(per_draw):
            bad.append("forward_attempts_per_draw missing or not a count per retained draw")
        elif np.sum(per_draw) != t["forward_attempts"] - w["forward_attempts"]:
            bad.append("per-draw forward attempts do not add up to the attempts of the sampling phase")
        if not 0 <= info.get("accepted_fraction", -1) <= 1:
            bad.append("accepted fraction missing")
        if (info.get("adapted") is False) != str(info.get("tuning", "")).startswith("none"):
            bad.append(f"tuning label '{info.get('tuning')}' inconsistent with adapted = {info.get('adapted')}")
        if t["adjoint"] != 0:
            bad.append("pCN made adjoint solves")
        if w["forward"] + w["forward_failed_evaluations"] != (info.get("n_warmup", 0) + info.get("n_burn", 0)) * calls + 1:
            bad.append("warm-up forward evaluations do not match warm-up + burn-in + 1")
        if not isinstance(info.get("adapted"), bool):
            bad.append("tuning flag missing")
    return bad


def check_run(run_dir, data_path, setting, n_chains, k=None, n_kl=None, expect=(), method=None, draws=None,
              seed=0, tau=None, mean_a=None, production=False, manifest=None, cp_bounds=None, cn_bounds=None,
              inversion_mesh=None):
    try:
        chains = load_chains(run_dir)
    except (FileNotFoundError, ValueError) as err:
        return False, str(err)
    if len(chains) != n_chains:
        return False, f"{len(chains)} chains instead of {n_chains}"
    ids = sorted(ch["info"].get("chain") for ch in chains)
    if ids != list(range(n_chains)):
        return False, f"chain IDs {ids} instead of 0..{n_chains - 1}"
    for ch in chains:
        i = ch["info"]
        if i.get("seed") != seed + i["chain"]:
            return False, f"chain {i['chain']} has seed {i.get('seed')} instead of {seed + i['chain']}"
        if method is not None and ch.get("method") != method:
            return False, f"chain {i['chain']} was sampled by {ch.get('method')} instead of {method}"
        if draws is not None and len(ch["z"]) != draws:
            return False, f"chain {i['chain']} has {len(ch['z'])} draws instead of {draws}"
        bad = _schema(ch)
        if not bad and production:
            bad = _production_schema(ch, manifest, seed)
        if bad:
            return False, f"chain {i['chain']}: {'; '.join(bad)}"
    c = chains[0]
    with open(data_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    if (c.get("provenance") or {}).get("data_sha256") != sha:
        return False, "sampled for a different data file"
    if c["info"].get("setting") != setting:
        return False, f"setting {c['info'].get('setting')} instead of {setting}"
    post = build_problem(load_data(data_path), setting, k, n_kl, tau, mean_a, cp_bounds, cn_bounds, inversion_mesh)
    meshes = {ch["info"].get("inversion_mesh_n", post.model.n) for ch in chains}   # not recorded before 30 Oct 2026
    if meshes != {post.model.n}:
        return False, f"inversion mesh {sorted(meshes)} instead of {post.model.n}"
    prior, expected = post.prior, prior_settings(post.prior)
    for key in ("n_kl", "tau", "alpha", "k", "mean_a"):
        if not np.isclose(c["prior"].get(key, 0.0), expected[key]):
            return False, f"prior setting {key} = {c['prior'].get(key)} instead of {expected[key]}"
    if c.get("kl_basis") is None or not np.allclose(c["kl_basis"], prior.basis.numpy(), rtol=0, atol=1e-12):
        return False, "different KL basis"
    if c.get("latent_mean") is not None and not np.allclose(c["latent_mean"], prior.mean.numpy(), rtol=0, atol=1e-12):
        return False, "different latent mean"
    if c.get("plateau_bounds") != plateau_bounds(post):
        return False, f"plateau bounds {c.get('plateau_bounds')} instead of {plateau_bounds(post)}"
    for item in expect:
        key, value = item.split("=", 1)
        bad = [ch["info"].get(key) for ch in chains if not _matches(ch["info"].get(key), value)]
        if bad:
            return False, f"sampler setting {key} = {bad[0]} instead of {value}"
    return True, f"{n_chains} consistent chains of {len(c['z'])} draws"


def _get(summary, *keys, present=False):
    """summary[k1][k2]...; with present=True, whether that key exists (its value may be None)."""
    value = summary
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return False if present else None
        value = value[key]
    return True if present else value


def _number(summary, *keys):
    """summary[k1][k2]... as a finite float, or None."""
    value = summary
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


REQUIRED = [("RE",), ("SSIM",), ("depth_mean",), ("depth_std",)]
DIAGNOSTICS = [("z", "r_hat_max"), ("z", "ess_bulk_min"), ("z", "ess_bulk_median"), ("z", "ess_tail_min"),
               ("depth", "r_hat"), ("depth", "ess_bulk"), ("depth", "ess_tail"), ("depth", "mcse_mean")]


def check_summary(run_dir, n_chains, draws=None, production=False, data_path=None):
    """
    The postprocessing output exists, was written after the chains (fields.npz before summary.json,
    which postprocess.py writes last), holds finite values for every reported quantity, and
    describes exactly the chain files of the folder.
    """
    pp = run_dir.rstrip("/") + "_postprocess"
    path, fields_path = os.path.join(pp, "summary.json"), os.path.join(pp, "fields.npz")
    for p in (path, fields_path):
        if not os.path.isfile(p):
            return False, f"{p} is missing"
    files = _chain_files(run_dir)
    if len(files) != n_chains:
        return False, f"{len(files)} chain files in {run_dir} instead of {n_chains}"
    newest_chain = max(os.path.getmtime(f) for f in files)
    if not newest_chain <= os.path.getmtime(fields_path) <= os.path.getmtime(path):
        return False, f"{pp} is older than the chain files, or summary.json is older than fields.npz"
    with open(path) as f:
        s = json.load(f)
    listed = sorted(os.path.normpath(p) for p in s.get("files", []))
    if listed != [os.path.normpath(p) for p in files]:
        return False, f"{path} describes other chain files"
    if s.get("n_chains") != n_chains:
        return False, f"{path} has {s.get('n_chains')} chains instead of {n_chains}"
    if draws is not None and s.get("draws_per_chain") != draws:
        return False, f"{path} has {s.get('draws_per_chain')} draws per chain instead of {draws}"
    missing = [".".join(keys) for keys in REQUIRED if _number(s, *keys) is None]
    # Diagnostics may be undefined (None) only if the summary records them as undefined: the output is
    # then complete and valid, and the run "not passed" (statistical adequacy is not decided here).
    declared_undefined = isinstance(s.get("convergence"), dict) and s["convergence"].get("complete") is False
    missing += [".".join(keys) for keys in DIAGNOSTICS
                if _number(s, *keys) is None and not (declared_undefined and _get(s, *keys, present=True))]
    if s.get("method") == "nuts":
        if _number(s, "divergences", "count") is None:
            missing.append("divergences.count")
        if len(s.get("sampler", [])) != n_chains:
            missing.append("sampler (one entry per chain)")
    if "cost" in s and not _all_finite(s["cost"]):
        missing.append("cost")
    if missing:
        return False, f"{path} lacks finite values for {', '.join(missing)}"
    with np.load(fields_path) as fields:
        keys = ("c_mean", "c_std", "hpd_low", "hpd_high", "depth")
        if any(key not in fields.files for key in keys):
            return False, f"{fields_path} lacks one of {keys}"
        sizes = {fields[key].size for key in keys[:4]}
        if len(sizes) != 1 or not all(_all_finite(fields[key]) for key in keys):
            return False, f"{fields_path} holds fields of different sizes or non-finite values"
        if draws is not None and fields["depth"].size != n_chains * draws:
            return False, f"{fields_path} holds {fields['depth'].size} depths instead of {n_chains * draws}"
        field_hashes = list(fields["chain_content_sha256"]) if "chain_content_sha256" in fields.files else None
    if production:
        bad = _production_summary(run_dir, s, field_hashes, data_path)
        if bad:
            return False, f"{path}: {'; '.join(bad)}"
    return True, f"summary of {n_chains} chains"


def _production_summary(run_dir, s, field_hashes, data_path):
    """The summary is complete and bound to the data file and to the exact chain contents."""
    bad = []
    chains = load_chains(run_dir)
    hashes = [chain_content_sha256(c) for c in chains]
    if s.get("chain_content_sha256") != hashes:
        bad.append("summary does not describe the current chain contents")
    if field_hashes != hashes:
        bad.append("fields.npz does not belong to the current chain contents")
    if s.get("draws_per_chain") != len(chains[0]["z"]):
        bad.append("draw count differs from the chains")
    used = {(c.get("provenance") or {}).get("data_sha256") for c in chains}
    if data_path is None or used != {file_sha256(data_path)} or s.get("data_sha256") not in used:
        bad.append("summary, chains and data file do not share one data hash")
    conv = s.get("convergence") or {}
    if not {"monitored", "undefined", "complete", "passes", "r_hat_max", "ess_bulk_min", "ess_tail_min"} <= conv.keys():
        bad.append("convergence record incomplete")
    if not isinstance(s.get("cost"), dict) or not _all_finite(s["cost"]) or not s.get("efficiency"):
        bad.append("cost or efficiency missing")
    if s.get("method") == "nuts" and not {"count", "fraction", "follow_up"} <= (s.get("divergences") or {}).keys():
        bad.append("divergence assessment missing")
    if chains[0].get("r") is not None and not all(name in s and name in conv["monitored"] for name in ("cp", "cn")):
        bad.append("plateau diagnostics missing (joint setting)")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--data", required=True)
    ap.add_argument("--setting", type=int, default=1)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--k", type=float, default=None)
    ap.add_argument("--n_kl", type=int, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--mean_a", type=float, default=None)
    ap.add_argument("--cp_bounds", type=float, nargs=2, default=None, help="required c_p prior support (setting 2)")
    ap.add_argument("--cn_bounds", type=float, nargs=2, default=None, help="required c_n prior support (setting 2)")
    ap.add_argument("--inversion_mesh_n", type=int, default=None, help="required inversion mesh (default: the data file's)")
    ap.add_argument("--method", choices=("pcn", "gibbs_pcn", "nuts"), default=None)
    ap.add_argument("--draws", type=int, default=None, help="required number of retained draws per chain")
    ap.add_argument("--seed", type=int, default=0, help="base seed; chain i must have seed + i")
    ap.add_argument("--expect", action="append", default=[], help="required sampler setting key=value")
    ap.add_argument("--summary", action="store_true", help="validate the postprocessing output instead")
    ap.add_argument("--production", action="store_true", help="apply the stricter production checks")
    ap.add_argument("--manifest", default=None, help="production manifest the chains must have been run with")
    a = ap.parse_args()
    if a.summary:
        ok, msg = check_summary(a.run_dir, a.chains, a.draws, a.production, a.data)
    else:
        ok, msg = check_run(a.run_dir, a.data, a.setting, a.chains, a.k, a.n_kl, a.expect, a.method, a.draws, a.seed,
                            a.tau, a.mean_a, a.production, a.manifest, a.cp_bounds, a.cn_bounds, a.inversion_mesh_n)
    print(("valid: " if ok else "not valid: ") + f"{a.run_dir}: {msg}", flush=True)
    sys.exit(0 if ok else 1)
