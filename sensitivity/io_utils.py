# io_utils.py
"""Reading and writing data sets and MCMC samples."""
import datetime
import glob
import hashlib
import importlib.metadata
import os
import pickle
import platform
import subprocess

import numpy as np


def save_pickle(path, obj):
    """Write obj to path atomically: to a temporary file, renamed to path only once it is complete, so that an
    interrupted write never leaves a truncated file under the final name."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print("Saved:", path)


def _numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def load_data(path):
    """
    Synthetic data set as written by generate_synthetic_data.py. The file is read once; data["_source"] holds
    its path and the SHA-256 of exactly the bytes that were loaded, so results can be bound to the data actually
    used.
    """
    with open(path, "rb") as f:
        raw = f.read()
    data = pickle.loads(raw)
    if not isinstance(data, dict) or "c_true" not in data or "y_obs" not in data:
        raise ValueError(f"{path} is not a data set written by generate_synthetic_data.py")
    data["_source"] = {"path": os.path.abspath(path), "sha256": hashlib.sha256(raw).hexdigest()}
    return data


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def code_version(code_dir=CODE_DIR):
    """
    Identity of the code: the git commit of code_dir (None outside a repository), whether tracked
    files differ from it, and a SHA-256 fingerprint of all *.py and *.sh files in code_dir (which
    identifies the code even with uncommitted changes).
    """
    def git(*args):
        try:
            return subprocess.run(["git", "-C", code_dir, *args], capture_output=True, text=True,
                                  timeout=20, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    h = hashlib.sha256()
    for name in sorted(os.listdir(code_dir)):
        if name.endswith((".py", ".sh")):
            h.update(name.encode())
            with open(os.path.join(code_dir, name), "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status),
            "source_sha256": h.hexdigest()}


THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


def run_identity(data, init_policy):
    """
    Identity of a run, captured at launch (before sampling): the data actually loaded (path and
    SHA-256 of its bytes), the code version, the initialisation policy, the launch time and host,
    and the thread settings.
    """
    out = {"data_path": data["_source"]["path"], "data_sha256": data["_source"]["sha256"],
           "data_hashed_at": "load", "code": code_version(), "init_policy": init_policy,
           "launched_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "host": platform.node(), "threads": {v: os.environ.get(v) for v in THREAD_VARIABLES}}
    try:
        import torch
        out["threads"]["torch"] = torch.get_num_threads()
    except ImportError:
        pass
    return out


def provenance(identity):
    """
    Package versions and the identity of the run (run_identity(), captured at launch), stored with every
    result: data_sha256 is the hash of the bytes actually loaded, and data_sha256_at_save records the file
    on disk when the result is saved (a difference means the file was changed during the run).
    """
    out = {"python": platform.python_version()}
    for pkg in ("fenics-dolfin", "dolfin-adjoint", "torch", "pyro-ppl", "numpy", "scipy", "arviz"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    out.update(identity)
    out["data_sha256_at_save"] = file_sha256(identity["data_path"]) if os.path.isfile(identity["data_path"]) else None
    return out


def save_samples(path, post, theta, method, identity, times=None, cost=None, **info):
    """
    Store one chain theta (n_samples, dim) with everything postprocess.py needs.
    times: wall-clock seconds since the start of sampling (after warm-up) for each draw.
    cost:  {"warmup": model.stats at the end of warm-up, "total": model.stats at the end}.
    The KL basis and the latent mean are stored, so the chain can be interpreted later
    even if the prior definition in the code changes. identity: run_identity() at launch.
    """
    theta = np.asarray(theta, dtype=float)
    cp, cn = post.plateaus(theta)
    n = theta.shape[0]
    save_pickle(path, {
        "method": method,
        "z": theta[:, : post.n_kl],
        "r": None if post.known_plateaus else theta[:, post.n_kl:],
        "cp": np.broadcast_to(_numpy(cp), (n,)).copy(),
        "cn": np.broadcast_to(_numpy(cn), (n,)).copy(),
        "times": None if times is None else np.asarray(times, dtype=float),
        "cost": cost,
        "kl_basis": post.prior.basis.numpy(),
        "latent_mean": post.prior.mean.numpy(),
        "prior": prior_settings(post.prior),
        "plateau_bounds": plateau_bounds(post),
        "info": info,
        "provenance": provenance(identity),
    })


def prior_settings(prior):
    return dict(n_kl=prior.n_kl, tau=prior.tau, alpha=prior.alpha, k=prior.k, mean_a=prior.mean_a,
                mean_type="trend")


def plateau_bounds(post):
    """Supports of the uniform plateau priors (joint setting); None when the plateaus are known."""
    return None if post.known_plateaus else {"cp": tuple(post.cp_bounds), "cn": tuple(post.cn_bounds)}


def load_samples(path):
    """One chain, as written by save_samples()."""
    with open(path, "rb") as f:
        out = pickle.load(f)
    if not isinstance(out, dict) or not {"method", "z", "prior", "info"} <= out.keys():
        raise ValueError(f"{path} is not a chain file written by run_nuts.py or run_pcn.py")
    return out


def chain_prior(chain, V):
    """
    The prior a chain was sampled with, rebuilt on the function space V from its stored settings; the stored
    KL basis and latent mean replace the rebuilt ones when the chain has them.
    """
    import torch
    from prior import MaternKLPrior

    prior = MaternKLPrior(V, **{k: chain["prior"][k] for k in ("n_kl", "tau", "alpha", "k", "mean_a")})
    if chain.get("kl_basis") is not None:
        prior.basis = torch.from_numpy(chain["kl_basis"])
    if chain.get("latent_mean") is not None:
        prior.mean = torch.from_numpy(chain["latent_mean"])
    return prior


def _sha(array):
    return None if array is None else hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _init_kind(chain):
    """The initialisation policy without its per-chain part (seed or row): what all chains share."""
    policy = (chain.get("provenance") or {}).get("init_policy")
    if policy is None:
        return None
    return tuple(sorted((k, v) for k, v in policy.items() if k not in ("rng_seed", "row")))


def chain_identity(chain):
    """What must agree between chains of one experiment: sampler, setting, data, prior (including the
    plateau bounds of the joint setting), basis, mean, initialisation policy and code."""
    info, prov = chain["info"], chain.get("provenance") or {}
    sampler = {k: info.get(k) for k in ("setting", "adapted", "target_acceptance", "target_accept_prob",
                                         "mass_matrix", "max_tree_depth", "n_warmup", "n_burn", "warmup_steps")}
    if info.get("adapted") is False:
        sampler["beta"] = info.get("beta")
    return (chain["method"], tuple(sorted(sampler.items())), tuple(sorted(chain["prior"].items())),
            tuple(sorted((chain.get("plateau_bounds") or {}).items())),
            prov.get("data_sha256"), _sha(chain.get("kl_basis")), _sha(chain.get("latent_mean")),
            _init_kind(chain), (prov.get("code") or {}).get("source_sha256"))


def chain_content_sha256(chain):
    """SHA-256 of a chain's draws (z, r, c_p, c_n, times): binds derived output to exact chain contents."""
    h = hashlib.sha256()
    for key in ("z", "r", "cp", "cn", "times"):
        value = chain.get(key)
        h.update(key.encode())
        if value is not None:
            h.update(np.ascontiguousarray(np.asarray(value, dtype=float)).tobytes())
    return h.hexdigest()


def load_chains(paths, allow_unequal=False):
    """
    Chains of one experiment: a run folder (all chain*.pickle in it), or a list of files.
    Raises if the chains differ in sampler settings, data, prior or KL basis, if a chain is
    duplicated, or if their lengths differ (unless allow_unequal, which truncates them).
    """
    if isinstance(paths, str):
        paths = [paths]
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "chain*.pickle"))) if os.path.isdir(p) else [p]
    if not files:
        raise FileNotFoundError(f"no chain files in {paths}")
    chains = [load_samples(f) for f in files]
    for c, f in zip(chains, files):
        c["file"] = f

    if len({chain_identity(c) for c in chains}) > 1:
        raise ValueError(f"chains of different experiments mixed: {files}")
    seeds = [c["info"].get("seed") for c in chains if c["info"].get("seed") is not None]
    if len(seeds) != len(set(seeds)) or len({c["z"][-1].tobytes() for c in chains}) < len(chains):
        raise ValueError(f"duplicated chains: {files}")
    lengths = [len(c["z"]) for c in chains]
    if len(set(lengths)) > 1:
        if not allow_unequal:
            raise ValueError(f"chains have different lengths {lengths}: {files}")
        n = min(lengths)
        print(f"warning: truncating chains to {n} draws; the stored costs refer to the full runs")
        for c in chains:
            for key in ("z", "r", "cp", "cn", "times"):
                if c.get(key) is not None:
                    c[key] = c[key][:n]
    return chains
