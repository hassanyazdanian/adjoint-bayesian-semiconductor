# run_chains.py
"""
Runs the chains of one experiment in parallel, one process per chain (run_nuts.py or run_pcn.py with --chain 0, 1,
..., chains - 1), with one thread per process (OMP, MKL, OpenBLAS, vecLib, numexpr), and records the wall-clock time
of the batch, from the launch of the first process to the end of the last, in results/<run>_batch.json. Every
argument after the method is passed to the sampler script. The output of chain c is in results/<run>_chain<c>.log.
At the end every chain file is read back and checked: chain index, seed and number of draws.
Usage: python run_chains.py nuts|pcn [--chains 4] <arguments of run_nuts.py or run_pcn.py>
Example: python run_chains.py nuts --example 1 --setting 1 --seed 0 --num_samples 2000 --full_mass True --tag _test
"""
import argparse
import json
import os
import subprocess
import sys
import time

import config as cfg
from io_utils import load_samples

HERE = os.path.dirname(os.path.abspath(__file__))
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def check_chain(path, chain, seed, draws):
    """None if path holds chain `chain` (seed seed + chain) with `draws` draws, else what is wrong with it."""
    if not os.path.exists(path):
        return "no chain file"
    try:
        c = load_samples(path)
    except Exception as e:                          # truncated, or not a chain file
        return f"unreadable chain file ({type(e).__name__}: {e})"
    if c["info"].get("chain") != chain or c["info"].get("seed") != seed + chain:
        return f"the file holds chain {c['info'].get('chain')} with seed {c['info'].get('seed')}"
    if len(c["z"]) != draws:
        return f"{len(c['z'])} draws instead of {draws}"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=["nuts", "pcn"])
    ap.add_argument("--chains", type=int, default=4)
    a, rest = ap.parse_known_args()
    if a.chains < 1:
        ap.error("--chains must be at least 1")
    defaults = cfg.NUTS if a.method == "nuts" else cfg.PCN
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--example", type=int, default=1)
    p.add_argument("--setting", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=defaults["seed"])
    if a.method == "nuts":
        p.add_argument("--num_samples", dest="draws", type=int, default=defaults["num_samples"])
    else:
        p.add_argument("--n_samples", dest="draws", type=int, default=defaults["n_samples"])
    r, _ = p.parse_known_args(rest)
    run = cfg.run_dir(a.method, r.setting, r.example, r.tag)
    if os.path.exists(run):
        sys.exit(f"{run} exists; remove it or choose another --tag")
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    env = dict(os.environ, **{v: "1" for v in THREADS})
    command = [sys.executable, os.path.join(HERE, f"run_{a.method}.py"), *rest]
    print(f"{a.chains} chains -> {run}")
    t0 = time.time()
    procs = []
    for c in range(a.chains):
        log = open(f"{run}_chain{c}.log", "w")
        procs.append((subprocess.Popen(command + ["--chain", str(c)], env=env, stdout=log, stderr=subprocess.STDOUT), log))
    status = []
    for proc, log in procs:
        status.append(proc.wait())
        log.close()
    t1 = time.time()
    # some MPI builds exit with a non-zero status from MPI_Finalize after the chain is saved, so the chain
    # files decide whether a chain finished (they are written atomically, see io_utils.save_pickle)
    problems = {c: check_chain(cfg.chain_file(a.method, r.setting, r.example, c, r.tag), c, r.seed, r.draws)
                for c in range(a.chains)}
    problems = {c: msg for c, msg in problems.items() if msg is not None}
    with open(f"{run}_batch.json", "w") as f:
        json.dump(dict(command=command, chains=a.chains, launch_epoch=t0, end_epoch=t1, elapsed_s=t1 - t0, exit_status=status,
                       threads={v: env[v] for v in THREADS}, chain_problems={str(c): p for c, p in problems.items()}),
                  f, indent=2)
    print(f"batch finished in {t1 - t0:.0f} s; exit status {status}")
    if problems:
        sys.exit("\n".join(f"chain {c}: {msg}; see {run}_chain{c}.log" for c, msg in problems.items()))


if __name__ == "__main__":
    main()
