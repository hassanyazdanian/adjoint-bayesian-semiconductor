# run_chains.py
"""
Runs the chains of one experiment in parallel, one process per chain (run_nuts.py or run_pcn.py with --chain 0, 1,
..., chains - 1), with one thread per process (OMP, MKL, OpenBLAS, vecLib, numexpr), and records the wall-clock time
of the batch, from the launch of the first process to the end of the last, in results/<run>_batch.json. Every
argument after the method is passed to the sampler script. The output of chain c is in results/<run>_chain<c>.log.
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

HERE = os.path.dirname(os.path.abspath(__file__))
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=["nuts", "pcn"])
    ap.add_argument("--chains", type=int, default=4)
    a, rest = ap.parse_known_args()
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--example", type=int, default=1)
    p.add_argument("--setting", type=int, default=1)
    p.add_argument("--tag", default="")
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
    with open(f"{run}_batch.json", "w") as f:
        json.dump(dict(command=command, chains=a.chains, launch_epoch=t0, end_epoch=t1, elapsed_s=t1 - t0, exit_status=status,
                       threads={v: env[v] for v in THREADS}), f, indent=2)
    missing = [c for c in range(a.chains) if not os.path.exists(cfg.chain_file(a.method, r.setting, r.example, c, r.tag))]
    print(f"batch finished in {t1 - t0:.0f} s; exit status {status}")
    # some MPI builds exit with a non-zero status from MPI_Finalize after the chain is saved; the chain files decide
    if missing:
        sys.exit(f"no chain file for chain(s) {missing}; see {run}_chain<c>.log")


if __name__ == "__main__":
    main()
