#!/bin/bash
# Production runs from an experiment manifest (default experiments/production_v1.json).
#     conda activate fenics-env && bash production.sh ex1_nuts ex1_pcn ex2_nuts ex2_pcn joint_nuts
# Stages run one after another, each as one batch of four chains (one process per chain) with the
# thread limits of the manifest and no other stage running, so that times are comparable.
# PRODUCTION_MANIFEST=<file> selects another manifest (e.g. a small test manifest).
# A stage may override prior settings of the manifest ("prior": {"k": 10}; keys tau, n_kl, k, mean_a) and
# add a suffix to the run folder ("tag": "_k10"); without these keys every stage uses the manifest's prior.
# A stage may also set its base seed ("seed": 100; chain c uses seed + c) and, in Setting 2, the supports of
# the uniform plateau priors ("plateau_bounds": {"cp": [-3, -1], "cn": [0, 2]}); otherwise the manifest's
# chains.seed and model.plateau_bounds_joint apply. Setting-2 bounds are always passed explicitly.
# A stage may use another data set of its example ("data": "ex1_mesh96", a key of the manifest's "data"
# that is ex<example> or starts with ex<example>_); otherwise ex<example>. Every data set's hash is checked.
# A NUTS stage may set its inversion mesh ("inversion_mesh_n": 96, a multiple of 48); otherwise the data file's (48).
#
# Before anything runs: the code must be committed (no changes to tracked files), the manifest must
# have status "final" and, if it lies in the repository, be committed; the observation files must
# have the hashes, and config.py the model constants, of the manifest.
# For every stage: every setting is passed explicitly; batch launch and end times are recorded in
# <run>_batch.json, with the load averages and the most CPU-intensive processes just before the launch
# and just after the last chain ended; the chains are validated (check_run.py --production --manifest), postprocessed
# (summary bound to data and chains, validated), and checkpoint diagnostics are computed.
# An existing run folder is never overwritten: if it is valid it is kept and only (re)postprocessed,
# otherwise the driver stops for inspection. A valid run that fails the diagnostics or needs
# divergence follow-up is kept and flagged; settings are never changed silently.
set -u
cd "$(dirname "$0")" || exit 1
MANIFEST=${PRODUCTION_MANIFEST:-experiments/production_v1.json}
[ $# -ge 1 ] || { echo "usage: bash production.sh <stage> [<stage> ...]"; exit 1; }

mget() { python -c "import json; m = json.load(open('$MANIFEST')); print($1)"; }

# ---- preconditions ---------------------------------------------------------------------------
[ -f "$MANIFEST" ] || { echo "REFUSED: no manifest $MANIFEST"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "REFUSED: uncommitted changes to tracked files"; exit 1; }
if git ls-files --error-unmatch "$MANIFEST" > /dev/null 2>&1; then
  [ -z "$(git status --porcelain -- "$MANIFEST")" ] || { echo "REFUSED: $MANIFEST has uncommitted changes"; exit 1; }
elif [[ "$(cd "$(dirname "$MANIFEST")" && pwd)" == "$(pwd)"* ]]; then
  echo "REFUSED: $MANIFEST lies in the repository but is not committed"; exit 1
else
  echo "note: $MANIFEST lies outside the repository (test manifest)"
fi
[ "$(mget "m['status']")" = final ] || { echo "REFUSED: manifest status is '$(mget "m['status']")', not 'final'"; exit 1; }
python - "$MANIFEST" <<'EOF' || exit 1
import hashlib, json, sys
import config as cfg
m = json.load(open(sys.argv[1]))
bad = []
for ex in [k for k in m["data"] if k.startswith("ex")]:
    d = m["data"][ex]
    sha = hashlib.sha256(open(d["path"], "rb").read()).hexdigest()
    if sha != d["sha256"]:
        bad.append(f"{d['path']} has SHA-256 {sha[:12]}..., the manifest {d['sha256'][:12]}...")
p, mm = m["model"]["prior"], m["model"]
expected = {"MESH_N": mm["inversion_mesh_n"], "EPS": mm["eps"], "DELTA": mm["delta"], "TAU": p["tau"], "ALPHA": p["alpha"],
            "N_KL": p["n_kl"], "K_SIGMOID": p["k"], "PRIOR_MEAN_A": p["mean_a"], "CP_TRUE": mm["plateaus_known"]["cp"],
            "CN_TRUE": mm["plateaus_known"]["cn"], "CP_BOUNDS": tuple(mm["plateau_bounds_joint"]["cp"]),
            "CN_BOUNDS": tuple(mm["plateau_bounds_joint"]["cn"]), "INIT_FROM_PRIOR": m["chains"]["initialisation"] == "prior_draw"}
for name, value in expected.items():
    if getattr(cfg, name) != value:
        bad.append(f"config.{name} = {getattr(cfg, name)}, the manifest {value}")
if bad:
    print("REFUSED: " + "; ".join(bad)); sys.exit(1)
EOF
for v in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS VECLIB_MAXIMUM_THREADS NUMEXPR_NUM_THREADS; do
  export $v="$(mget "m['timing']['threads']['$v']")"
done
NCH=$(mget "m['chains']['count']"); SEED=$(mget "m['chains']['seed']"); TAG=$(mget "m['output_tag']")
BOUNDS=""
MESHARG=""
PRIOR="--tau $(mget "m['model']['prior']['tau']") --n_kl $(mget "m['model']['prior']['n_kl']") --k $(mget "m['model']['prior']['k']") --mean_a $(mget "m['model']['prior']['mean_a']")"
now() { python -c "import time; print(repr(time.time()))"; }

snapshot() {   # snapshot <file>: load averages and the most CPU-intensive processes, as JSON
  python - "$1" <<'EOF'
import json, os, subprocess, sys, time
ps = subprocess.run(["ps", "-A", "-o", "pid=,pcpu=,etime=,comm="], capture_output=True, text=True).stdout.splitlines()
rows = sorted((l.split(None, 3) for l in ps if len(l.split(None, 3)) == 4), key=lambda r: -float(r[1]))[:8]
json.dump(dict(epoch=time.time(), loadavg_1_5_15_min=list(os.getloadavg()),
               top_cpu_processes=[dict(pid=int(r[0]), pcpu=float(r[1]), etime=r[2], command=r[3]) for r in rows]),
          open(sys.argv[1], "w"), indent=2)
EOF
}

validate() {    # validate <run folder> <data> <setting> <check_run options ...>
  local dir=$1 data=$2 setting=$3; shift 3
  python check_run.py "$dir" --data "$data" --setting "$setting" --chains "$NCH" --seed "$SEED" --production \
    --manifest "$MANIFEST" $PRIOR $BOUNDS $MESHARG "$@" 2>/dev/null | grep "^valid:\|^not valid:"
}

postprocess_run() {   # postprocess_run <run folder> <data> <setting> <draws> <checkpoint interval>
  local d=$1 data=$2 setting=$3 draws=$4 every=$5 pp="${1}_postprocess" result f
  for f in summary.json fields.npz checkpoints.json checkpoints.md; do [ -f "$pp/$f" ] && mv "$pp/$f" "$pp/prev_$f"; done
  python postprocess.py "$d" --data "$data" --out "$pp" > "$d.log" 2>&1 || echo "note: postprocess exited with status $?"
  result=$(validate "$d" "$data" "$setting" --summary --draws "$draws")
  [[ $result == valid:* ]] || { echo "STOP: ${result:-postprocessing of $d failed} (see $d.log)"; return 1; }
  python checkpoint_diagnostics.py "$d" --data "$data" --every "$every" > "$d.checkpoints.log" 2>&1 \
    && [ -f "$pp/checkpoints.json" ] || { echo "STOP: checkpoint diagnostics failed (see $d.checkpoints.log)"; return 1; }
  python - "$pp" <<'EOF'
import json, sys
s = json.load(open(f"{sys.argv[1]}/summary.json")); c = json.load(open(f"{sys.argv[1]}/checkpoints.json"))
conv, div, ms = s["convergence"], s.get("divergences", {}), c["milestone"]
print(f"  diagnostics: passes = {conv['passes']} (max R-hat {conv['r_hat_max']}, min bulk ESS {conv['ess_bulk_min']}, "
      f"min tail ESS {conv['ess_tail_min']}, undefined {conv['undefined']})")
if div:
    print(f"  divergences: {div['count']} ({100 * div['fraction']:.2f} %), per chain {div['per_chain']}, follow-up: {div['follow_up']}")
print(f"  checkpoints: {ms['statement']}" + ("" if ms["first_pass"] is None else
      f"; first pass at {ms['first_pass_draws']} draws per chain, persists: {ms['persists']}"))
EOF
}

run_stage() {   # run_stage <stage>
  local stage=$1 example setting method data dir args checks draws every pids=() statuses=() c p t0 t1 result
  read -r example setting method <<< "$(mget "' '.join(str(m['stages']['$stage'][k]) for k in ('example', 'setting', 'method'))" 2>/dev/null)"
  [ -n "${method:-}" ] || { echo "STOP: no stage '$stage' in $MANIFEST"; return 1; }
  # stage-level data set (default ex<example>)
  local dkey dok
  dkey=$(mget "m['stages']['$stage'].get('data', 'ex$example')")
  dok=$(mget "'$dkey' in m['data'] and ('$dkey' == 'ex$example' or '$dkey'.startswith('ex${example}_'))")
  [ "$dok" = True ] || { echo "STOP: stage $stage uses data '$dkey', not a data set of example $example in $MANIFEST"; return 1; }
  data=$(mget "m['data']['$dkey']['path']")
  # stage-level prior overrides and run-folder suffix (PRIOR is local: validate and postprocess_run see it)
  local PRIOR stag
  PRIOR=$(mget "(lambda s, p: ' '.join(f'--{x} {s.get(x, p[x])}' for x in ('tau', 'n_kl', 'k', 'mean_a')) if set(s) <= {'tau', 'n_kl', 'k', 'mean_a'} else 'BAD')(m['stages']['$stage'].get('prior', {}), m['model']['prior'])")
  [ "$PRIOR" != BAD ] || { echo "STOP: stage $stage overrides a prior key other than tau, n_kl, k, mean_a"; return 1; }
  stag=$(mget "m['stages']['$stage'].get('tag', '')")
  # assignments, not "$(...)" inside [ ]: /bin/bash 3.2 mis-parses quoted command substitutions with braces
  local extra pb
  extra=$(mget "sorted(set(m['stages']['$stage']).difference(['example', 'setting', 'method', 'prior', 'tag', 'seed', 'plateau_bounds', 'data', 'inversion_mesh_n']))")
  [ "$extra" = "[]" ] || { echo "STOP: stage $stage has unknown keys $extra"; return 1; }
  # stage-level inversion mesh (local: validate sees it)
  local MESHARG imesh
  imesh=$(mget "m['stages']['$stage'].get('inversion_mesh_n', '')")
  if [ -n "$imesh" ]; then
    [ "$method" = nuts ] || { echo "STOP: an inversion mesh is supported for NUTS stages only"; return 1; }
    [ $((imesh % 48)) -eq 0 ] && [ "$imesh" -ge 48 ] || { echo "STOP: inversion mesh $imesh is not a multiple of 48"; return 1; }
    MESHARG="--inversion_mesh_n $imesh"
  else
    MESHARG=""
  fi
  # stage-level seed and plateau bounds (local: validate and postprocess_run see them)
  local SEED BOUNDS
  SEED=$(mget "m['stages']['$stage'].get('seed', m['chains']['seed'])")
  if [ "$setting" = 2 ]; then
    [ "$method" = nuts ] || { echo "STOP: this driver runs the joint setting with NUTS only"; return 1; }
    BOUNDS=$(mget "(lambda b: f\"--cp_bounds {float(b['cp'][0])} {float(b['cp'][1])} --cn_bounds {float(b['cn'][0])} {float(b['cn'][1])}\")(m['stages']['$stage'].get('plateau_bounds', m['model']['plateau_bounds_joint']))")
  else
    pb=$(mget "'plateau_bounds' in m['stages']['$stage']")
    [ "$pb" = False ] || { echo "STOP: plateau bounds given for a known-plateau stage"; return 1; }
    BOUNDS=""
  fi
  dir=results/${method}_Setting${setting}_Ex${example}${TAG}${stag}
  if [ "$method" = nuts ]; then
    draws=$(mget "m['nuts']['num_samples_cap']"); every=$(mget "m['nuts']['checkpoint_every']")
    local w=$(mget "m['nuts']['warmup_steps']") t=$(mget "m['nuts']['target_accept_prob']") dpt=$(mget "m['nuts']['max_tree_depth']")
    local full=$( [ "$(mget "m['nuts']['mass_matrix']")" = dense ] && echo True || echo False )
    args="run_nuts.py --warmup_steps $w --num_samples $draws --full_mass $full --target_accept_prob $t --max_tree_depth $dpt"
    checks="--method nuts --draws $draws --expect warmup_steps=$w --expect target_accept_prob=$t --expect max_tree_depth=$dpt --expect mass_matrix=$(mget "m['nuts']['mass_matrix']")"
  else
    local beta=$(mget "m['pcn']['beta']['ex$example']")
    [ "$beta" != None ] || { echo "STOP: no pCN beta for example $example in $MANIFEST"; return 1; }
    draws=$(mget "m['pcn']['n_samples_cap']"); every=$(mget "m['pcn']['checkpoint_every']")
    local nw=$(mget "m['pcn']['n_warmup']") nb=$(mget "m['pcn']['n_burn']")
    args="run_pcn.py --n_warmup $nw --n_burn $nb --n_samples $draws --beta $beta"
    checks="--method pcn --draws $draws --expect n_warmup=$nw --expect n_burn=$nb --expect adapted=False --expect beta=$beta"
  fi
  echo "== stage $stage: $method, example $example, setting $setting, data $data, seed $SEED${BOUNDS:+, $BOUNDS}${MESHARG:+, $MESHARG} -> $dir ($(date '+%Y-%m-%d %H:%M:%S'))"
  if [ -d "$dir" ]; then
    result=$(validate "$dir" "$data" "$setting" $checks)
    if [[ $result == valid:* ]]; then
      echo "  existing valid run kept (not rerun): $result"
    else
      echo "STOP: $dir exists but is not valid (${result:-validation did not run}); inspect it, it is not overwritten"
      return 1
    fi
  else
    snapshot "${dir}_load_start.json"
    t0=$(now)
    for ((c = 0; c < NCH; c++)); do
      python $args --example "$example" --setting "$setting" --data "$data" --seed "$SEED" $PRIOR $BOUNDS $MESHARG \
        --manifest "$MANIFEST" --tag "$TAG$stag" --chain $c > "${dir}_chain$c.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; statuses+=($?); done
    t1=$(now)
    snapshot "${dir}_load_end.json"
    python -c "import json, os; json.dump(dict(stage='$stage', manifest='$MANIFEST', launch_epoch=$t0, end_epoch=$t1,
elapsed_s=$t1 - $t0, exit_status=[$(IFS=,; echo "${statuses[*]}")], processes=$NCH, prior_arguments='$PRIOR',
seed=$SEED, plateau_bounds_arguments='$BOUNDS', data='$data', inversion_mesh_arguments='$MESHARG',
threads={v: os.environ.get(v) for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}, load_start=json.load(open('${dir}_load_start.json')),
load_end=json.load(open('${dir}_load_end.json'))), open('${dir}_batch.json', 'w'), indent=2)" \
      && rm -f "${dir}_load_start.json" "${dir}_load_end.json"
    echo "  batch finished in $(python -c "print(round($t1 - $t0))") s; process exit statuses ${statuses[*]} (validated next)"
    result=$(validate "$dir" "$data" "$setting" $checks)
    [[ $result == valid:* ]] || { echo "STOP: ${result:-validation did not run} (see ${dir}_chain*.log)"; return 1; }
    echo "  $result"
  fi
  postprocess_run "$dir" "$data" "$setting" "$draws" "$every" || return 1
}

echo "manifest $MANIFEST ($(mget "m['id']")), code $(git rev-parse --short HEAD), threads OMP=$OMP_NUM_THREADS"
for stage in "$@"; do
  run_stage "$stage" || { echo "production stopped at stage $stage"; exit 1; }
done
echo "done: $*"
