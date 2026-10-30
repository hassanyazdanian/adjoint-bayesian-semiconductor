#!/bin/bash
# Pilot runs for the sampler comparison (Example 1, known plateaus), 4 chains per configuration.
# Run from this folder with the fenics-env environment active:
#     conda activate fenics-env && bash pilot.sh
# Every stage runs exactly 4 processes at a time, so worker times are comparable across samplers.
# Every stage, whether rerun or reused, is validated by check_run.py: chain IDs 0-3 with seeds
# 0-3, the requested sampler, settings and number of draws, finite arrays, and the data, prior
# and KL basis of the current configuration. An invalid existing folder is moved aside and rerun;
# an invalid fresh stage stops the script. A non-zero exit status of a process is reported but
# is not decisive by itself (in some environments MPI_Finalize fails after the results are
# saved); the validation of the outputs is. Before each run is postprocessed, its previous
# summary.json and fields.npz are moved to prev_*; the new output is validated, so a failed
# postprocessing cannot pass on an earlier summary. The script exits with status 1 if any is invalid.
# Results: results/<method>_Setting1_Ex1_pilot*/ and their *_postprocess/summary.json
export OMP_NUM_THREADS=1
EX=1
CHAINS="0 1 2 3"
DATA=obs/obs_Ex${EX}.pickle
R=results
PCN_DRAWS=100000
NUTS_DRAWS=300

validate() {    # validate <run folder> <check_run options ...>
  local dir=$1; shift
  python check_run.py "$dir" --data $DATA --setting 1 --chains 4 "$@" 2>/dev/null | grep "^valid:\|^not valid:"
}

run_stage() {   # run_stage <run folder> "<check_run options>" <runner command without --chain>
  local dir=$1 checks=$2; shift 2
  local result
  result=$(validate "$dir" $checks)
  if [[ $result == valid:* ]]; then
    echo "reusing $dir"; return 0
  fi
  [ -d "$dir" ] && { echo "${result:-not valid: $dir}"; mv "$dir" "${dir}_stale_$(date +%Y%m%d_%H%M%S)"; }
  echo "running $dir"
  local pids=()
  for c in $CHAINS; do "$@" --chain $c > "${dir}_chain$c.log" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p || echo "note: process $p exited with status $? (outputs are validated next)"; done
  result=$(validate "$dir" $checks)
  echo "${result:-not valid: $dir: validation did not run}"
  [[ $result == valid:* ]] || { echo "FAILED: $dir (see ${dir}_chain*.log)"; exit 1; }
}

postprocess_run() {   # postprocess_run <run folder> <draws>: recompute and validate the postprocessing
  local d=$1 draws=$2 pp="${1}_postprocess" result f
  for f in summary.json fields.npz; do [ -f "$pp/$f" ] && mv "$pp/$f" "$pp/prev_$f"; done
  python postprocess.py "$d" --data $DATA --out "$pp" > "$d.log" 2>&1 || echo "note: postprocess of $d exited with status $?"
  result=$(validate "$d" --summary --draws $draws)
  [[ $result == valid:* ]] && return 0
  echo "FAILED: ${result:-postprocessing of $d} (see $d.log; the previous output is kept as $pp/prev_*)"
  return 1
}

PCN="python run_pcn.py --example $EX --setting 1 --n_samples $PCN_DRAWS --n_burn 10000 --n_warmup 5000"
NUTS="python run_nuts.py --example $EX --setting 1 --warmup_steps 300 --num_samples $NUTS_DRAWS"
PCN_CHECKS="--method pcn --draws $PCN_DRAWS --expect n_warmup=5000 --expect n_burn=10000"
NUTS_CHECKS="--method nuts --draws $NUTS_DRAWS --expect warmup_steps=300 --expect max_tree_depth=10"

# 1) pCN with tuned beta (target acceptance 0.234)
run_stage $R/pcn_Setting1_Ex${EX}_pilot "$PCN_CHECKS --expect adapted=True --expect target_acceptance=0.234" \
  $PCN --tag _pilot

# 2) pCN step-size sweep: fixed multiples of the mean tuned beta (Reviewer 1, item vi)
BETA=$(python -c "import glob, pickle, numpy as np; print(np.mean([pickle.load(open(f, 'rb'))['info']['beta'] for f in glob.glob('$R/pcn_Setting1_Ex${EX}_pilot/chain*.pickle')]))")
echo "mean tuned beta = $BETA"
for f in 0.5 2 3 4 6 8 12; do
  b=$(python -c "print($BETA * $f)")
  run_stage $R/pcn_Setting1_Ex${EX}_pilot_beta_x$f "$PCN_CHECKS --expect adapted=False --expect beta=$b" \
    $PCN --beta $b --tag _pilot_beta_x$f
done

# 3) NUTS: diagonal and dense mass matrices (target acceptance 0.8), dense with targets 0.9 and 0.95
run_stage $R/nuts_Setting1_Ex${EX}_pilot_diag "$NUTS_CHECKS --expect mass_matrix=diagonal --expect target_accept_prob=0.8" \
  $NUTS --full_mass False --target_accept_prob 0.8 --tag _pilot_diag
run_stage $R/nuts_Setting1_Ex${EX}_pilot_dense "$NUTS_CHECKS --expect mass_matrix=dense --expect target_accept_prob=0.8" \
  $NUTS --full_mass True --target_accept_prob 0.8 --tag _pilot_dense
for t in 0.9 0.95; do
  run_stage $R/nuts_Setting1_Ex${EX}_pilot_dense_t$t "$NUTS_CHECKS --expect mass_matrix=dense --expect target_accept_prob=$t" \
    $NUTS --full_mass True --target_accept_prob $t --tag _pilot_dense_t$t
done

# 4) Diagnostics and cost-normalised efficiency for every pilot run, each summary validated
failed=0
for d in $R/*_Setting1_Ex${EX}_pilot*; do
  [ -d "$d" ] || continue
  case $d in *_postprocess|*_stale_*) continue ;; esac
  case $d in $R/pcn_*) draws=$PCN_DRAWS ;; *) draws=$NUTS_DRAWS ;; esac
  postprocess_run "$d" $draws || failed=1
done
[ $failed -eq 0 ] || { echo "postprocessing incomplete"; exit 1; }
echo "done: see $R/*_pilot*_postprocess/summary.json"
