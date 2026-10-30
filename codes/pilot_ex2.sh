#!/bin/bash
# pCN step-size sweep for Example 2, known plateaus (Reviewer 1, item vi), declared in
# experiments/production_v2.json before it is run. Same lengths as the Example 1 sweep (pilot.sh):
# 4 chains of 5000 + 10 000 discarded and 100 000 retained steps.
#     conda activate fenics-env && bash pilot_ex2.sh
# 1) pCN tuned per chain to acceptance 0.234; beta_0 = mean tuned beta.
# 2) Fixed beta = f x beta_0 for f in 6, 8, 12. If the best (beta_sweep.py) is the smallest or the
#    largest tested f, the sweep is extended on that side (4, 3, 2 or 16, 24) until the best is interior.
# Every stage is validated with check_run.py and postprocessed (validated summary); one batch at a time.
set -u
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
EX=2
DATA=obs/obs_Ex${EX}.pickle
PRIOR="--tau 16 --n_kl 12 --k 14 --mean_a 2"
R=results
DRAWS=100000
PCN="python run_pcn.py --example $EX --setting 1 --data $DATA $PRIOR --n_samples $DRAWS --n_burn 10000 --n_warmup 5000"
CHECKS="--method pcn --draws $DRAWS --expect n_warmup=5000 --expect n_burn=10000"

validate() {    # validate <run folder> <check_run options ...>
  local dir=$1; shift
  python check_run.py "$dir" --data $DATA --setting 1 --chains 4 $PRIOR "$@" 2>/dev/null | grep "^valid:\|^not valid:"
}

run_stage() {   # run_stage <run folder> "<check_run options>" <runner command without --chain>
  local dir=$1 checks=$2 result pids=() p; shift 2
  result=$(validate "$dir" $checks)
  if [[ $result == valid:* ]]; then echo "reusing $dir"; return 0; fi
  [ -d "$dir" ] && { echo "STOP: $dir exists but is not valid (${result:-no validation}); inspect it"; exit 1; }
  echo "running $dir ($(date '+%H:%M:%S'))"
  for c in 0 1 2 3; do "$@" --chain $c > "${dir}_chain$c.log" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p; done
  result=$(validate "$dir" $checks)
  echo "  ${result:-not valid: $dir: validation did not run}"
  [[ $result == valid:* ]] || { echo "STOP: $dir (see ${dir}_chain*.log)"; exit 1; }
  python postprocess.py "$dir" --data $DATA > "$dir.log" 2>&1
  result=$(validate "$dir" --summary --draws $DRAWS)
  [[ $result == valid:* ]] || { echo "STOP: postprocessing of $dir: $result (see $dir.log)"; exit 1; }
}

run_stage $R/pcn_Setting1_Ex${EX}_pilot "$CHECKS --expect adapted=True --expect target_acceptance=0.234" $PCN --tag _pilot
BETA=$(python -c "import glob, pickle, numpy as np; print(repr(float(np.mean([pickle.load(open(f, 'rb'))['info']['beta'] for f in sorted(glob.glob('$R/pcn_Setting1_Ex${EX}_pilot/chain*.pickle'))]))))")
echo "mean tuned beta = $BETA"

run_multiple() {   # run_multiple <f>
  local b; b=$(python -c "print(repr($BETA * $1))")
  python -c "import sys; sys.exit(0 if $b < 1 else 1)" || { echo "STOP: beta = $b >= 1 for multiple $1"; exit 1; }
  run_stage $R/pcn_Setting1_Ex${EX}_pilot_beta_x$1 "$CHECKS --expect adapted=False --expect beta=$b" $PCN --beta $b --tag _pilot_beta_x$1
}

FS="6 8 12"
LOWER="4 3 2"
UPPER="16 24"
for f in $FS; do run_multiple $f; done
while true; do
  args=(); for f in $FS; do args+=(--run "$f=$R/pcn_Setting1_Ex${EX}_pilot_beta_x$f"); done
  out=$(python beta_sweep.py "${args[@]}" --json $R/beta_sweep_Ex${EX}.json 2>/dev/null) || { echo "STOP: beta_sweep.py failed"; exit 1; }
  echo "$out"
  read -r status f <<< "$(echo "$out" | tail -1)"
  [ "$status" = selected ] && { echo "selected multiple $f, beta = $(python -c "print(repr($BETA * $f))")"; break; }
  if [ "$f" = "$(echo $FS | awk '{print $1}')" ]; then next=$(echo $LOWER | awk '{print $1}'); LOWER=$(echo $LOWER | cut -s -d' ' -f2-)
  else next=$(echo $UPPER | awk '{print $1}'); UPPER=$(echo $UPPER | cut -s -d' ' -f2-); fi
  [ -n "$next" ] || { echo "STOP: best multiple $f at the end of the planned grid; decide before continuing"; exit 1; }
  echo "best at the boundary ($f): adding $next"
  run_multiple $next
  FS=$(echo $FS $next | tr ' ' '\n' | sort -g | tr '\n' ' ')
done
