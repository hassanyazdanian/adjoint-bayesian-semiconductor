# Bayesian reconstruction of pn-junction doping profiles

Python code for the paper
**"Adjoint-Based Bayesian Uncertainty Quantification for PDE-Constrained Inverse Problems with Application to Semiconductor Imaging"**.

The code reconstructs approximately piecewise-constant pn-junction doping profiles from boundary flux measurements. It contains:

- a nonlinear Poisson-Boltzmann forward model solved with FEniCS, with adjoint gradients from dolfin-adjoint;
- a Whittle-Matérn KL prior with a trend mean and a sigmoid pushforward to near piecewise-constant doping fields;
- MAP estimation, NUTS (Pyro) and pCN posterior sampling, for known plateau values and for plateau values inferred jointly;
- the scripts that produce every table and figure of the numerical section of the paper.

## Requirements

The paper's results were computed with Python 3.11.13, FEniCS 2019.1.0, dolfin-adjoint 2023.0.0, PyTorch 2.7.1, Pyro 1.9.1, ArviZ 0.23.4, NumPy 2.3.3 and SciPy 1.16.2, together with pandas, matplotlib and scikit-image, on an Apple M4 Pro. Every chain file records the package versions it was run with.

All commands below are run from `codes/`:

```bash
cd codes
```

## Files

| File | Content |
|---|---|
| `config.py` | Model, prior, data and sampler settings |
| `forward_model.py` | FEM Poisson-Boltzmann model, electrode observation operator, one-solve adjoint gradient |
| `prior.py` | Matérn KL basis, trend mean, sigmoid pushforward |
| `posterior.py` | Posterior for known plateaus (z) and for inferred plateaus (z, r_p, r_n); PyTorch bridge |
| `samplers.py` | pCN (tuned or fixed step) |
| `diagnostics.py` | Relative L2 error, SSIM, credible intervals, junction depth |
| `io_utils.py`, `plotting.py` | Data and chain files, run identity; figure helpers |
| `generate_synthetic_data.py` | Ground truth and noisy data (Examples 1 and 2), computed on a finer mesh |
| `run_map.py`, `run_nuts.py`, `run_pcn.py` | MAP estimate; one NUTS or pCN chain per process |
| `postprocess.py` | All chains of a run: R-hat, ESS, MCSE, cost-normalised ESS, field metrics |
| `production.sh` | Runs one stage of a manifest as a batch of four chains, then validates (`check_run.py`), postprocesses and computes checkpoint diagnostics (`checkpoint_diagnostics.py`) |
| `experiments/*.json` | Frozen manifests: settings, seeds and data hashes of every run in the paper |
| `obs/*.pickle` | The data sets used in the paper |
| `pilot.sh`, `pilot_ex2.sh`, `beta_sweep.py` | pCN step-size sweeps for Examples 1 and 2 |
| `production_table.py` | Diagnostics and costs of the production runs (`tables.json`) |
| `production_figures.py` | Posterior mean, std and MAP panels, trace plots |
| `plot_prior_samples.py` | Prior samples |
| `joint_report.py`, `make_joint_marginals.py` | Joint reconstruction: diagnostics, posterior summaries, data fit, figures |
| `compare_samplers.py` | Posterior quantiles of the junction depth with MCSE, between runs |
| `sensitivity_study.py` | Sensitivity and N_KL scaling study (Table 3 and Section 6.3.3) |
| `verify_gradients.py`, `gradient_cost.py`, `representation_error.py` | Taylor test, gradient cost against N_KL, best sigmoid approximation of the truth |
| `make_paper_tables.py` | LaTeX Tables 1-3 from the JSON outputs |

## Reproducing the paper

Outputs are written to `codes/results/`, which is not tracked.

**Runs.** Every sampling run of the paper is a stage of a manifest in `experiments/`. `production.sh` runs one stage at a time as four chains in parallel (one process each), with the seeds, settings and thread limits of the manifest. Before it starts, it checks that the code and the manifest are committed and that the data files have the SHA-256 of the manifest. The run folder is `results/<method>_Setting<s>_Ex<e><tag>/`. One batch of the paper took between 26 minutes and 13.4 hours on the machine above. Run nothing else on the machine during a batch if the timings matter.

```bash
bash production.sh ex1_pcn ex1_nuts                                              # Example 1: pCN, NUTS at target 0.99
PRODUCTION_MANIFEST=experiments/production_v1_t095.json bash production.sh ex1_nuts  # Example 1: NUTS at target 0.95
PRODUCTION_MANIFEST=experiments/production_v2_pcn2m.json bash production.sh ex2_pcn  # Example 2: pCN
PRODUCTION_MANIFEST=experiments/production_v2.json bash production.sh ex2_nuts       # Example 2: NUTS at target 0.95
PRODUCTION_MANIFEST=experiments/production_v2_t099.json bash production.sh ex2_nuts  # Example 2: NUTS at target 0.99
PRODUCTION_MANIFEST=experiments/production_v3_joint.json bash production.sh joint_A joint_B   # joint reconstruction
```

**Data.** The data sets of the paper are in `obs/` (192 x 192 data mesh; for Table 3 also 96 and 384). `generate_synthetic_data.py` regenerates them and overwrites the files in `obs/`:

```bash
python generate_synthetic_data.py                                # obs/obs_Ex1.pickle, obs/obs_Ex2.pickle
python generate_synthetic_data.py --examples 1 --data_mesh 96    # obs/obs_Ex1_mesh96.pickle
python generate_synthetic_data.py --examples 1 --data_mesh 384   # obs/obs_Ex1_mesh384.pickle
```

**pCN step size (Sections 6.3.1, 6.3.2).** Four chains of 100 000 draws for the tuned step and for fixed multiples of it. `pilot.sh` also runs NUTS pilots, which the paper does not use.

```bash
bash pilot.sh
bash pilot_ex2.sh
```

**Tables 1 and 2, MAP estimates and Example 2 depth quantiles.**

```bash
P=results/pcn_Setting1_Ex1_pilot
python production_table.py --stages ex1_pcn experiments/production_v1_t095.json:ex1_nuts ex1_nuts \
  --out results/production_v1_report/ex1 \
  --pilot "pCN tuned=$P" --pilot "pCN 0.5x=${P}_beta_x0.5" --pilot "pCN 2x=${P}_beta_x2" --pilot "pCN 3x=${P}_beta_x3" \
  --pilot "pCN 4x=${P}_beta_x4" --pilot "pCN 6x=${P}_beta_x6" --pilot "pCN 8x=${P}_beta_x8" --pilot "pCN 12x=${P}_beta_x12"
P=results/pcn_Setting1_Ex2_pilot
python production_table.py --manifest experiments/production_v2.json \
  --stages experiments/production_v2_pcn2m.json:ex2_pcn ex2_nuts experiments/production_v2_t099.json:ex2_nuts \
  --out results/production_v2_report/ex2 \
  --pilot "pCN tuned=$P" --pilot "pCN 6x=${P}_beta_x6" --pilot "pCN 8x=${P}_beta_x8" --pilot "pCN 12x=${P}_beta_x12"
python joint_report.py --manifest experiments/production_v3_joint.json --out results/production_v3_report/joint \
  --known_plateau_run results/nuts_Setting1_Ex1_prod_v1_t095
python run_map.py --example 1 --data obs/obs_Ex1.pickle --tau 16 --n_kl 12 --k 14 --mean_a 2 --out results/map_Setting1_Ex1_prod_v1
python run_map.py --example 2 --data obs/obs_Ex2.pickle --tau 16 --n_kl 12 --k 14 --mean_a 2 --out results/map_Setting1_Ex2_prod_v2
python compare_samplers.py results/pcn_Setting1_Ex2_prod_v2_pcn2m results/nuts_Setting1_Ex2_prod_v2 \
  results/nuts_Setting1_Ex2_prod_v2_t099 --example 2 --out results/production_v2_report/ex2/comparison
```

`production_table.py` writes `tables.json` and `tables.md`. The pCN sweep values of Sections 6.3.1 and 6.3.2 are in their tuning-pilot table. `joint_report.py` writes the joint diagnostics, posterior summaries, correlations and data misfit of Section 6.4, for both prior supports (`joint_A`, `joint_B`).

**Figures.**

```bash
python plot_prior_samples.py --out results/figures/prior_samples.png                      # Figure 3
python production_figures.py --example 1 --map results/map_Setting1_Ex1_prod_v1 \
  --nuts_manifest experiments/production_v1_t095.json --out results/figures/ex1           # Figure 4
python production_figures.py --example 2 --manifest experiments/production_v2_pcn2m.json \
  --nuts_manifest experiments/production_v2.json --map results/map_Setting1_Ex2_prod_v2 \
  --out results/figures/ex2                                                               # Figure 6
python production_figures.py --example 1 --map results/map_Setting1_Ex1_prod_v1 --nuts_manifest experiments/production_v1_t095.json \
  --quantities depth --panel_letters a b --no_legend --out results/figures/ex1_depth      # Figure 5 (a, b)
python production_figures.py --example 2 --manifest experiments/production_v2_pcn2m.json --nuts_manifest experiments/production_v2.json \
  --map results/map_Setting1_Ex2_prod_v2 --quantities depth --panel_letters c d --out results/figures/ex2_depth   # Figure 5 (c, d)
python make_joint_marginals.py results/nuts_Setting2_Ex1_prod_v3_joint_A \
  results/production_v3_report/joint/tables.json results/production_v3_report/joint                         # Figure 7 (c, d)
```

Figure 7 (a, b) is `joint_mean_A.png` and `joint_std_A.png` of `joint_report.py`.

**Table 3 and the sensitivity and scaling results (Section 6.3.3).**

```bash
PRODUCTION_MANIFEST=experiments/study_k_v1.json bash production.sh k14 k20 k28
PRODUCTION_MANIFEST=experiments/study_k_v1_k10r.json bash production.sh k10
PRODUCTION_MANIFEST=experiments/study_timing_v1.json bash production.sh data96 data384 nkl24 nkl48 nkl96
PRODUCTION_MANIFEST=experiments/study_pcn_nkl_v1.json bash production.sh pcn_nkl24 pcn_nkl48 pcn_nkl96
python sensitivity_study.py rows      # Table 3 -> results/sensitivity_ex1/sensitivity.{json,md}
python sensitivity_study.py pcn       # pCN at N_KL = 12, 24, 48, 96 -> results/sensitivity_ex1/pcn_nkl.{json,md}
python verify_gradients.py            # Taylor test in both inference settings; prints "Gradient checks passed."
python gradient_cost.py               # adjoint gradient cost for N_KL = 12 to 192 -> results/scaling_Ex1/
python representation_error.py        # best sigmoid approximation of the truth for k = 10, 14, 20, 28 -> results/kstudy_Ex1/
```

The reference row of Table 3 is stage `k14` of `study_k_v1.json`. The `k10` stage of that manifest is not used: its batch ran while other heavy processes were running, and it was rerun under `study_k_v1_k10r.json`.

**LaTeX tables.**

```bash
python make_paper_tables.py           # Tables 1-3 -> results/paper_tables/
```

**Note.** In some MPI environments a process exits with a non-zero status (for example 143) from `MPI_Finalize` after it has written its output. The scripts validate the outputs instead of relying on the exit status.

## Citation

If you use this code, please cite the accompanying paper.

## License

MIT License, see `LICENSE`.
