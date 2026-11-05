# Bayesian reconstruction of pn-junction doping profiles

Python code for the paper
**"Adjoint-Based Bayesian Uncertainty Quantification for PDE-Constrained Inverse Problems with Application to Semiconductor Imaging"**.

The code reconstructs approximately piecewise-constant pn-junction doping profiles from boundary flux measurements. It contains a nonlinear Poisson-Boltzmann forward model solved with FEniCS and adjoint gradients from dolfin-adjoint, a Whittle-Matérn KL prior with a sigmoid pushforward, MAP estimation, and NUTS (Pyro) and pCN posterior sampling.

## Folders

| Folder | Paper | Content |
|---|---|---|
| `examples/` | Sections 6.3.1, 6.3.2, 6.4; Tables 1 and 2; Figures 3 to 7 | Example 1 (straight junction), Example 2 (curved junction), joint reconstruction of the field and the plateau values |
| `sensitivity/` | Section 6.3.3; Table 3 | Sensitivity to the data mesh, the KL truncation and the sigmoid steepness; Taylor test; gradient cost; pCN against N_KL |

Each folder is self-contained, with its own copy of the model code and the data. Run every command from inside the folder; the outputs go to `results/` in that folder.

## Requirements

The paper's results were computed on an Apple M4 Pro (macOS, arm64) with Python 3.11.13, FEniCS 2019.1.0, PyTorch 2.7.1, Pyro 1.9.1, ArviZ 0.23.4, NumPy 2.3.3, SciPy 1.16.2, pandas 3.0.6, matplotlib 3.10.6 and scikit-image 0.26.0. The adjoint code that is imported is dolfin-adjoint/pyadjoint 2018.1.0 from the `torch-fenics` branch of `https://bitbucket.org/barkm/pyadjoint` (commit 3d02816), installed with pip over the conda-forge package dolfin-adjoint 2023.0.0; the package metadata, and so the provenance stored with the chains, report 2023.0.0.

`environment.yml` is an export of that conda environment, with the two packages installed from git pinned to their commits. It is specific to macOS arm64 and has not been tested on another machine:

```bash
conda env create -f environment.yml
conda activate fenics-env
```

## How a run works

- `run_chains.py nuts|pcn <options>` runs four chains in parallel, one process per chain, with one thread per process. Chain c uses the seed `--seed` + c and starts from a prior draw. All other options are passed to `run_nuts.py` or `run_pcn.py`. The chains are written to `results/<method>_Setting<s>_Ex<e><tag>/`, and the wall-clock time of the batch to `results/<method>_Setting<s>_Ex<e><tag>_batch.json`.
- `postprocess.py <run folder> --data <data file>` computes R-hat, bulk and tail ESS, MCSE, cost-normalised ESS, the posterior mean and std fields, RE, SSIM and the junction depth. It writes `<run folder>_postprocess/`.
- The default settings of the scripts (`config.py`) are for short trial runs, not the paper's: for example, `run_nuts.py` keeps 1000 draws per chain, with target acceptance 0.8 and a diagonal mass matrix, and `run_pcn.py` tunes the pCN step. The commands below give every setting of the paper explicitly.
- The data sets of the paper are in `obs/`. `generate_synthetic_data.py` regenerates them (`--examples`, `--data_mesh`) and overwrites the files in `obs/`.
- The times given below are those of the paper's runs. Run nothing else on the machine during a run if the timings matter.
- In some MPI environments a process exits with a non-zero status (for example 143) from `MPI_Finalize` after it has written its output. `run_chains.py` therefore reads every chain file back (chain index, seed and number of draws) instead of relying on the exit status. Chain files are written to a temporary file first and renamed when complete.

## Example 1: straight junction (Section 6.3.1)

```bash
cd examples
python plot_prior_samples.py --out results/figures/prior_samples.png                               # Figure 3
python run_map.py --example 1 --data obs/obs_Ex1.pickle --tau 16 --n_kl 12 --k 14 --mean_a 2 \
  --out results/map_Setting1_Ex1                                                                    # MAP estimate
python pcn_sweep.py --example 1 --factors 0.5 2 3 4 6 8 12                                          # pCN step-size sweep

# pCN with the step of the sweep, 8 x beta_0 (99 min)
python run_chains.py pcn --example 1 --setting 1 --data obs/obs_Ex1.pickle --seed 0 --tau 16 --n_kl 12 --k 14 --mean_a 2 \
  --n_warmup 5000 --n_burn 30000 --n_samples 500000 --beta 0.43131272512554686
python postprocess.py results/pcn_Setting1_Ex1 --data obs/obs_Ex1.pickle

# NUTS, target acceptance probability 0.95 (36 min)
python run_chains.py nuts --example 1 --setting 1 --data obs/obs_Ex1.pickle --seed 0 --tau 16 --n_kl 12 --k 14 --mean_a 2 \
  --warmup_steps 500 --num_samples 2000 --full_mass True --target_accept_prob 0.95 --max_tree_depth 10
python postprocess.py results/nuts_Setting1_Ex1 --data obs/obs_Ex1.pickle

# NUTS, target 0.99, compared in the text (52 min)
python run_chains.py nuts --example 1 --setting 1 --data obs/obs_Ex1.pickle --seed 0 --tau 16 --n_kl 12 --k 14 --mean_a 2 \
  --warmup_steps 500 --num_samples 2000 --full_mass True --target_accept_prob 0.99 --max_tree_depth 10 --tag _t099
python postprocess.py results/nuts_Setting1_Ex1_t099 --data obs/obs_Ex1.pickle

# Figure 4, and Figure 5 (a, b)
python make_figures.py --example 1 --pcn results/pcn_Setting1_Ex1 --nuts results/nuts_Setting1_Ex1 \
  --map results/map_Setting1_Ex1 --out results/figures/ex1
python make_figures.py --example 1 --pcn results/pcn_Setting1_Ex1 --nuts results/nuts_Setting1_Ex1 \
  --map results/map_Setting1_Ex1 --quantities depth --panel_letters a b --no_legend --out results/figures/ex1_depth
```

## Example 2: curved junction (Section 6.3.2)

```bash
cd examples
python run_map.py --example 2 --data obs/obs_Ex2.pickle --tau 16 --n_kl 12 --k 14 --mean_a 2 \
  --out results/map_Setting1_Ex2                                                                    # MAP estimate
python pcn_sweep.py --example 2 --factors 6 8 12                                                    # pCN step-size sweep

# pCN with the step of the sweep, 8 x beta_0 (378 min)
python run_chains.py pcn --example 2 --setting 1 --data obs/obs_Ex2.pickle --seed 0 --tau 16 --n_kl 12 --k 14 --mean_a 2 \
  --n_warmup 5000 --n_burn 30000 --n_samples 2000000 --beta 0.231701399076914
python postprocess.py results/pcn_Setting1_Ex2 --data obs/obs_Ex2.pickle

# NUTS, target 0.95 (75 min)
python run_chains.py nuts --example 2 --setting 1 --data obs/obs_Ex2.pickle --seed 0 --tau 16 --n_kl 12 --k 14 --mean_a 2 \
  --warmup_steps 500 --num_samples 2000 --full_mass True --target_accept_prob 0.95 --max_tree_depth 10
python postprocess.py results/nuts_Setting1_Ex2 --data obs/obs_Ex2.pickle

# NUTS, target 0.99, compared in the text (88 min)
python run_chains.py nuts --example 2 --setting 1 --data obs/obs_Ex2.pickle --seed 0 --tau 16 --n_kl 12 --k 14 --mean_a 2 \
  --warmup_steps 500 --num_samples 2000 --full_mass True --target_accept_prob 0.99 --max_tree_depth 10 --tag _t099
python postprocess.py results/nuts_Setting1_Ex2_t099 --data obs/obs_Ex2.pickle

# Figure 6, Figure 5 (c, d), and the quantiles of the junction depth quoted in the text
python make_figures.py --example 2 --pcn results/pcn_Setting1_Ex2 --nuts results/nuts_Setting1_Ex2 \
  --map results/map_Setting1_Ex2 --out results/figures/ex2
python make_figures.py --example 2 --pcn results/pcn_Setting1_Ex2 --nuts results/nuts_Setting1_Ex2 \
  --map results/map_Setting1_Ex2 --quantities depth --panel_letters c d --out results/figures/ex2_depth
python compare_samplers.py results/pcn_Setting1_Ex2 results/nuts_Setting1_Ex2 results/nuts_Setting1_Ex2_t099 \
  --example 2 --out results/ex2_depth_quantiles
```

## Joint reconstruction (Section 6.4)

Stage A uses the paper's plateau intervals, Stage B intervals twice as wide with the same centres.

```bash
cd examples
# Stage A (668 min)
python run_chains.py nuts --example 1 --setting 2 --data obs/obs_Ex1.pickle --seed 100 --tau 16 --n_kl 12 --k 14 --mean_a 2 \
  --cp_bounds -2.5 -1.5 --cn_bounds 0.5 1.5 \
  --warmup_steps 500 --num_samples 2000 --full_mass True --target_accept_prob 0.99 --max_tree_depth 10 --tag _A
python postprocess.py results/nuts_Setting2_Ex1_A --data obs/obs_Ex1.pickle

# Stage B (806 min)
python run_chains.py nuts --example 1 --setting 2 --data obs/obs_Ex1.pickle --seed 200 --tau 16 --n_kl 12 --k 14 --mean_a 2 \
  --cp_bounds -3 -1 --cn_bounds 0 2 \
  --warmup_steps 500 --num_samples 2000 --full_mass True --target_accept_prob 0.99 --max_tree_depth 10 --tag _B
python postprocess.py results/nuts_Setting2_Ex1_B --data obs/obs_Ex1.pickle

# Posterior summaries, correlations and data fit of both stages; Figure 7 (a, b): joint_mean_A.png, joint_std_A.png
python joint_report.py --run_a results/nuts_Setting2_Ex1_A --run_b results/nuts_Setting2_Ex1_B \
  --known_plateau_run results/nuts_Setting1_Ex1 --out results/joint
# Figure 7 (c, d)
python make_joint_marginals.py results/nuts_Setting2_Ex1_A results/joint/tables.json results/joint
```

## Tables 1 and 2

After the runs of Examples 1 and 2 and Stage A of the joint reconstruction:

```bash
cd examples
python make_tables.py          # results/tables/table1_ex1.tex, table2_ex2.tex; tables.md compares all runs
```

## Section 6.3.3 and Table 3

```bash
cd sensitivity
python verify_gradients.py              # Taylor tests and central differences in both settings; prints "Gradient checks passed."
python gradient_cost.py                 # time of one adjoint gradient for N_KL = 12 to 192 -> results/scaling_Ex1/
python representation_error.py          # best sigmoid approximation of the Example 1 truth, k = 10, 14, 20, 28 -> results/kstudy_Ex1/

# Table 3: NUTS, four chains of 500 draws (26 to 145 min each). Reference run:
python run_chains.py nuts --example 1 --setting 1 --data obs/obs_Ex1.pickle --seed 0 --tau 16 --n_kl 12 --k 14 --mean_a 2 \
  --warmup_steps 500 --num_samples 500 --full_mass True --target_accept_prob 0.95 --max_tree_depth 10 --tag _ref
python postprocess.py results/nuts_Setting1_Ex1_ref --data obs/obs_Ex1.pickle
```

The other rows of Table 3 use the same command with one change each. Run each command, then `python postprocess.py results/nuts_Setting1_Ex1<tag> --data <data file>`:

| Row | Change to the reference command | Data file for postprocess.py |
|---|---|---|
| Data 96 | `--data obs/obs_Ex1_mesh96.pickle --tag _data96` | `obs/obs_Ex1_mesh96.pickle` |
| Data 384 | `--data obs/obs_Ex1_mesh384.pickle --tag _data384` | `obs/obs_Ex1_mesh384.pickle` |
| N_KL = 24 | `--n_kl 24 --tag _nkl24` | `obs/obs_Ex1.pickle` |
| N_KL = 48 | `--n_kl 48 --tag _nkl48` | `obs/obs_Ex1.pickle` |
| N_KL = 96 | `--n_kl 96 --tag _nkl96` | `obs/obs_Ex1.pickle` |
| k = 10 | `--k 10 --tag _k10` | `obs/obs_Ex1.pickle` |
| k = 20 | `--k 20 --tag _k20` | `obs/obs_Ex1.pickle` |
| k = 28 | `--k 28 --tag _k28` | `obs/obs_Ex1.pickle` |

For example, the row Data 96:

```bash
python run_chains.py nuts --example 1 --setting 1 --data obs/obs_Ex1_mesh96.pickle --seed 0 --tau 16 --n_kl 12 --k 14 --mean_a 2 \
  --warmup_steps 500 --num_samples 500 --full_mass True --target_accept_prob 0.95 --max_tree_depth 10 --tag _data96
python postprocess.py results/nuts_Setting1_Ex1_data96 --data obs/obs_Ex1_mesh96.pickle
```

Then:

```bash
python sensitivity_study.py rows        # results/sensitivity/table3_sensitivity.tex, sensitivity.{json,md}
```

pCN against N_KL, with the step and lengths of Example 1 (about 100 min each):

```bash
python run_chains.py pcn --example 1 --setting 1 --data obs/obs_Ex1.pickle --seed 0 --tau 16 --n_kl 12 --k 14 --mean_a 2 \
  --n_warmup 5000 --n_burn 30000 --n_samples 500000 --beta 0.43131272512554686
python postprocess.py results/pcn_Setting1_Ex1 --data obs/obs_Ex1.pickle
```

Repeat with `--n_kl 24 --tag _nkl24`, `--n_kl 48 --tag _nkl48` and `--n_kl 96 --tag _nkl96`, postprocess each run, then:

```bash
python sensitivity_study.py pcn         # results/sensitivity/pcn_nkl.{json,md}
```

The N_KL = 12 run is the pCN run of Example 1. Instead of running it again, you can use the one in `examples/` with `python sensitivity_study.py pcn --run "N_KL 12=../examples/results/pcn_Setting1_Ex1"`.

## Citation

If you use this code, please cite the accompanying paper.

## License

MIT License, see `LICENSE`.
