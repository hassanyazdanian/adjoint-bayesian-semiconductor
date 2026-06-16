# Bayesian reconstruction of pn-junction doping profiles

This repository contains Python code accompanying the paper
**"Adjoint-Based Bayesian Uncertainty Quantification for PDE-Constrained Inverse Problems with Application to Semiconductor Imaging"**.

The code implements a Bayesian PDE-constrained inverse problem for reconstructing approximately piecewise-constant pn-junction doping profiles from boundary flux measurements. The main components are:

- a nonlinear Poisson-Boltzmann-type forward model solved with FEniCS,
- a Whittle-Matérn/KL latent Gaussian prior,
- a differentiable sigmoid pushforward map for near piecewise-constant doping fields,
- a FEniCS/dolfin-adjoint to PyTorch bridge for adjoint gradients,
- MAP estimation and NUTS posterior sampling,
- posterior post-processing and uncertainty visualization.

## Repository layout

```text
src/semiconductor_bayes/
    top_flux_pde.py          # nonlinear PDE model and top-flux observations
    matern_prior.py          # Matérn-type KL prior and sigmoid/Heaviside pushforward
    fenics_torch_bridge.py   # custom PyTorch autograd bridge to dolfin-adjoint
    synthetic_data.py        # synthetic pn-junction data generation
    map_estimation.py        # MAP objective and optimization utilities
    nuts_sampling.py         # NUTS posterior sampling utilities
    postprocess.py           # posterior summaries and plots

scripts/
    generate_synthetic_data.py
    run_map.py
    run_nuts.py
    run_postprocess.py

verification/
    verify_gradients.py      # finite-difference checks for adjoint/PyTorch gradients
```

## Installation

The code was developed for the legacy FEniCS/dolfin-adjoint stack. A working environment should include:

- Python 3.11
- FEniCS 2019.1.0
- dolfin-adjoint 2018.1.0
- PyTorch
- Pyro
- NumPy, SciPy, Matplotlib

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

Because FEniCS and dolfin-adjoint can be platform-dependent, using a conda/mamba environment or container is recommended.

## Quick start

From the repository root:

```bash
python scripts/generate_synthetic_data.py
python scripts/run_map.py
python scripts/run_nuts.py
python scripts/run_postprocess.py
```

Generated data, samples, and figures are written to local output folders such as `obs/`, `stat_*`, and `stat_postprocess_*`. These folders are ignored by Git.

## Gradient verification

Before running long NUTS chains, verify the FEniCS--PyTorch gradient bridge:

```bash
python verification/verify_gradients.py
```

This script compares adjoint and PyTorch gradients against finite-difference approximations.

## Citation

If you use this code, please cite the accompanying paper. A `CITATION.cff` file is included and should be updated with the final publication metadata.

## License

License information is not finalized yet. Add a license before making the repository public.
