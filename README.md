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

## Main files

- `generate_synthetic_data.py`: generate synthetic pn-junction data.
- `pde_model.py`: nonlinear Poisson--Boltzmann-type PDE model and top-boundary flux observations.
- `matern_torch.py`: Matérn-type KL prior and sigmoid/heaviside pushforward.
- `fenics_torch_bridge.py`: PyTorch/FEniCS bridge for adjoint-based gradients.
- `run_map.py`: MAP estimation.
- `run_pcn.py`: pCN and Gibbs-pCN sampling.
- `run_nuts.py`: NUTS sampling using Pyro.
- `postprocess_pcn.py`: pCN post-processing.
- `postprocess_nuts.py`: NUTS post-processing.
- `verify_gradients.py`: finite-difference checks for adjoint and PyTorch gradients.


## Installation

The code was developed for the legacy FEniCS/dolfin-adjoint stack. A working environment should include:

- Python 3.11
- FEniCS 2019.1.0
- dolfin-adjoint
- PyTorch
- Pyro
- NumPy
- SciPy
- Matplotlib
- scikit-image
- progressbar2

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
