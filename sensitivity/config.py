# config.py
"""
Experiment settings (Section 6 of the revised manuscript), collected in one place.

Section/equation numbers refer to
"Adjoint-based Bayesian uncertainty quantification for PDE-constrained inverse
problems with application to semiconductor imaging".
"""
import os

# ----------------------------------------------------------------------------
# Forward model (Sections 2-3, 6.1)
# ----------------------------------------------------------------------------
MESH_N = 48            # inversion mesh of the unit square, 49^2 = 2401 CG1 dofs
DATA_MESH_N = 192      # finer mesh for the synthetic data (no inverse crime); multiple of 48
EPS = 1.0              # permittivity epsilon in (2.2)
DELTA = 1.0            # scaling parameter delta in (2.2)
# N_meas = 16 electrodes I_l = [3l/48, (3l+1)/48] on y = 1 (width 1/48, centres 1/96 + l/16)
ELECTRODES = tuple((3 * l / 48, (3 * l + 1) / 48) for l in range(16))

# ----------------------------------------------------------------------------
# Ground truth and synthetic data (Section 6.1)
# ----------------------------------------------------------------------------
CP_TRUE = -2.0
CN_TRUE = 1.0
INTERFACE_WIDTH = 0.035   # tanh width of the true junction
NOISE_LEVEL = 0.01        # rho in sigma = rho * ||y_true||_2 / sqrt(N_meas)
DATA_SEED = 123

# ----------------------------------------------------------------------------
# Prior (Sections 4.1-4.3, 6.3, 6.4)
# ----------------------------------------------------------------------------
TAU = 0.25 ** -2          # C = (tau I - Laplace)^(-alpha), Neumann BCs
ALPHA = 4.0
N_KL = 12                 # 99.9 % of the prior variance
K_SIGMOID = 14.0          # median 10-90 % interface width of prior draws ~ 3.5 mesh cells
# Latent prior mean m(y) = a (2y - 1): c = c_p at the bottom contact and c = c_n at the top
# contact, consistent with the Dirichlet data (2.2), for any plateau values.
PRIOR_MEAN_A = 2.0
CP_BOUNDS = (-2.5, -1.5)  # c_p ~ U(CP_BOUNDS) in the joint setting
CN_BOUNDS = (0.5, 1.5)    # c_n ~ U(CN_BOUNDS) in the joint setting

# ----------------------------------------------------------------------------
# Samplers (Sections 6.3, 6.4). One chain per process; run 4 chains with --chain 0..3.
# ----------------------------------------------------------------------------
# Chains start from independent prior draws (z ~ N(0, I), c_p and c_n uniform),
# so that R-hat compares dispersed initialisations.
INIT_FROM_PRIOR = True
NUTS = dict(
    warmup_steps=500,
    num_samples=1000,
    target_accept_prob=0.8,
    max_tree_depth=10,
    full_mass=False,
    seed=0,
)
PCN = dict(
    n_warmup=5000,        # discarded; beta is tuned (per-step Robbins-Monro) during n_warmup + n_burn
    n_samples=300000,
    n_burn=30000,         # discarded
    beta0=0.1,
    target_acceptance=0.234,
    seed=0,
)
MAP = dict(z0=1.0, maxiter=5000, gtol=1e-5, ftol=1e-12)   # z^(0) = 1 (Section 6.3.1)

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OBS_DIR = os.path.join(HERE, "obs")
RESULTS_DIR = os.path.join(HERE, "results")


def data_file(example, data_mesh_n=DATA_MESH_N, noise_level=NOISE_LEVEL, width=None):
    suffix = "" if data_mesh_n == DATA_MESH_N else f"_mesh{data_mesh_n}"
    if noise_level != NOISE_LEVEL:
        suffix += f"_rho{100 * noise_level:g}"            # e.g. obs_Ex1_rho0.5.pickle for rho = 0.5 %
    if width is not None and width != INTERFACE_WIDTH:
        suffix += f"_w{width:g}"                          # e.g. obs_Ex1_w0.018.pickle for a sharper truth
    return os.path.join(OBS_DIR, f"obs_Ex{example}{suffix}.pickle")


def run_dir(method, setting, example, tag=""):
    """Folder holding the chain files chain0.pickle, chain1.pickle, ... of one experiment."""
    return os.path.join(RESULTS_DIR, f"{method}_Setting{setting}_Ex{example}{tag}")


def chain_file(method, setting, example, chain, tag=""):
    return os.path.join(run_dir(method, setting, example, tag), f"chain{chain}.pickle")


def known_plateaus(setting):
    """Setting 1: plateau values known; Setting 2: inferred jointly (Section 6.2)."""
    if setting not in (1, 2):
        raise ValueError("setting must be 1 (known plateaus) or 2 (joint)")
    return setting == 1
