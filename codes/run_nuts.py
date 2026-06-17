# run_nuts.py
import os
import pickle
import numpy as np
import torch
import dolfin as dl

import pyro
from pyro.infer.mcmc import MCMC, NUTS
import torch.nn.functional as F

from matern_torch import TorchMaternCov
from pde_model import TopFluxPDE
from fenics_torch_bridge import FenicsTopFluxMisfit


# --------------------------------------------------
# Global settings
# --------------------------------------------------
torch.set_default_dtype(torch.float64)
dl.set_log_level(50)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def bounded(raw, low, high):
    """
    map unconstrained raw variable to [low, high].
    """
    return low + (high - low) * torch.sigmoid(raw)


def neg_log_uniform_pushforward(raw):
    """
    Negative log-Jacobian term for a uniform prior on the bounded
    physical variable c = low + (high-low) sigmoid(raw).

    Up to additive constants:

        -log(sigmoid(raw) * (1 - sigmoid(raw)))
        = softplus(raw) + softplus(-raw)
    """
    return F.softplus(raw) + F.softplus(-raw)


def make_output_dir(out_dir):
    os.makedirs(out_dir, exist_ok=True)


def get_observation_subset(pde, y_obs_full, stride):
    """
    Set full or sparse top-flux observations in the PDE object.
    """
    y_obs_full = np.asarray(y_obs_full, dtype=float)
    m = y_obs_full.size

    if stride is None or int(stride) <= 1:
        obs_idx = np.arange(m, dtype=int)
        y_obs_sub = y_obs_full.copy()
        pde.set_obs(y_obs_sub, obs_idx=None)
        stride = 1
    else:
        stride = int(stride)
        obs_idx = np.arange(0, m, stride, dtype=int)
        y_obs_sub = y_obs_full[obs_idx]
        pde.set_obs(y_obs_sub, obs_idx=obs_idx)

    return obs_idx, y_obs_sub, stride


def build_pde_from_obs(obs):
    """
    Build TopFluxPDE from observation dictionary.
    """
    pde = TopFluxPDE(
        N_points=int(obs["N_x"]),
        Lx=float(obs["Lx"]),
        lam=float(obs["lambda_val"]),
        delta=float(obs["delta_val"]),
        V_top=float(obs["V_top"]),
        V_bottom=float(obs["V_bottom"]),
    )

    return pde


def build_prior_from_obs(
    obs,
    pde,
    corr_len=0.15,
    nu=3.0,
):
    """
    Build TorchMaternCov on the same mesh as the PDE.
    """
    prior = TorchMaternCov(
        pde.mesh,
        push_forward_method="sigmoid",
        C_p = float(obs["C_p"]),
        C_n = float(obs["C_n"]),
        k=float(obs["k"]),
        device="cpu",
    )

    prior.compute_eigen_decomp(
        int(obs["N_KL"]),
        corr_len=corr_len,
        nu=nu,
    )

    return prior


def get_plateau_bounds(obs, CP = None, CN = None):
    """
    Return bounds for cp and cn.

    CP corresponds to the p-type / negative plateau.
    CN corresponds to the n-type / positive plateau.
    """
    if CP is None:
        CP = obs.get(
            "cp_bounds",
            (float(obs["C_p"]) - 0.5, float(obs["C_p"]) + 0.5),
        )

    if CN is None:
        CN = obs.get(
            "cn_bounds",
            (float(obs["C_n"]) - 0.5, float(obs["C_n"]) + 0.5),
        )

    CP = tuple(map(float, CP))
    CN = tuple(map(float, CN))

    return CP, CN


def make_initial_params(
    N_KL,
    infer_plateaus=True,
    p_init_scale=1e-2,
    cp_raw0=0.0,
    cn_raw0=0.0,
):
    """
    Initial parameters for Pyro NUTS.
    """
    init = {
        "p": p_init_scale * torch.randn(N_KL, dtype=torch.float64),
    }

    if infer_plateaus:
        init["cp_raw"] = torch.tensor(cp_raw0, dtype=torch.float64)
        init["cn_raw"] = torch.tensor(cn_raw0, dtype=torch.float64)

    return init


def make_potential_fn(
    pde,
    prior,
    sigma2,
    infer_plateaus=True,
    CP=None,
    CN=None,
    cp_fixed=None,
    cn_fixed=None,
):
    """
    Create negative log-posterior potential function for Pyro NUTS.

    Case 1: infer_plateaus=False

        params = {"p": p}

        U(p) =
            J_data(C(p), cp_fixed, cn_fixed)
            + 0.5 ||p||^2

    Case 2: infer_plateaus=True

        params = {"p": p, "cp_raw": cp_raw, "cn_raw": cn_raw}

        cp = bounded(cp_raw, CP[0], CP[1])
        cn = bounded(cn_raw, CN[0], CN[1])

        U(p, cp_raw, cn_raw) =
            J_data(C(p,cp_raw,cn_raw), cp, cn)
            + 0.5 ||p||^2
            + neg_log_uniform_pushforward(cp_raw)
            + neg_log_uniform_pushforward(cn_raw)
    """
    if infer_plateaus:
        if CP is None or CN is None:
            raise ValueError("CP and CN must be provided when infer_plateaus=True.")

        CP = tuple(map(float, CP))
        CN = tuple(map(float, CN))

    else:
        if cp_fixed is None or cn_fixed is None:
            raise ValueError(
                "cp_fixed and cn_fixed must be provided when infer_plateaus=False."
            )

        cp_fixed = float(cp_fixed)
        cn_fixed = float(cn_fixed)

    def potential_fn(params):
        p = params["p"]

        if infer_plateaus:
            cp = bounded(params["cp_raw"], CP[0], CP[1])
            cn = bounded(params["cn_raw"], CN[0], CN[1])

            J_prior_plateaus = (
                neg_log_uniform_pushforward(params["cp_raw"])
                + neg_log_uniform_pushforward(params["cn_raw"])
            )

        else:
            cp = torch.tensor(cp_fixed, dtype=torch.float64)
            cn = torch.tensor(cn_fixed, dtype=torch.float64)

            J_prior_plateaus = torch.tensor(0.0, dtype=torch.float64)

        # Update prior plateau levels.
        prior.C_p = cp.to(prior.device)
        prior.C_n = cn.to(prior.device)

        # Interior doping field.
        C = prior.assemble_torch(p)

        # PDE misfit, including parameter-dependent BCs.
        J_data = FenicsTopFluxMisfit.apply(
            pde,
            C,
            sigma2,
            cp,
            cn,
        )

        # Gaussian prior on KL coefficients.
        J_prior_p = 0.5 * (p @ p)

        return J_data + J_prior_p + J_prior_plateaus

    return potential_fn


def get_derived_plateau_samples(
    samples,
    infer_plateaus=True,
    CP=None,
    CN=None,
    cp_fixed=None,
    cn_fixed=None,
):
    """
    Return derived physical plateau samples.

    For fixed plateau case, returns constant arrays with length S.
    """
    S = samples["p"].shape[0]

    if infer_plateaus:
        cp_s = bounded(samples["cp_raw"], CP[0], CP[1])
        cn_s = bounded(samples["cn_raw"], CN[0], CN[1])
    else:
        cp_s = torch.full(
            (S,),
            float(cp_fixed),
            dtype=torch.float64,
        )
        cn_s = torch.full(
            (S,),
            float(cn_fixed),
            dtype=torch.float64,
        )

    return cp_s, cn_s


def save_nuts_results(
    out_dir,
    out_name,
    samples,
    derived,
    config,
):
    """
    Save NUTS samples and metadata.
    """
    make_output_dir(out_dir)

    out_path = os.path.join(out_dir, out_name)

    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "samples": samples,
                "derived": derived,
                "config": config,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("\nSaved:", out_path)

    return out_path


# --------------------------------------------------
# Main NUTS routine
# --------------------------------------------------
def run_nuts(
    data_file="./obs/obs_synth_top.pickle",
    out_dir="./stat",
    out_name=None,
    infer_plateaus=True,
    num_samples=200,
    warmup_steps=200,
    num_chains=1,
    corr_len=0.15,
    nu=3.0,
    stride=1,
    seed=0,
    CP=None,
    CN=None,
    cp_fixed=None,
    cn_fixed=None,
    p_init_scale=1e-2,
    cp_raw0=0.0,
    cn_raw0=0.0,
    target_accept_prob=0.8,
    max_tree_depth=10,
    full_mass=False,
    jit_compile=False,
):
    """
    Run NUTS for the top-flux inverse problem.

    Parameters
    ----------
    infer_plateaus : bool
        If True:
            sample p, cp_raw, cn_raw.
        If False:
            sample only p, with fixed cp, cn.

    CP, CN : tuple or list
        Bounds for cp and cn when infer_plateaus=True.

    cp_fixed, cn_fixed : float
        Fixed values for cp and cn when infer_plateaus=False.
        If None, obs["C_p"] and obs["C_n"] are used.

    full_mass : bool
    """
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    make_output_dir(out_dir)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    with open(data_file, "rb") as f:
        obs = pickle.load(f)

    N_KL = int(obs["N_KL"])
    y_obs_full = np.asarray(obs["y_obs"], dtype=float)

    sigma = float(obs["sigma"])
    sigma2 = sigma * sigma

    print("\nNoise:")
    print("  sigma  =", sigma)
    print("  sigma2 =", sigma2)

    # --------------------------------------------------
    # PDE and observations
    # --------------------------------------------------
    pde = build_pde_from_obs(obs)

    obs_idx, y_obs_sub, stride = get_observation_subset(
        pde=pde,
        y_obs_full=y_obs_full,
        stride=stride,
    )

    print("\nObservation setup:")
    print("  full data size =", y_obs_full.size)
    print("  used data size =", y_obs_sub.size)
    print("  stride =", stride)

    # --------------------------------------------------
    # Prior
    # --------------------------------------------------
    prior = build_prior_from_obs(
        obs=obs,
        pde=pde,
        corr_len=corr_len,
        nu=nu,
    )

    # --------------------------------------------------
    # Plateau values
    # --------------------------------------------------
    CP, CN = get_plateau_bounds(obs, CP=CP, CN=CN)

    cp_true = float(obs["C_p"])
    cn_true = float(obs["C_n"])
    
    if infer_plateaus:
        print("\nNUTS mode: unknown cp and cn")
        print("  cp bounds =", CP)
        print("  cn bounds =", CN)
        print("  true/default cp =", cp_true)
        print("  true/default cn =", cn_true)
    
    else:
        if cp_fixed is None:
            cp_fixed = cp_true
    
        if cn_fixed is None:
            cn_fixed = cn_true
    
        print("\nNUTS mode: known cp and cn")
        print("  fixed cp =", cp_fixed)
        print("  fixed cn =", cn_fixed)

    # --------------------------------------------------
    # Potential function
    # --------------------------------------------------
    potential_fn = make_potential_fn(
        pde=pde,
        prior=prior,
        sigma2=sigma2,
        infer_plateaus=infer_plateaus,
        CP=CP,
        CN=CN,
        cp_fixed=cp_fixed,
        cn_fixed=cn_fixed,
    )

    # --------------------------------------------------
    # Initial parameters
    # --------------------------------------------------
    init = make_initial_params(
        N_KL=N_KL,
        infer_plateaus=infer_plateaus,
        p_init_scale=p_init_scale,
        cp_raw0=cp_raw0,
        cn_raw0=cn_raw0,
    )

    print("\nInitial parameters:")
    print("  p_init_scale =", p_init_scale)
    print("  p shape =", tuple(init["p"].shape))

    if infer_plateaus:
        print("  cp_raw0 =", float(init["cp_raw"]))
        print("  cn_raw0 =", float(init["cn_raw"]))
        print("  initial cp =", float(bounded(init["cp_raw"], CP[0], CP[1])))
        print("  initial cn =", float(bounded(init["cn_raw"], CN[0], CN[1])))

    # --------------------------------------------------
    # NUTS kernel
    # --------------------------------------------------
    nuts = NUTS(
        potential_fn=potential_fn,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        full_mass=full_mass,
        jit_compile=jit_compile,
    )

    mcmc = MCMC(
        nuts,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
        initial_params=init,
    )

    # --------------------------------------------------
    # Run NUTS
    # --------------------------------------------------
    mcmc.run()

    print("\nNUTS finished.")

    try:
        print("Final step size:", float(mcmc.kernel.step_size))
    except Exception:
        pass

    try:
        mcmc.summary()
    except Exception:
        pass

    samples = mcmc.get_samples()

    # --------------------------------------------------
    # Derived physical plateau samples
    # --------------------------------------------------
    cp_s, cn_s = get_derived_plateau_samples(
        samples=samples,
        infer_plateaus=infer_plateaus,
        CP=CP,
        CN=CN,
        cp_fixed=cp_fixed,
        cn_fixed=cn_fixed,
    )

    derived = {
        "cp": cp_s,
        "cn": cn_s,
    }

    # --------------------------------------------------
    # Output name
    # --------------------------------------------------
    if out_name is None:
        if infer_plateaus:
            out_name = "nuts_unknown_cp_cn.pickle"
        else:
            out_name = "nuts_known_cp_cn.pickle"

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    config = {
        "data_file": data_file,
        "infer_plateaus": infer_plateaus,
        "corr_len": corr_len,
        "nu": nu,
        "stride": stride,
        "sigma": sigma,
        "sigma2": sigma2,
        "num_samples": num_samples,
        "warmup_steps": warmup_steps,
        "num_chains": num_chains,
        "seed": seed,
        "N_KL": N_KL,
        "k_sigmoid": float(obs["k"]),
        "CP": CP,
        "CN": CN,
        "target_accept_prob": target_accept_prob,
        "max_tree_depth": max_tree_depth,
        "full_mass": full_mass,
        "p_init_scale": p_init_scale,
        "cp_raw0": cp_raw0 if infer_plateaus else None,
        "cn_raw0": cn_raw0 if infer_plateaus else None,
        "cp_true": cp_true,
        "cn_true": cn_true,
        "cp_fixed": cp_fixed if not infer_plateaus else None,
        "cn_fixed": cn_fixed if not infer_plateaus else None,
    }

    save_nuts_results(
        out_dir=out_dir,
        out_name=out_name,
        samples=samples,
        derived=derived,
        config=config,
    )

    return samples, derived, config


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    
    Example = 2
    Setting = 1
    
    if Setting == 1:
        infer_plateaus=False
        cp_fixed=-2.0
        cn_fixed=1.0
        CP = None
        CN = None
    elif Setting == 2:
        infer_plateaus=True
        cp_fixed = None
        cn_fixed = None
        CP = (-2.5, -1.5)
        CN = (0.5, 1.5)
    else:
        raise ValueError("Setting value must be 1 or 2")
        
    # --------------------------------------------------
    # Setting 1: known cP and cN
    # Setting 2: unknown cP and cN
    # --------------------------------------------------
    run_nuts(
        data_file=f"./obs/obs_synth_top_Ex{Example}.pickle",
        out_dir=f"./stat_Setting{Setting}",
        out_name=f"nuts_Setting{Setting}_Ex{Example}.pickle",
        infer_plateaus=infer_plateaus,
        num_samples = 1000,
        warmup_steps =500,
        num_chains=1,
        corr_len=0.15,
        nu=3.0,
        stride=3,
        seed=0,
        cp_fixed = cp_fixed,
        cn_fixed = cn_fixed,
        CP = CP,
        CN = CN,
        p_init_scale=1e-2,
        target_accept_prob=0.8,
        max_tree_depth=10,
        full_mass=False,
    )


   