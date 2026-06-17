# run_pcn.py

import os
import pickle
import numpy as np
import dolfin as dl
import torch
from matern_torch import TorchMaternCov
from pde_model import TopFluxPDE
from fenics_torch_bridge import FenicsTopFluxMisfit
from sampler import Gibbs, pCN


# --------------------------------------------------
# Global settings
# --------------------------------------------------
dl.set_log_level(50)
torch.set_default_dtype(torch.float64)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def bounded(raw, low, high):
    """
    Map unconstrained raw scalar to bounded interval [low, high].
    """
    return low + (high - low) * torch.sigmoid(raw)


def neg_log_uniform_pushforward(raw):
    """
    Negative log-density induced on the raw variable when the
    bounded physical parameter has a uniform prior on its interval.

    If

        c = low + (high-low) sigmoid(raw),

    then, up to an additive constant,

        -log pi(raw) = softplus(raw) + softplus(-raw).
    """
    return torch.nn.functional.softplus(raw) + torch.nn.functional.softplus(-raw)


def make_output_dir(out_dir):
    os.makedirs(out_dir, exist_ok=True)



def get_plateau_bounds(obs, CP=None, CN=None):
    """
    Return bounds for cp and cn.

    CP: bounds for p-type / negative plateau.
    CN: bounds for n-type / positive plateau.
    """
    cp_true = float(obs["C_p"])
    cn_true = float(obs["C_n"])
    
    if CP is None:
        CP = obs.get("cp_bounds", (cp_true - 0.5, cp_true + 0.5))

    if CN is None:
        CN = obs.get("cn_bounds", (cn_true - 0.5, cn_true + 0.5))

    CP = tuple(map(float, CP))
    CN = tuple(map(float, CN))

    return CP, CN


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


# --------------------------------------------------
# Log posterior for pCN/Gibbs sampler
# --------------------------------------------------
class LogPostTopFlux:
    """
    Log-posterior up to an additive constant for the mixed Gibbs sampler.

    - The KL coefficients p are updated by pCN.
      Therefore the N(0,I) prior on p is NOT included here.
      pCN preserves the Gaussian prior by construction.

    - If infer_plateaus=True, the plateau variables are updated by
      random-walk MH. Therefore their prior contribution MUST be included.

    - If infer_plateaus=False, cp and cn are fixed and only p is sampled.
    """

    def __init__(
        self,
        pde,
        prior,
        sigma2,
        infer_plateaus=True,
        CP=None,
        CN=None,
        cp_fixed=None,
        cn_fixed=None,
        add_raw_gaussian_prior=False,
        raw_sd=2.0,
    ):
        self.pde = pde
        self.prior = prior
        self.sigma2 = float(sigma2)

        self.infer_plateaus = bool(infer_plateaus)

        self.add_raw_gaussian_prior = bool(add_raw_gaussian_prior)
        self.raw_sd = float(raw_sd)

        if self.infer_plateaus:
            if CP is None or CN is None:
                raise ValueError("CP and CN must be provided when infer_plateaus=True.")

            self.CP = tuple(map(float, CP))
            self.CN = tuple(map(float, CN))

            self.cp_fixed = None
            self.cn_fixed = None

        else:
            if cp_fixed is None or cn_fixed is None:
                raise ValueError(
                    "cp_fixed and cn_fixed must be provided when infer_plateaus=False."
                )

            self.CP = None
            self.CN = None

            self.cp_fixed = float(cp_fixed)
            self.cn_fixed = float(cn_fixed)

    def __call__(self, p_np, h_np=None):
        """
        Parameters
        ----------
        p_np : ndarray, shape (N_KL,)
            KL coefficients.

        h_np : ndarray, shape (2,), optional
            Raw plateau variables [cp_raw, cn_raw].
            Used only when infer_plateaus=True.

        Returns
        -------
        float
            Log-posterior up to a constant, excluding the Gaussian prior on p.
        """
        p_t = torch.tensor(p_np, dtype=torch.float64)

        # --------------------------------------------------
        # Plateau values
        # --------------------------------------------------
        if self.infer_plateaus:
            if h_np is None:
                raise ValueError("h_np must be provided when infer_plateaus=True.")

            cp_raw = torch.tensor(h_np[0], dtype=torch.float64)
            cn_raw = torch.tensor(h_np[1], dtype=torch.float64)

            cp = bounded(cp_raw, self.CP[0], self.CP[1])
            cn = bounded(cn_raw, self.CN[0], self.CN[1])

            # Uniform prior on physical cp, cn induces this prior on raw variables.
            log_prior_plateaus = -float(
                neg_log_uniform_pushforward(cp_raw)
                + neg_log_uniform_pushforward(cn_raw)
            )

            if self.add_raw_gaussian_prior:
                s2 = self.raw_sd * self.raw_sd
                log_prior_plateaus += -0.5 * float(cp_raw**2 + cn_raw**2) / s2

        else:
            cp = torch.tensor(self.cp_fixed, dtype=torch.float64)
            cn = torch.tensor(self.cn_fixed, dtype=torch.float64)

            log_prior_plateaus = 0.0

        # --------------------------------------------------
        # Assemble field
        # --------------------------------------------------      
        self.prior.C_p = cp
        self.prior.C_n = cn

        C = self.prior.assemble_torch(p_t)

        # --------------------------------------------------
        # Data misfit
        # --------------------------------------------------
        J_data = FenicsTopFluxMisfit.apply(
            self.pde,
            C,
            self.sigma2,
            cp,
            cn,
        )

        log_like = -float(J_data)

        return log_like + log_prior_plateaus


# --------------------------------------------------
# Derived physical plateau samples
# --------------------------------------------------
def get_derived_plateau_samples(
    samples,
    infer_plateaus=True,
    CP=None,
    CN=None,
    cp_fixed=None,
    cn_fixed=None,
):
    """
    Return physical cp and cn samples.
    """
    S = samples["p"].shape[0]

    if infer_plateaus:
        cp_raw = torch.tensor(samples["cp_raw"], dtype=torch.float64)
        cn_raw = torch.tensor(samples["cn_raw"], dtype=torch.float64)

        cp_s = bounded(cp_raw, CP[0], CP[1]).detach().cpu().numpy()
        cn_s = bounded(cn_raw, CN[0], CN[1]).detach().cpu().numpy()

    else:
        cp_s = np.full(S, float(cp_fixed))
        cn_s = np.full(S, float(cn_fixed))

    return cp_s, cn_s


def save_pcn_results(
    out_dir,
    out_name,
    samples,
    derived,
    config,
):
    """
    Save pCN/Gibbs samples and metadata.
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
# Main pCN/Gibbs routine
# --------------------------------------------------
def run_pcn(
    data_file="./obs/obs_synth_top.pickle",
    out_dir="./stat",
    out_name=None,
    infer_plateaus=True,
    warmup=5000,
    skip_len=100,
    nsamples=20000,
    burn=10000,
    corr_len=0.15,
    nu=3.0,
    stride=1,
    seed=0,
    CP=None,
    CN=None,
    cp_fixed=None,
    cn_fixed=None,
    pcn_scale_init=0.05,
    mh_scale_init=0.2,
    add_raw_gaussian_prior=False,
    raw_sd=2.0,
):
    """
    Run mixed pCN/Gibbs sampler.

    Setting 1:
        infer_plateaus=False
        sample only p with fixed cp, cn.

    Setting 2:
        infer_plateaus=True
        sample p by pCN and h=(cp_raw, cn_raw) by random-walk MH.

    Parameters
    ----------
    beta : float
        pCN scale for KL coefficients.

    mh_scale : float
        Random-walk MH scale for raw plateau variables.
        Used only when infer_plateaus=True.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    make_output_dir(out_dir)

    # --------------------------------------------------
    # Load observation file
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
    # Plateau setup
    # --------------------------------------------------
    CP, CN = get_plateau_bounds(obs, CP=CP, CN=CN)

    if cp_fixed is None:
        cp_fixed = float(obs["C_p"])

    if cn_fixed is None:
        cn_fixed = float(obs["C_n"])

    if infer_plateaus:
        print("\npCN mode: unknown cp and cn")
        print("  cp bounds =", CP)
        print("  cn bounds =", CN)
    else:
        print("\npCN mode: known cp and cn")
        print("  fixed cp =", cp_fixed)
        print("  fixed cn =", cn_fixed)

    # --------------------------------------------------
    # Log-posterior
    # --------------------------------------------------
    logpost = LogPostTopFlux(
        pde=pde,
        prior=prior,
        sigma2=sigma2,
        infer_plateaus=infer_plateaus,
        CP=CP,
        CN=CN,
        cp_fixed=cp_fixed,
        cn_fixed=cn_fixed,
        add_raw_gaussian_prior=add_raw_gaussian_prior,
        raw_sd=raw_sd,
    )

    # --------------------------------------------------
    # Initial state
    # --------------------------------------------------
    p0 = np.zeros(N_KL)
    
    print("\nInitial sampler state:")
    print("  p0 shape =", p0.shape)
    print("  initial pCN scale =", pcn_scale_init)
    
    if infer_plateaus:
        h0 = np.array([0.0, 0.0], dtype=float)  # [cp_raw, cn_raw]
    
        print("  h0 =", h0)
        print("  initial MH scale =", mh_scale_init)
        print("  initial cp =", float(bounded(torch.tensor(h0[0]), CP[0], CP[1])))
        print("  initial cn =", float(bounded(torch.tensor(h0[1]), CN[0], CN[1])))

   # --------------------------------------------------
    # Run sampler
    # --------------------------------------------------
    if infer_plateaus:
        # --------------------------------------------------
        # Setting 2:
        # Gibbs sampler:
        #   p-block       : pCN
        #   plateau block : random-walk MH for h = [cp_raw, cn_raw]
        # --------------------------------------------------
        gibbs = Gibbs(
            inits=[p0, h0],
            target=logpost,
            scales=[pcn_scale_init, mh_scale_init],
        )
    
        gibbs.warm_up(
            N_outer=warmup // skip_len,
            N_inner=skip_len,
        )
    
        pcn_scale_adapted = gibbs.samplers[0].scale
        mh_scale_adapted = gibbs.samplers[1].scale
    
        print("Adapted pCN scale =", pcn_scale_adapted)
        print("Adapted MH scale  =", mh_scale_adapted)
    
        gibbs.sample(
            N_outer=nsamples,
            N_inner=1,
        )
    
        p_samps, h_samps = gibbs.get_samples()
    
        p_samps = p_samps[burn:]
        h_samps = h_samps[burn:]
    
        samples = {
            "p": p_samps,
            "cp_raw": h_samps[:, 0],
            "cn_raw": h_samps[:, 1],
        }

    else:
        # --------------------------------------------------
        # Setting 1:
        # Pure pCN sampler for p only.
        # cp and cn are fixed.
        # --------------------------------------------------
        target_p = lambda p: logpost(p, None)
    
        pcn_sampler = pCN(x0=p0, target=target_p, scale = pcn_scale_init)
    
        # Warm-up and then remove warm-up samples.
        pcn_sampler.warm_up(N=warmup, skip_len=skip_len)
        
        pcn_scale_adapted = pcn_sampler.scale
        mh_scale_adapted = None
        
        print("Adapted pCN scale =", pcn_scale_adapted)
        
        # reset storage after adaptation/warmup
        pcn_sampler.samples = [pcn_sampler.current_sample.copy()]
        pcn_sampler.acc = []
        
    
        # Draw posterior samples.
        pcn_sampler.sample(N=nsamples)
    
        p_samps = pcn_sampler.get_samples()
        p_samps = p_samps[burn:]
    
        samples = {
            "p": p_samps,
        }
        
        
        
    print("\nSampling finished:")
    print("  retained p samples =", samples["p"].shape)

    if infer_plateaus:
        print("  retained cp_raw samples =", samples["cp_raw"].shape)
        print("  retained cn_raw samples =", samples["cn_raw"].shape)

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
            out_name = "pcn_unknown_cp_cn.pickle"
        else:
            out_name = "pcn_known_cp_cn.pickle"

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
        "N_KL": N_KL,
        "k_sigmoid": float(obs["k"]),
        "seed": seed,
        "warmup": warmup,
        "skip_len": skip_len,
        "nsamples": nsamples,
        "burn": burn,
    
        # pCN adaptation
        "pcn_scale_init": pcn_scale_init,
        "pcn_scale_adapted": pcn_scale_adapted,
    
        # MH adaptation for plateau block
        "mh_scale_init": mh_scale_init if infer_plateaus else None,
        "mh_scale_adapted": mh_scale_adapted if infer_plateaus else None,
    
        # plateau setup
        "CP": CP,
        "CN": CN,
        "cp_fixed": cp_fixed,
        "cn_fixed": cn_fixed,
    
        # optional extra prior on raw plateau variables
        "add_raw_gaussian_prior": add_raw_gaussian_prior,
        "raw_sd": raw_sd if add_raw_gaussian_prior else None,
    }

    save_pcn_results(
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

    # --------------------------------------------------
    # Setting 1: known cp and cn
    # Setting 2: unknown cp and cn
    # --------------------------------------------------
    if Setting == 1:
        infer_plateaus = False

        cp_fixed = -2.0
        cn_fixed = 1.0

        CP = None
        CN = None

    elif Setting == 2:
        infer_plateaus = True

        cp_fixed = None
        cn_fixed = None

        CP = (-2.5, -1.5)
        CN = (0.5, 1.5)

    else:
        raise ValueError("Setting value must be 1 or 2.")

    run_pcn(
        data_file=f"./obs/obs_synth_top_Ex{Example}.pickle",
        out_dir=f"./stat_Setting{Setting}",
        out_name=f"pcn_Setting{Setting}_Ex{Example}.pickle",
        infer_plateaus=infer_plateaus,
        warmup=5000,
        skip_len=100,
        nsamples=300000,
        burn=30000,
        corr_len=0.15,
        nu=3.0,
        stride=3,
        seed=0,
        CP=CP,
        CN=CN,
        cp_fixed=cp_fixed,
        cn_fixed=cn_fixed,
        pcn_scale_init=1,
        mh_scale_init=0.2,
        add_raw_gaussian_prior=False,
    )