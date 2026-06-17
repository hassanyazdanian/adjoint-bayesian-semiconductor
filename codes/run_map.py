# run_map.py

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import dolfin as dl
from scipy.optimize import minimize
import torch.nn.functional as F

from pde_model import TopFluxPDE
from matern_torch import TorchMaternCov
from fenics_torch_bridge import FenicsTopFluxMisfit

from skimage.metrics import structural_similarity as ssim

# --------------------------------------------------
# Global settings
# --------------------------------------------------
dl.set_log_level(50)
torch.set_default_dtype(torch.float64)


# --------------------------------------------------
# Compute SSIM and RE
# --------------------------------------------------
def compute_re_ssim(C_est, C_true):
    re = np.linalg.norm(C_est - C_true) / np.linalg.norm(C_true)
    data_range = float(C_true.max() - C_true.min())
    ssim_val = ssim(C_true, C_est, data_range=data_range)

    return float(re), float(ssim_val)

# --------------------------------------------------
# Helper maps for cP, cN
# --------------------------------------------------
def sigmoid_torch(x):
    return 1.0 / (1.0 + torch.exp(-x))


def bounded_torch(z, a, b):
    """
    Map unconstrained z in R to interval [a,b].
    """
    return a + (b - a) * sigmoid_torch(z)


def logistic_log_jacobian(z):
    """
    log(sigma(z) * (1 - sigma(z))).

    This is used for the change-of-variables term when z parametrizes
    a uniform prior on [a,b] through c = a + (b-a) sigmoid(z).

    Up to additive constants, the negative log-prior contributes

        -log(sigma(z) * (1 - sigma(z))).
    """
    return F.logsigmoid(z) + F.logsigmoid(-z)


# --------------------------------------------------
# MAP objective
# --------------------------------------------------
class MAPObjective:
    """
    MAP objective with two inference modes.

    Case 1: known plateau values

        infer_plateaus = False
        theta = p
        cP and cN are fixed.

        J(theta) =
            J_data(C(p), cP, cN)
            + 0.5 ||p||^2

    Case 2: unknown plateau values

        infer_plateaus = True
        theta = (p, zP, zN)

        cP = bounded_torch(zP, cp_bounds[0], cp_bounds[1])
        cN = bounded_torch(zN, cn_bounds[0], cn_bounds[1])

        J(theta) =
            J_data(C(p,zP,zN), cP, cN)
            + 0.5 ||p||^2
            - log(sigma(zP)(1-sigma(zP)))
            - log(sigma(zN)(1-sigma(zN)))
    """

    def __init__(
        self,
        pde,
        prior,
        sigma2,
        infer_plateaus=True,
        cp_bounds=None,
        cn_bounds=None,
        cP_fixed=None,
        cN_fixed=None,
        obs_idx=None,
        stride=1,
    ):
        self.pde = pde
        self.prior = prior
        self.sigma2 = float(sigma2)
        self.infer_plateaus = bool(infer_plateaus)
        self.obs_idx = obs_idx
        self.stride = int(stride)

        if self.infer_plateaus:
            if cp_bounds is None or cn_bounds is None:
                raise ValueError(
                    "cp_bounds and cn_bounds must be provided when infer_plateaus=True."
                )

            self.cp_bounds = tuple(map(float, cp_bounds))
            self.cn_bounds = tuple(map(float, cn_bounds))

            self.cP_fixed = None
            self.cN_fixed = None

        else:
            if cP_fixed is None or cN_fixed is None:
                raise ValueError(
                    "cP_fixed and cN_fixed must be provided when infer_plateaus=False."
                )

            self.cp_bounds = None
            self.cn_bounds = None

            self.cP_fixed = float(cP_fixed)
            self.cN_fixed = float(cN_fixed)

            # Fixed plateau values in the prior push-forward.
            self.prior.C_p = self.cP_fixed
            self.prior.C_n = self.cN_fixed

        # Cache to avoid repeated evaluations by scipy.
        self._x = None
        self._f = None
        self._g = None
        self._cache = None

        self.it = 0
        self.nevals = 0

    @property
    def dim_p(self):
        return self.prior.dim

    @property
    def dim_theta(self):
        if self.infer_plateaus:
            return self.prior.dim + 2
        return self.prior.dim

    def unpack_theta_torch(self, x):
        """
        Convert scipy vector to torch variables.

        If infer_plateaus=True:
            x = [p_1, ..., p_dim, zP, zN]

        If infer_plateaus=False:
            x = [p_1, ..., p_dim]
        """
        x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)

        p = x_t[: self.dim_p]

        if self.infer_plateaus:
            zP = x_t[self.dim_p]
            zN = x_t[self.dim_p + 1]
        else:
            zP = None
            zN = None

        return x_t, p, zP, zN

    def get_plateaus_torch(self, zP, zN):
        """
        Return cP, cN as torch scalars.
        """
        if self.infer_plateaus:
            cP = bounded_torch(zP, self.cp_bounds[0], self.cp_bounds[1])
            cN = bounded_torch(zN, self.cn_bounds[0], self.cn_bounds[1])
        else:
            cP = torch.tensor(self.cP_fixed, dtype=torch.float64)
            cN = torch.tensor(self.cN_fixed, dtype=torch.float64)

        return cP, cN

    def _eval(self, x):
        x = np.asarray(x, dtype=float)

        if self._x is not None and np.array_equal(x, self._x):
            return

        x_t, p, zP, zN = self.unpack_theta_torch(x)

        # --------------------------------------------------
        # Plateau values
        # --------------------------------------------------
        cP, cN = self.get_plateaus_torch(zP, zN)

        # Update prior plateau levels.
        # This is important because assemble_torch uses prior.C_p/C_n.
        self.prior.C_p = cP
        self.prior.C_n = cN

        # --------------------------------------------------
        # Assemble doping field
        # --------------------------------------------------
        C = self.prior.assemble_torch(p)

        # --------------------------------------------------
        # Data misfit through FEniCS bridge
        # --------------------------------------------------
        J_data = FenicsTopFluxMisfit.apply(
            self.pde,
            C,
            self.sigma2,
            cP,
            cN,
        )

        # --------------------------------------------------
        # Prior terms
        # --------------------------------------------------
        J_prior_p = 0.5 * (p @ p)

        if self.infer_plateaus:
            J_prior_z = (
                -logistic_log_jacobian(zP)
                -logistic_log_jacobian(zN)
            )
        else:
            J_prior_z = torch.tensor(0.0, dtype=torch.float64)

        J = J_data + J_prior_p + J_prior_z

        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------
        J.backward()

        self._x = x.copy()
        self._f = float(J.detach().cpu().numpy())
        self._g = x_t.grad.detach().cpu().numpy().astype(float)
        self.nevals += 1

        self._cache = {
            "p": p.detach().cpu().numpy().copy(),
            "cP": float(cP.detach().cpu().numpy()),
            "cN": float(cN.detach().cpu().numpy()),
            "J_data": float(J_data.detach().cpu().numpy()),
            "J_prior_p": float(J_prior_p.detach().cpu().numpy()),
            "J_prior_z": float(J_prior_z.detach().cpu().numpy()),
        }

        if self.infer_plateaus:
            self._cache["zP"] = float(zP.detach().cpu().numpy())
            self._cache["zN"] = float(zN.detach().cpu().numpy())

    def fun(self, x):
        self._eval(x)
        return self._f

    def jac(self, x):
        self._eval(x)
        return self._g

    def callback(self, xk):
        self._eval(xk)
        self.it += 1

        if self.infer_plateaus:
            print(
                f"[it {self.it:3d}] "
                f"f={self._f:.6e}  "
                f"||g||={np.linalg.norm(self._g):.3e}  "
                f"cP={self._cache['cP']:.4f}  "
                f"cN={self._cache['cN']:.4f}  "
                f"Jdata={self._cache['J_data']:.3e}  "
                f"Jp={self._cache['J_prior_p']:.3e}  "
                f"Jz={self._cache['J_prior_z']:.3e}  "
                f"evals={self.nevals}"
            )
        else:
            print(
                f"[it {self.it:3d}] "
                f"f={self._f:.6e}  "
                f"||g||={np.linalg.norm(self._g):.3e}  "
                f"fixed cP={self._cache['cP']:.4f}  "
                f"fixed cN={self._cache['cN']:.4f}  "
                f"Jdata={self._cache['J_data']:.3e}  "
                f"Jp={self._cache['J_prior_p']:.3e}  "
                f"evals={self.nevals}"
            )


# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def make_output_dir(out_dir):
    os.makedirs(out_dir, exist_ok=True)


def get_observation_subset(pde, y_obs_full, stride):
    """
    Set observation data inside the PDE object and return
    obs_idx, y_obs_sub, stride.
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


def unpack_map_solution(theta_map, prior, obs, infer_plateaus, cp_bounds, cn_bounds):
    """
    Convert optimized scipy vector into p_MAP, zP_MAP, zN_MAP, cP_MAP, cN_MAP.
    """
    p_map = theta_map[: prior.dim]

    if infer_plateaus:
        zP_map = theta_map[prior.dim]
        zN_map = theta_map[prior.dim + 1]

        with torch.no_grad():
            zP_t = torch.tensor(zP_map, dtype=torch.float64)
            zN_t = torch.tensor(zN_map, dtype=torch.float64)

            cP_map = float(
                bounded_torch(zP_t, cp_bounds[0], cp_bounds[1]).cpu().numpy()
            )
            cN_map = float(
                bounded_torch(zN_t, cn_bounds[0], cn_bounds[1]).cpu().numpy()
            )

    else:
        zP_map = None
        zN_map = None

        cP_map = float(obs["C_p"])
        cN_map = float(obs["C_n"])

    return p_map, zP_map, zN_map, cP_map, cN_map


def assemble_map_field(prior, p_map, cP_map, cN_map):
    """
    Assemble MAP field C_MAP from p_MAP and plateau values.
    """
    with torch.no_grad():
        prior.C_p = torch.tensor(cP_map, dtype=torch.float64)
        prior.C_n = torch.tensor(cN_map, dtype=torch.float64)

        p_t = torch.tensor(p_map, dtype=torch.float64)
        C_map = prior.assemble_torch(p_t).detach().cpu().numpy()

    return C_map


def assemble_heaviside_map_field(prior, p_map, cP_map, cN_map):
    """
    Assemble Heaviside version of the MAP field using the same p_MAP.
    """
    pf_old = prior.push_forward_method

    prior.push_forward_method = "heaviside"

    with torch.no_grad():
        prior.C_p = torch.tensor(cP_map, dtype=torch.float64)
        prior.C_n = torch.tensor(cN_map, dtype=torch.float64)

        p_t = torch.tensor(p_map, dtype=torch.float64)
        C_H = prior.assemble_torch(p_t).detach().cpu().numpy()

    prior.push_forward_method = pf_old

    return C_H


def save_map_arrays(out_dir, theta_map, p_map, cP_map, cN_map, infer_plateaus, zP_map, zN_map):
    """
    Save MAP arrays.
    """
    np.save(os.path.join(out_dir, "theta_map.npy"), theta_map)
    np.save(os.path.join(out_dir, "p_map.npy"), p_map)
    np.save(os.path.join(out_dir, "cP_cN_map.npy"), np.array([cP_map, cN_map]))

    if infer_plateaus:
        np.save(os.path.join(out_dir, "zP_map.npy"), np.array([zP_map]))
        np.save(os.path.join(out_dir, "zN_map.npy"), np.array([zN_map]))


def plot_data_vs_prediction(
    out_dir,
    obs_idx,
    y_obs_sub,
    y_obs_full,
    y_map_sub,
    y_map_full,
):
    """
    Plot observed data and MAP prediction.
    """
    m = y_obs_full.size

    plt.figure()
    plt.plot(obs_idx, y_obs_sub, "ko", ms=4, label="Observed subset")
    plt.plot(np.arange(m), y_obs_full, "k--", lw=1, alpha=0.35, label="Full obs")
    plt.plot(obs_idx, y_map_sub, "ro", ms=4, label="MAP pred subset")
    plt.plot(np.arange(m), y_map_full, "r--", lw=1, alpha=0.35, label="MAP pred full")
    plt.xlabel("top facet index")
    plt.ylabel("u_y on top facets")
    plt.title("Data vs MAP prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "map_data_vs_prediction.png"), dpi=200)


def plot_map_fields(
    out_dir,
    pde,
    obs,
    C_map,
    C_H,
    cP_map,
    cN_map,
):
    """
    Plot True C, MAP C, and Heaviside MAP C.
    """
    C_true = obs.get("C_true", None)

    C_true_fun = dl.Function(pde.V)
    if C_true is not None:
        C_true_fun.vector().set_local(np.asarray(C_true, dtype=float))
        C_true_fun.vector().apply("insert")
        true_title = "True C(x)"
    else:
        true_title = "True C(x) not in file"

    C_map_fun = dl.Function(pde.V)
    C_map_fun.vector().set_local(C_map)
    C_map_fun.vector().apply("insert")

    C_H_fun = dl.Function(pde.V)
    C_H_fun.vector().set_local(C_H)
    C_H_fun.vector().apply("insert")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    plt.sca(axes[0])
    im0 = dl.plot(C_true_fun, mode="color")
    axes[0].set_title(true_title)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    plt.sca(axes[1])
    im1 = dl.plot(C_map_fun, mode="color")
    axes[1].set_title(f"MAP C(x)\n(cP={cP_map:.3f}, cN={cN_map:.3f})")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.sca(axes[2])
    im2 = dl.plot(C_H_fun, mode="color")
    axes[2].set_title("Heaviside MAP")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "map_C_3panel.png"), dpi=200)


def plot_thresholded_map(
    out_dir,
    pde,
    C_map,
    cP_map,
    cN_map,
):
    """
    Explicit binary thresholding of MAP C.
    """
    threshold = 0.5 * (cP_map + cN_map)
    C_thresholded = np.where(C_map >= threshold, cN_map, cP_map)

    C_thresholded_fun = dl.Function(pde.V)
    C_thresholded_fun.vector().set_local(C_thresholded)
    C_thresholded_fun.vector().apply("insert")

    plt.figure()
    img = dl.plot(C_thresholded_fun, mode="color")
    plt.colorbar(img)
    plt.title("Thresholded MAP C(x)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "map_C_thresholded.png"), dpi=200)

    return C_thresholded

# ============================================================
# Plot functions
# ============================================================  
def plot_field(
    V,
    vec,
    save_path,
    title=None,
    tick_format_set=None,
    cmap="viridis",
    cbar_label=r"$c(\boldsymbol{x})$",
    contour_zero=False,
    vmin=None,
    vmax=None,
    extra_contours=None,
):
    """
    Plot and save a FEniCS field.
    """
    fun = dl.Function(V)
    fun.vector().set_local(np.asarray(vec, dtype=float))
    fun.vector().apply("insert")

    fig, ax = plt.subplots(figsize=(4.2, 3.75))

    im = dl.plot(
        fun,
        axes=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    mesh = V.mesh()
    coords = mesh.coordinates()
    cells = mesh.cells()

    # --------------------------------------------------------
    # Optional zero contour of the plotted field itself
    # --------------------------------------------------------
    if contour_zero:
        vals = fun.compute_vertex_values(mesh)

        try:
            ax.tricontour(
                coords[:, 0],
                coords[:, 1],
                cells,
                vals,
                levels=[0.0],
                colors="black",
                linewidths=1.5,
            )
        except Exception:
            pass

    # --------------------------------------------------------
    # Additional contours
    # --------------------------------------------------------
    if extra_contours is not None:
        handles_for_legend = []

        for item in extra_contours:
            vec_i = item["vec"]
            level_i = item.get("level", 0.0)
            color_i = item.get("color", "black")
            linestyle_i = item.get("linestyle", "-")
            linewidth_i = item.get("linewidth", 1.5)
            label_i = item.get("label", None)

            fun_i = dl.Function(V)
            fun_i.vector().set_local(np.asarray(vec_i, dtype=float))
            fun_i.vector().apply("insert")

            vals_i = fun_i.compute_vertex_values(mesh)

            try:
                cs = ax.tricontour(
                    coords[:, 0],
                    coords[:, 1],
                    cells,
                    vals_i,
                    levels=[level_i],
                    colors=color_i,
                    linestyles=linestyle_i,
                    linewidths=linewidth_i,
                )

                if label_i is not None:
                    # proxy handle for clean legend
                    handle = plt.Line2D(
                        [0],
                        [0],
                        color=color_i,
                        linestyle=linestyle_i,
                        linewidth=linewidth_i,
                        label=label_i,
                    )
                    handles_for_legend.append(handle)

            except Exception:
                pass

        # if len(handles_for_legend) > 0:
        #     ax.legend(handles=handles_for_legend, fontsize=9, loc="best")

    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$y$", fontsize=16)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])

    if title is not None:
        ax.set_title(title, fontsize=14)

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("black")
        sp.set_linewidth(1.0)

    ax.tick_params(direction="out", length=3, width=1, labelsize=14)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=14)

    if vmin is not None and vmax is not None:
        cb.set_ticks(np.linspace(vmin, vmax, 4))

    cb.ax.tick_params(labelsize=14)

    if tick_format_set is not None:
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)
# --------------------------------------------------
# Main MAP routine
# --------------------------------------------------
def compute_map(
    data_file="./obs/obs_synth_top.pickle",
    stride=1,
    out_dir="./stat",
    infer_plateaus=True,
    cP_fixed=None,
    cN_fixed=None,
    corr_len=0.15,
    nu=3.0,
    maxiter=200,
    gtol=1e-4,
    ftol=1e-6,
    p0_value=1.0,
    zP0=-0.1,
    zN0=0.1,
    show_plots=True,
):
    """
    Compute MAP estimate.

    Parameters
    ----------
    infer_plateaus : bool
        If True:
            infer p, zP, zN.
        If False:
            infer only p, while cP and cN are fixed.

    cP_fixed, cN_fixed : float or None
        Fixed plateau values used when infer_plateaus=False.
        If None, obs["C_p"] and obs["C_n"] are used.

    corr_len, nu : float
        Prior parameters for TorchMaternCov.compute_eigen_decomp.

    p0_value : float
        Initial value for all KL coefficients.

    zP0, zN0 : float
        Initial values for plateau latent variables when infer_plateaus=True.
    """
    make_output_dir(out_dir)

    # --------------------------------------------------
    # Load observation file
    # --------------------------------------------------
    with open(data_file, "rb") as f:
        obs = pickle.load(f)

    # --------------------------------------------------
    # PDE
    # --------------------------------------------------
    pde = TopFluxPDE(
        N_points=obs["N_x"],
        Lx=obs["Lx"],
        V_top=obs["V_top"],
        V_bottom=obs["V_bottom"],
        lam=obs["lambda_val"],
        delta=obs["delta_val"],
    )

    # --------------------------------------------------
    # Observations
    # --------------------------------------------------
    y_obs_full = np.asarray(obs["y_obs"], dtype=float)

    obs_idx, y_obs_sub, stride = get_observation_subset(
        pde=pde,
        y_obs_full=y_obs_full,
        stride=stride,
    )

    sigma2 = float(obs["sigma"] ** 2)

    # --------------------------------------------------
    # Plateau bounds and fixed values
    # --------------------------------------------------
    cp_bounds = obs.get(
        "cp_bounds",
        (float(obs["C_p"]) - 0.5, float(obs["C_p"]) + 0.5),
    )
    cn_bounds = obs.get(
        "cn_bounds",
        (float(obs["C_n"]) - 0.5, float(obs["C_n"]) + 0.5),
    )

    if cP_fixed is None:
        cP_fixed = float(obs["C_p"])

    if cN_fixed is None:
        cN_fixed = float(obs["C_n"])

    # --------------------------------------------------
    # Prior
    # --------------------------------------------------
    prior = TorchMaternCov(
        pde.mesh,
        push_forward_method="sigmoid",
        C_p = float(obs["C_p"]),
        C_n = float(obs["C_n"]),
        k=obs["k"],
        device="cpu",
    )

    prior.compute_eigen_decomp(
        obs["N_KL"],
        corr_len=corr_len,
        nu=nu,
    )

    # --------------------------------------------------
    # Objective
    # --------------------------------------------------
    if infer_plateaus:
        print("\nMAP mode: unknown cP and cN")
        print("Using cP bounds:", cp_bounds)
        print("Using cN bounds:", cn_bounds)

        obj = MAPObjective(
            pde=pde,
            prior=prior,
            sigma2=sigma2,
            infer_plateaus=True,
            cp_bounds=cp_bounds,
            cn_bounds=cn_bounds,
            obs_idx=obs_idx,
            stride=stride,
        )

    else:
        print("\nMAP mode: known cP and cN")
        print("Fixed cP =", cP_fixed)
        print("Fixed cN =", cN_fixed)

        obj = MAPObjective(
            pde=pde,
            prior=prior,
            sigma2=sigma2,
            infer_plateaus=False,
            cP_fixed=cP_fixed,
            cN_fixed=cN_fixed,
            obs_idx=obs_idx,
            stride=stride,
        )

    # --------------------------------------------------
    # Initial guess
    # --------------------------------------------------
    x0 = np.ones(obj.dim_theta)
    x0[: prior.dim] = p0_value * np.ones(prior.dim)

    if infer_plateaus:
        x0[prior.dim] = zP0
        x0[prior.dim + 1] = zN0

    print("\nInitial theta:")
    print("  p0 shape =", x0[: prior.dim].shape)
    print("  p0 value =", p0_value)

    if infer_plateaus:
        print("  zP0 =", x0[prior.dim])
        print("  zN0 =", x0[prior.dim + 1])
    else:
        print("  cP and cN are fixed")

    # --------------------------------------------------
    # Optimize
    # --------------------------------------------------
    res = minimize(
        obj.fun,
        x0,
        jac=obj.jac,
        method="L-BFGS-B",
        callback=obj.callback,
        options={
            "gtol": gtol,
            "maxiter": maxiter,
            "ftol": ftol,
        },
    )

    print("\nOptimization finished.")
    print("  success =", res.success)
    print("  message =", res.message)
    print("  iters   =", obj.it)
    print(f"  f*      = {res.fun:.6e}")
    print(f"  ||g*||  = {np.linalg.norm(obj.jac(res.x)):.3e}")

    # --------------------------------------------------
    # Unpack MAP
    # --------------------------------------------------
    theta_map = res.x.copy()

    p_map, zP_map, zN_map, cP_map, cN_map = unpack_map_solution(
        theta_map=theta_map,
        prior=prior,
        obs=obs,
        infer_plateaus=infer_plateaus,
        cp_bounds=cp_bounds,
        cn_bounds=cn_bounds,
    )

    C_map = assemble_map_field(
        prior=prior,
        p_map=p_map,
        cP_map=cP_map,
        cN_map=cN_map,
    )

    print("\nMAP parameters:")

    if infer_plateaus:
        print("  zP_MAP =", zP_map)
        print("  zN_MAP =", zN_map)
    else:
        print("  zP_MAP = not used")
        print("  zN_MAP = not used")

    print("  cP_MAP =", cP_map)
    print("  cN_MAP =", cN_map)
    print("  ||p_MAP|| =", np.linalg.norm(p_map))

    # --------------------------------------------------
    # Save arrays
    # --------------------------------------------------
    save_map_arrays(
        out_dir=out_dir,
        theta_map=theta_map,
        p_map=p_map,
        cP_map=cP_map,
        cN_map=cN_map,
        infer_plateaus=infer_plateaus,
        zP_map=zP_map,
        zN_map=zN_map,
    )

    np.save(os.path.join(out_dir, "C_map.npy"), C_map)

    # --------------------------------------------------
    # Predictions at MAP
    # --------------------------------------------------
    pde.solve_state(
        C_map,
        cP=cP_map,
        cN=cN_map,
        zero_guess=True,
    )

    y_map_full = pde.top_flux_facet_vec()
    y_map_sub = y_map_full[obs_idx]

    np.save(os.path.join(out_dir, "y_map_full.npy"), y_map_full)
    np.save(os.path.join(out_dir, "y_map_sub.npy"), y_map_sub)
    np.save(os.path.join(out_dir, "obs_idx.npy"), obs_idx)

    # --------------------------------------------------
    # Heaviside MAP field
    # --------------------------------------------------
    C_H = assemble_heaviside_map_field(
        prior=prior,
        p_map=p_map,
        cP_map=cP_map,
        cN_map=cN_map,
    )

    np.save(os.path.join(out_dir, "C_map_heaviside.npy"), C_H)

    # --------------------------------------------------
    # Explicit thresholded MAP field
    # --------------------------------------------------
    threshold = 0.5 * (cP_map + cN_map)
    C_thresholded = np.where(C_map >= threshold, cN_map, cP_map)
    np.save(os.path.join(out_dir, "C_map_thresholded.npy"), C_thresholded)

    # --------------------------------------------------
    # Plots
    # --------------------------------------------------
    plot_data_vs_prediction(
        out_dir=out_dir,
        obs_idx=obs_idx,
        y_obs_sub=y_obs_sub,
        y_obs_full=y_obs_full,
        y_map_sub=y_map_sub,
        y_map_full=y_map_full,
    )

    plot_map_fields(
        out_dir=out_dir,
        pde=pde,
        obs=obs,
        C_map=C_map,
        C_H=C_H,
        cP_map=cP_map,
        cN_map=cN_map,
    )

    plot_thresholded_map(
        out_dir=out_dir,
        pde=pde,
        C_map=C_map,
        cP_map=cP_map,
        cN_map=cN_map,
    )
    
    # --------------------------------------------------------
    # Extra contours for plotting
    # --------------------------------------------------------
    C_true = obs.get("C_true", None)
    
    extra_true_contours = []
    if C_true is not None:
        extra_true_contours.append({
            "vec": C_true,
            "level": 0.0,
            "color": "black",
            "linestyle": "-",
            "linewidth": 2.0,
            "label": "True interface",
        })
    
    plot_field(
        V = pde.V,
        vec=C_map,
        save_path=out_dir,
        cmap="viridis",
        cbar_label=r"$c_{MAP}(\boldsymbol{x})$",
        contour_zero=False,
        extra_contours=extra_true_contours,
        vmin=float(np.min(C_map)),
        vmax=float(np.max(C_map)),
        tick_format_set=True,
    )
    
   
    
    
    
    if show_plots:
        plt.show()
    else:
        plt.close("all")
    
    C_true = obs.get("C_true", None)
    if C_true is not None:
        re_mean, ssim_mean = compute_re_ssim(C_map, C_true)
    
        print("\nPosterior mean vs truth:")
        print(f"  RE   = {re_mean:.6f}")
        print(f"  SSIM = {ssim_mean:.6f}")


    return res, pde, prior


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    
    Example =2
    Setting = 1
    
    if Setting == 1:
        infer_plateaus=False
    elif Setting == 2:
        infer_plateaus=True
    else:
        raise ValueError("Setting value must be 1 or 2")
        
    # --------------------------------------------------
    # Setting 1: known cP and cN
    # Setting 2: unknown cP and cN
    # --------------------------------------------------
    compute_map(
        data_file=f"./obs/obs_synth_top_Ex{Example}.pickle",
        stride=3,
        out_dir=f"./stat_Setting{Setting}_Ex{Example}",
        infer_plateaus=infer_plateaus,
        cP_fixed=-2.0,
        cN_fixed=1.0,
        corr_len=0.15,
        nu=3.0,
        maxiter=200,
    )

    