# postprocess_nuts.py
import os
import pickle
import numpy as np
import torch
import dolfin as dl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde

from matern_torch import TorchMaternCov
from pde_model import TopFluxPDE
from skimage.metrics import structural_similarity as ssim

torch.set_default_dtype(torch.float64)
dl.set_log_level(50)


# ============================================================
# Compute SSIM and RE
# ============================================================

def compute_re_ssim(C_est, C_true):
    re = np.linalg.norm(C_est - C_true) / np.linalg.norm(C_true)
    data_range = float(C_true.max() - C_true.min())
    ssim_val = ssim(C_true, C_est, data_range=data_range)

    return float(re), float(ssim_val)


# ============================================================
# Basic utilities
# ============================================================

def to_numpy(x):
    """
    Convert torch tensor or numpy array to numpy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _autocorr_1d(x, max_lag):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = x.size
    if n < 2:
        return np.ones(max_lag + 1)
    var = np.dot(x, x) / n
    if var <= 0:
        return np.ones(max_lag + 1)

    ac = np.empty(max_lag + 1, float)
    ac[0] = 1.0
    for k in range(1, max_lag + 1):
        ac[k] = np.dot(x[:-k], x[k:]) / (n - k) / var
    return ac

def _iat_from_acf(ac):
    tau = 1.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        tau += 2.0 * ac[k]
    return float(max(tau, 1.0))


def _accept_rate_proxy(samples, tol=0.0):
    X = np.asarray(samples)
    if X.shape[0] < 2:
        return np.nan
    d = np.max(np.abs(X[1:] - X[:-1]), axis=1)
    same = d <= float(tol)
    return float(1.0 - same.mean())

def iat_ess_1d(series, max_lag):
    """
    Compute IAT and ESS using autocorrelation truncation
    at the first non-positive autocorrelation.
    """
    series = np.asarray(series, dtype=float).reshape(-1)
    n = series.size

    if n < 2:
        return 1.0, float(n), np.ones(1)

    max_lag_eff = int(min(int(max_lag), n // 2))

    ac = _autocorr_1d(series, max_lag=max_lag_eff)
    tau = _iat_from_acf(ac)
    ess = n / tau

    return tau, ess, ac

def hpd_interval_1d(samples, cred_mass=0.95):
    """
    Shortest 1D HPD interval.
    """
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)

    if n < 2:
        return np.nan, np.nan

    m = int(np.floor(cred_mass * n))

    if m < 1:
        return np.nan, np.nan

    widths = x[m:] - x[: n - m]

    if len(widths) == 0:
        return float(x[0]), float(x[-1])

    j = int(np.argmin(widths))

    return float(x[j]), float(x[j + m])


def pointwise_hpd_bounds(C_samples, cred_mass=0.95):
    """
    Compute pointwise HPD bounds for field samples.

    """
    C_sorted = np.sort(C_samples, axis=0)

    n_samples = C_sorted.shape[0]
    m = int(np.floor(cred_mass * n_samples))

    if m < 1:
        raise ValueError("Not enough samples for HPD interval.")

    widths = C_sorted[m:, :] - C_sorted[: n_samples - m, :]
    j = np.argmin(widths, axis=0)

    cols = np.arange(C_sorted.shape[1])

    hpd_low = C_sorted[j, cols]
    hpd_high = C_sorted[j + m, cols]

    return hpd_low, hpd_high


def save_fenics_vector(path, vec):
    """
    Save vector as .npy file.
    """
    np.save(path, np.asarray(vec, dtype=float))


def make_function(V, vec):
    """
    Create FEniCS Function from vector.
    """
    fun = dl.Function(V)
    fun.vector().set_local(np.asarray(vec, dtype=float))
    fun.vector().apply("insert")
    return fun


# ============================================================
# Sample extraction
# ============================================================

def extract_samples(stat, obs):
    """
    Extract p, cp, cn samples.
    """
    samples = stat["samples"]
    derived = stat.get("derived", {})

    P = to_numpy(samples["p"])

    if P.ndim == 2:
        S = P
        n_chains = 1
        n_draws, dim = S.shape

    elif P.ndim == 3:
        n_chains, n_draws, dim = P.shape
        S = P.reshape(n_chains * n_draws, dim)

    else:
        raise ValueError(f"Unexpected p sample shape: {P.shape}")

    # --------------------------------------------------------
    # Derived physical plateau samples
    # --------------------------------------------------------
    if "cp" in derived and "cn" in derived:
        cp_samples = to_numpy(derived["cp"]).reshape(-1)
        cn_samples = to_numpy(derived["cn"]).reshape(-1)

    else:
        # fixed plateau case without derived values
        cp_samples = np.full(S.shape[0], float(obs["C_p"]))
        cn_samples = np.full(S.shape[0], float(obs["C_n"]))

    # --------------------------------------------------------
    # Raw plateau variables, if present
    # --------------------------------------------------------
    cp_raw = None
    cn_raw = None

    if "cp_raw" in samples and "cn_raw" in samples:
        cp_raw = to_numpy(samples["cp_raw"]).reshape(-1)
        cn_raw = to_numpy(samples["cn_raw"]).reshape(-1)


    info = {
        "n_chains": n_chains,
        "n_draws_per_chain": n_draws,
        "n_total": S.shape[0],
        "dim": dim,
        "has_unknown_plateaus": cp_raw is not None,
    }

    return S, cp_samples, cn_samples, cp_raw, cn_raw, info


# ============================================================
# PDE and prior
# ============================================================

def build_pde(obs):
    return TopFluxPDE(
        N_points=int(obs["N_x"]),
        Lx=float(obs["Lx"]),
        lam=float(obs["lambda_val"]),
        delta=float(obs["delta_val"]),
        V_top=float(obs["V_top"]),
        V_bottom=float(obs["V_bottom"]),
    )


def build_prior(obs, pde, stat):
    cfg = stat.get("config", {})

    k_sigmoid = float(cfg.get("k_sigmoid", obs["k"]))
    corr_len = float(cfg.get("corr_len", 0.15))
    nu = float(cfg.get("nu", 3.0))

    prior = TorchMaternCov(
        pde.mesh,
        push_forward_method="sigmoid",
        C_p=float(obs["C_p"]),
        C_n=float(obs["C_n"]),
        k=k_sigmoid,
        device="cpu",
    )

    prior.compute_eigen_decomp(
        int(obs["N_KL"]),
        corr_len=corr_len,
        nu=nu,
    )

    return prior, cfg


# ============================================================
# Interface depth
# ============================================================

def estimate_interface_depth_from_array(
    C_vec,
    V,
    cp,
    cn,
    x_probe=0.5,
    y_min=0.0,
    y_max=1.0,
    n_y=400,
):
    """
    Estimate interface depth y_Gamma(x_probe) from one field sample.

    The interface is defined by

        C(x,y) = 0.5 * (cp + cn).
    """
    C_fun = make_function(V, C_vec)

    level = 0.5 * (float(cp) + float(cn))
    ys = np.linspace(y_min, y_max, n_y)

    vals = []

    for y in ys:
        try:
            vals.append(C_fun(dl.Point(float(x_probe), float(y))))
        except RuntimeError:
            vals.append(np.nan)

    vals = np.asarray(vals, dtype=float)
    good = np.isfinite(vals)

    ys = ys[good]
    vals = vals[good]

    if len(vals) < 2:
        return np.nan

    diff = vals - level

    crossing = np.where(diff[:-1] * diff[1:] < 0.0)[0]

    if len(crossing) == 0:
        return np.nan

    j = crossing[0]

    y1, y2 = ys[j], ys[j + 1]
    d1, d2 = diff[j], diff[j + 1]

    return float(y1 - d1 * (y2 - y1) / (d2 - d1))


def estimate_mean_interface_depth_from_array(
    C_vec,
    V,
    cp,
    cn,
    x_list,
    y_min=0.0,
    y_max=1.0,
    n_y=400,
):
    """
    Estimate mean interface depth over several x locations.
    """
    depths = []

    for x_probe in x_list:
        y = estimate_interface_depth_from_array(
            C_vec=C_vec,
            V=V,
            cp=cp,
            cn=cn,
            x_probe=x_probe,
            y_min=y_min,
            y_max=y_max,
            n_y=n_y,
        )

        if np.isfinite(y):
            depths.append(y)

    if len(depths) == 0:
        return np.nan

    return float(np.mean(depths))

# ============================================================
# Trace-plot
# ============================================================
def plot_trace5(
    S,
    save_path,
    idxs=None,
):
    """
    Save one trace plot containing 5 selected KL coefficients.
    """
    import matplotlib.pyplot as plt

    S = np.asarray(S, dtype=float)
    n_samples, dim = S.shape

    if idxs is None:
        idxs = list(range(min(5, dim)))

    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    for j in idxs:
        ax.plot(
            np.arange(n_samples),
            S[:, j],
            linewidth=0.6,
            label=rf"$x_{{{j+1}}}$",
        )

    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_ylabel("KL coefficient value", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, ncol=1, loc = 'lower right')
    ax.tick_params(labelsize=13)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
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
    
    
def plot_plateau_histogram(
    samples,
    save_path,
    xlabel,
    true_value=None,
    bins="auto",
):
    """
    Plot and save histogram + KDE for cp or cn.
    """
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]

    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    ax.hist(
        samples,
        bins=bins,
        density=True,
        edgecolor="black",
        linewidth=0.25,
        color="lightgray",
    )

    if samples.size > 2 and np.std(samples) > 0:
        x = np.linspace(samples.min(), samples.max(), 400)
        kde = gaussian_kde(samples)
        ax.plot(x, kde(x), color="black", linewidth=1.2)

    sample_mean = float(np.mean(samples))
    ax.axvline(
        sample_mean,
        color="green",
        linestyle="-",
        linewidth=2.0,
        label="Mean",
    )

    if true_value is not None:
        ax.axvline(
            true_value,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="True value",
        )

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=13)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    
    
# ============================================================
# Main post-processing
# ============================================================

def postprocess_nuts(
    sample_file="./stat_unknown_cp_cn/nuts_unknown_cp_cn.pickle",
    data_file="./obs/obs_synth_top.pickle",
    outdir="./stat_postprocess",
    cred_mass=0.95,
    x_probe=0.5,
    use_mean_depth=False,
    x_depth_range=(0.2, 0.8),
    nx_depth=7,
    trace_idxs=None,
    max_lag = 200,
    accept_tol = 0.0,
):
    
    os.makedirs(outdir, exist_ok=True)

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------
    with open(data_file, "rb") as f:
        obs = pickle.load(f)

    with open(sample_file, "rb") as f:
        stat = pickle.load(f)

    # --------------------------------------------------------
    # Extract samples
    # --------------------------------------------------------
    S, cp_samples, cn_samples, cp_raw, cn_raw, info = extract_samples(stat, obs)

    total_samples, dim = S.shape

    print("\nLoaded NUTS samples:")
    print(f"  chains          = {info['n_chains']}")
    print(f"  draws per chain = {info['n_draws_per_chain']}")
    print(f"  total samples   = {total_samples}")
    print(f"  KL dimension    = {dim}")
    print(f"  unknown cp/cn   = {info['has_unknown_plateaus']}")
    
    # --------------------------------------------------------
    # Acceptance proxy from repeated full states
    # --------------------------------------------------------
    if cp_raw is not None:
        state_for_acc = np.column_stack([S, cp_raw, cn_raw])
    else:
        state_for_acc = S
    
    acc_proxy = _accept_rate_proxy(state_for_acc, tol=accept_tol)
    
    print(f"\naccept-rate proxy full state (from repeats, tol={accept_tol}): {acc_proxy:.3f}")
    
    
    # --------------------------------------------------------
    # ESS diagnostics for all KL coefficients
    # --------------------------------------------------------
    ess_p = np.array([
        iat_ess_1d(S[:, j], max_lag=max_lag)[1]
        for j in range(dim)
    ])
    
    ess_summary = {
        "p_min": float(np.min(ess_p)),
        "p_median": float(np.median(ess_p)),
        "p_mean": float(np.mean(ess_p)),
        "p_max": float(np.max(ess_p)),
    }
    
    np.save(os.path.join(outdir, "ess_p.npy"), ess_p)
    
    print("\nESS summary for KL coefficients:")
    print(f"  min    = {ess_summary['p_min']:.1f}")
    print(f"  median = {ess_summary['p_median']:.1f}")
    print(f"  mean   = {ess_summary['p_mean']:.1f}")
    print(f"  max    = {ess_summary['p_max']:.1f}")
    
    
    # --------------------------------------------------------
    # ESS diagnostics for plateau variables
    # --------------------------------------------------------
    if cp_raw is not None:
        tau_cp_raw, ess_cp_raw, _ = iat_ess_1d(cp_raw, max_lag=max_lag)
        tau_cn_raw, ess_cn_raw, _ = iat_ess_1d(cn_raw, max_lag=max_lag)
    
        tau_cp, ess_cp, _ = iat_ess_1d(cp_samples, max_lag=max_lag)
        tau_cn, ess_cn, _ = iat_ess_1d(cn_samples, max_lag=max_lag)
    
        ess_summary["cp_raw"] = float(ess_cp_raw)
        ess_summary["cn_raw"] = float(ess_cn_raw)
        ess_summary["cp"] = float(ess_cp)
        ess_summary["cn"] = float(ess_cn)
    
        ess_summary["iat_cp_raw"] = float(tau_cp_raw)
        ess_summary["iat_cn_raw"] = float(tau_cn_raw)
        ess_summary["iat_cp"] = float(tau_cp)
        ess_summary["iat_cn"] = float(tau_cn)
    
        print("\nESS summary for plateau variables:")
        print(f"  cp_raw: IAT={tau_cp_raw:.2f}, ESS={ess_cp_raw:.1f}, ESS/n={ess_cp_raw/total_samples:.3f}")
        print(f"  cn_raw: IAT={tau_cn_raw:.2f}, ESS={ess_cn_raw:.1f}, ESS/n={ess_cn_raw/total_samples:.3f}")
        print(f"  cp    : IAT={tau_cp:.2f}, ESS={ess_cp:.1f}, ESS/n={ess_cp/total_samples:.3f}")
        print(f"  cn    : IAT={tau_cn:.2f}, ESS={ess_cn:.1f}, ESS/n={ess_cn/total_samples:.3f}")
    # --------------------------------------------------------
    # Trace plot for 5 KL coefficients
    # --------------------------------------------------------
    trace_path = os.path.join(outdir, "trace_5_coefficients.png")
    
    plot_trace5(
        S=S,
        save_path=trace_path,
        idxs=trace_idxs,
    )
    
    print("\nSaved trace plot:")
    print(" ", trace_path)


    # --------------------------------------------------------
    # Plateau summaries and HPD intervals
    # --------------------------------------------------------
    cp_hpd = hpd_interval_1d(cp_samples, cred_mass=cred_mass)
    cn_hpd = hpd_interval_1d(cn_samples, cred_mass=cred_mass)

    plateau_summary = {
        "cp_mean": float(np.mean(cp_samples)),
        "cp_std": float(np.std(cp_samples, ddof=1)),
        "cp_hpd_low": float(cp_hpd[0]),
        "cp_hpd_high": float(cp_hpd[1]),
        "cn_mean": float(np.mean(cn_samples)),
        "cn_std": float(np.std(cn_samples, ddof=1)),
        "cn_hpd_low": float(cn_hpd[0]),
        "cn_hpd_high": float(cn_hpd[1]),
    }

    print(f"\nPlateau posterior summaries ({100*cred_mass:.1f}% HPD):")
    print(
        f"  cp: mean={plateau_summary['cp_mean']:.4f}, "
        f"std={plateau_summary['cp_std']:.4f}, "
        f"HPD=[{plateau_summary['cp_hpd_low']:.4f}, "
        f"{plateau_summary['cp_hpd_high']:.4f}]"
    )
    print(
        f"  cn: mean={plateau_summary['cn_mean']:.4f}, "
        f"std={plateau_summary['cn_std']:.4f}, "
        f"HPD=[{plateau_summary['cn_hpd_low']:.4f}, "
        f"{plateau_summary['cn_hpd_high']:.4f}]"
    )
    
    
    # --------------------------------------------------------
    # Save and plot cp/cn histograms
    # --------------------------------------------------------
    hist_cp_path = os.path.join(outdir, "hist_cp.png")
    hist_cn_path = os.path.join(outdir, "hist_cn.png")
    
    plot_plateau_histogram(
        samples=cp_samples,
        save_path=hist_cp_path,
        xlabel=r"$c_{\mathrm{p}}$",
        true_value=float(obs["C_p"]) if "C_p" in obs else None,
    )
    
    plot_plateau_histogram(
        samples=cn_samples,
        save_path=hist_cn_path,
        xlabel=r"$c_{\mathrm{n}}$",
        true_value=float(obs["C_n"]) if "C_n" in obs else None,
    )
    
    print("\nSaved plateau histograms:")
    print(" ", hist_cp_path)
    print(" ", hist_cn_path)

    # --------------------------------------------------------
    # Build PDE and prior
    # --------------------------------------------------------
    pde = build_pde(obs)
    prior, cfg = build_prior(obs, pde, stat)

    V = prior.V
    ndof = V.dim()

    
    field_indices = np.arange(total_samples)
    n_field_samples = len(field_indices)

    print("\nField statistics:")
    print(f"  number of field samples used = {n_field_samples}")
    print(f"  number of dofs               = {ndof}")

    C_samples = np.zeros((n_field_samples, ndof), dtype=float)
    depth_samples = np.full(n_field_samples, np.nan, dtype=float)

    if use_mean_depth:
        x_list = np.linspace(x_depth_range[0], x_depth_range[1], nx_depth)
    else:
        x_list = None

    # --------------------------------------------------------
    # Assemble field samples and depth samples
    # --------------------------------------------------------
    with torch.no_grad():
        for kk, ii in enumerate(field_indices):
            p_t = torch.tensor(S[ii], dtype=torch.float64)

            cp_i = float(cp_samples[ii])
            cn_i = float(cn_samples[ii])

            prior.C_p = torch.tensor(cp_i, dtype=torch.float64, device=prior.device)
            prior.C_n = torch.tensor(cn_i, dtype=torch.float64, device=prior.device)

            C_i = prior.assemble_torch(p_t).detach().cpu().numpy()

            C_samples[kk, :] = C_i

            if use_mean_depth:
                depth_samples[kk] = estimate_mean_interface_depth_from_array(
                    C_vec=C_i,
                    V=V,
                    cp=cp_i,
                    cn=cn_i,
                    x_list=x_list,
                    y_min=0.0,
                    y_max=1.0,
                    n_y=400,
                )
            else:
                depth_samples[kk] = estimate_interface_depth_from_array(
                    C_vec=C_i,
                    V=V,
                    cp=cp_i,
                    cn=cn_i,
                    x_probe=x_probe,
                    y_min=0.0,
                    y_max=1.0,
                    n_y=400,
                )

    cp_used = cp_samples[field_indices]
    cn_used = cn_samples[field_indices]

    # --------------------------------------------------------
    # Mean, std, HPD field bounds
    # --------------------------------------------------------
    C_mean = np.mean(C_samples, axis=0)
    C_std = np.std(C_samples, axis=0, ddof=1)

    C_hpd_low, C_hpd_high = pointwise_hpd_bounds(
        C_samples,
        cred_mass=cred_mass,
    )
    
    # --------------------------------------------------------
    # Extra contours for plotting
    # --------------------------------------------------------
    C_true = obs.get("C_true", None)
    
    extra_mean_contours = []
    extra_true_contours = []
    # True interface: C_true = 0
    if C_true is not None:
        extra_mean_contours.append({
            "vec": C_true,
            "level": 0.0,
            "color": "black",
            "linestyle": "-",
            "linewidth": 2.0,
            "label": "True interface",
        })
    
    # Mean interface: C_mean = 0
    extra_mean_contours.append({
        "vec": C_mean,
        "level": 0.0,
        "color": "red",
        "linestyle": "--",
        "linewidth": 2.0,
        "label": "Mean interface",
    })
    
    if C_true is not None:
        extra_true_contours.append({
            "vec": C_true,
            "level": 0.0,
            "color": "black",
            "linestyle": "-",
            "linewidth": 2.0,
            "label": "True interface",
        })
    
    # --------------------------------------------------------
    # Save and plot posterior mean and std fields
    # --------------------------------------------------------
    mean_fig_path = os.path.join(outdir, "C_mean.png")
    std_fig_path = os.path.join(outdir, "C_std.png")
    
    plot_field(
        V=V,
        vec=C_mean,
        save_path=mean_fig_path,
        # title="Posterior mean",
        cmap="viridis",
        cbar_label=r"$\mathbb{E}[c(\boldsymbol{x}) \mid y]$",
        contour_zero=False,
        extra_contours=extra_mean_contours,
        vmin=float(np.min(C_mean)),
        vmax=float(np.max(C_mean)),
        tick_format_set=True,
    )
    
    plot_field(
        V=V,
        vec=C_std,
        save_path=std_fig_path,
        # title="Posterior standard deviation",
        cmap="PuRd",
        cbar_label=r"$\mathrm{std}(c(\boldsymbol{x}) \mid y)$",
        contour_zero=False,
        extra_contours=extra_true_contours,
        vmin=float(np.min(C_std)),
        vmax=float(np.max(C_std)),
        tick_format_set=True,
    )
    
    print("\nSaved field plots:")
    print(" ", mean_fig_path)
    print(" ", std_fig_path)
    
    
    
    
    save_fenics_vector(os.path.join(outdir, "C_mean.npy"), C_mean)
    save_fenics_vector(os.path.join(outdir, "C_std.npy"), C_std)
    save_fenics_vector(os.path.join(outdir, "C_hpd_low.npy"), C_hpd_low)
    save_fenics_vector(os.path.join(outdir, "C_hpd_high.npy"), C_hpd_high)

    print(f"\nPointwise field summaries saved:")
    print("  C_mean.npy")
    print("  C_std.npy")
    print("  C_hpd_low.npy")
    print("  C_hpd_high.npy")
    
    
    C_true = obs.get("C_true", None)
    if C_true is not None:
        re_mean, ssim_mean = compute_re_ssim(C_mean, C_true)
    
        print("\nPosterior mean vs truth:")
        print(f"  RE   = {re_mean:.6f}")
        print(f"  SSIM = {ssim_mean:.6f}")

    # --------------------------------------------------------
    # Correlation with interface depth
    # --------------------------------------------------------
    good_depth = np.isfinite(depth_samples)

    if np.sum(good_depth) >= 3:
        corr_cp_depth = float(
            np.corrcoef(cp_used[good_depth], depth_samples[good_depth])[0, 1]
        )
        corr_cn_depth = float(
            np.corrcoef(cn_used[good_depth], depth_samples[good_depth])[0, 1]
        )

        depth_hpd = hpd_interval_1d(depth_samples[good_depth], cred_mass=cred_mass)

        depth_summary = {
            "depth_mean": float(np.mean(depth_samples[good_depth])),
            "depth_std": float(np.std(depth_samples[good_depth], ddof=1)),
            "depth_hpd_low": float(depth_hpd[0]),
            "depth_hpd_high": float(depth_hpd[1]),
            "corr_cp_depth": corr_cp_depth,
            "corr_cn_depth": corr_cn_depth,
            "n_valid_depth": int(np.sum(good_depth)),
        }

        if use_mean_depth:
            depth_label = (
                f"mean interface depth over "
                f"x in [{x_depth_range[0]}, {x_depth_range[1]}]"
            )
        else:
            depth_label = f"interface depth at x={x_probe:.2f}"

        print(f"\n{depth_label}:")
        print(
            f"  mean={depth_summary['depth_mean']:.4f}, "
            f"std={depth_summary['depth_std']:.4f}, "
            f"HPD=[{depth_summary['depth_hpd_low']:.4f}, "
            f"{depth_summary['depth_hpd_high']:.4f}]"
        )
        print(f"  corr(cp, depth) = {corr_cp_depth:.4f}")
        print(f"  corr(cn, depth) = {corr_cn_depth:.4f}")

    else:
        depth_summary = {
            "depth_mean": np.nan,
            "depth_std": np.nan,
            "depth_hpd_low": np.nan,
            "depth_hpd_high": np.nan,
            "corr_cp_depth": np.nan,
            "corr_cn_depth": np.nan,
            "n_valid_depth": int(np.sum(good_depth)),
        }

        print("\nInterface depth:")
        print("  Not enough valid depth samples.")

    np.save(os.path.join(outdir, "depth_samples.npy"), depth_samples)
    np.save(os.path.join(outdir, "cp_samples_used_for_depth.npy"), cp_used)
    np.save(os.path.join(outdir, "cn_samples_used_for_depth.npy"), cn_used)

    # --------------------------------------------------------
    # Save text summary
    # --------------------------------------------------------
    summary_path = os.path.join(outdir, "nuts_clean_summary.txt")

    with open(summary_path, "w") as f:
        f.write("NUTS clean post-processing summary\n")
        f.write("==================================\n\n")

        f.write("Sample information\n")
        f.write("------------------\n")
        for key, value in info.items():
            f.write(f"{key}: {value}\n")

        f.write("\nESS summary\n")
        f.write("-----------\n")
        
        f.write("\nTrace plot\n")
        f.write("----------\n")
        f.write("trace_5_coefficients.png\n")

        f.write("\nPlateau summaries\n")
        f.write("-----------------\n")
        for key, value in plateau_summary.items():
            f.write(f"{key}: {value}\n")

        f.write("\nDepth summary\n")
        f.write("-------------\n")
        for key, value in depth_summary.items():
            f.write(f"{key}: {value}\n")

        f.write("\nField outputs\n")
        f.write("-------------\n")
        f.write("C_mean.npy\n")
        f.write("C_std.npy\n")
        f.write("C_hpd_low.npy\n")
        f.write("C_hpd_high.npy\n")
        
        f.write("\nFigures\n")
        f.write("-------\n")
        f.write("trace_5_coefficients.png\n")
        f.write("C_mean.png\n")
        f.write("C_std.png\n")
        f.write("hist_cp.png\n")
        f.write("hist_cn.png\n")
        

    print("\nSaved summary:")
    print(" ", summary_path)

    results = {
        "info": info,
        "plateau_summary": plateau_summary,
        "depth_summary": depth_summary,
        "C_mean": C_mean,
        "C_std": C_std,
        "C_hpd_low": C_hpd_low,
        "C_hpd_high": C_hpd_high,
        "depth_samples": depth_samples,
        "cp_samples": cp_samples,
        "cn_samples": cn_samples,
    }

    return results


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    
    Example = 1
    Setting = 1
    
    postprocess_nuts(
        sample_file=f"./stat_Setting{Setting}/nuts_Setting{Setting}_Ex{Example}.pickle",
        data_file= f"./obs/obs_synth_top_Ex{Example}.pickle",
        outdir=f"./postprocess/nuts_Setting{Setting}_Ex{Example}",
        cred_mass=0.95,
        x_probe=0.5,
        use_mean_depth=False,
        trace_idxs=[0, 1, 2, 3, 4],
        max_lag = 200,
        accept_tol = 0.0,
    )