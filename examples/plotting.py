# plotting.py
"""Figure helpers in the style of the paper figures."""
import numpy as np
import dolfin as dl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.gridspec import GridSpec              # noqa: E402
from matplotlib.ticker import FormatStrFormatter      # noqa: E402
from scipy.stats import gaussian_kde                  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

TRUE_INTERFACE = dict(color="black", linestyle="-", linewidth=2.0)
MEAN_INTERFACE = dict(color="red", linestyle="--", linewidth=2.0)


def _function(V, vec):
    f = dl.Function(V)
    f.vector().set_local(np.asarray(vec, dtype=float))
    f.vector().apply("insert")
    return f


def _style_axes(ax, ylabel=True):
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_xticks([0.0, 0.5, 1.0])
    if ylabel:
        ax.set_ylabel("$y$", fontsize=16)
        ax.set_yticks([0.0, 0.5, 1.0])
    else:
        ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("black")
        sp.set_linewidth(1.0)
    ax.tick_params(direction="out", length=3, width=1, labelsize=14)


def _zero_contour(ax, V, vec, level=0.0, **style):
    mesh = V.mesh()
    vals = _function(V, vec).compute_vertex_values(mesh)
    if vals.min() < level < vals.max():
        xy = mesh.coordinates()
        ax.tricontour(xy[:, 0], xy[:, 1], mesh.cells(), vals, levels=[level],
                      colors=style["color"], linestyles=style["linestyle"], linewidths=style["linewidth"])


def plot_field(V, vec, path, cbar_label=r"$c(\boldsymbol{x})$", cmap="viridis", contours=(), vlim=None):
    """Field with colorbar; `contours` is a sequence of (vec, style) drawn at level 0; `vlim` = (vmin, vmax)
    gives several panels one color scale (default: the field's range)."""
    fig, ax = plt.subplots(figsize=(4.2, 3.75))
    vmin, vmax = (float(np.min(vec)), float(np.max(vec))) if vlim is None else map(float, vlim)
    im = dl.plot(_function(V, vec), axes=ax, cmap=cmap, vmin=vmin, vmax=vmax)
    for c_vec, style in contours:
        _zero_contour(ax, V, c_vec, **style)
    _style_axes(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=14)
    cb.set_ticks(np.linspace(vmin, vmax, 4))
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cb.ax.tick_params(labelsize=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_prior_samples(V, fields, path):
    """Row of prior samples with their zero-level sets (Figure 3)."""
    n = len(fields)
    fig = plt.figure(figsize=(12.5, 3.75))
    gs = GridSpec(1, n + 1, figure=fig, width_ratios=[1] * n + [0.05], wspace=0.17)
    axs = [fig.add_subplot(gs[0, j]) for j in range(n)]
    cax = fig.add_subplot(gs[0, n])
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + 0.10, pos.width, pos.height * 0.74])
    for j, (ax, c) in enumerate(zip(axs, fields)):
        plt.sca(ax)   # dolfin.plot draws on the current axes
        im = dl.plot(_function(V, c), axes=ax, cmap="viridis", mode="color")
        _zero_contour(ax, V, c, color="black", linestyle="-", linewidth=1.5)
        _style_axes(ax, ylabel=(j == 0))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$c(\boldsymbol{x})$", fontsize=14)
    cb.ax.tick_params(labelsize=14)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_trace(chain, path, idxs=(0, 1, 2, 3, 4), x=None, xlabel="Iteration"):
    """Trace plot of selected KL coefficients (Figures 5 and 8) against x (default: draw index)."""
    x = np.arange(chain.shape[0]) if x is None else x
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for j in idxs:
        ax.plot(x, chain[:, j], linewidth=0.6, label=rf"$z_{{{j + 1}}}$")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("KL coefficient value", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="lower right")
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# chains 1-3: the first three slots of a CVD-validated categorical palette; chain 4: neutral gray
CHAIN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#52514e"]


def plot_trace_chains(columns, path, xlabel, row_labels, true_values=None, xlim=None, legend=True):
    """
    Four-chain trace plots (Figure 5 of the revision): one column per sampler, one row per quantity.
    `columns` is a list of dicts with "title", "x" (list of per-chain x arrays), "y" (list of per-chain
    arrays of shape (n_draws, n_rows)) and optionally "marks" (list of per-chain (x, y) pairs of draws
    marked with a cross, e.g. NUTS divergences). Rows share their y-range across the columns; `xlim`
    gives all panels one x-range. legend=False omits the legend row below the panels.
    """
    n_rows, n_cols = len(row_labels), len(columns)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 2.1 * n_rows), sharey="row", squeeze=False)
    for j, col in enumerate(columns):
        for i in range(n_rows):
            ax = axs[i, j]
            for c, (x, y) in enumerate(zip(col["x"], col["y"])):
                ax.plot(x, y[:, i], color=CHAIN_COLORS[c % len(CHAIN_COLORS)], linewidth=0.5, alpha=0.85,
                        label=f"chain {c + 1}")
                if col.get("marks") is not None and len(col["marks"][c][0]):
                    xm, ym = col["marks"][c]
                    ax.plot(xm, ym[:, i], linestyle="none", marker="x", zorder=5,
                            markersize=5, markeredgewidth=1.0, color="black", label="divergent transition")
            if true_values is not None and true_values[i] is not None:
                ax.axhline(true_values[i], color="black", linestyle="--", linewidth=0.8, zorder=4)
            if xlim is not None:
                ax.set_xlim(*xlim)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.tick_params(labelsize=10)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=12)
            if i == 0:
                ax.set_title(col["title"], fontsize=12)
            if i == n_rows - 1:
                ax.set_xlabel(xlabel, fontsize=11)
    if legend:
        handles, labels = [], []
        for ax in axs.ravel():
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l and l not in labels:
                    handles.append(h)
                    labels.append(l)
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=10, frameon=False,
                   bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04 if legend else 0, 1, 1))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_histogram(samples, path, xlabel, true_value=None):
    """Marginal posterior histogram with KDE, mean and true value (Figure 7c-d)."""
    samples = np.asarray(samples, dtype=float)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.hist(samples, bins="auto", density=True, edgecolor="black", linewidth=0.25, color="lightgray")
    if samples.std() > 0:
        x = np.linspace(samples.min(), samples.max(), 400)
        ax.plot(x, gaussian_kde(samples)(x), color="black", linewidth=1.2)
    ax.axvline(samples.mean(), color="green", linestyle="-", linewidth=2.0, label="Mean")
    if true_value is not None:
        ax.axvline(true_value, color="red", linestyle="--", linewidth=1.5, label="True value")
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=13)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_data_fit(x, y_obs, y_pred, path, sigma=None):
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    if sigma is None:
        ax.plot(x, y_obs, "ko", ms=4, label="data")
    else:
        ax.errorbar(x, y_obs, yerr=2 * sigma, fmt="ko", ms=4, capsize=2, label=r"data $\pm 2\sigma$")
    ax.plot(x, y_pred, "r.-", label="prediction")
    ax.set_xlabel("electrode position $x$", fontsize=13)
    ax.set_ylabel(r"$\int_{I_\ell} \partial_y u \, ds$", fontsize=13)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
