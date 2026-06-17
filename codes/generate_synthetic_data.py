# generate_synthetic_data.py
import os
import pickle
import numpy as np
import dolfin as dl

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

from pde_model import TopFluxPDE
from matern_torch import TorchMaternCov


norm = mcolors.PowerNorm(gamma=0.7)  # < 1 → brighter

dl.set_log_level(50)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

def resize_cax(cax, shrink=0.85, shift=0.03):
    """
    shrink < 1 makes colorbar shorter.
    shift moves it upward (positive) / downward (negative) as a fraction of its height.
    """
    pos = cax.get_position()
    new_h = pos.height * shrink
    new_y = pos.y0 + (pos.height - new_h) * 0.5 + shift * pos.height
    cax.set_position([pos.x0, new_y, pos.width, new_h])


class SmoothTwoPhase(dl.UserExpression):
    """
    Smooth 2-phase doping profile C(x,y).

    example = 1:
        y_curve = base + slope * x

    example = 2:
        y_curve = base + slope * x + 0.2 sin(pi x)
    """

    def __init__(
        self,
        C_n=1.0,
        C_p=-2.0,
        base=0.6,
        slope=-0.1,
        width=0.05,
        example=1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.C_n = float(C_n)
        self.C_p = float(C_p)
        self.base = float(base)
        self.slope = float(slope)
        self.width = float(width)
        self.example = int(example)

    def eval(self, values, x):
        xc, yc = float(x[0]), float(x[1])

        if self.example == 1:
            y_curve = self.base + self.slope * xc

        elif self.example == 2:
            y_curve = self.base + self.slope * xc + 0.2 * np.sin(np.pi * xc)

        else:
            raise ValueError("example must be 1 or 2.")

        s = 0.5 * (1.0 + np.tanh((yc - y_curve) / self.width))

        values[0] = self.C_p + (self.C_n - self.C_p) * s

    def value_shape(self):
        return ()

def plot_and_save_C_field(C_fun, save_path="Fig_true_C.png", title=None):
    """
    Plot a single C-field with a dedicated colorbar axis, similar in style
    to the multi-panel example.
    """
    fig = plt.figure(figsize=(4.2, 3.75))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.05])
    gs.update(wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    # make colorbar a bit shorter and vertically centered
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + 0.10, pos.width, pos.height * 0.74])

    # main field plot
    im = dl.plot(
        C_fun,
        axes=ax,
        cmap="viridis"
    )

    # contour for interface c(x)=0
    mesh = C_fun.function_space().mesh()
    coords = mesh.coordinates()
    cells = mesh.cells()
    C_vals = C_fun.compute_vertex_values(mesh)

    ax.tricontour(
        coords[:, 0],
        coords[:, 1],
        cells,
        C_vals,
        levels=[0.0],
        colors="black",
        linewidths=2.0,
    )

    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel("$y$", fontsize=14)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("black")
        sp.set_linewidth(1)

    ax.tick_params(direction="out", length=3, width=1, labelsize=14)

    if title is not None:
        ax.set_title(title, fontsize=14)

    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$c(\boldsymbol{x})$", fontsize=14)
    cb.ax.tick_params(labelsize=14)

    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    fmt.set_useOffset(False)
    cb.formatter = fmt
    cb.set_ticks([-2.0, -1.0, 0.0, 1.0])
    cb.update_ticks()

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved C-field figure: {save_path}")
    
    
def make_synth_data(
    save_as="obs_synth_top.pickle",
    true_fig_save_path="./obs/Fig_true_C",
    mesh_N=48,
    Lx=1.0,
    C_n=1.0,
    C_p=-2.0,
    lam=1.0,
    delta=1.0,
    k_sigmoid=2e1,
    noise_level=0.05,
    width=0.035,
    example=1,
    show=True,
    seed=123,
):
    
        
    # BCs consistent with your earlier choice
    V_top = np.asinh(C_n / (2.0 * delta**2))
    V_bottom = np.asinh(C_p / (2.0 * delta**2))

    # # PDE (takes C as input)
    # build PDE first
    pde = TopFluxPDE(
        N_points=mesh_N,
        Lx=Lx,
        V_top=V_top,
        V_bottom=V_bottom,
        lam=lam,
        delta=delta,
    )
    
    # prior on SAME mesh
    prior = TorchMaternCov(
        pde.mesh,
        push_forward_method="sigmoid",
        C_p = C_p,
        C_n=C_n,
        k=k_sigmoid,
    )
    
    # N_full = pde.V.dim()
    # n9978, r9978 = prior.variance_retention(rho=0.9978, N_full=N_full, corr_len=0.15, nu=3)
    # N_KL = n9978
    # print("Modes for 99.9% variance:", n9978, " achieved:", r9978)
    
    N_KL = 30
    
   
     
    # True C
    expr = SmoothTwoPhase(
        C_n=C_n,
        C_p=C_p,
        width=width,
        example=example,
        degree=2,
    )
    
    C_fun = dl.interpolate(expr, pde.V)
    C_true = C_fun.vector().get_local()
    
    vtk_file = dl.File(true_fig_save_path + '.pvd')
    C_fun.rename("C", "doping")
    vtk_file << C_fun

    
    if show: 
        fig, ax = plt.subplots(figsize=(4.2, 3.75))

        # --- enforce correct color scaling ---
        im = dl.plot(
            C_fun,
            axes=ax,
            cmap="viridis",
            norm=norm,
            vmin=-2.0,
            vmax=1.0
        )
        
        # --- interface (c = 0 contour) ---
        mesh = C_fun.function_space().mesh()
        coords = mesh.coordinates()
        cells = mesh.cells()
        C_vals = C_fun.compute_vertex_values(mesh)
        
        ax.tricontour(
            coords[:, 0],
            coords[:, 1],
            cells,
            C_vals,
            levels=[0.0],
            colors="black",
            linewidths = 1.5,
        )
        
        # --- axis styling ---
        ax.set_xlabel("$x$", fontsize=16)
        ax.set_ylabel("$y$", fontsize=16)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("black")
            sp.set_linewidth(1.0)
        
        ax.tick_params(direction="out", length=3, width=1, labelsize=14)
        
        # --- colorbar (attached to same axis) ---
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # cb.set_label(r"$c(\boldsymbol{x})$", fontsize=14)
        cb.set_ticks([-2.0, -1.0, 0.0, 1.0])
        cb.ax.tick_params(labelsize=14)
        plt.tight_layout()
        # --- save ---
        plt.savefig( true_fig_save_path + '.png', bbox_inches="tight", dpi=300)
        
        # =========================================================
        # 3D plot
        # =========================================================
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import matplotlib.tri as mtri
        
        fig3d = plt.figure(figsize=(5.5, 4.6))
        ax3d = fig3d.add_subplot(111, projection="3d")
        
        triang = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)
        
        surf = ax3d.plot_trisurf(
            triang,
            C_vals,
            cmap="viridis",
            vmin=-2.0,
            vmax=1.0,
            linewidth=0.1,
            edgecolor="k",
            antialiased=True,
        )
        
        # --- axis styling ---
        ax3d.set_xlabel("$x$", fontsize=13, labelpad=6)
        ax3d.set_ylabel("$y$", fontsize=13, labelpad=6)
        ax3d.set_zlabel(r"$c(\boldsymbol{x})$", fontsize=13, labelpad=8)
        
        ax3d.set_xlim(0.0, 1.0)
        ax3d.set_ylim(0.0, 1.0)
        ax3d.set_zlim(-2.0, 1.0)
        
        ax3d.set_xticks([0.0, 0.5, 1.0])
        ax3d.set_yticks([0.0, 0.5, 1.0])
        ax3d.set_zticks([-2.0, -1.0, 0.0, 1.0])
        
        # good MATLAB-like viewing angle
        ax3d.view_init(elev=28, azim=-135)
        
        # cleaner panes/grid
        ax3d.tick_params(labelsize=11)
        
        # --- colorbar ---
        cb3d = fig3d.colorbar(surf, ax=ax3d, fraction=0.046, pad=0.08)
        cb3d.set_label(r"$c(\boldsymbol{x})$", fontsize=13)
        cb3d.set_ticks([-2.0, -1.0, 0.0, 1.0])
        cb3d.ax.tick_params(labelsize=11)
        
        # --- save 3D ---
        plt.savefig(true_fig_save_path + "_3D.png", bbox_inches="tight", dpi=300)
        plt.show()

    # Solve & measure y = u_y on top dofs
    pde.solve_state(C_true, cP = C_p, cN = C_n, zero_guess=True)
    y_true = pde.top_flux_facet_vec()
    
    if show:
        plt.figure()
        plt.plot(y_true, "-o")
        plt.title("Top data (noiseless): y = u_y on top dofs")
        plt.xlabel("top dof index")
        plt.ylabel("u_y")
        plt.tight_layout()
        plt.show()

    # Noise (noise_level is % of the RMS variability (not the mean))
    np.random.seed(seed)
    y0 = y_true - y_true.mean()
    sigma = noise_level * np.linalg.norm(y0) / np.sqrt(y0.size)   # = noise_level * RMS(demeaned)
    noise = sigma * np.random.randn(y_true.size)
    y_obs = y_true + noise
    
    err = 100*np.linalg.norm(y_obs - y_true)/np.linalg.norm(y0)   # consistent
    print(err)
    
    if show:
        plt.figure()
        plt.plot(y_true, label="y_true")
        plt.plot(y_obs, label="y_obs")
        plt.legend()
        plt.title("Noisy observations on top boundary")
        plt.tight_layout()
        plt.show()

    obs = {
        "mesh_N": mesh_N,
        "N_x": mesh_N,           # keep backward compatibility
        "N_KL": N_KL,
        "Lx": float(Lx),
        "lambda_val": float(lam),
        "delta_val": float(delta),
        "k": float(k_sigmoid),
        "V_top": float(V_top),
        "V_bottom": float(V_bottom),
        "C_n": float(C_n),
        "C_p": float(C_p),
        "C_true": C_true,
        "y_true": y_true,
        "y_obs": y_obs,
        "noise": noise,
        "sigma": float(sigma),
        "obs_grid": "top_facets",
        "meta": {
        "source": "smooth_two_phase",
        "example": int(example),
        "width": float(width),
        "noise_level": float(noise_level),},
        "seed": int(seed),
    }

    os.makedirs("./obs", exist_ok=True)
    path = os.path.join("./obs", save_as)
    with open(path, "wb") as f:
        pickle.dump(obs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved: {path}")
    return obs


if __name__ == "__main__":
    
    Example = 1
    # --------------------------------------------------
    # Example 1: Straigh Junction
    # Example 2: Curved Junction
    # --------------------------------------------------
    make_synth_data(
        save_as=f"obs_synth_top_Ex{Example}.pickle",
        true_fig_save_path=f"./obs/Fig_true_C_Ex{Example}",
        example=Example,
        show=True,
    )

    
