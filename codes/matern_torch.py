#matern_torch.py

import numpy as np
import dolfin as dl
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

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


class TorchMaternCov:
    """
    Matérn-type KL prior where the mapping p -> C is implemented in PyTorch.

    - Eigenpairs are computed with FEniCS/SLEPc (same as your current code).
    - After that, we build a torch matrix Phi = evecs * sqrt_lmbda (ndof x N_KL)
      so that:
          eta = Phi @ p
      and (optionally) push-forward to C via sigmoid/heaviside.
    """

    def __init__(self, mesh_or_N, push_forward_method=None, C_p=-2.0, C_n=1.0, k=1000.0, device="cpu"):
        if isinstance(mesh_or_N, dl.Mesh):
            self.mesh = mesh_or_N
        else:
            N = int(mesh_or_N)
            self.mesh = dl.UnitSquareMesh(dl.MPI.comm_self, N, N)

        self.V = dl.FunctionSpace(self.mesh, "CG", 1)
        self.u = dl.TrialFunction(self.V)
        self.v = dl.TestFunction(self.V)

        self.push_forward_method = push_forward_method
        self.C_p = float(C_p)
        self.C_n = float(C_n)
        self.k = float(k)

        self.dim = None
        self.tau2 = None

        # Stored in NumPy (from eigen solve)
        self.evecs_np = None          # (ndof, N_KL)
        self.sqrt_lmbda_np = None     # (N_KL,)

        # Stored in torch (for fast/differentiable assemble)
        self.Phi = None               # (ndof, N_KL) torch
        self.device = torch.device(device)
        
        self.lmbda_np = None          # (N_KL,)
        self.eigvals_np = None        # optional: raw PDE eigenvalues

    # ------------------------------------------------------------
    # Eigenpairs (FEniCS/SLEPc) -> then convert to torch
    # ------------------------------------------------------------
    def compute_eigen_decomp(self, N_eig, corr_len=0.1, nu=1.0):
        corr_len = float(corr_len)
        self.tau2 = 1.0 / (corr_len * corr_len)

        a_form = (
            self.tau2 * self.u * self.v * dl.dx
            + dl.inner(dl.grad(self.u), dl.grad(self.v)) * dl.dx
        )
        
        # ell = corr_len
        # a_form = (
        #     self.tau2 * self.u * self.v * dl.dx
        #     + dl.inner(dl.grad(self.u), dl.grad(self.v)) * dl.dx
        #     + (1.0 / ell) * self.u * self.v * dl.ds )     # Robin term
        
        m_form = self.u * self.v * dl.dx
        
        
        A = dl.assemble(a_form)
        M = dl.assemble(m_form)
        
        eigen_solver = dl.SLEPcEigenSolver(dl.as_backend_type(A),  dl.as_backend_type(M))
        # eigen_solver = dl.SLEPcEigenSolver(dl.as_backend_type(A))
        eigen_solver.parameters["spectrum"] = "smallest magnitude"
        eigen_solver.solve(int(N_eig))

        tmp_fun = dl.Function(self.V)
        n_dofs = tmp_fun.vector().local_size()

        eigvals = np.zeros(N_eig)
        eigvecs = np.zeros((n_dofs, N_eig))

        # for i in range(N_eig):
        #     val, _, vec, _ = eigen_solver.get_eigenpair(i)
        #     eigvals[i] = val
        #     eigvecs[:, i] = vec.get_local()
        
        for i in range(N_eig):
            val, _, vec, _ = eigen_solver.get_eigenpair(i)
            eigvals[i] = val
        
            v_np = vec.get_local()
        
            v_fun = dl.Function(self.V)
            v_fun.vector().set_local(v_np)
            v_fun.vector().apply("insert")
        
            norm_M = np.sqrt(dl.assemble(v_fun * v_fun * dl.dx))
        
            eigvecs[:, i] = v_np / norm_M

        # KL eigenvalues (your choice)
        alpha = nu + 1  # for 2D, matches your code
        lmbda = np.float_power(eigvals, -alpha)
        lmbda /= np.sum(lmbda)
        # lmbda /= np.linalg.norm(lmbda)

        self.dim = int(N_eig)
        self.eigvals_np = eigvals
        self.lmbda_np = lmbda
        self.evecs_np = eigvecs
        self.sqrt_lmbda_np = np.sqrt(lmbda)

        # Build Phi = evecs * sqrt_lmbda  (ndof x N_KL)
        Phi_np = self.evecs_np * self.sqrt_lmbda_np[None, :]
        self.Phi = torch.tensor(Phi_np, dtype=torch.float64, device=self.device)
        
       
    # ------------------------------------------------------------
    # Push-forward in torch
    # ------------------------------------------------------------
    def _push_forward_torch(self, eta_t: torch.Tensor) -> torch.Tensor:
        if self.push_forward_method is None:
            return eta_t

        if self.push_forward_method == "sigmoid":
            # sigmoid(k*eta) but with safe clipping (like your numpy version)
            z = (-self.k) * eta_t
            z = torch.clamp(z, -60.0, 60.0)
            sig = 1.0 / (1.0 + torch.exp(z))
            return self.C_p + (self.C_n - self.C_p) * sig

        if self.push_forward_method == "heaviside":
            H = (eta_t >= 0.0).to(eta_t.dtype)
            return self.C_p + (self.C_n - self.C_p) * H

        raise ValueError(f"Unknown push_forward_method: {self.push_forward_method}")

    # ------------------------------------------------------------
    # Torch assemble: p -> C (differentiable)
    # ------------------------------------------------------------
    def assemble_torch(self, p_t: torch.Tensor) -> torch.Tensor:
        """
        p_t: torch tensor shape (N_KL,) or (batch, N_KL)
        returns: C_t shape (ndof,) or (batch, ndof)
        """
        if self.Phi is None:
            raise RuntimeError("Call compute_eigen_decomp() first.")

        if p_t.dtype != torch.float64:
            p_t = p_t.to(torch.float64)
        p_t = p_t.to(self.device)

        # eta = Phi @ p
        if p_t.ndim == 1:
            eta = self.Phi @ p_t                    # (ndof,)
            return self._push_forward_torch(eta)    # (ndof,)
        elif p_t.ndim == 2:
            eta = p_t @ self.Phi.T                  # (batch, ndof)
            return self._push_forward_torch(eta)    # (batch, ndof)
        else:
            raise ValueError("p_t must have shape (N_KL,) or (batch, N_KL).")

    # ------------------------------------------------------------
    # Torch directional derivative (optional; mostly for debugging)
    # ------------------------------------------------------------
    def derivative_torch(self, p_t: torch.Tensor, t_t: torch.Tensor) -> torch.Tensor:
        """
        Returns dC/dp[t] in torch, same shape as C.
        Implemented analytically to match your numpy version.
        """
        if self.Phi is None:
            raise RuntimeError("Call compute_eigen_decomp() first.")

        p_t = p_t.to(torch.float64).to(self.device)
        t_t = t_t.to(torch.float64).to(self.device)

        # delta_eta = Phi @ t
        if p_t.ndim != 1 or t_t.ndim != 1:
            raise ValueError("derivative_torch expects 1D p_t and t_t.")

        delta_eta = self.Phi @ t_t  # (ndof,)

        if self.push_forward_method is None:
            return delta_eta

        if self.push_forward_method == "sigmoid":
            eta = self.Phi @ p_t
            z = (-self.k) * eta
            z = torch.clamp(z, -60.0, 60.0)
            sig = 1.0 / (1.0 + torch.exp(z))
            dC_deta = (self.C_n - self.C_p) * self.k * sig * (1.0 - sig)
            return dC_deta * delta_eta

        raise ValueError(f"Unknown push_forward_method: {self.push_forward_method}")

    # ------------------------------------------------------------
    # Convenience: numpy wrappers (so old code still works)
    # ------------------------------------------------------------
    def assemble(self, p_np: np.ndarray) -> np.ndarray:
        p_t = torch.tensor(np.asarray(p_np, dtype=float), dtype=torch.float64, device=self.device)
        C_t = self.assemble_torch(p_t)
        return C_t.detach().cpu().numpy()

    def derivative(self, p_np: np.ndarray, t_np: np.ndarray) -> np.ndarray:
        p_t = torch.tensor(np.asarray(p_np, dtype=float), dtype=torch.float64, device=self.device)
        t_t = torch.tensor(np.asarray(t_np, dtype=float), dtype=torch.float64, device=self.device)
        dC_t = self.derivative_torch(p_t, t_t)
        return dC_t.detach().cpu().numpy()
    
    def variance_retention(self, rho, N_full=2000, corr_len=0.1, nu=1.0, return_curve=False):
        """
        Compute the number of KL modes needed to retain a fraction rho of the variance.
        return smallest number of modes such that cumulative variance >= rho.
 
        """
        if not (0.0 < rho <= 1.0):
            raise ValueError("rho must satisfy 0 < rho <= 1.")
    
        ndof = self.V.dim()
        N_full = min(int(N_full), ndof)
    
        # If current spectrum is missing or too short, recompute with N_full modes
        if self.lmbda_np is None or len(self.lmbda_np) < N_full:
            self.compute_eigen_decomp(N_full, corr_len=corr_len, nu=nu)
    
        lmbda = self.lmbda_np[:N_full]
    
        total_var = np.sum(lmbda)
        if total_var <= 0:
            raise RuntimeError("Total variance is non-positive.")
    
        cumvar = np.cumsum(lmbda) / total_var
        n_retained = int(np.searchsorted(cumvar, rho) + 1)
        retained_ratio = float(cumvar[n_retained - 1])
    
        if return_curve:
            return n_retained, retained_ratio, cumvar
        return n_retained, retained_ratio


def _test_derivative():
    """
    Simple finite-difference check for MaternCov.derivative().
    """
    dl.set_log_level(50)

    N_mesh = 48
    C_p = -2
    C_n = 1
    k = 2e1
    
    prior = TorchMaternCov(N_mesh, push_forward_method="sigmoid", C_p=C_p, C_n=C_n, k=k)
    
    # N_full= (N_mesh + 1)**2
    
    # n95, r95 = prior.variance_retention(rho=0.95, N_full= N_full, corr_len=0.15, nu=3)
    # print("Modes for 95% variance:", n95, " achieved:", r95)

    # n99, r99 = prior.variance_retention(rho=0.99, N_full= N_full, corr_len=0.15, nu=3)
    # print("Modes for 99% variance:", n99, " achieved:", r99)
    
    
    N_KL = 30
    prior.compute_eigen_decomp(N_KL, corr_len=0.15, nu=3)
    
    p = torch.randn(N_KL)
    c = prior.assemble_torch(p)   

    t = torch.randn(N_KL)

    eps = 1e-6
    C_plus = prior.assemble_torch(p + eps * t)    # (ndof,)
    C_minus = prior.assemble_torch(p - eps * t)   # (ndof,)
    fd = (C_plus - C_minus) / (2.0 * eps)

    exact = prior.derivative_torch(p, t)

    fd_np = fd.detach().cpu().numpy()
    exact_np = exact.detach().cpu().numpy()

    rel_err = np.linalg.norm(exact_np - fd_np) / (np.linalg.norm(exact_np) + 1e-14)
    max_err = np.max(np.abs(exact_np - fd_np))

    print(f"[TorchMaternCov] derivative FD check:")
    print(f"  rel_err = {rel_err:.3e}")
    print(f"  max_err = {max_err:.3e}")

   
    from matplotlib.gridspec import GridSpec  
    import matplotlib.ticker as ticker
    
    fig = plt.figure(figsize=(12.5, 3.75))
    n_fig = 4
    gs = GridSpec(1, n_fig + 1, figure=fig, width_ratios=[1] * n_fig + [0.05])
    gs.update(wspace=0.17)
    
    axs = [fig.add_subplot(gs[0, j]) for j in range(n_fig)]
    cax = fig.add_subplot(gs[0, n_fig])  # dedicated colorbar axis
    
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + 0.10, pos.width, pos.height * 0.74])
    
    ims = []
    
    C_fun = dl.Function(prior.V)
    
    torch.manual_seed(1368)
    for j in range(n_fig):
        p_i = torch.randn(prior.dim)
        C_i = prior.assemble_torch(p_i).detach().cpu().numpy()
    
        C_fun.vector().set_local(C_i)
        C_fun.vector().apply("insert")
    
        plt.sca(axs[j])                     # make axs[j] the active axis
        # im = dl.plot(C_fun, mode="color")  # plot on that axis
        im = dl.plot(C_fun, axes=axs[j], cmap="viridis", mode="color")

        mesh = prior.V.mesh()
        coords = mesh.coordinates()
        cells = mesh.cells()
        C_vals = C_fun.compute_vertex_values(mesh)
        
        axs[j].tricontour(
            coords[:,0],
            coords[:,1],
            cells,
            C_vals,
            levels=[0.0],
            colors="black",
            # linestyles="--",
            linewidths=1.5,
        )
        
        ims.append(im)
        
        axs[j].set_xlabel("$x$", fontsize=14)
        axs[j].set_xticks([0.0, 0.5, 1.0])


        if j == 0:
            axs[j].set_ylabel("$y$", fontsize=14)
            axs[j].set_yticks([0.0, 0.5, 1.0])
        else:
            axs[j].set_yticks([])
       
        # Black frame
        for sp in axs[j].spines.values():
            sp.set_visible(True)
            sp.set_color("black")
            sp.set_linewidth(1.3)

        axs[j].tick_params(direction="out", length=3, width=1, labelsize=14)

            
    # one shared colorbar    
    cb = fig.colorbar(ims[-1], cax=cax)
    # cb.set_label(r"$c(\mathbf{x})$", fontsize=14)
    cb.set_label(r"$c(\boldsymbol{x})$", fontsize=14)
    cb.ax.tick_params(labelsize=14)
    
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    fmt.set_useOffset(False)
    cb.formatter = fmt
    cb.set_ticks([-2.0, -1.0, 0.0, 1.0])
    cb.update_ticks()
    
    # cax = fig.add_subplot(gs[4])
    # resize_cax(cax, shrink=0.92, shift=0.00)
    
    plt.savefig("Fig2_prior_samples.png", bbox_inches="tight", dpi=300)
    # plt.close(fig)
    
    plt.show()
    

def compute_prior_mean_variance(
    n_samples=100000,
    unbiased=True,
    make_fenics_functions=True,
):
    dl.set_log_level(50)

    N_mesh = 48
    C_p = -2
    C_n = 1
    k = 2e1
    
    prior = TorchMaternCov(N_mesh, push_forward_method="sigmoid", C_p=C_p, C_n=C_n, k=k)
    N_KL = 30
    prior.compute_eigen_decomp(N_KL, corr_len=0.5, nu=3)
    


    # One sample to get size, dtype, device
    p0 = torch.randn(prior.dim)
    c0 = prior.assemble_torch(p0).detach()

    mean = torch.zeros_like(c0)
    M2 = torch.zeros_like(c0)

    for s in range(n_samples):
        p = torch.randn(prior.dim)

        with torch.no_grad():
            c = prior.assemble_torch(p).detach()

        # Welford online update
        delta = c - mean
        mean = mean + delta / (s + 1)
        delta2 = c - mean
        M2 = M2 + delta * delta2

        if (s + 1) % 200 == 0:
            print(f"Generated {s + 1}/{n_samples} prior samples")

    if unbiased:
        var = M2 / (n_samples - 1)
    else:
        var = M2 / n_samples

    mean_np = mean.cpu().numpy()
    var_np = var.cpu().numpy()

    print("[Prior samples]")
    print(f"  n_samples = {n_samples}")
    print(f"  mean: min={mean_np.min():.4e}, max={mean_np.max():.4e}")
    print(f"  var : min={var_np.min():.4e}, max={var_np.max():.4e}")
    
    
    mean_fun = dl.Function(prior.V)
    var_fun = dl.Function(prior.V)

    mean_fun.vector().set_local(mean_np)
    mean_fun.vector().apply("insert")

    var_fun.vector().set_local(var_np)
    var_fun.vector().apply("insert")
    
    plt.figure(figsize=(6, 5))
    im = dl.plot(mean_fun, mode="color", cmap="viridis")
    plt.colorbar(im)
    plt.title("Prior sample mean")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(6, 5))
    im = dl.plot(var_fun, mode="color", cmap="viridis")
    plt.colorbar(im)
    plt.title("Prior sample variance")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.tight_layout()
    plt.show()

    return mean_np, var_np, mean_fun, var_fun

    
    
if __name__ == '__main__':
   _test_derivative()
   
   # compute_prior_mean_variance()
    
    