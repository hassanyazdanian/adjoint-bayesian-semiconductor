# prior.py
"""
Truncated Whittle-Matern KL prior (4.3)-(4.5) and the pushforward maps (4.6)-(4.7).
"""
import numpy as np
import scipy.linalg
import scipy.sparse as sp
import dolfin as dl
import torch

torch.set_default_dtype(torch.float64)


def _operator_matrices(V, tau):
    """FE matrices of (tau I - Laplace) with natural (homogeneous Neumann) BCs, and the mass matrix."""
    u, v = dl.TrialFunction(V), dl.TestFunction(V)
    A = dl.assemble(tau * u * v * dl.dx + dl.inner(dl.grad(u), dl.grad(v)) * dl.dx)
    M = dl.assemble(u * v * dl.dx)
    return A, M


def to_scipy(A):
    indptr, indices, data = dl.as_backend_type(A).mat().getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=(A.size(0), A.size(1)))


def trend_mean(y, a):
    """
    Prior mean m(y) = a (2y - 1) of the latent field: m = -a at the p-contact (y = 0) and
    +a at the n-contact (y = 1), so prior draws have c = c_p and c = c_n at the contacts assumed
    by the Dirichlet data (2.2), whatever the plateau values. a = 0 gives a zero mean.
    """
    return a * (2.0 * y - 1.0)


class MaternKLPrior:
    """
    X = R_N(z) = m + sum_{j <= N_KL} sqrt(lambda_j) z_j e_j,   z ~ N(0, I),      (4.3)

    with (gamma_j, e_j) the smallest generalized FE eigenpairs of (tau I - Laplace)
    (L2-normalised e_j) and lambda_j = s gamma_j^(-alpha)                          (4.5)

    The constant s is fixed by sum_j lambda_j = 1, i.e. E ||X - m||^2_{L2} = 1 (centred
    variance), so that the fluctuation has O(1) amplitude and k in (4.7) sets the
    interface width. With the trend mean, E ||X||^2_{L2} = 1 + ||m||^2_{L2} = 1 + a^2/3.
    (Without it, sqrt(lambda_1) ~ 5e-4 and k = 20 would give a nearly constant c.)

    m is the trend mean a (2y - 1) (mean_a = 0: zero mean, as in the original code).

    legacy_basis=True reproduces the basis of the original code, only to read chains
    generated with it. That basis is NOT M-orthonormal: e_2 and e_3 (gamma_2 = gamma_3)
    are both ~ cos(pi y), so cos(pi x) is missing from the prior.
    """

    def __init__(self, V, n_kl=30, tau=0.15**-2, alpha=4.0, k=20.0, mean_a=0.0, legacy_basis=False):
        self.V = V
        self.n_kl = int(n_kl)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.k = float(k)
        self.mean_a = float(mean_a)
        self.legacy_basis = bool(legacy_basis)
        y = V.tabulate_dof_coordinates()[:, 1]
        self.mean = torch.from_numpy(trend_mean(y, self.mean_a))

        gamma, E = self._eigenpairs()
        lam = gamma ** (-self.alpha)
        lam /= lam.sum()

        self.gamma = gamma
        self.lam = lam
        # Columns sqrt(lambda_j) e_j, so that R_N(z) = basis @ z.
        self.basis = torch.from_numpy(E * np.sqrt(lam))

    def _eigenpairs(self):
        A, M = _operator_matrices(self.V, self.tau)
        solver = dl.SLEPcEigenSolver(dl.as_backend_type(A), dl.as_backend_type(M))
        if self.legacy_basis:
            solver.parameters["spectrum"] = "smallest magnitude"
        else:
            # Symmetric-definite pencil: SLEPc then returns M-orthonormal eigenvectors,
            # also inside degenerate eigenspaces; shift-and-invert targets the smallest gamma.
            solver.parameters["problem_type"] = "gen_hermitian"
            solver.parameters["spectrum"] = "target magnitude"
            solver.parameters["spectral_transform"] = "shift-and-invert"
            solver.parameters["spectral_shift"] = 0.0
        solver.solve(self.n_kl)
        if solver.get_number_converged() < self.n_kl:
            raise RuntimeError("SLEPc did not converge all requested eigenpairs")

        gamma = np.empty(self.n_kl)
        E = np.empty((self.V.dim(), self.n_kl))
        for j in range(self.n_kl):
            gamma[j], _, e, _ = solver.get_eigenpair(j)
            E[:, j] = e.get_local()
        Ms = to_scipy(M)
        E /= np.sqrt(np.einsum("ij,ij->j", E, Ms @ E))   # ||e_j||_{L2} = 1
        if not self.legacy_basis:
            As = to_scipy(A)
            orth = np.abs(E.T @ (Ms @ E) - np.eye(self.n_kl)).max()
            res = np.max(np.linalg.norm(As @ E - (Ms @ E) * gamma, axis=0) / np.linalg.norm(As @ E, axis=0))
            if orth > 1e-6 or res > 1e-6:
                raise RuntimeError(f"inaccurate KL basis: max |E^T M E - I| = {orth:.1e}, "
                                   f"max relative residual = {res:.1e}")
        return gamma, E

    def latent(self, z):
        """R_N(z) = m + basis @ z for z of shape (N_KL,) or (batch, N_KL)."""
        return self.mean + torch.as_tensor(z) @ self.basis.T

    def pushforward(self, z, cp, cn):
        """Sigmoid pushforward F_2 (4.7): c = c_p + (c_n - c_p) / (1 + exp(-k X))."""
        return cp + (cn - cp) * torch.sigmoid(self.k * self.latent(z))

    def pushforward_heaviside(self, z, cp, cn):
        """Heaviside pushforward F_1 (4.6)."""
        X = self.latent(z)
        return cp + (cn - cp) * (X >= 0).to(X.dtype)


def kl_variance_fraction(V, tau, alpha):
    """Cumulative fraction of the FE prior variance captured by the leading KL modes."""
    A, M = _operator_matrices(V, tau)
    gamma = scipy.linalg.eigh(A.array(), M.array(), eigvals_only=True)
    lam = gamma ** (-alpha)
    return np.cumsum(lam) / lam.sum()
