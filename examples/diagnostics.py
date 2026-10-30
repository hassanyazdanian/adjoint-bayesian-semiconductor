# diagnostics.py
"""Reconstruction metrics and MCMC diagnostics used in Section 6."""
import numpy as np
import scipy.sparse as sp
import dolfin as dl
from skimage.metrics import structural_similarity

from prior import to_scipy


# ----------------------------------------------------------------------------
# Reconstruction quality
# ----------------------------------------------------------------------------
def mass_matrix(V):
    u, v = dl.TrialFunction(V), dl.TestFunction(V)
    return to_scipy(dl.assemble(u * v * dl.dx))


def relative_l2_error(M, c, c_ref):
    """RE = ||c - c_ref||_{L2(Omega)} / ||c_ref||_{L2(Omega)}, M the FE mass matrix."""
    e = np.asarray(c) - np.asarray(c_ref)
    return float(np.sqrt(e @ (M @ e) / (c_ref @ (M @ c_ref))))


def to_grid(V, c):
    """Nodal values of a CG1 field on a structured mesh as an (ny, nx) image (row 0: y = 0)."""
    coords = V.mesh().coordinates()
    order = np.lexsort((coords[:, 0], coords[:, 1]))
    nx = np.unique(coords[:, 0]).size
    return np.asarray(c)[..., dl.vertex_to_dof_map(V)[order]].reshape(*np.shape(c)[:-1], -1, nx)


def ssim(V, c, c_ref, gaussian=False):
    """
    Structural similarity of the two fields as 2-D (49 x 49 nodal) images, data range
    of c_ref. gaussian=False: scikit-image defaults (7 x 7 uniform window, sample
    covariance, K1 = 0.01, K2 = 0.03). gaussian=True: the parameters of Wang et al.
    (2004) (11 x 11 Gaussian window with sigma = 1.5, population covariance).
    """
    a, b = to_grid(V, c_ref), to_grid(V, c)
    kw = dict(gaussian_weights=True, sigma=1.5, use_sample_covariance=False) if gaussian else {}
    return float(structural_similarity(a, b, data_range=float(a.max() - a.min()), **kw))


# ----------------------------------------------------------------------------
# Field summaries (convergence diagnostics: ArviZ, see postprocess.py)
# ----------------------------------------------------------------------------
def pointwise_hpd(C, mass=0.95):
    """Pointwise shortest intervals for field samples C of shape (n_samples, ndof)."""
    C = np.sort(C, axis=0)
    m = int(np.floor(mass * C.shape[0]))
    j = np.argmin(C[m:] - C[: C.shape[0] - m], axis=0)
    cols = np.arange(C.shape[1])
    return C[j, cols], C[j + m, cols]


def field_moments(batches):
    """Pointwise mean and std (1/(N-1), Section 5.3) over an iterable of (batch, ndof) arrays."""
    n, mean, m2 = 0, 0.0, 0.0
    for C in batches:
        nb = C.shape[0]
        mb = C.mean(axis=0)
        delta = mb - mean
        m2 = m2 + ((C - mb) ** 2).sum(axis=0) + delta**2 * n * nb / (n + nb)
        mean = mean + delta * nb / (n + nb)
        n += nb
    return mean, np.sqrt(m2 / (n - 1))


# ----------------------------------------------------------------------------
# Interface depth (Section 6.4)
# ----------------------------------------------------------------------------
def point_evaluation_matrix(V, points):
    """Sparse E with (E c)_i = c_h(points[i]) for a scalar Lagrange space V."""
    mesh, element, dofmap = V.mesh(), V.element(), V.dofmap()
    tree = mesh.bounding_box_tree()
    rows, cols, vals = [], [], []
    for i, x in enumerate(np.asarray(points, dtype=float)):
        cell_id = tree.compute_first_entity_collision(dl.Point(*x))
        cell = dl.Cell(mesh, cell_id)
        phi = element.evaluate_basis_all(x, cell.get_vertex_coordinates(), cell.orientation())
        dofs = dofmap.cell_dofs(cell_id)
        rows += [i] * len(dofs)
        cols += list(dofs)
        vals += list(phi)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(points), V.dim()))


def interface_depth(V, C, x0=0.5, level=0.0, n_y=401, clip=False):
    """
    Depth d = y-coordinate of the level set {c = level} on the vertical line x = x0
    for each field sample in C (n_samples, ndof). Convention: the lowest crossing
    (smallest y) is taken. Without a crossing d is NaN, or with clip=True 0 if c > level
    on the whole line (junction below the domain) and 1 if c < level (above it).
    Also returns the number of crossings per sample (> 1: several interface points).
    """
    ys = np.linspace(0.0, 1.0, n_y)
    E = point_evaluation_matrix(V, np.column_stack([np.full(n_y, x0), ys]))
    vals = np.atleast_2d(C) @ E.T - level                  # (n_samples, n_y)
    s = np.sign(vals)
    # crossing in [y_j, y_j+1): strict sign change, or an exact zero at y_j (not continuing a zero run)
    at_zero = (s == 0) & np.concatenate([np.ones((s.shape[0], 1), bool), s[:, :-1] != 0], axis=1)
    hit = at_zero.copy()
    hit[:, :-1] |= s[:, :-1] * s[:, 1:] < 0
    n_cross = hit.sum(axis=1)
    j = np.argmax(hit, axis=1)
    rows = np.arange(s.shape[0])
    jn = np.minimum(j + 1, n_y - 1)
    v1, v2 = vals[rows, j], vals[rows, jn]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(v1 == 0, ys[j], ys[j] - v1 * (ys[jn] - ys[j]) / (v2 - v1))
    missing = np.where(vals[:, 0] > 0, 0.0, 1.0) if clip else np.nan
    return np.where(n_cross > 0, d, missing), n_cross
