# verify_gradients.py
import numpy as np
import torch
import dolfin as dl
import dolfin_adjoint as da

from pde_model import TopFluxPDE
from fenics_torch_bridge import FenicsTopFluxMisfit

torch.set_default_dtype(torch.float64)
dl.set_log_level(50)


def relerr(a, b, eps=1e-12):
    return abs(a - b) / (abs(b) + eps)


def vec_relerr(a, b, eps=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + eps)


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def make_problem():
    # -----------------------------
    # user settings: adapt as needed
    # -----------------------------
    N_points = 32
    Lx = 1.0
    lam = 1.0
    delta = 0.1

    # base plateau values
    cP = -2.0
    cN = 1.0

    V_bottom = np.arcsinh(cP / (2.0 * delta**2))
    V_top = np.arcsinh(cN / (2.0 * delta**2))

    pde = TopFluxPDE(
        N_points=N_points,
        Lx=Lx,
        V_top=V_top,
        V_bottom=V_bottom,
        lam=lam,
        delta=delta,
    )

    return pde, cP, cN


def make_test_field(pde, seed=0):
    rng = np.random.default_rng(seed)

    # smooth-ish random C in V
    C = dl.Function(pde.V)
    C_vec = rng.normal(size=pde.V.dim())
    C.vector().set_local(C_vec)
    C.vector().apply("insert")

    # optional: scale down to avoid too wild states
    C_arr = C.vector().get_local()
    C_arr = 0.2 * C_arr
    C.vector().set_local(C_arr)
    C.vector().apply("insert")

    return C.vector().get_local().copy()


def choose_measurements(pde, n_meas=16):
    # choose approximately uniform measurement facets by x-location
    x_targets = np.linspace(0.0, pde.Lx, n_meas)
    idx = []
    for xt in x_targets:
        idx.append(np.argmin(np.abs(pde.top_xmid - xt)))
    idx = np.unique(np.array(idx, dtype=int))
    return idx


def setup_observations(pde, C_vec, cP, cN, sigma2, obs_idx):
    # Solve once and generate synthetic data
    J0, _, _ = pde.forward_record(C_vec, sigma2=sigma2, cP=cP, cN=cN)
    y_true_full = pde.top_flux_facet_vec()

    y_obs = y_true_full[obs_idx].copy()
    pde.set_obs(y_obs, obs_idx=obs_idx)

    # recompute J with consistent obs; should be ~0
    J_clean, _, _ = pde.forward_record(C_vec, sigma2=sigma2, cP=cP, cN=cN)
    return y_obs, J_clean


def check_forward(pde, C_vec, cP, cN, sigma2):
    print_header("1) Forward solve / misfit sanity")

    J, _, _ = pde.forward_record(C_vec, sigma2=sigma2, cP=cP, cN=cN)
    y_pred = pde.top_flux_facet_vec()

    print("J =", J)
    print("len(y_pred) =", len(y_pred))
    print("||y_pred|| =", np.linalg.norm(y_pred))

    if not np.isfinite(J):
        print("FAIL: J is not finite")
    else:
        print("OK: J is finite")


def check_adjoint_wrt_C(pde, C_vec, cP, cN, sigma2, seed=1, eps=1e-4):
    print_header("2) Adjoint gradient wrt C vs finite difference")

    # base
    J0, tape, J_adj = pde.forward_record(C_vec, sigma2=sigma2, cP=cP, cN=cN)

    # adjoint gradient wrt C
    ctrl_C = da.Control(pde.C)
    gC_fun = da.compute_gradient(J_adj, ctrl_C, tape=tape)
    gC = gC_fun.vector().get_local()

    rng = np.random.default_rng(seed)
    dC = rng.normal(size=C_vec.shape)
    dC /= np.linalg.norm(dC)

    # FD directional derivative
    Jp, _, _ = pde.forward_record(C_vec + eps * dC, sigma2=sigma2, cP=cP, cN=cN)
    Jm, _, _ = pde.forward_record(C_vec - eps * dC, sigma2=sigma2, cP=cP, cN=cN)
    dJ_fd = (Jp - Jm) / (2.0 * eps)

    # adjoint directional derivative
    dJ_ad = np.dot(gC, dC)
    
    print("dJ_fd =", dJ_fd)
    print("dJ_ad =", dJ_ad)
    print("relerr =", relerr(dJ_fd, dJ_ad))


def check_tangent_state_bc(pde, C_vec, cP, cN, sigma2, which="cP", eps=1e-3):
    print_header(f"3) Tangent state wrt {which} vs finite-difference state")
    
    # base state
    pde.forward_record(C_vec, sigma2=sigma2, cP=cP, cN=cN)
    if which == "cP":
        du = pde.tangent_solve(
            dC_vec=np.zeros_like(C_vec),
            cP=cP, cN=cN,
            dcP=1.0, dcN=0.0,
        )
        up, _, _ = pde.forward_record(C_vec, sigma2=sigma2, cP=cP + eps, cN=cN)
        u_plus = pde.get_state_vector()
        um, _, _ = pde.forward_record(C_vec, sigma2=sigma2, cP=cP - eps, cN=cN)
        u_minus = pde.get_state_vector()

    elif which == "cN":
        du = pde.tangent_solve(
            dC_vec=np.zeros_like(C_vec),
            cP=cP, cN=cN,
            dcP=0.0, dcN=1.0,
        )
        up, _, _ = pde.forward_record(C_vec, sigma2=sigma2, cP=cP, cN=cN + eps)
        u_plus = pde.get_state_vector()
        um, _, _ = pde.forward_record(C_vec, sigma2=sigma2, cP=cP, cN=cN - eps)
        u_minus = pde.get_state_vector()
    else:
        raise ValueError("which must be cP or cN")

    du_fd = (u_plus - u_minus) / (2.0 * eps)
    du_tan = du.vector().get_local()

    print("||du_fd||   =", np.linalg.norm(du_fd))
    print("||du_tan||  =", np.linalg.norm(du_tan))
    print("state relerr =", vec_relerr(du_fd, du_tan))


def check_tangent_observation_bc(pde, C_vec, cP, cN, sigma2, which="cP", eps=1e-6):
    print_header(f"4) Tangent observation wrt {which} vs finite-difference observation")

    # base state
    pde.forward_record(C_vec, sigma2=sigma2, cP=cP, cN=cN)

    if which == "cP":
        du = pde.tangent_solve(
            dC_vec=np.zeros_like(C_vec),
            cP=cP, cN=cN,
            dcP=1.0, dcN=0.0,
        )
        dG_tan = pde.observe_top_flux_tangent(du)

        pde.forward_record(C_vec, sigma2=sigma2, cP=cP + eps, cN=cN)
        G_plus = pde.top_flux_facet_vec()

        pde.forward_record(C_vec, sigma2=sigma2, cP=cP - eps, cN=cN)
        G_minus = pde.top_flux_facet_vec()

    elif which == "cN":
        du = pde.tangent_solve(
            dC_vec=np.zeros_like(C_vec),
            cP=cP, cN=cN,
            dcP=0.0, dcN=1.0,
        )
        dG_tan = pde.observe_top_flux_tangent(du)

        pde.forward_record(C_vec, sigma2=sigma2, cP=cP, cN=cN + eps)
        G_plus = pde.top_flux_facet_vec()

        pde.forward_record(C_vec, sigma2=sigma2, cP=cP, cN=cN - eps)
        G_minus = pde.top_flux_facet_vec()
    else:
        raise ValueError("which must be cP or cN")

    dG_fd = (G_plus - G_minus) / (2.0 * eps)

    print("||dG_fd||   =", np.linalg.norm(dG_fd))
    print("||dG_tan||  =", np.linalg.norm(dG_tan))
    print("obs relerr  =", vec_relerr(dG_fd, dG_tan))


def check_tangent_misfit_bc(pde, C_vec, cP, cN, sigma2, which="cP", eps=1e-6):
    print_header(f"5) Tangent misfit wrt {which} vs finite-difference misfit")

    # IMPORTANT: base state must be current and dudy_c consistent
    J0, _, _ = pde.forward_record(C_vec, sigma2=sigma2, cP=cP, cN=cN)

    if which == "cP":
        dJ_tan, du = pde.tangent_misfit_from_params(
            dC_vec=np.zeros_like(C_vec),
            cP=cP, cN=cN,
            dcP=1.0, dcN=0.0,
        )
        Jp, _, _ = pde.forward_record(C_vec, sigma2=sigma2, cP=cP + eps, cN=cN)
        Jm, _, _ = pde.forward_record(C_vec, sigma2=sigma2, cP=cP - eps, cN=cN)

    elif which == "cN":
        dJ_tan, du = pde.tangent_misfit_from_params(
            dC_vec=np.zeros_like(C_vec),
            cP=cP, cN=cN,
            dcP=0.0, dcN=1.0,
        )
        Jp, _, _ = pde.forward_record(C_vec, sigma2=sigma2, cP=cP, cN=cN + eps)
        Jm, _, _ = pde.forward_record(C_vec, sigma2=sigma2, cP=cP, cN=cN - eps)
    else:
        raise ValueError("which must be cP or cN")

    dJ_fd = (Jp - Jm) / (2.0 * eps)

    print("dJ_fd  =", dJ_fd)
    print("dJ_tan =", dJ_tan)
    print("relerr =", relerr(dJ_fd, dJ_tan))


def check_torch_backward(pde, C_vec, cP, cN, sigma2, eps=1e-4):
    print_header("6) Torch backward sanity check")

    C_torch = torch.tensor(C_vec, requires_grad=True, dtype=torch.float64)
    cP_torch = torch.tensor(cP, requires_grad=True, dtype=torch.float64)
    cN_torch = torch.tensor(cN, requires_grad=True, dtype=torch.float64)

    J_torch = FenicsTopFluxMisfit.apply(pde, C_torch, sigma2, cP_torch, cN_torch)
    J_torch.backward()

    gC_autograd = C_torch.grad.detach().cpu().numpy().copy()
    gcP_autograd = float(cP_torch.grad.item())
    gcN_autograd = float(cN_torch.grad.item())

    print("||grad_C|| =", np.linalg.norm(gC_autograd))
    print("grad_cP   =", gcP_autograd)
    print("grad_cN   =", gcN_autograd)

    # FD scalar checks
    Jp, _, _ = pde.forward_record(C_vec, sigma2=sigma2, cP=cP + eps, cN=cN)
    Jm, _, _ = pde.forward_record(C_vec, sigma2=sigma2, cP=cP - eps, cN=cN)
    gcP_fd = (Jp - Jm) / (2.0 * eps)

    Jp, _, _ = pde.forward_record(C_vec, sigma2=sigma2, cP=cP, cN=cN + eps)
    Jm, _, _ = pde.forward_record(C_vec, sigma2=sigma2, cP=cP, cN=cN - eps)
    gcN_fd = (Jp - Jm) / (2.0 * eps)

    print("\nScalar checks:")
    print("gcP_fd       =", gcP_fd)
    print("gcP_autograd =", gcP_autograd)
    print("gcP relerr   =", relerr(gcP_fd, gcP_autograd))

    print("gcN_fd       =", gcN_fd)
    print("gcN_autograd =", gcN_autograd)
    print("gcN relerr   =", relerr(gcN_fd, gcN_autograd))

    # one random directional FD check wrt C
    rng = np.random.default_rng(123)
    dC = rng.normal(size=C_vec.shape)
    dC /= np.linalg.norm(dC)

    Jp, _, _ = pde.forward_record(C_vec + eps * dC, sigma2=sigma2, cP=cP, cN=cN)
    Jm, _, _ = pde.forward_record(C_vec - eps * dC, sigma2=sigma2, cP=cP, cN=cN)
    dJ_fd = (Jp - Jm) / (2.0 * eps)

    dJ_ad = np.dot(gC_autograd, dC)

    print("\nField directional check:")
    print("dJ_fd =", dJ_fd)
    print("dJ_ad =", dJ_ad)
    print("relerr =", relerr(dJ_fd, dJ_ad))


if __name__ == "__main__":
    pde, cP, cN = make_problem()

    sigma2 = 1e-4
    C_vec = make_test_field(pde, seed=0)

    obs_idx = choose_measurements(pde, n_meas=16)
    y_obs, J_clean = setup_observations(pde, C_vec, cP, cN, sigma2, obs_idx)

    print_header("0) Observation setup")
    print("n_top facets =", len(pde.top_cells))
    print("n_meas       =", len(obs_idx))
    print("obs_idx      =", obs_idx)
    print("clean J      =", J_clean, "(should be close to 0)")
    

    check_forward(pde, C_vec, cP, cN, sigma2)
    check_adjoint_wrt_C(pde, C_vec, cP, cN, sigma2)
    check_tangent_state_bc(pde, C_vec, cP, cN, sigma2, which="cP")
    check_tangent_state_bc(pde, C_vec, cP, cN, sigma2, which="cN")
    check_tangent_observation_bc(pde, C_vec, cP, cN, sigma2, which="cP")
    check_tangent_observation_bc(pde, C_vec, cP, cN, sigma2, which="cN")
    check_tangent_misfit_bc(pde, C_vec, cP, cN, sigma2, which="cP")
    check_tangent_misfit_bc(pde, C_vec, cP, cN, sigma2, which="cN")
    check_torch_backward(pde, C_vec, cP, cN, sigma2)