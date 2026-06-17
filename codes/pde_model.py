# pde_model.py
import numpy as np
import dolfin as dl
import dolfin_adjoint as da
import matplotlib.pyplot as plt
dl.set_log_level(50)


class TopFluxPDE:
    """
    Nonlinear PDE model with parameter-dependent Dirichlet boundary conditions.

    PDE:
        -div(lam^2 grad(u)) + delta^2 (exp(u) - exp(-u)) = C   in Omega

    BCs:
        u = V_bottom on y = 0
        u = V_top    on y = Ly
        natural Neumann elsewhere

    Here V_bottom and V_top may be updated from unknown plateau values cP, cN via
        V_bottom = asinh(cP / (2 delta^2))
        V_top    = asinh(cN / (2 delta^2))
    """

    def __init__(self, N_points, Lx, V_top, V_bottom, lam, delta):
        self.Lx = float(Lx)
        self.Ly = 1.0
        self.lam = float(lam)
        self.delta = float(delta)

        Ny = int(N_points)
        Nx = max(1, int(self.Lx * Ny))

        self.mesh = dl.RectangleMesh(dl.MPI.comm_self,dl.Point(0.0, 0.0), 
                                     dl.Point(self.Lx, self.Ly),
                                     Nx, Ny)

        self.V = dl.FunctionSpace(self.mesh, "CG", 1)
        self.Q = dl.FunctionSpace(self.mesh, "DG", 0)  # cellwise constants

        # Observation / mask fields on DG0
        self.yc = da.Function(self.Q, name="y_obs_cell")
        self.qc = da.Function(self.Q, name="mask_cell")

        # State and parameter field
        self.u = da.Function(self.V, name="u")
        self.C = da.Function(self.V, name="C")

        # Store current plateau values if using parameter-dependent BCs
        self.cP_value = None
        self.cN_value = None

        v = dl.TestFunction(self.V)
        w = dl.TrialFunction(self.V)

        def top(x, on_boundary):
            return on_boundary and dl.near(x[1], self.Ly)

        def bottom(x, on_boundary):
            return on_boundary and dl.near(x[1], 0.0)

        self._top_boundary = top
        self._bottom_boundary = bottom

        # IMPORTANT:
        # Use adjoint-aware Constants so BC values can be updated before each solve.
        self.V_top = da.Constant(float(V_top), name="V_top")
        self.V_bottom = da.Constant(float(V_bottom), name="V_bottom")

        # self.bcs = [
        #     da.DirichletBC(self.V, self.V_top, top),
        #     da.DirichletBC(self.V, self.V_bottom, bottom),
        # ]

        # Residual and Jacobian
        self.F = (
            self.lam**2 * dl.inner(dl.grad(self.u), dl.grad(v)) * dl.dx
            + self.delta**2 * (dl.exp(self.u) - dl.exp(-self.u)) * v * dl.dx
            - self.C * v * dl.dx
        )
        self.Jac = dl.derivative(self.F, self.u, w)

        # Boundary markers
        self.boundaries = dl.MeshFunction("size_t", self.mesh, 
                                          self.mesh.topology().dim() - 1, 0)

        Ly = self.Ly

        class Top(dl.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and dl.near(x[1], Ly)

        self.top_id = 1
        Top().mark(self.boundaries, self.top_id)
        self.ds = dl.Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)

        # Misfit weight
        self.inv_sigma2 = da.Constant(1.0)

        # DG0 projection of u_y
        self.dudy_c = da.Function(self.Q, name="dudy_cell")

        # Boundary-flux misfit
        self.J_form = (
            0.5
            * self.inv_sigma2
            * self.qc
            * (self.dudy_c - self.yc) ** 2
            * self.ds(self.top_id)
        )

        # Top boundary dofs (nodal view, if needed elsewhere)
        coords = self.V.tabulate_dof_coordinates().reshape(-1, 2)
        self.top_dofs = np.where(np.isclose(coords[:, 1], self.Ly))[0]

        # Top boundary cells and x-midpoints
        tdim = self.mesh.topology().dim()
        self.mesh.init(tdim - 1, tdim)  # facet -> cell connectivity

        top_cells = []
        top_xmid = []

        for f in dl.facets(self.mesh):
            if self.boundaries[f] == self.top_id:
                c = f.entities(tdim)[0]
                top_cells.append(c)
                mp = f.midpoint()
                top_xmid.append(mp.x())

        self.top_cells = np.asarray(top_cells, dtype=int)
        self.top_xmid = np.asarray(top_xmid, dtype=float)

        order = np.argsort(self.top_xmid)
        self.top_cells = self.top_cells[order]
        self.top_xmid = self.top_xmid[order]

    # ------------------------------------------------------------------
    # Boundary-condition helpers
    # ------------------------------------------------------------------
    def _contact_potential(self, c_val):
        return np.arcsinh(float(c_val) / (2.0 * self.delta**2))

    def _d_contact_potential_dc(self, c_val):
        """
        Derivative of asinh(c / (2 delta^2)) with respect to c.
        """
        s = float(c_val) / (2.0 * self.delta**2)
        return 1.0 / (2.0 * self.delta**2 * np.sqrt(1.0 + s**2))

    def set_boundary_potentials(self, V_top=None, V_bottom=None):
        """
        Directly set the Dirichlet boundary values.
        """
        if V_top is not None:
            self.V_top.assign(float(V_top))
        if V_bottom is not None:
            self.V_bottom.assign(float(V_bottom))

    def set_contact_potentials_from_doping(self, cP, cN):
        """
        Update boundary values from plateau parameters cP and cN:
            V_bottom = asinh(cP / (2 delta^2))
            V_top    = asinh(cN / (2 delta^2))
        """
        self.cP_value = float(cP)
        self.cN_value = float(cN)

        self.V_bottom.assign(self._contact_potential(cP))
        self.V_top.assign(self._contact_potential(cN))

    # ------------------------------------------------------------------
    # Observation setup
    # ------------------------------------------------------------------
    def set_obs(self, y_obs, obs_idx=None):
        y_obs = np.asarray(y_obs, dtype=float)

        yc = np.zeros(self.Q.dim())
        qc = np.zeros(self.Q.dim())

        if obs_idx is None:
            # Full observation on all top boundary cells
            assert y_obs.shape[0] == self.top_cells.shape[0]
            yc[self.top_cells] = y_obs
            qc[self.top_cells] = 1.0
        else:
            # Sparse observation indexed into the top boundary cell ordering
            obs_idx = np.asarray(obs_idx, dtype=int)
            assert y_obs.shape[0] == obs_idx.shape[0]
            cells = self.top_cells[obs_idx]
            yc[cells] = y_obs
            qc[cells] = 1.0

        self.yc.vector().set_local(yc)
        self.yc.vector().apply("insert")

        self.qc.vector().set_local(qc)
        self.qc.vector().apply("insert")

    # ------------------------------------------------------------------
    # Forward / observation utilities
    # ------------------------------------------------------------------
    def top_flux_facet_vec(self, u_fun=None):
        """
        Return projected u_y on top boundary cells in the order of self.top_cells.
        """
        if u_fun is None:
            u_fun = self.u
        dudy_c = dl.project(u_fun.dx(1), self.Q)
        return dudy_c.vector().get_local()[self.top_cells]
    
    
    def _make_bcs_adjoint(self):
        return [
            da.DirichletBC(self.V, self.V_top, self._top_boundary),
            da.DirichletBC(self.V, self.V_bottom, self._bottom_boundary),
        ]
    
    def _make_bcs_plain(self):
        return [
            dl.DirichletBC(self.V, dl.Constant(float(self.V_top.values()[0])), self._top_boundary),
            dl.DirichletBC(self.V, dl.Constant(float(self.V_bottom.values()[0])), self._bottom_boundary),
        ]
    
    def _solve(self):
    
        bcs = self._make_bcs_adjoint()
        
        da.solve(
            self.F == 0,
            self.u,
            bcs,
            J=self.Jac,
            solver_parameters={
                "newton_solver": {
                    "absolute_tolerance": 1e-8,
                    "relative_tolerance": 1e-6,
                    "maximum_iterations": 40,
                    "linear_solver": "lu",
                    "report": False,
                }
            },
        )

    def forward_record(self, C_vec, sigma2, cP=None, cN=None):
        """
        Run forward once on a fresh adjoint tape and return (J_float, tape, J_adjfloat).

        Parameters
        ----------
        C_vec : array-like
            State-source / doping field values in V.
        sigma2 : float
            Noise variance.
        cP, cN : float, optional
            If given, boundary values are updated via the equilibrium relation.
        """
        tape = da.Tape()
        da.set_working_tape(tape)

        self.C.vector().set_local(np.asarray(C_vec, dtype=float))
        self.C.vector().apply("insert")

        self.inv_sigma2.assign(1.0 / float(sigma2))

        if cP is not None and cN is not None:
            self.set_contact_potentials_from_doping(cP, cN)

        self._solve()

        # Differentiable DG0 projection
        self.dudy_c.assign(da.project(self.u.dx(1), self.Q))

        J = da.assemble(self.J_form)
        return float(J), tape, J

   
    def solve_state(self, C_vec, cP=None, cN=None, zero_guess=False):
        """
        Solve the state equation without recording a tape.
        """
        self.C.vector().set_local(np.asarray(C_vec, dtype=float))
        self.C.vector().apply("insert")
    
        if cP is not None and cN is not None:
            self.cP_value = float(cP)
            self.cN_value = float(cN)
            V_bottom_val = self._contact_potential(cP)
            V_top_val = self._contact_potential(cN)
    
            # keep stored values in sync for diagnostics
            self.V_bottom.assign(V_bottom_val)
            self.V_top.assign(V_top_val)
        else:
            V_bottom_val = float(self.V_bottom.values()[0])
            V_top_val = float(self.V_top.values()[0])
    
        if zero_guess:
            self.u.vector().zero()
            self.u.vector().apply("insert")
    
        bcs_plain = [
            dl.DirichletBC(self.V, dl.Constant(V_top_val), self._top_boundary),
            dl.DirichletBC(self.V, dl.Constant(V_bottom_val), self._bottom_boundary),
        ]
    
        dl.solve(
            self.F == 0,
            self.u,
            bcs_plain,
            J=self.Jac,
            solver_parameters={
                "newton_solver": {
                    "absolute_tolerance": 1e-10,
                    "relative_tolerance": 1e-9,
                    "maximum_iterations": 40,
                    "linear_solver": "lu",
                    "report": False,
                }
            },
        )
    # ------------------------------------------------------------------
    # Tangent solve
    # ------------------------------------------------------------------
    def tangent_solve(self, dC_vec, cP=None, cN=None, dcP=0.0, dcN=0.0):
        """
        Solve the tangent equation including parameter-dependent Dirichlet BCs.

        If u depends on:
            - interior field C through dC
            - plateau values cP, cN through boundary data

        then du satisfies the linearized PDE with
            du|bottom = d/dcP [asinh(cP / (2 delta^2))] * dcP
            du|top    = d/dcN [asinh(cN / (2 delta^2))] * dcN

        Notes
        -----
        - This assumes the current Jacobian self.Jac corresponds to the already-solved state.
        - So typically call solve_state(...) first, then tangent_solve(...).
        """
        du = dl.Function(self.V)

        if cP is None:
            cP = self.cP_value
        if cN is None:
            cN = self.cN_value

        if cP is None or cN is None:
            raise ValueError(
                "cP and cN must be provided (or previously set) for tangent_solve "
                "when using parameter-dependent boundary conditions."
            )

        dV_bottom = self._d_contact_potential_dc(cP) * float(dcP)
        dV_top = self._d_contact_potential_dc(cN) * float(dcN)

        bcs_du = [
            dl.DirichletBC(self.V, dl.Constant(dV_top), self._top_boundary),
            dl.DirichletBC(self.V, dl.Constant(dV_bottom), self._bottom_boundary),
        ]

        dC = dl.Function(self.V)
        dC.vector().set_local(np.asarray(dC_vec, dtype=float))
        dC.vector().apply("insert")

        v = dl.TestFunction(self.V)
        rhs = dC * v * dl.dx

        dl.solve(self.Jac == rhs, du, bcs_du)
        return du

    # # ------------------------------------------------------------------
    # # Convenience helpers
    # # ------------------------------------------------------------------
    def get_current_boundary_values(self):
        return {
            "V_bottom": float(self.V_bottom.values()[0]),
            "V_top": float(self.V_top.values()[0]),
            "cP": self.cP_value,
            "cN": self.cN_value,
        }

    def get_state_vector(self):
        return self.u.vector().get_local().copy()

    def get_parameter_vector(self):
        return self.C.vector().get_local().copy()
    
    def observe_top_flux_tangent(self, du_fun):
        """
        Linearized observation G'(u)[du].
        Since the observation is linear in u through the DG0 projection of u_y,
        this is simply the top-cell values of the DG0 projection of du_y.
        """
        ddu_c = dl.project(du_fun.dx(1), self.Q)
        return ddu_c.vector().get_local()[self.top_cells]

    def tangent_misfit(self, du_fun):
          """
          Given a tangent state du, compute the directional derivative dJ.
          Assumes self.u is the converged state and self.dudy_c corresponds to self.u.
          """
          ddudy_c = dl.project(du_fun.dx(1), self.Q)
        
          integrand = (
              self.inv_sigma2
              * self.qc
              * (self.dudy_c - self.yc)
              * ddudy_c
              * self.ds(self.top_id)
          )
          return float(dl.assemble(integrand))
    
    def tangent_misfit_from_params(self, dC_vec=None, cP=None, cN=None, dcP=0.0, dcN=0.0):
        """
        Solve the tangent equation and return the directional derivative dJ
        for the perturbation (dC, dcP, dcN).
        """
        if dC_vec is None:
            dC_vec = np.zeros(self.V.dim(), dtype=float)
    
        du = self.tangent_solve(dC_vec=dC_vec, cP=cP, cN=cN, dcP=dcP, dcN=dcN)
        dJ = self.tangent_misfit(du)
        return dJ, du



def plot_measurement_facets(pde, idx_meas):
    fig, ax = plt.subplots(figsize=(6, 4))

    # plot mesh
    dl.plot(pde.mesh, linewidth=0.3)

    # all top facet midpoints
    ax.plot(pde.top_xmid,
            pde.Ly * np.ones_like(pde.top_xmid),
            'o', markersize=4, label='all top facets')

    # selected measurement facets
    x_meas = pde.top_xmid[idx_meas]
    y_meas = pde.Ly * np.ones_like(x_meas)

    ax.plot(x_meas, y_meas,
            'ro', markersize=7, label='measurement facets')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Selected measurement facets")
    ax.legend()
    ax.set_aspect("equal")
    plt.show()
    
def plot_measurement_facet_segments(pde, idx_meas):
    fig, ax = plt.subplots(figsize=(6, 4))
    dl.plot(pde.mesh, linewidth=0.3)

    selected_cells = set(pde.top_cells[idx_meas])

    for f in dl.facets(pde.mesh):
        if pde.boundaries[f] == pde.top_id:
            c = f.entities(pde.mesh.topology().dim())[0]
            x = f.entities(0)  # vertex ids
            if c in selected_cells:
                verts = [dl.Vertex(pde.mesh, int(i)).point().array() for i in x]
                xs = [verts[0][0], verts[1][0]]
                ys = [verts[0][1], verts[1][1]]
                ax.plot(xs, ys, 'r-', linewidth=1)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Measurement facets")
    ax.set_aspect("equal")
    plt.show()
    
if __name__=="__main__":
    
    
    mesh_N = 48
    Lx = 1
    V_top = 1
    V_bottom = -1
    lam = 1
    delta = 1
    
    pde = TopFluxPDE(
        N_points=mesh_N,
        Lx=Lx,
        V_top=V_top,
        V_bottom=V_bottom,
        lam=lam,
        delta=delta,
    )
    idx_meas = np.linspace(0, len(pde.top_xmid)-1, 16, dtype=int)
    # plot_measurement_facets(pde, idx_meas)
    plot_measurement_facet_segments(pde, idx_meas)
