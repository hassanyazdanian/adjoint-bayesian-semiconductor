# forward_model.py
"""
Finite element model of the Poisson-Boltzmann problem (2.2), the discrete
observation operator (3.7) and the adjoint gradient of Section 5.1.
"""
import contextlib
import time
from collections import namedtuple

import numpy as np
import scipy.sparse as sp
import dolfin as dl
import dolfin_adjoint as da
import pyadjoint
import ufl

from config import ELECTRODES

dl.set_log_level(50)

# Tape, controls (c, f_p, f_n) and state of one recorded solve.
SolveRecord = namedtuple("SolveRecord", ["tape", "controls", "u"])

BOTTOM, TOP = 2, 3   # boundary ids of dOmega_2 (p-contact) and dOmega_3 (n-contact)

NEWTON = {
    "newton_solver": {
        "absolute_tolerance": 1e-10,
        "relative_tolerance": 1e-9,
        "maximum_iterations": 50,
        "linear_solver": "lu",
        "report": False,
    }
}


class PoissonBoltzmannModel:
    """
    -div(eps grad u) + 2 delta^2 sinh(u) = c          in Omega = (0, 1)^2,
    grad u . n = 0                                     on dOmega_1 (x = 0, 1),
    u = f_p = asinh(c_p / (2 delta^2))                 on dOmega_2 (y = 0),
    u = f_n = asinh(c_n / (2 delta^2))                 on dOmega_3 (y = 1),

    observed through the electrode fluxes [O(u)]_l = int_{I_l} du/dy ds  (2.3),
    where the electrodes are segments I_l = [a_l, b_l] of y = 1 whose end points lie
    on mesh vertices. The doping c is a CG1 dof vector; u is discretised with CG1.

    `stats` counts the PDE work: nonlinear forward solves, Newton iterations (one
    Jacobian solve each), adjoint solves, and the wall-clock time spent in them.
    """

    def __init__(self, n=48, eps=1.0, delta=1.0, electrodes=ELECTRODES):
        self.n = int(n)
        self.eps = float(eps)
        self.delta = float(delta)
        self.electrodes = tuple((float(a), float(b)) for a, b in electrodes)
        self._check_electrodes()

        self.mesh = dl.UnitSquareMesh(dl.MPI.comm_self, self.n, self.n)
        self.V = dl.FunctionSpace(self.mesh, "CG", 1)
        self._u0 = np.zeros(self.V.dim())          # Newton initial guess (last solution)
        self._v = dl.TestFunction(self.V)

        self._boundaries = dl.MeshFunction("size_t", self.mesh, 1, 0)
        dl.CompiledSubDomain("on_boundary && near(x[1], 0.0)").mark(self._boundaries, BOTTOM)
        dl.CompiledSubDomain("on_boundary && near(x[1], 1.0)").mark(self._boundaries, TOP)

        self._build_observation_operator()
        self.reset_stats()

    def reset_stats(self):
        # forward: successful logical evaluations; forward_attempts: Newton solves started (a failed warm-started
        # solve is retried once from u = 0); forward_failures: attempts that raised; forward_restarts: retries;
        # forward_failed_evaluations: evaluations whose both attempts failed. newton_iterations counts the
        # iterations of converged solves only (a diverging solve does not report its count). Times include
        # failed attempts.
        self.stats = dict(forward=0, newton_iterations=0, adjoint=0, forward_time=0.0, adjoint_time=0.0,
                          forward_attempts=0, forward_failures=0, forward_restarts=0, forward_failed_evaluations=0,
                          adjoint_failures=0)

    def _check_electrodes(self):
        """Electrodes must be disjoint segments of positive length with end points on mesh vertices."""
        tol = 1e-9
        ab = np.array(self.electrodes)
        if np.any(ab[:, 0] < -tol) or np.any(ab[:, 1] > 1 + tol) or np.any(ab[:, 1] - ab[:, 0] <= tol):
            raise ValueError("electrodes must satisfy 0 <= a < b <= 1")
        if np.any(np.abs(ab * self.n - np.round(ab * self.n)) > tol * self.n):
            raise ValueError(f"electrode end points must lie on vertices of the {self.n} x {self.n} mesh")
        s = ab[np.argsort(ab[:, 0])]
        if np.any(s[1:, 0] < s[:-1, 1] - tol):
            raise ValueError("electrodes must not overlap")

    # ------------------------------------------------------------------
    # Observation operator (3.7)
    # ------------------------------------------------------------------
    def _build_observation_operator(self):
        """Sparse B with B[l, j] = int_{I_l} d(phi_j)/dy ds, so that O(u) = B u."""
        markers = dl.MeshFunction("size_t", self.mesh, 1, 0)
        covered = np.zeros(len(self.electrodes))
        xy = self.mesh.coordinates()
        for f in dl.facets(self.mesh):
            if self._boundaries[f] == TOP:
                for i, (a, b) in enumerate(self.electrodes):
                    if a < f.midpoint().x() < b:
                        markers[f] = i + 1
                        v0, v1 = f.entities(0)
                        covered[i] += np.linalg.norm(xy[v0] - xy[v1])
        lengths = np.array([b - a for a, b in self.electrodes])
        if not np.allclose(covered, lengths, rtol=0, atol=1e-12):
            raise ValueError("electrode end points must lie on mesh vertices of y = 1")
        ds = dl.Measure("ds", domain=self.mesh, subdomain_data=markers)

        rows = [dl.assemble(self._v.dx(1) * ds(i + 1)).get_local() for i in range(len(self.electrodes))]
        self.B = sp.csr_matrix(np.array(rows))
        self.electrode_x = np.array([0.5 * (a + b) for a, b in self.electrodes])

    @property
    def n_meas(self):
        return self.B.shape[0]

    def observe(self, u):
        return self.B @ u

    # ------------------------------------------------------------------
    # Forward solve
    # ------------------------------------------------------------------
    def dirichlet_data(self, cp, cn):
        """(f_p, f_n) = (asinh(c_p / 2 delta^2), asinh(c_n / 2 delta^2)), see (2.2)."""
        s = 2.0 * self.delta**2
        return np.arcsinh(cp / s), np.arcsinh(cn / s)

    def _residual(self, u, c):
        v = self._v
        return (
            self.eps * dl.inner(dl.grad(u), dl.grad(v)) * dl.dx
            + 2.0 * self.delta**2 * ufl.sinh(u) * v * dl.dx
            - c * v * dl.dx
        )

    def solve(self, c, f_p, f_n):
        """Newton solve of (3.3) for the doping dof vector c and Dirichlet data (f_p, f_n); returns u."""
        return self._solve_with_restart(c, f_p, f_n, annotate=False)[0]

    def solve_recorded(self, c, f_p, f_n):
        """As solve(), but recorded on its own dolfin-adjoint tape; returns (u, record) for adjoint_gradient()."""
        return self._solve_with_restart(c, f_p, f_n, annotate=True)

    def _solve_with_restart(self, c, f_p, f_n, annotate):
        """One logical forward evaluation: a warm-started Newton solve, restarted once from u = 0 if it fails."""
        t0 = time.perf_counter()
        try:
            self.stats["forward_attempts"] += 1
            try:
                out = self._solve(c, f_p, f_n, annotate)
            except RuntimeError:
                # Newton diverged from the warm start: restart from u = 0.
                self.stats["forward_failures"] += 1
                self.stats["forward_restarts"] += 1
                self._u0[:] = 0.0
                self.stats["forward_attempts"] += 1
                try:
                    out = self._solve(c, f_p, f_n, annotate)
                except RuntimeError:
                    self.stats["forward_failures"] += 1
                    self.stats["forward_failed_evaluations"] += 1
                    raise
            self.stats["forward"] += 1
            return out
        finally:
            self.stats["forward_time"] += time.perf_counter() - t0

    def _solve(self, c, f_p, f_n, annotate):
        if annotate:
            tape = da.Tape()
            da.set_working_tape(tape)
        with contextlib.nullcontext() if annotate else pyadjoint.stop_annotating():
            u = da.Function(self.V)                       # a fresh state for every solve
            u.vector().set_local(self._u0)                # warm start from the previous solution
            u.vector().apply("insert")
            c_fun = da.Function(self.V)
            c_fun.vector().set_local(np.asarray(c, dtype=float))
            c_fun.vector().apply("insert")
            g_p, g_n = da.Constant(float(f_p)), da.Constant(float(f_n))
            bcs = [
                da.DirichletBC(self.V, g_p, self._boundaries, BOTTOM),
                da.DirichletBC(self.V, g_n, self._boundaries, TOP),
            ]
            F = self._residual(u, c_fun)
            solver = da.NonlinearVariationalSolver(
                da.NonlinearVariationalProblem(F, u, bcs, J=dl.derivative(F, u)))
            solver.parameters.update(NEWTON)
            n_iterations, _ = solver.solve()
        self.stats["newton_iterations"] += n_iterations

        self._u0 = u.vector().get_local()
        record = SolveRecord(tape, [da.Control(c_fun), da.Control(g_p), da.Control(g_n)], u) if annotate else None
        return self._u0.copy(), record

    def forward(self, c, cp, cn):
        """Discrete forward map G^h(c) = O^h(W^h(c)) with Dirichlet data from (c_p, c_n)."""
        return self.observe(self.solve(c, *self.dirichlet_data(cp, cn)))

    # ------------------------------------------------------------------
    # Adjoint gradient (Section 5.1)
    # ------------------------------------------------------------------
    def adjoint_gradient(self, record, dJdu):
        """
        Gradient of a functional J(u) with respect to (c, f_p, f_n) at a recorded
        solve, given dJ/du. One adjoint solve (5.6); dolfin-adjoint evaluates the
        adjoint identity (5.9), including the Dirichlet lifting.
        """
        t0 = time.perf_counter()
        try:
            adj = record.u.vector().copy()
            adj.set_local(np.asarray(dJdu, dtype=float))
            adj.apply("insert")
            g_c, g_p, g_n = da.compute_gradient(record.u, record.controls, tape=record.tape, adj_value=adj)
        except Exception:
            self.stats["adjoint_failures"] += 1
            raise
        finally:
            self.stats["adjoint_time"] += time.perf_counter() - t0
        self.stats["adjoint"] += 1
        return g_c.vector().get_local(), float(g_p), float(g_n)
