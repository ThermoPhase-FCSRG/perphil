# %% [markdown]
# # 2D DPP conforming Galerkin FEM

# %%
import os
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"

import firedrake as fd

# import petsc4py
# import numpy as np
# from scipy.sparse import csr_matrix
# from scipy.linalg import svd
# from scipy.sparse.linalg import svds
import logging

from perphil.forms.spaces import create_function_spaces
from perphil.mesh.builtin import create_mesh
from perphil.models.dpp.parameters import DPPParameters
from perphil.solvers.solver import (
    solve_dpp,
    solve_dpp_nonlinear,
    solve_dpp_picard,
    logger,
)
from perphil.solvers.parameters import (
    LINEAR_SOLVER_PARAMS,
    GMRES_PARAMS,
    FIELDSPLIT_LU_PARAMS,
    GMRES_JACOBI_PARAMS,
    RICHARDSON_SOLVER_PARAMS,
)
from perphil.utils.plotting import plot_2d_mesh, plot_scalar_field, plot_vector_field
from perphil.utils.manufactured_solutions import interpolate_exact
from perphil.utils.postprocessing import (
    split_dpp_solution,
    calculate_darcy_velocity_from_pressure,
    slice_along_x,
)

# %% [markdown]
# For convenience, we define the operators from Firedrake:

# %%
grad = fd.grad
div = fd.div
dx = fd.dx
inner = fd.inner
pi = fd.pi
sin = fd.sin
exp = fd.exp
cos = fd.cos

# %% [markdown]
# ## Case 1

# %% [markdown]
# ### Mesh

# %%
mesh = create_mesh(20, 20, quadrilateral=True)

# %%
plot_2d_mesh(mesh)

# %% [markdown]
# ### Exact solutions

# %%
U, V = create_function_spaces(
    mesh,
    velocity_deg=1,
    pressure_deg=1,
    velocity_family="CG",
    pressure_family="CG",
)

dpp_params = DPPParameters(k1=1.0, k2=1 / 1e2, beta=1.0, mu=1)
u1_exact, p1_exact, u2_exact, p2_exact = interpolate_exact(mesh, U, V, dpp_params)

# %%
plot_scalar_field(p1_exact)
plot_scalar_field(p2_exact)
plot_vector_field(u1_exact)
plot_vector_field(u2_exact)

# %% [markdown]
# ### Conforming Galerkin FEM approximations

# %% [markdown]
# #### Monolithic (fully coupled) approximation

# %%
W = V * V  # Mixed function space with both scales

# Dirichlet BCs
bc_macro = fd.DirichletBC(W.sub(0), p1_exact, "on_boundary")
bc_micro = fd.DirichletBC(W.sub(1), p2_exact, "on_boundary")
bcs = [bc_macro, bc_micro]

solver_parameters = LINEAR_SOLVER_PARAMS
solution_monolithic = solve_dpp(W, dpp_params, bcs, solver_parameters=solver_parameters)
p1_monolithic, p2_monolithic = split_dpp_solution(solution_monolithic)

u1_monolithic = calculate_darcy_velocity_from_pressure(p1_monolithic, dpp_params.k1)

u2_monolithic = calculate_darcy_velocity_from_pressure(p2_monolithic, dpp_params.k2)

# %%
plot_scalar_field(p1_monolithic, title=r"$p_1$ scalar field")
plot_scalar_field(p2_monolithic, title=r"$p_2$ scalar field")
plot_vector_field(u1_monolithic, title=r"$u_1$ vector field")
plot_vector_field(u2_monolithic, title=r"$u_2$ vector field")

# %%
x_mid_point = 0.5
y_points, p1_mono_at_x_mid_point = slice_along_x(p1_monolithic, x_value=x_mid_point)
_, p1_exact_at_x_mid_point = slice_along_x(p1_exact, x_value=x_mid_point)
_, p2_mono_at_x_mid_point = slice_along_x(p2_monolithic, x_value=x_mid_point)
_, p2_exact_at_x_mid_point = slice_along_x(p2_exact, x_value=x_mid_point)

y_points, p1_mono_at_x_mid_point, p2_mono_at_x_mid_point

# %%
figsize = (7, 7)
plt.figure(figsize=figsize)
plt.plot(
    y_points, p1_mono_at_x_mid_point, "x", ms=10, lw=4, c="k", label="Monolithic LU"
)
plt.plot(y_points, p1_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Macro Pressure $(p_{1,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

plt.figure(figsize=figsize)
plt.plot(
    y_points, p2_mono_at_x_mid_point, "x", ms=10, lw=4, c="k", label="Monolithic LU"
)
plt.plot(y_points, p2_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Micro Pressure $(p_{2,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

# %% [markdown]
# #### Scale-splitting

# %% [markdown]
# Pre-conditioner by scale:

# %%
solver_monitoring_param = {
    "ksp_monitor": None,
}
solver_parameters = {**GMRES_PARAMS, **FIELDSPLIT_LU_PARAMS, **solver_monitoring_param}
solution_preconditioned = solve_dpp(
    W, dpp_params, bcs, solver_parameters=solver_parameters
)
p1_preconditioned, p2_preconditioned = split_dpp_solution(solution_preconditioned)

u1_preconditioned = calculate_darcy_velocity_from_pressure(
    p1_preconditioned, dpp_params.k1
)

u2_preconditioned = calculate_darcy_velocity_from_pressure(
    p2_preconditioned, dpp_params.k2
)

# %%
solver_monitoring_param = {
    "ksp_monitor": None,
}
solver_parameters = {**GMRES_JACOBI_PARAMS, **solver_monitoring_param}
solution_gmres_jacobi = solve_dpp(
    W, dpp_params, bcs, solver_parameters=solver_parameters
)
p1_gmres_jacobi, p2_gmres_jacobi = split_dpp_solution(solution_gmres_jacobi)

u1_gmres_jacobi = calculate_darcy_velocity_from_pressure(p1_gmres_jacobi, dpp_params.k1)

u2_gmres_jacobi = calculate_darcy_velocity_from_pressure(p2_gmres_jacobi, dpp_params.k2)

# %%
y_points, p1_pc_at_x_mid_point = slice_along_x(p1_preconditioned, x_value=x_mid_point)
_, p2_pc_at_x_mid_point = slice_along_x(p2_preconditioned, x_value=x_mid_point)

y_points, p1_pc_at_x_mid_point, p2_pc_at_x_mid_point

# %%
figsize = (7, 7)
plt.figure(figsize=figsize)
plt.plot(y_points, p1_pc_at_x_mid_point, "x", ms=10, lw=4, c="k", label="Fieldsplit PC")
plt.plot(y_points, p1_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Macro Pressure $(p_{1,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

plt.figure(figsize=figsize)
plt.plot(y_points, p2_pc_at_x_mid_point, "x", ms=10, lw=4, c="k", label="Fieldsplit PC")
plt.plot(y_points, p2_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Micro Pressure $(p_{2,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

# %% [markdown]
# PETSc's Richardson iterations:

# %%
solver_monitoring_param = {
    "snes_monitor": None,
}
solver_parameters = {**RICHARDSON_SOLVER_PARAMS, **solver_monitoring_param}
solution_richardson = solve_dpp_nonlinear(
    W, dpp_params, bcs, solver_parameters=solver_parameters
)
p1_richardson, p2_richardson = split_dpp_solution(solution_richardson)

u1_richardson = calculate_darcy_velocity_from_pressure(p1_richardson, dpp_params.k1)

u2_richardson = calculate_darcy_velocity_from_pressure(p2_richardson, dpp_params.k2)

# %%
y_points, p1_richardson_at_x_mid_point = slice_along_x(
    p1_richardson, x_value=x_mid_point
)
_, p2_richardson_at_x_mid_point = slice_along_x(p2_richardson, x_value=x_mid_point)

y_points, p1_richardson_at_x_mid_point, p2_richardson_at_x_mid_point

# %%
figsize = (7, 7)
plt.figure(figsize=figsize)
plt.plot(
    y_points, p1_richardson_at_x_mid_point, "x", ms=10, lw=4, c="k", label="Richardson"
)
plt.plot(y_points, p1_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Macro Pressure $(p_{1,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

plt.figure(figsize=figsize)
plt.plot(y_points, p2_pc_at_x_mid_point, "x", ms=10, lw=4, c="k", label="Richardson")
plt.plot(y_points, p2_richardson_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Micro Pressure $(p_{2,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

# %% [markdown]
# Loop-based Picard fixed-point iterations:

# %%
# Set the logger in Picard loop-based to INFO level.
# This way, iterations are displayed in cell outputs.
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
logger.addHandler(handler)

# %%
solver_parameters = {
    **LINEAR_SOLVER_PARAMS,
}
bc_macro = fd.DirichletBC(V, p1_exact, "on_boundary")
bc_micro = fd.DirichletBC(V, p2_exact, "on_boundary")
p1_picard, p2_picard = solve_dpp_picard(
    V,
    V,
    dpp_params,
    bcs_macro=[bc_macro],
    bcs_micro=[bc_micro],
    macro_solver_parameters=solver_parameters,
    micro_solver_parameters=solver_parameters,
)

u1_picard = calculate_darcy_velocity_from_pressure(p1_picard, dpp_params.k1)

u2_picard = calculate_darcy_velocity_from_pressure(p2_picard, dpp_params.k2)

# %%
y_points, p1_picard_at_x_mid_point = slice_along_x(p1_picard, x_value=x_mid_point)
_, p2_picard_at_x_mid_point = slice_along_x(p2_picard, x_value=x_mid_point)

y_points, p1_picard_at_x_mid_point, p2_picard_at_x_mid_point

# %%
figsize = (7, 7)
plt.figure(figsize=figsize)
plt.plot(
    y_points,
    p1_picard_at_x_mid_point,
    "x",
    ms=10,
    lw=4,
    c="k",
    label="Loop-based Picard",
)
plt.plot(y_points, p1_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Macro Pressure $(p_{1,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

plt.figure(figsize=figsize)
plt.plot(
    y_points, p2_pc_at_x_mid_point, "x", ms=10, lw=4, c="k", label="Loop-based Picard"
)
plt.plot(y_points, p2_picard_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Micro Pressure $(p_{2,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

# %% [markdown]
# #### Conditioning Analysis

# %% [markdown]
# ##### Monolithic system

# %% [markdown]
# Define the variational form:

# %%
# # Approximation degree
# degree = 1

# # Function space declaration
# pressure_family = "CG"
# velocity_family = "CG"
# U = fd.VectorFunctionSpace(mesh, velocity_family, degree)
# V = fd.FunctionSpace(mesh, pressure_family, degree)
# W = V * V

# # Trial and test functions
# dpp_fields = fd.Function(W)
# p1, p2 = fd.TrialFunctions(W)
# q1, q2 = fd.TestFunctions(W)

# # Forcing function
# f = fd.Constant(0.0)

# # Dirichlet BCs
# bc_macro = fd.DirichletBC(W.sub(0), p1_exact, "on_boundary")
# bc_micro = fd.DirichletBC(W.sub(1), p2_exact, "on_boundary")
# bcs = [bc_macro, bc_micro]

# # Variational form
# ## Mass transfer term
# xi = -beta / mu * (p1 - p2)

# ## Macro terms
# a = (k1 / mu) * inner(grad(p1), grad(q1)) * dx - xi * q1 * dx
# L = f * q1 * dx

# ## Micro terms
# a += (k2 / mu) * inner(grad(p2), grad(q2)) * dx + xi * q2 * dx
# L += f * q2 * dx

# # Isolate LHS
# F = a - L
# a_form = fd.lhs(F)

# %% [markdown]
# Assemble and get the associated matrix:

# %%
# A = fd.assemble(a_form, bcs=bcs, mat_type="aij")
# petsc_mat = A.M.handle
# is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
# size = petsc_mat.getSize()
# Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)

# %% [markdown]
# Get DoF number and clean matrix:

# %%
# Mnp.eliminate_zeros()
# nnz = Mnp.nnz
# number_of_dofs = W.dim()
# num_of_factors = int(number_of_dofs) - 1

# print(f"Number of Degrees of Freedom: {number_of_dofs}")
# print(f"Number of non-zero entries: {nnz}")
# print(f"Is operator symmetric? {is_symmetric}")
# print(f"Number of factors to compute in SVD: {num_of_factors}")

# %% [markdown]
# Convenient function to calculate spectral Condition Number using `scipy`:

# %%
# def calculate_condition_number(
#     A: petsc4py.PETSc.Mat,
#     num_of_factors: int,
#     use_sparse: bool = False,
#     zero_tol: float = 1e-5,
# ) -> float | np.float64:
#     size = A.getSize()
#     Mnp = csr_matrix(A.getValuesCSR()[::-1], shape=size)
#     Mnp.eliminate_zeros()

#     if use_sparse:
#         singular_values = svds(
#             A=Mnp,
#             k=num_of_factors,
#             which="LM",
#             maxiter=5000,
#             return_singular_vectors=False,
#             solver="lobpcg",
#         )
#     else:
#         M = Mnp.toarray()
#         singular_values = svd(M, compute_uv=False, check_finite=False)

#     singular_values = singular_values[singular_values > zero_tol]

#     condition_number = singular_values.max() / singular_values.min()

#     return condition_number

# %% [markdown]
# Condition Number for the monolithic (all scales) matrix system:

# %%
# monolithic_system_condition_number = calculate_condition_number(
#     A=petsc_mat, num_of_factors=num_of_factors
# )

# print(f"Monolithic system Condition Number: {monolithic_system_condition_number}")

# %% [markdown]
# ##### Scale-splitting

# %%
# # Approximation degree
# degree = 1

# # Function space declaration
# pressure_family = "CG"
# velocity_family = "CG"
# U = fd.VectorFunctionSpace(mesh, velocity_family, degree)
# V = fd.FunctionSpace(mesh, pressure_family, degree)

# # Trial and test functions
# p1 = fd.TrialFunction(V)
# p2 = fd.TrialFunction(V)
# q1 = fd.TestFunction(V)
# q2 = fd.TestFunction(V)

# # Forcing function
# f = fd.Constant(0.0)

# # Dirichlet BCs
# bc_macro = fd.DirichletBC(V, p1_exact, "on_boundary")
# bc_micro = fd.DirichletBC(V, p2_exact, "on_boundary")

# # Staggered pressures
# p1_old = fd.interpolate(fd.Constant(0), V)
# p2_old = fd.interpolate(fd.Constant(0), V)

# # Variational form
# ## Mass transfer term
# xi_macro = -beta / mu * (p1 - p2_old)
# xi_micro = -beta / mu * (p1_old - p2)

# ## Macro terms
# a_1 = (k1 / mu) * inner(grad(p1), grad(q1)) * dx - xi_macro * q1 * dx
# L_1 = f * q1 * dx
# F_macro = a_1 - L_1
# a_macro = fd.lhs(F_macro)
# L_macro = fd.rhs(F_macro)

# ## Micro terms
# a_2 = (k2 / mu) * inner(grad(p2), grad(q2)) * dx + xi_micro * q2 * dx
# L_2 = f * q2 * dx
# F_micro = a_2 - L_2
# a_micro = fd.lhs(F_micro)
# L_micro = fd.rhs(F_micro)

# %%
# # Macro
# A_macro = fd.assemble(a_macro, bcs=bc_macro, mat_type="aij")
# petsc_mat_macro = A_macro.M.handle
# is_symmetric_macro = petsc_mat_macro.isSymmetric(tol=1e-8)
# size_macro = petsc_mat_macro.getSize()
# Mnp_macro = csr_matrix(petsc_mat_macro.getValuesCSR()[::-1], shape=size_macro)

# # Micro
# A_micro = fd.assemble(a_micro, bcs=bc_micro, mat_type="aij")
# petsc_mat_micro = A_micro.M.handle
# is_symmetric_micro = petsc_mat_micro.isSymmetric(tol=1e-8)
# size_micro = petsc_mat_micro.getSize()
# Mnp_micro = csr_matrix(petsc_mat_micro.getValuesCSR()[::-1], shape=size_micro)

# %%
# Mnp_macro.eliminate_zeros()
# nnz = Mnp_macro.nnz
# number_of_dofs = V.dim()
# num_of_factors = int(number_of_dofs) - 1

# print(f"Number of Degrees of Freedom (Macro): {number_of_dofs}")
# print(f"Number of non-zero entries: {nnz}")
# print(f"Is operator symmetric? {is_symmetric}")
# print(f"Number of factors to compute in SVD: {num_of_factors}")

# %%
# Mnp_micro.eliminate_zeros()
# nnz = Mnp_micro.nnz
# number_of_dofs = V.dim()
# num_of_factors = int(number_of_dofs) - 1

# print(f"Number of Degrees of Freedom (Micro): {number_of_dofs}")
# print(f"Number of non-zero entries: {nnz}")
# print(f"Is operator symmetric? {is_symmetric}")
# print(f"Number of factors to compute in SVD: {num_of_factors}")

# %%
# macro_system_condition_number = calculate_condition_number(
#     A=petsc_mat_macro, num_of_factors=num_of_factors
# )
# micro_system_condition_number = calculate_condition_number(
#     A=petsc_mat_micro, num_of_factors=num_of_factors
# )

# print(f"Macro system Condition Number: {macro_system_condition_number}")
# print(f"Micro system Condition Number: {micro_system_condition_number}")
