# %%
import os

os.environ["OMP_NUM_THREADS"] = "1"

import firedrake as fd

# %% [markdown]
# For convenience, we define the operators from Firedrake:

# %%
grad = fd.grad
div = fd.div
dx = fd.dx
inner = fd.inner


# %%
def exact_solutions_expressions(mesh):
    x, y = fd.SpatialCoordinate(mesh)
    p_exact = fd.sin(2 * fd.pi * x) * fd.sin(2 * fd.pi * y)  # noqa: F405
    flux_exact = -grad(p_exact)
    return p_exact, flux_exact


def calculate_exact_solution(
    mesh, pressure_family, velocity_family, pressure_degree, velocity_degree, is_hdiv_space=False
):
    """
    For compatibility only. Should be removed.
    """
    return exact_solutions_expressions(mesh)


# %%
# Mesh
num_elements_x, num_elements_y = 10, 10
enable_run_on_quads = False
mesh = fd.UnitSquareMesh(
    num_elements_x,
    num_elements_y,
    quadrilateral=enable_run_on_quads,
)

# Approximation
degree = 1

# Function space declaration
pressure_family = "CG"
velocity_family = "CG"
U = fd.VectorFunctionSpace(mesh, velocity_family, degree)
V = fd.FunctionSpace(mesh, pressure_family, degree)

# Trial and test functions
u = fd.TrialFunction(V)
v = fd.TestFunction(V)

# Exact solution
exact_solution, sigma_e = calculate_exact_solution(
    mesh, pressure_family, velocity_family, degree + 3, degree + 3
)

# Forcing function
f = div(-grad(exact_solution))

# Dirichlet BCs
bcs = fd.DirichletBC(V, fd.project(exact_solution, V), "on_boundary")

# Variational form
a = inner(grad(u), grad(v)) * dx
L = f * v * dx

# Solving the problem
solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
solution = fd.Function(V)
problem = fd.LinearVariationalProblem(a, L, solution, bcs=bcs, constant_jacobian=False)
solver = fd.LinearVariationalSolver(problem, solver_parameters=solver_parameters)
solver.solve()

# Retrieving the solution
p_h = solution
sigma_h = fd.project(-grad(p_h), U)
