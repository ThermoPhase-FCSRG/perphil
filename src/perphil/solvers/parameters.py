# Linear monolithic solver parameters (direct solver via MUMPS)
LINEAR_SOLVER_PARAMS: dict = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# Pure GMRES parameters
GMRES_PARAMS: dict = {
    "mat_type": "aij",
    "ksp_type": "gmres",
    "ksp_rtol": 1.0e-12,
    "ksp_atol": 1.0e-12,
    "ksp_max_it": 5000,
}

# GMRES + Jacobi parameters for scale-splitting comparison
GMRES_JACOBI_PARAMS: dict = {"pc_type": "jacobi", **GMRES_PARAMS}

# Field-split preconditioner (multiplicative) with LU in each block
FIELDSPLIT_LU_PARAMS: dict = {
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    "pc_fieldsplit_0_fields": "0",
    "pc_fieldsplit_1_fields": "1",
    "fieldsplit_0": LINEAR_SOLVER_PARAMS,
    "fieldsplit_1": LINEAR_SOLVER_PARAMS,
}

# Picard (nonlinear Richardson) solver parameters with field-split
RICHARDSON_SOLVER_PARAMS: dict = {
    "snes_type": "nrichardson",
    "snes_max_it": 50000,
    "snes_linesearch_type": "basic",
    "snes_linesearch_damping": 0.5,
    "snes_rtol": 1e-5,
    "snes_atol": 1e-12,
    **FIELDSPLIT_LU_PARAMS,
}

# Picard (with nonlinear Gauss-Siedel) solver parameters with field-split
NGS_SOLVER_PARAMS = {
    "snes_type": "ngs",
    "snes_max_it": 10000,
    "snes_rtol": 1e-5,
    "snes_atol": 1e-12,
    **FIELDSPLIT_LU_PARAMS,
}

# SNES with KSP-only (for preconditioner analysis)
KSP_PREONLY_PARAMS: dict = {
    "snes_type": "ksponly",
    "ksp_monitor": None,
    **FIELDSPLIT_LU_PARAMS,
}
