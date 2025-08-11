from perphil.solvers import parameters as P


def test_linear_solver_params_have_expected_keys():
    p = P.LINEAR_SOLVER_PARAMS
    assert p["pc_type"] == "lu"
    assert p["ksp_type"] == "preonly"


def test_plain_gmres_explicitly_disables_pc():
    p = P.PLAIN_GMRES_PARAMS
    assert p["ksp_type"] == "gmres"
    assert p["pc_type"] == "none"


def test_fieldsplit_lu_has_block_configs():
    p = P.FIELDSPLIT_LU_PARAMS
    assert p["pc_type"] == "fieldsplit"
    assert p["pc_fieldsplit_type"] == "multiplicative"
    assert "fieldsplit_0" in p and "fieldsplit_1" in p


def test_picard_params_extend_fieldsplit():
    p = P.PICARD_LU_SOLVER_PARAMS
    assert p["snes_type"] in {"ngs", "nrichardson", "ksponly"}
    assert p["pc_type"] == "fieldsplit"
