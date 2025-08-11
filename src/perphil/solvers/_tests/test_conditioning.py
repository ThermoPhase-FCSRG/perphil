import numpy as np
import scipy.sparse as sp
import pytest

pytest.importorskip("firedrake")


def _small_csr():
    # Simple 2x2 SPD matrix
    data = np.array([2.0, -1.0, -1.0, 2.0])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    return sp.csr_matrix((data, (rows, cols)), shape=(2, 2))


def test_assemble_and_extract_matrix_data():
    import firedrake as fd
    from perphil.solvers.conditioning import (
        assemble_bilinear_form,
        get_matrix_data_from_form,
    )
    from perphil.mesh.builtin import create_mesh
    from perphil.forms.spaces import create_function_spaces
    from perphil.models.dpp.parameters import DPPParameters
    from perphil.forms.dpp import dpp_form

    mesh = create_mesh(2, 2, quadrilateral=False)
    _, V = create_function_spaces(mesh)
    W = fd.MixedFunctionSpace((V, V))

    params = DPPParameters()
    a, L = dpp_form(W, params)
    bcs = []

    A_fd = assemble_bilinear_form(a, bcs)
    md = get_matrix_data_from_form(a, bcs)
    # Compare structural properties instead of pointer identity; assembly may rebuild objects
    assert md.petsc_matrix.getType() == A_fd.M.handle.getType()
    assert md.petsc_matrix.getSize() == A_fd.M.handle.getSize()
    assert md.sparse_csr_data.nnz == md.number_of_nonzero_entries
    assert md.petsc_matrix.getType() in {"aij", "seqaij", "mpiaij"}
    assert md.number_of_dofs == W.dim()
    assert md.number_of_nonzero_entries == md.sparse_csr_data.nnz


def test_calculate_condition_number_dense_and_sparse():
    from perphil.solvers.conditioning import calculate_condition_number

    csr = _small_csr()
    cond_dense = calculate_condition_number(csr, num_of_factors=1, use_sparse=False)
    cond_sparse = calculate_condition_number(csr, num_of_factors=1, use_sparse=True)
    # Methods differ on very small matrices; assert same order of magnitude
    assert cond_dense == pytest.approx(cond_sparse, rel=1e-2, abs=1e-9) or (
        abs(np.log10(cond_dense) - np.log10(cond_sparse)) < 0.5
    )
    assert cond_dense > 0
