import firedrake as fd
import numpy as np
from petsc4py import PETSc
import attr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import svd


@attr.define(frozen=True)
class MatrixData:
    """
    TODO.
    """

    assembled_matrix: fd.Matrix
    petsc_matrix: PETSc.Mat
    is_symmetric: bool
    sparse_csr_data: csr_matrix
    number_of_nonzero_entries: int
    number_of_dofs: int
    symmetry_tolerance: float


def assemble_bilinear_form(form: fd.Form, boundary_conditions: list[fd.DirichletBC]) -> fd.Matrix:
    """
    TODO.

    :param form: _description_
    :type form: fd.Form
    :param boundary_conditions: _description_
    :type boundary_conditions: list[fd.DirichletBC]
    :return: _description_
    :rtype: fd.Matrix
    """
    assembled_matrix = fd.assemble(form, bcs=boundary_conditions, mat_type="aij")
    return assembled_matrix


def get_matrix_data_from_form(
    form: fd.Form, boundary_conditions: list[fd.DirichletBC], symmetry_tolerance=1e-8
) -> MatrixData:
    """
    TODO.

    :param form: _description_
    :type form: fd.Form
    :param boundary_conditions: _description_
    :type boundary_conditions: list[fd.DirichletBC]
    :param symmetry_tolerance: _description_, defaults to 1e-8
    :type symmetry_tolerance: _type_, optional
    :return: _description_
    :rtype: MatrixData
    """
    assembled_matrix = assemble_bilinear_form(form, boundary_conditions)
    petsc_matrix = assembled_matrix.M.handle
    is_symmetric = petsc_matrix.isSymmetric(tol=symmetry_tolerance)
    petsc_data_memory_size = petsc_matrix.getSize()
    sparse_csr_data = csr_matrix(petsc_matrix.getValuesCSR()[::-1], shape=petsc_data_memory_size)
    sparse_csr_data.eliminate_zeros()  # Note: in-place operation
    number_of_nonzero_entries = sparse_csr_data.nnz
    number_of_dofs = petsc_data_memory_size

    matrix_data = MatrixData(
        assembled_matrix,
        petsc_matrix,
        is_symmetric,
        sparse_csr_data,
        number_of_nonzero_entries,
        number_of_dofs,
        symmetry_tolerance,
    )
    return matrix_data


def calculate_condition_number(
    scipy_csr_sparse_matrix: csr_matrix,
    num_of_factors: int,
    use_sparse: bool = False,
    zero_tol: float = 1e-5,
) -> float | np.float64:
    """
    TODO.

    :param scipy_csr_sparse_matrix: _description_
    :param num_of_factors: _description_
    :type num_of_factors: int
    :param use_sparse: _description_, defaults to False
    :type use_sparse: bool, optional
    :param zero_tol: _description_, defaults to 1e-5
    :type zero_tol: float, optional
    :return: _description_
    :rtype: float | np.float64
    """
    if use_sparse:
        singular_values = svds(
            A=scipy_csr_sparse_matrix,
            k=num_of_factors,
            which="LM",
            maxiter=5000,
            return_singular_vectors=False,
            solver="lobpcg",
        )
    else:
        M = scipy_csr_sparse_matrix.toarray()
        singular_values = svd(M, compute_uv=False, check_finite=False)

    singular_values = singular_values[singular_values > zero_tol]
    condition_number = singular_values.max() / singular_values.min()
    return condition_number
