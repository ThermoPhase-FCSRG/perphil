import firedrake as fd
import numpy as np
import attr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds, eigsh
from scipy.linalg import svd


DEFAULT_CONDITION_NUMBER_TOLERANCE = 1e-7


@attr.define(frozen=True)
class MatrixData:
    """
    A class to hold data related to a matrix, including its
    assembled_form, PETSc matrix handle, symmetry status,
    sparse CSR representation, number of non-zero entries,
    number of degrees of freedom (DoFs), and symmetry tolerance.

    :param assembled_matrix: The assembled matrix in Firedrake format.
    :type assembled_matrix: fd.Matrix

    :param petsc_matrix: The PETSc matrix handle.
    :type petsc_matrix: PETSc.Mat

    :param is_symmetric: Boolean indicating if the matrix is symmetric.
    :type is_symmetric: bool

    :param sparse_csr_data: Sparse CSR representation of the matrix.
    :type sparse_csr_data: csr_matrix

    :param number_of_nonzero_entries: Number of non-zero entries in the sparse matrix.
    :type number_of_nonzero_entries: int

    :param number_of_dofs: Number of degrees of freedom (DoFs) in the matrix.
    :type number_of_dofs: int

    :param symmetry_tolerance: Tolerance for symmetry checks.
    :type symmetry_tolerance: float
    """

    assembled_matrix: fd.Matrix
    petsc_matrix: fd.PETSc.Mat
    is_symmetric: bool
    sparse_csr_data: csr_matrix
    number_of_nonzero_entries: int
    number_of_dofs: int
    symmetry_tolerance: float


def assemble_bilinear_form(form: fd.Form, boundary_conditions: list[fd.DirichletBC]) -> fd.Matrix:
    """
    Assembles a bilinear form into a Firedrake matrix, applying the specified boundary conditions.

    :param form: The bilinear form to assemble.
    :type form: fd.Form
    :param boundary_conditions: List of Dirichlet boundary conditions to apply during assembly.
    :type boundary_conditions: list[fd.DirichletBC]
    :return: The assembled Firedrake matrix.
    :rtype: fd.Matrix
    """
    assembled_matrix = fd.assemble(form, bcs=boundary_conditions, mat_type="aij")
    return assembled_matrix


def get_matrix_data_from_form(
    form: fd.Form, boundary_conditions: list[fd.DirichletBC], symmetry_tolerance: float = 1e-8
) -> MatrixData:
    """
    Assembles a bilinear form and extracts matrix data, including symmetry and sparsity information.

    :param form: The bilinear form to assemble.
    :type form: fd.Form
    :param boundary_conditions: List of Dirichlet boundary conditions to apply during assembly.
    :type boundary_conditions: list[fd.DirichletBC]
    :param symmetry_tolerance: Tolerance used to check matrix symmetry, defaults to 1e-8.
    :type symmetry_tolerance: float, optional
    :return: MatrixData object containing the assembled matrix, PETSc matrix handle, symmetry status, sparse CSR data, number of non-zero entries, number of DoFs, and symmetry tolerance.
    :rtype: MatrixData
    """
    assembled_matrix = assemble_bilinear_form(form, boundary_conditions)
    petsc_matrix = assembled_matrix.M.handle
    is_symmetric = petsc_matrix.isSymmetric(tol=symmetry_tolerance)
    petsc_data_memory_size = petsc_matrix.getSize()
    sparse_csr_data = csr_matrix(petsc_matrix.getValuesCSR()[::-1], shape=petsc_data_memory_size)
    sparse_csr_data.eliminate_zeros()  # Note: in-place operation
    number_of_nonzero_entries = sparse_csr_data.nnz
    matrix_number_of_rows, matrix_number_of_columns = petsc_data_memory_size
    assert matrix_number_of_rows == matrix_number_of_columns
    # DoFs must be equal to the number of rows/cols of the resulting algebraic systen
    number_of_dofs = matrix_number_of_rows

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
    num_singular_values: int | None,
    use_sparse: bool = False,
    zero_tol: float = DEFAULT_CONDITION_NUMBER_TOLERANCE,
) -> float | np.float64:
    """
    Computes the condition number of a matrix using its singular values.

    Depending on the 'use_sparse' flag, this function either computes the singular values using a sparse or dense method.
    In sparse mode, it computes the largest and smallest singular values using iterative methods (no need for full spectrum),
    which is scalable and accurate. The condition number is defined as the ratio of the largest to the smallest singular
    value above a given zero tolerance.

    :param scipy_csr_sparse_matrix: The matrix in SciPy CSR sparse format for which to compute the condition number.
    :type scipy_csr_sparse_matrix: csr_matrix
    :param num_of_factors:
        Number of singular values to compute (used only if use_sparse is True).
        If None or <= 0, compute the full dense SVD (recommended when you want to
        use all DoFs of the target matrix). If the requested number is >= min(A.shape)-1,
        this function will also fall back to dense SVD for robustness and accuracy.
    :type num_of_factors: int | None
    :param use_sparse: Whether to use sparse SVD computation (svds) or dense SVD (svd), defaults to False.
    :type use_sparse: bool, optional
    :param zero_tol: Tolerance below which singular values are ignored, defaults to 1e-5.
    :type zero_tol: float, optional
    :return: The computed condition number of the matrix.
    :rtype: float | np.float64
    """
    nrows, ncols = scipy_csr_sparse_matrix.shape
    nmin = min(nrows, ncols)

    # Guard against degenerate shapes
    if nmin == 0:
        return float("nan")

    # If full spectrum requested or dense mode, compute dense SVD directly
    if (
        (not use_sparse)
        or (num_singular_values is None)
        or (num_singular_values <= 0)
        or (int(num_singular_values) >= nmin - 1)
    ):
        M = scipy_csr_sparse_matrix.toarray()
        svals = svd(M, compute_uv=False, check_finite=False)
        svals = np.asarray(svals)
        svals = svals[svals > zero_tol]
        if svals.size == 0:
            return float("inf")
        return float(svals.max() / svals.min())

    if use_sparse:
        # For accurate condition numbers we only need the extreme singular values.
        # Compute largest singular value (||A||2) using svds with which='LM'.
        try:
            smax_arr = svds(
                A=scipy_csr_sparse_matrix,
                k=1,
                which="LM",
                maxiter=10000,
                return_singular_vectors=False,
                solver="arpack",
            )
            smax = float(np.max(smax_arr))
        except Exception:
            # Dense fallback for robustness
            M = scipy_csr_sparse_matrix.toarray()
            svals = svd(M, compute_uv=False, check_finite=False)
            smax = float(np.max(svals)) if svals.size else float("nan")

        # Compute smallest singular value using svds with which='SM' (ARPACK),
        # falling back to eigsh on A^T A if needed.
        smin: float | None = None
        try:
            smin_arr = svds(
                A=scipy_csr_sparse_matrix,
                k=1,
                which="SM",
                maxiter=20000,
                return_singular_vectors=False,
                solver="arpack",
                tol=1e-8,
            )
            smin = float(np.min(smin_arr))
        except Exception:
            # Build normal equations A^T A (symmetric PSD) and compute its smallest eigenvalue
            try:
                AtA = (scipy_csr_sparse_matrix.T).dot(scipy_csr_sparse_matrix)
                # Using shift-invert around sigma=0 helps find the smallest eigenvalue
                evals, _ = eigsh(AtA, k=1, which="SM")
                lam_min = float(evals[0])
                smin = float(np.sqrt(max(lam_min, 0.0)))
            except Exception:
                # Last resort: dense SVD
                M = scipy_csr_sparse_matrix.toarray()
                svals = svd(M, compute_uv=False, check_finite=False)
                if svals.size:
                    smin = float(np.min(svals))

        if smin is None or not np.isfinite(smax):
            return float("nan")
        # Apply zero tolerance filtering
        if smin <= zero_tol:
            return float("inf")
        return float(smax / smin)
    else:
        # This branch is unreachable due to the guard above; kept for clarity.
        M = scipy_csr_sparse_matrix.toarray()
        svals = svd(M, compute_uv=False, check_finite=False)
        svals = np.asarray(svals)
        svals = svals[svals > zero_tol]
        if svals.size == 0:
            return float("inf")
        return float(svals.max() / svals.min())
