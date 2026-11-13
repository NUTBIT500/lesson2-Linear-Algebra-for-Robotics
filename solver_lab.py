import numpy as np
import math
import numbers


def solve_least_squares(A, b, method, rcond=1e-12):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if(np.size(A) == 0 or np.size(b) == 0):
        raise ValueError
    if(np.ndim(A) != 2 or np.ndim(b) != 1):
        raise ValueError
    if(np.shape(A)[0] != np.shape(b)[0]):
        raise ValueError
    if(not np.isfinite(A).all() or not np.isfinite(b).all()):
        raise ValueError
    if (
        not isinstance(rcond, numbers.Real)
        or isinstance(rcond, (bool, np.bool_))
        or not np.isfinite(rcond)
        or rcond < 0
        or rcond >= 1
    ):
        raise ValueError
    
    
    # Determine numerical rank using singular values.
    rank = np.linalg.matrix_rank(A, rtol=rcond)
    m = A.shape[0]
    n = A.shape[1]
    # Implement the requested solver.
    if(method == "normal"):
        if(m >= n and rank == n):
            return np.linalg.solve(A.T @ A, A.T @ b)
        else:
            raise ValueError
    elif(method == "qr"):
        if(m >= n and rank == n):
            Q, R = np.linalg.qr(A)
            return np.linalg.solve(R, Q.T @ b)
        else:
            raise ValueError
    elif(method == "svd"):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        s_inv = np.zeros_like(s)
        nonzero_mask = s > np.max(s) * rcond
        s_inv[nonzero_mask] = 1.0 / s[nonzero_mask]
        return Vt.T @ (s_inv * (U.T @ b))
    elif(method == "pinv"):
        A_pinv= np.linalg.pinv(A, rcond = rcond)
        return A_pinv @ b
    else:
        raise ValueError



def calculate_metrics(A, b, x, x_true=None):
    # Calculate:
    # - residual norm
    residual_norm = math.sqrt(np.sum((b - A @ x) ** 2))
    # - residual orthogonality norm
    residual_orthogonality_norm = math.sqrt(np.sum((A.T @ (b - A @ x)) ** 2))
    # - relative forward error (only when x_true is known)
    forward_error = np.nan
    if x_true is not None:
        if(np.shape(x_true) != np.shape(x)):
            raise ValueError
        x_true = np.asarray(x_true, dtype=float)
        denominator = np.linalg.norm(x_true)

        if denominator > 0:
            forward_error = np.linalg.norm(x - x_true) / denominator
    # - condition number
    condition_number = np.linalg.cond(A)
    return residual_norm, residual_orthogonality_norm, forward_error, condition_number