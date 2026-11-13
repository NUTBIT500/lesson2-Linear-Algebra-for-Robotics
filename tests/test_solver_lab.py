import numpy as np
import pytest

from solver_lab import solve_least_squares, calculate_metrics


# Test 1: Full-rank QR versus NumPy lstsq
def test_Full_rank_QR():
    A = np.array([[1, 0], [0, 1], [1, 1], [2, -1]]) 
    b = [2, -1, 1.2, 2]
    x_qr = solve_least_squares(A, b, method="qr")
    x_lstsq = np.linalg.lstsq(A, b, rcond=1e-12)[0]
    assert np.allclose(x_qr, x_lstsq, 0, 1e-10)

# Test 2: Full-rank SVD versus pseudoinverse
def test_Full_rank_SVD():
    A = np.array([[1, 0], [0, 1], [1, 1], [2, -1]]) 
    b = [2, -1, 1.2, 2]
    x_svd = solve_least_squares(A, b, method="svd")
    x_pinv = solve_least_squares(A, b, method="pinv")
    assert np.allclose(x_svd, x_pinv, 0, 1e-10)

# Test 3: Residual orthogonality
# def test_Residual_Orthogonality():
def test_Residual_Orthogonality():
    A = np.array([[1, 0], [0, 1], [1, 1], [2, -1]]) 
    b = [2, -1, 1.2, 2]
    x_qr = solve_least_squares(A, b, method="normal")
    residual_orthogonality_norm = calculate_metrics(A, b, x_qr, None)[1]
    assert np.isclose(residual_orthogonality_norm, 0, 0, 1e-10)
    
# Test 4: QR orthogonality
def test_QR_Orthogonality():
    A = np.array([[1, 0], [0, 1], [1, 1], [2, -1]]) 
    Q, R = np.linalg.qr(A, mode="reduced")
    assert np.allclose(
        Q.T @ Q,
        np.eye(A.shape[1]),
        rtol=0,
        atol=1e-12
    )

# Test 5: Rank-deficient numerical rank
def test_Rank_deficient():
    A = np.array([[1, 2], [2, 4], [3, 6]]) 
    rank = np.linalg.matrix_rank(A, rtol=1e-12)
    assert rank == 1

# Test 6: Minimum-norm SVD versus pseudoinverse
def test_Minimum_norm_SVD():
    A = np.array([[1, 2], [2, 4], [3, 6]]) 
    b = [1, 2, 4]
    x_svd = solve_least_squares(A, b, method="svd")
    x_pinv = solve_least_squares(A, b, method="pinv")
    assert np.allclose(x_svd, x_pinv, 0, 1e-10)

    z = np.array([-2, 1])
    t = 1.0
    x_pert = x_svd + t * z

    res_svd = np.linalg.norm(A @ x_svd - b)
    res_pert = np.linalg.norm(A @ x_pert - b)
    assert np.isclose(res_svd, res_pert, atol=1e-10)

    norm_svd = np.linalg.norm(x_svd)
    norm_pert = np.linalg.norm(x_pert)
    assert norm_pert > norm_svd

# Test 7: Near-singular QR versus SVD
@pytest.mark.parametrize("eps", [1e-2, 1e-4, 1e-6])
def test_Near_singular_QR(eps):
    A = np.array([
                [1.0, 1.0],
                [1.0, 1.0 + eps]
            ])
    x_true = np.array([1.0, 0.0])
    b = A @ x_true 
    x_qr = solve_least_squares(A, b, method="qr")
    x_svd = solve_least_squares(A, b, method="svd")
    assert np.allclose(x_qr, x_svd, 0, 1e-8)

# Test 8: Invalid dimensions
def test_Invalid_dimensions():
    A = np.array([[[1, 0], [0, 1]], [[1, 1], [2, -1]]]) 
    b = [2, -1, 1.2, 2]
    with pytest.raises(ValueError):
        x_qr = solve_least_squares(A, b, method="normal")

    A2 = np.array([[1, 0], [0, 1], [1, 1], [2, -1]]) 
    b2 = [[2, -1], [1.2, 2]]
    with pytest.raises(ValueError):
        x_qr = solve_least_squares(A2, b2, method="normal")
    

# Test 9: Nonfinite inputs
def test_Nonfinite_inputs():
    A = np.array([[1, 0], [0, 1], [None, 1], [2, -1]]) 
    b = [2, -1, 1.2, 2]
    with pytest.raises(ValueError):
        x_qr = solve_least_squares(A, b, method="normal")
    A2 = np.array([[1, 0], [0, 1], [np.inf, 1], [2, -1]]) 
    b2 = [2, -1, 1.2, 2]
    with pytest.raises(ValueError):
        x_qr = solve_least_squares(A2, b2, method="normal")
    A3 = np.array([[1, 0], [0, 1], [np.nan, 1], [2, -1]]) 
    b3 = [2, -1, 1.2, 2]
    with pytest.raises(ValueError):
        x_qr = solve_least_squares(A3, b3, method="normal")

# Test 10: Unsupported method
def test_Unsupported_method():
    A = np.array([[1, 0], [0, 1], [1, 1], [2, -1]]) 
    b = [2, -1, 1.2, 2]
    with pytest.raises(ValueError):
        x_qr = solve_least_squares(A, b, method="xxx")

# Test 11: Normal equations reject rank deficiency
def test_rank_deficiency1():
    A = np.array([[1, 2], [2, 4], [3, 6]]) 
    b = [1, 2, 4]
    with pytest.raises(ValueError):
        x_qr = solve_least_squares(A, b, method="normal")
# Test 12: QR rejects rank deficiency
def test_rank_deficiency2():
    A = np.array([[1, 2], [2, 4], [3, 6]]) 
    b = [1, 2, 4]
    with pytest.raises(ValueError):
        x_qr = solve_least_squares(A, b, method="qr")