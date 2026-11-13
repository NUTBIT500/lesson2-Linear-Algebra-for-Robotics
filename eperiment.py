import csv
from pathlib import Path

import numpy as np

from solver_lab import solve_least_squares, calculate_metrics


METHODS = ("normal", "qr", "svd", "pinv")
RCOND = 1e-12

OUTPUT_PATH = Path("outputs/results.csv")


def run_experiment(name, A, b, epsilon=None, x_true=None):
    rows = []

    # Determine numerical rank.
    rank = np.linalg.matrix_rank(A, rtol = RCOND)

    # Calculate cond(A) and cond(A.T @ A).
    cond_A = np.linalg.cond(A)
    cond_ATA = np.linalg.cond(A.T @ A)

    # 3: Call each method in METHODS.
    x_list = []
    for method in METHODS:
        if method in ("normal", "qr") and (
            A.shape[0] < A.shape[1] or rank < A.shape[1]
        ):
            x_list.append({method:None})
            continue
        x_list.append({method:solve_least_squares(A, b, method, rcond=RCOND)})

    # Calculate numerical metrics.
    numerical_metrics = [
    calculate_metrics(A, b, x) if x is not None else None
    for d in x_list 
    for x in d.values()
    ]

    # Return the result records.
    rows = [{"name":name, "rank":rank, "cond_A":cond_A, "cond_ATA":cond_ATA, "x_list":x_list, "numerical_metrics":numerical_metrics}]
    return rows 
    


def main():
    # Construct Experiment 1.
    A1 = np.array([[1, 0], [0, 1], [1, 1], [2, -1]]) 
    b1 = [2, -1, 1.2, 2]
    result1 = run_experiment("experiment 1", A1, b1)

    # Construct Experiment 2.
    A2 = np.array([[1, 2], [2, 4], [3, 6]]) 
    b2 = [1, 2, 4]
    result2 = run_experiment("experiment 2", A2, b2)

    # Construct Experiment 3 for all three epsilons.
    eps_values = [1e-2, 1e-4, 1e-6]
    result3 = []
    for eps in eps_values:
        A = np.array([
            [1.0, 1.0],
            [1.0, 1.0 + eps]
        ])

        x_true = np.array([1.0, 0.0])
        b = A @ x_true
        result3.append(run_experiment("experiment 3", A, b, x_true=x_true))

    # Aggregate records.
    rows = result1 + result2 + [row for experiment in result3 for row in experiment]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
