"""
Synthetic data generation from Structural Causal Models (SCMs).
Implements the 3-variable (Section III) and 8-variable (Section IV) causal graphs
from Takahashi et al. (2024).
"""

import numpy as np
import pandas as pd


def generate_error_terms(n_samples, distribution="uniform", size=1):
    """Generate error terms from specified distribution."""
    if distribution == "uniform":
        return np.random.uniform(0, 1, size=(n_samples, size))
    elif distribution == "gaussian":
        return np.random.normal(0, 1, size=(n_samples, size))
    elif distribution == "bernoulli":
        return np.random.binomial(1, 0.5, size=(n_samples, size))
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def apply_function(values, func_type="linear"):
    """Apply linear or nonlinear transformation."""
    if func_type == "linear":
        return values
    elif func_type == "nonlinear":
        return values ** 2
    else:
        raise ValueError(f"Unknown function type: {func_type}")


# ---------------------------------------------------------------------------
# 3-variable structures (Section III, Figure 2)
# ---------------------------------------------------------------------------

def generate_structure_a(n_samples, func_type="linear", distribution="uniform"):
    """Structure A (Collider): X -> Y <- Z, coefficients X=1.0, Z=1.5."""
    U = generate_error_terms(n_samples, distribution, size=3)
    X = U[:, 0]
    Z = U[:, 1]
    Y = 1.0 * apply_function(X, func_type) + 1.5 * apply_function(Z, func_type) + U[:, 2]
    return pd.DataFrame({"X": X, "Z": Z, "Y": Y})


def generate_structure_b(n_samples, func_type="linear", distribution="uniform"):
    """Structure B (Chain): X -> Z -> Y, all coefficients 1.0."""
    U = generate_error_terms(n_samples, distribution, size=3)
    X = U[:, 0]
    Z = 1.0 * apply_function(X, func_type) + U[:, 1]
    Y = 1.0 * apply_function(Z, func_type) + U[:, 2]
    return pd.DataFrame({"X": X, "Z": Z, "Y": Y})


def generate_structure_c(n_samples, func_type="linear", distribution="uniform"):
    """Structure C (Confounding): X -> Z, X -> Y, Z -> Y, coefficients 1.0."""
    U = generate_error_terms(n_samples, distribution, size=3)
    X = U[:, 0]
    Z = 1.0 * apply_function(X, func_type) + U[:, 1]
    Y = 1.0 * apply_function(X, func_type) + 1.0 * apply_function(Z, func_type) + U[:, 2]
    return pd.DataFrame({"X": X, "Z": Z, "Y": Y})


def generate_structure_d(n_samples, func_type="linear", distribution="uniform"):
    """Structure D (Independent X): Z -> Y, X independent. Coefficient 1.0."""
    U = generate_error_terms(n_samples, distribution, size=3)
    X = U[:, 0]
    Z = U[:, 1]
    Y = 1.0 * apply_function(Z, func_type) + U[:, 2]
    return pd.DataFrame({"X": X, "Z": Z, "Y": Y})


def generate_structure_e(n_samples, func_type="linear", distribution="uniform"):
    """Structure E (Fork): Z -> X, Z -> Y, coefficients 1.0."""
    U = generate_error_terms(n_samples, distribution, size=3)
    Z = U[:, 0]
    X = 1.0 * apply_function(Z, func_type) + U[:, 1]
    Y = 1.0 * apply_function(Z, func_type) + U[:, 2]
    return pd.DataFrame({"X": X, "Z": Z, "Y": Y})


THREE_VAR_GENERATORS = {
    "A": generate_structure_a,
    "B": generate_structure_b,
    "C": generate_structure_c,
    "D": generate_structure_d,
    "E": generate_structure_e,
}

# True adjacency matrices for 3-variable structures (X=0, Z=1, Y=2)
# adj[i][j] = 1 means i -> j
THREE_VAR_TRUE_GRAPHS = {
    "A": np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]]),  # X->Y, Z->Y
    "B": np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),  # X->Z, Z->Y
    "C": np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),  # X->Z, X->Y, Z->Y
    "D": np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),  # Z->Y only
    "E": np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]),  # Z->X, Z->Y
}


# ---------------------------------------------------------------------------
# 8-variable structure (Section IV, Figure 4)
# ---------------------------------------------------------------------------

def generate_8var_data(n_samples, func_type="linear", distribution="uniform",
                       mixed=False):
    """
    Generate data from the 8-variable causal graph (Figure 4).

    Graph: X1 -> X3 (1.5), X2 -> X3 (4.0), X3 -> X4 (2.0),
           X4 -> X5 (1.0), X4 -> X6 (2.0), X6 -> X7 (3.0),
           X4 -> X7 (2.0), X5 -> Y (2.0), X7 -> Y (1.0)
    """
    if mixed:
        U1 = generate_error_terms(n_samples, "bernoulli", size=1).flatten()
        U7 = generate_error_terms(n_samples, "bernoulli", size=1).flatten()
        U_rest = generate_error_terms(n_samples, distribution, size=6)
        U2, U3, U4, U5, U6, UY = [U_rest[:, i] for i in range(6)]
    else:
        U_all = generate_error_terms(n_samples, distribution, size=8)
        U1, U2, U3, U4, U5, U6, U7, UY = [U_all[:, i] for i in range(8)]

    X1 = U1
    X2 = U2
    X3 = 1.5 * apply_function(X1, func_type) + 4.0 * apply_function(X2, func_type) + U3
    X4 = 2.0 * apply_function(X3, func_type) + U4
    X5 = 1.0 * apply_function(X4, func_type) + U5
    X6 = 2.0 * apply_function(X4, func_type) + U6
    X7 = 2.0 * apply_function(X4, func_type) + 3.0 * apply_function(X6, func_type) + U7
    Y = 2.0 * apply_function(X5, func_type) + 1.0 * apply_function(X7, func_type) + UY

    return pd.DataFrame({
        "X1": X1, "X2": X2, "X3": X3, "X4": X4,
        "X5": X5, "X6": X6, "X7": X7, "Y": Y,
    })


# True adjacency matrix for 8-variable graph
# Order: X1=0, X2=1, X3=2, X4=3, X5=4, X6=5, X7=6, Y=7
EIGHT_VAR_TRUE_GRAPH = np.array([
    # X1 X2 X3 X4 X5 X6 X7  Y
    [0,  0,  1,  0,  0,  0,  0,  0],  # X1 -> X3
    [0,  0,  1,  0,  0,  0,  0,  0],  # X2 -> X3
    [0,  0,  0,  1,  0,  0,  0,  0],  # X3 -> X4
    [0,  0,  0,  0,  1,  1,  1,  0],  # X4 -> X5, X6, X7
    [0,  0,  0,  0,  0,  0,  0,  1],  # X5 -> Y
    [0,  0,  0,  0,  0,  0,  1,  0],  # X6 -> X7
    [0,  0,  0,  0,  0,  0,  0,  1],  # X7 -> Y
    [0,  0,  0,  0,  0,  0,  0,  0],  # Y (sink)
])

EIGHT_VAR_FEATURE_NAMES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "Y"]
