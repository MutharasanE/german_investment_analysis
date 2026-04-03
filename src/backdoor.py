"""
Backdoor adjustment for computing P(Y|do(X)) from causal graphs.
Implements the core causal inference computation needed for LEWIS scores.
"""

import numpy as np
import pandas as pd
from itertools import combinations


def get_parents(adj_matrix, node):
    """Get parent nodes of a given node from adjacency matrix. adj[i][j]=1 means i->j."""
    return list(np.where(adj_matrix[:, node] == 1)[0])


def get_children(adj_matrix, node):
    """Get child nodes of a given node from adjacency matrix."""
    return list(np.where(adj_matrix[node, :] == 1)[0])


def get_ancestors(adj_matrix, node, visited=None):
    """Get all ancestors of a node (recursively)."""
    if visited is None:
        visited = set()
    parents = get_parents(adj_matrix, node)
    for p in parents:
        if p not in visited:
            visited.add(p)
            get_ancestors(adj_matrix, p, visited)
    return visited


def get_descendants(adj_matrix, node, visited=None):
    """Get all descendants of a node (recursively)."""
    if visited is None:
        visited = set()
    children = get_children(adj_matrix, node)
    for c in children:
        if c not in visited:
            visited.add(c)
            get_descendants(adj_matrix, c, visited)
    return visited


def find_backdoor_set(adj_matrix, treatment, outcome):
    """
    Find a valid backdoor adjustment set for estimating P(Y|do(X)).
    Uses parent adjustment: returns parents of treatment that are not descendants of treatment.
    """
    parents_of_treatment = set(get_parents(adj_matrix, treatment))
    descendants_of_treatment = get_descendants(adj_matrix, treatment)

    # Remove treatment and outcome from consideration
    backdoor = parents_of_treatment - descendants_of_treatment - {treatment, outcome}
    return sorted(list(backdoor))


def compute_do_probability(df, treatment_col, treatment_val, outcome_col, outcome_val,
                           adj_matrix, col_names, graph_outcome_col=None):
    """
    Compute P(outcome=outcome_val | do(treatment=treatment_val)) using backdoor adjustment.

    If a valid backdoor set exists:
        P(Y=y|do(X=x)) = sum_z P(Y=y|X=x, Z=z) * P(Z=z)

    If no backdoor set (no confounders):
        P(Y=y|do(X=x)) = P(Y=y|X=x)

    graph_outcome_col: column name in col_names for the graph lookup (e.g. "Y")
                       when outcome_col differs (e.g. "Y_pred").
    """
    treatment_idx = col_names.index(treatment_col)
    graph_out = graph_outcome_col if graph_outcome_col else outcome_col
    outcome_idx = col_names.index(graph_out)

    backdoor_set = find_backdoor_set(adj_matrix, treatment_idx, outcome_idx)
    backdoor_cols = [col_names[i] for i in backdoor_set]

    if not backdoor_cols:
        # No confounders: P(Y|do(X)) = P(Y|X)
        subset = df[df[treatment_col] == treatment_val]
        if len(subset) == 0:
            return 0.0
        return (subset[outcome_col] == outcome_val).mean()

    # Backdoor adjustment: sum over all values of Z
    prob = 0.0
    # Get unique combinations of backdoor variables
    if len(backdoor_cols) == 1:
        z_col = backdoor_cols[0]
        for z_val in df[z_col].unique():
            # P(Y=y | X=x, Z=z)
            mask_xz = (df[treatment_col] == treatment_val) & (df[z_col] == z_val)
            subset_xz = df[mask_xz]
            if len(subset_xz) == 0:
                continue
            p_y_given_xz = (subset_xz[outcome_col] == outcome_val).mean()
            # P(Z=z)
            p_z = (df[z_col] == z_val).mean()
            prob += p_y_given_xz * p_z
    else:
        # Multiple backdoor variables - group by all of them
        grouped = df.groupby(backdoor_cols)
        total = len(df)
        for z_vals, z_group in grouped:
            mask_x = z_group[treatment_col] == treatment_val
            subset_xz = z_group[mask_x]
            if len(subset_xz) == 0:
                continue
            p_y_given_xz = (subset_xz[outcome_col] == outcome_val).mean()
            p_z = len(z_group) / total
            prob += p_y_given_xz * p_z

    return prob


def compute_conditional_probability(df, outcome_col, outcome_val, condition_col, condition_val):
    """Compute P(outcome=outcome_val | condition=condition_val)."""
    subset = df[df[condition_col] == condition_val]
    if len(subset) == 0:
        return 0.0
    return (subset[outcome_col] == outcome_val).mean()
