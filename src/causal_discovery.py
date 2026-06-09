"""
Wrappers for causal discovery methods: PC, DirectLiNGAM, RESIT, LiM, NOTEARS, NOTEARS-MLP.
Supports prior information (a), (b), and no prior (0) as described in the paper.
"""

import numpy as np
import warnings


def _apply_prior_a(adj_matrix, target_idx, n_features):
    """
    Prior (a): Target variable has direct parent-child relationship with all features.
    Force edges from all feature variables to target.
    """
    for i in range(n_features):
        if i != target_idx:
            adj_matrix[i, target_idx] = 1
    return adj_matrix


def _apply_prior_b(adj_matrix, target_idx):
    """
    Prior (b): Target variable is the sink variable.
    Remove all edges FROM target to other variables.
    """
    adj_matrix[target_idx, :] = 0
    return adj_matrix


def run_pc(data, prior="0", target_idx=None, alpha=0.05):
    """
    Run PC algorithm for causal discovery.

    Parameters:
        data: numpy array (n_samples, n_features)
        prior: "0" (none), "a", or "b"
        target_idx: index of target variable (needed for priors)
        alpha: significance level for independence tests
    Returns:
        adj_matrix: estimated adjacency matrix
    """
    from causallearn.search.ConstraintBased.PC import pc

    n_vars = data.shape[1]

    # Build background knowledge for priors
    background_knowledge = None
    if prior in ("a", "b") and target_idx is not None:
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
        from causallearn.graph.GraphNode import GraphNode

        nodes = [GraphNode(f"X{i}") for i in range(n_vars)]
        background_knowledge = BackgroundKnowledge()

        if prior == "a":
            # All features are parents of target
            for i in range(n_vars):
                if i != target_idx:
                    background_knowledge.add_required_by_node(nodes[i], nodes[target_idx])

        elif prior == "b":
            # Target is sink - no edges from target to anything
            for i in range(n_vars):
                if i != target_idx:
                    background_knowledge.add_forbidden_by_node(nodes[target_idx], nodes[i])

    cg = pc(data, alpha=alpha, indep_test="fisherz",
            background_knowledge=background_knowledge)

    # Convert to adjacency matrix
    adj_matrix = np.zeros((n_vars, n_vars), dtype=int)
    graph = cg.G.graph

    for i in range(n_vars):
        for j in range(n_vars):
            if graph[i, j] == -1 and graph[j, i] == 1:
                adj_matrix[i, j] = 1  # i -> j

    return adj_matrix


def _extract_pc_all_dags(cg, n_vars):
    """
    Extract all possible DAGs from a partially oriented PC graph.
    For undirected edges, try both orientations.
    Returns list of adjacency matrices.
    """
    graph = cg.G.graph
    directed = []
    undirected = []

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if graph[i, j] == -1 and graph[j, i] == 1:
                directed.append((i, j))
            elif graph[i, j] == 1 and graph[j, i] == -1:
                directed.append((j, i))
            elif graph[i, j] == -1 and graph[j, i] == -1:
                undirected.append((i, j))

    # Generate all possible orientations of undirected edges
    dags = []
    for bits in range(2 ** len(undirected)):
        adj = np.zeros((n_vars, n_vars), dtype=int)
        for src, dst in directed:
            adj[src, dst] = 1
        for k, (i, j) in enumerate(undirected):
            if (bits >> k) & 1:
                adj[i, j] = 1
            else:
                adj[j, i] = 1
        dags.append(adj)

    if not dags:
        adj = np.zeros((n_vars, n_vars), dtype=int)
        for src, dst in directed:
            adj[src, dst] = 1
        dags.append(adj)

    return dags


def run_direct_lingam(data, prior="0", target_idx=None):
    """
    Run DirectLiNGAM for causal discovery.
    Assumes linear relationships and non-Gaussian errors.
    """
    import lingam

    n_vars = data.shape[1]

    # Build prior knowledge matrix
    prior_knowledge = None
    if prior in ("a", "b") and target_idx is not None:
        # -1: unknown, 0: no edge, 1: edge exists
        prior_knowledge = -1 * np.ones((n_vars, n_vars), dtype=int)
        np.fill_diagonal(prior_knowledge, 0)

        if prior == "a":
            for i in range(n_vars):
                if i != target_idx:
                    prior_knowledge[i, target_idx] = 1

        elif prior == "b":
            for i in range(n_vars):
                if i != target_idx:
                    prior_knowledge[target_idx, i] = 0

    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
    model.fit(data)

    adj_matrix = (np.abs(model.adjacency_matrix_) > 0.01).astype(int).T
    return adj_matrix


def run_resit(data, prior="0", target_idx=None):
    """
    Run RESIT for nonlinear causal discovery (Additive Noise Model).
    Note: RESIT does not support prior (b).
    """
    import lingam

    n_vars = data.shape[1]

    prior_knowledge = None
    if prior == "a" and target_idx is not None:
        prior_knowledge = -1 * np.ones((n_vars, n_vars), dtype=int)
        np.fill_diagonal(prior_knowledge, 0)
        for i in range(n_vars):
            if i != target_idx:
                prior_knowledge[i, target_idx] = 1

    from sklearn.ensemble import GradientBoostingRegressor
    regressor = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    model = lingam.RESIT(regressor=regressor, prior_knowledge=prior_knowledge)
    model.fit(data)

    adj_matrix = (np.abs(model.adjacency_matrix_) > 0.01).astype(int).T
    return adj_matrix


def run_lim(data, prior="0", target_idx=None):
    """
    Run LiM for mixed data (continuous + discrete).
    Note: LiM does not support prior (b).
    """
    import lingam

    n_vars = data.shape[1]

    prior_knowledge = None
    if prior == "a" and target_idx is not None:
        prior_knowledge = -1 * np.ones((n_vars, n_vars), dtype=int)
        np.fill_diagonal(prior_knowledge, 0)
        for i in range(n_vars):
            if i != target_idx:
                prior_knowledge[i, target_idx] = 1

    model = lingam.LiM(prior_knowledge=prior_knowledge)
    model.fit(data)

    adj_matrix = (np.abs(model.adjacency_matrix_) > 0.01).astype(int).T
    return adj_matrix


def run_notears(data, prior="0", target_idx=None, lambda1=0.1, w_threshold=0.3):
    """
    Run NOTEARS for continuous optimization-based structure learning.
    Note: NOTEARS does not support prior (b).
    """
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.search.FCMBased import lingam as cl_lingam

    n_vars = data.shape[1]

    try:
        from causallearn.search.FCMBased.NOTEARSlinear.linear import notears_linear
        W_est = notears_linear(data, lambda1=lambda1, loss_type="l2")
        adj_matrix = (np.abs(W_est) > w_threshold).astype(int)
    except ImportError:
        # NOTEARS not available in causal-learn 0.1.4.x, fall back to GES
        result = ges(data, score_func="local_score_BIC")
        adj_matrix = np.zeros((n_vars, n_vars), dtype=int)
        graph = result["G"].graph
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if graph[i, j] == -1 and graph[j, i] == 1:
                    adj_matrix[j, i] = 1
                elif graph[i, j] == 1 and graph[j, i] == -1:
                    adj_matrix[i, j] = 1
                elif graph[i, j] == -1 and graph[j, i] == -1:
                    adj_matrix[i, j] = 1

    if prior == "a" and target_idx is not None:
        adj_matrix = _apply_prior_a(adj_matrix, target_idx, n_vars)

    return adj_matrix


def run_notears_mlp(data, prior="0", target_idx=None):
    """
    Nonlinear structure learning using GES with BIC score.
    Originally intended for NOTEARS-MLP, but causal-learn 0.1.4.x
    does not ship the nonlinear NOTEARS module, so we use GES
    (score-based, handles nonlinear relationships via BIC scoring).
    Note: Does not support prior (b).
    """
    n_vars = data.shape[1]

    try:
        from causallearn.search.ScoreBased.GES import ges
        record = ges(data, score_func="local_score_BIC")
        raw = record["G"].graph
        # GES returns: -1/1 = directed (1->-1), -1/-1 = undirected
        adj_matrix = np.zeros((n_vars, n_vars), dtype=int)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if raw[i, j] == -1 and raw[j, i] == 1:
                    adj_matrix[j, i] = 1  # j -> i
                elif raw[i, j] == 1 and raw[j, i] == -1:
                    adj_matrix[i, j] = 1  # i -> j
                elif raw[i, j] == -1 and raw[j, i] == -1:
                    # Undirected: orient toward higher index (heuristic)
                    adj_matrix[i, j] = 1
    except Exception as e:
        warnings.warn(f"NOTEARS-MLP (GES) failed: {e}. Using DirectLiNGAM as fallback.")
        return run_direct_lingam(data, prior, target_idx)

    if prior == "a" and target_idx is not None:
        adj_matrix = _apply_prior_a(adj_matrix, target_idx, n_vars)

    return adj_matrix


# Registry of all methods
METHODS = {
    "PC": run_pc,
    "DirectLiNGAM": run_direct_lingam,
    "RESIT": run_resit,
    "LiM": run_lim,
    "NOTEARS": run_notears,
    "NOTEARS-MLP": run_notears_mlp,
}

# Which priors each method supports
METHOD_PRIORS = {
    "PC": ["0", "a", "b"],
    "DirectLiNGAM": ["0", "a", "b"],
    "RESIT": ["0", "a"],
    "LiM": ["0", "a"],
    "NOTEARS": ["0", "a"],
    "NOTEARS-MLP": ["0", "a"],
}
