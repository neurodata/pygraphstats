# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import pytest
from graspologic.cluster.gclust import GaussianCluster
from graspologic.embed.mase import MultipleASE
from graspologic.simulations.simulations import er_np, sbm
from numpy import array_equal


def make_train_undirected(n=[128, 128], m=10, alpha=1):
    """
    Make 4 class training dataset
    n = number of vertices
    m = number of graphs from each class
    """
    c1 = np.array([[0.1, 0], [0, 0.1]])
    c2 = -1 * c1
    c3 = np.array([[0.1, 0], [0, 0]])
    c4 = np.array([[0, 0], [0, 0.1]])

    A = [
        sbm(n, np.ones((2, 2)) * 0.25 + alpha * c)
        for _ in range(m)
        for c in [c1, c2, c3, c4]
    ]

    return A


def make_train_directed(n=[128, 128], m=10):
    p1 = [[0, 0.9], [0, 0]]
    p2 = [[0, 0], [0.9, 0]]
    p3 = [[0.9, 0.9], [0, 0]]
    p4 = [[0, 0], [0.9, 0.9]]

    A = [sbm(n, p, directed=True) for _ in range(m) for p in [p1, p2, p3, p4]]

    return A


def test_bad_inputs():
    np.random.seed(1)
    single_graph = er_np(100, 0.2)
    different_size_graphs = [er_np(100, 0.2)] + [er_np(200, 0.2)]

    with pytest.raises(TypeError):
        "Invalid unscaled"
        mase = MultipleASE(scaled="1")

    with pytest.raises(TypeError):
        wrong_diag_aug = "True"
        mase = MultipleASE(diag_aug=wrong_diag_aug)

    with pytest.raises(ValueError):
        "Test single graph input"
        MultipleASE().fit(single_graph)

    with pytest.raises(ValueError):
        "Test 3-d tensor with 1 graph"
        single_graph_tensor = single_graph.reshape(1, 100, -1)
        MultipleASE().fit(single_graph_tensor)

    with pytest.raises(ValueError):
        "Empty list"
        MultipleASE().fit([])

    with pytest.raises(ValueError):
        "Test graphs with different sizes"
        MultipleASE().fit(different_size_graphs)


def test_scores_equal():
    """
    This shows a better way of calculating scores, proving that the list comprehension
    and matrix multiplication method produce the same result.
    """
    A = np.asarray(make_train_undirected())
    mase = MultipleASE().fit(A)
    Uhat = mase.latent_left_

    scores = np.asarray([Uhat.T @ graph @ Uhat for graph in list(A)])
    scores_ = Uhat.T @ A @ Uhat

    assert np.array_equal(scores, scores_)


def test_diag_aug():
    np.random.seed(5)
    n = 100
    p = 0.25

    graphs_list = [er_np(n, p) for _ in range(2)]
    graphs_arr = np.array(graphs_list)

    # Test that array and list inputs results in same embeddings
    mase_arr = MultipleASE(diag_aug=True).fit_transform(graphs_arr)
    mase_list = MultipleASE(diag_aug=True).fit_transform(graphs_list)

    assert array_equal(mase_list, mase_arr)


def test_graph_clustering():
    """
    There should be 4 total clusters since 4 class problem.
    n_components = 2
    """
    n = [128, 128]
    m = 10

    def run(diag_aug, scaled):
        # undirected case
        np.random.seed(2 + diag_aug + scaled)
        X = make_train_undirected(n, m)

        res = (
            MultipleASE(2, diag_aug=diag_aug, scaled=scaled)
            .fit(X)
            .scores_.reshape((m * 4, -1))
        )
        gmm = GaussianCluster(10, covariance_type="all").fit(res)
        assert gmm.n_components_ == 4

        # directed case
        np.random.seed(3 + diag_aug + scaled)
        X = make_train_directed(n, m)

        res = MultipleASE(2, diag_aug=diag_aug).fit(X).scores_.reshape((m * 4, -1))
        gmm = GaussianCluster(10, covariance_type="all").fit(res)
        assert gmm.n_components_ == 4

    run(diag_aug=False, scaled=False)
    run(diag_aug=True, scaled=False)
    run(diag_aug=False, scaled=True)
    run(diag_aug=True, scaled=True)


def test_vertex():
    """
    There should be 2 clusters since each graph is a 2 block model
    """
    n = [128, 128]
    m = 10

    def run(diag_aug, scaled):
        # undirected case
        np.random.seed(4 + diag_aug + scaled)
        X = make_train_undirected(n, m)
        # undirected case
        res = MultipleASE(n_components=2).fit_transform(X)
        gmm = GaussianCluster(10, covariance_type="all").fit(res)
        assert gmm.n_components_ == 2

        # directed case
        np.random.seed(5 + diag_aug + scaled)
        X = make_train_directed(n, m)
        res = MultipleASE(n_components=2, concat=True).fit_transform(X)
        gmm = GaussianCluster(10, covariance_type="all").fit(res)
        assert gmm.n_components_ == 2

    run(diag_aug=False, scaled=False)
    run(diag_aug=True, scaled=False)
    run(diag_aug=False, scaled=True)
    run(diag_aug=True, scaled=True)
