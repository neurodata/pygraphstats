# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import pytest
import numpy as np
from numpy.testing import assert_allclose
from graspologic.models import (
    EREstimator,
    DCSBMEstimator,
    SBMEstimator,
    RDPGEstimator,
    DCEREstimator,
    HSBMEstimator,
)
from graspologic.simulations import er_np, sbm, sample_edges
from graspologic.utils import cartesian_product, is_symmetric
from sklearn.metrics import adjusted_rand_score
from sklearn.exceptions import NotFittedError


class TestER:
    @classmethod
    def setup_class(cls):
        np.random.seed(8888)
        cls.graph = er_np(1000, 0.5)
        cls.p = 0.5
        cls.p_mat = np.full((1000, 1000), 0.5)
        cls.estimator = EREstimator(directed=True, loops=False)
        cls.estimator.fit(cls.graph)
        cls.p_hat = cls.estimator.p_

    def test_ER_inputs(self):
        ere = EREstimator()

        with pytest.raises(TypeError):
            EREstimator(directed="hey")

        with pytest.raises(TypeError):
            EREstimator(loops=6)

        with pytest.raises(ValueError):
            ere.fit(self.graph[:, :99])

        with pytest.raises(ValueError):
            ere.fit(self.graph[..., np.newaxis])

    def test_ER_fit(self):
        assert self.p_hat - self.p < 0.001

    def test_ER_sample(self):
        with pytest.raises(ValueError):
            self.estimator.sample(n_samples=-1)

        with pytest.raises(TypeError):
            self.estimator.sample(n_samples="nope")
        g = er_np(100, 0.5)
        estimator = EREstimator(directed=True, loops=False)
        estimator.fit(g)
        p_mat = np.full((100, 100), 0.5)
        p_mat -= np.diag(np.diag(p_mat))
        _test_sample(estimator, p_mat)

    def test_ER_score(self):
        p_mat = self.p_mat
        graph = self.graph
        estimator = EREstimator(directed=False)
        _test_score(estimator, p_mat, graph)

        with pytest.raises(ValueError):
            estimator.score_samples(graph=er_np(500, 0.5))

    def test_ER_nparams(self):
        assert self.estimator._n_parameters() == 1


class TestDCER:
    @classmethod
    def setup_class(cls):
        np.random.seed(8888)
        n = 1000
        p = 0.5
        dc = np.random.beta(2, 5, size=n)
        p_mat = np.full((n, n), p)
        p_mat = p_mat * np.outer(dc, dc)
        p_mat -= np.diag(np.diag(p_mat))
        graph = sample_edges(p_mat, directed=True, loops=False)
        cls.p_mat = p_mat
        cls.graph = graph

    def test_DCER_score(self):
        p_mat = self.p_mat
        graph = self.graph
        estimator = DCEREstimator()
        _test_score(estimator, p_mat, graph)

        with pytest.raises(ValueError):
            estimator.score_samples(graph=graph[1:500, 1:500])

    def test_DCER_inputs(self):
        with pytest.raises(TypeError):
            DCEREstimator(directed="hey")

        with pytest.raises(TypeError):
            DCEREstimator(loops=6)

        graph = er_np(100, 0.5)
        dcere = DCEREstimator()

        with pytest.raises(ValueError):
            dcere.fit(graph[:, :99])

        with pytest.raises(ValueError):
            dcere.fit(graph[..., np.newaxis])

    def test_DCER_fit(self):
        np.random.seed(8888)
        graph = self.graph
        p_mat = self.p_mat
        dcsbe = DCSBMEstimator(directed=True, loops=False)
        dcsbe.fit(graph)
        assert_allclose(p_mat, dcsbe.p_mat_, atol=0.12)

    def test_DCER_sample(self):
        np.random.seed(8888)
        estimator = DCEREstimator(directed=True, loops=False)
        g = self.graph
        p_mat = self.p_mat
        with pytest.raises(NotFittedError):
            estimator.sample()

        estimator.fit(g)
        with pytest.raises(ValueError):
            estimator.sample(n_samples=-1)

        with pytest.raises(TypeError):
            estimator.sample(n_samples="nope")
        B = 0.5
        dc = np.random.uniform(0.25, 0.75, size=100)
        p_mat = np.outer(dc, dc) * B
        p_mat -= np.diag(np.diag(p_mat))
        g = sample_edges(p_mat, directed=True)
        estimator.fit(g)
        estimator.p_mat_ = p_mat
        _test_sample(estimator, p_mat, n_samples=1000, atol=0.2)

    def test_DCER_nparams(self):
        n_verts = 1000
        graph = self.graph
        e = DCEREstimator(directed=True)
        e.fit(graph)
        assert e._n_parameters() == (n_verts + 1)


class TestSBM:
    @classmethod
    def setup_class(cls):
        estimator = SBMEstimator(directed=True, loops=False)
        B = np.array([[0.9, 0.1], [0.1, 0.9]])
        g = sbm([50, 50], B, directed=True)
        labels = _n_to_labels([50, 50])
        p_mat = _block_to_full(B, labels, (100, 100))
        p_mat -= np.diag(np.diag(p_mat))
        cls.estimator = estimator
        cls.p_mat = p_mat
        cls.graph = g
        cls.labels = labels

    def test_SBM_inputs(self):
        with pytest.raises(TypeError):
            SBMEstimator(directed="hey")

        with pytest.raises(TypeError):
            SBMEstimator(loops=6)

        with pytest.raises(TypeError):
            SBMEstimator(n_components="XD")

        with pytest.raises(ValueError):
            SBMEstimator(n_components=-1)

        with pytest.raises(TypeError):
            SBMEstimator(min_comm="1")

        with pytest.raises(ValueError):
            SBMEstimator(min_comm=-1)

        with pytest.raises(TypeError):
            SBMEstimator(max_comm="ay")

        with pytest.raises(ValueError):
            SBMEstimator(max_comm=-1)

        with pytest.raises(ValueError):
            SBMEstimator(min_comm=4, max_comm=2)

        graph = er_np(100, 0.5)
        bad_y = np.zeros(99)
        sbe = SBMEstimator()
        with pytest.raises(ValueError):
            sbe.fit(graph, y=bad_y)

        with pytest.raises(ValueError):
            sbe.fit(graph[:, :99])

        with pytest.raises(ValueError):
            sbe.fit(graph[..., np.newaxis])

        with pytest.raises(TypeError):
            SBMEstimator(cluster_kws=1)

        with pytest.raises(TypeError):
            SBMEstimator(embed_kws=1)

    def test_SBM_fit_supervised(self):
        np.random.seed(8888)
        B = np.array(
            [
                [0.9, 0.2, 0.05, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.2, 0.4, 0.8, 0.5],
                [0.1, 0.2, 0.1, 0.7],
            ]
        )
        n = np.array([500, 500, 250, 250])
        g = sbm(n, B, directed=True, loops=False)
        sbe = SBMEstimator(directed=True, loops=False)
        labels = _n_to_labels(n)
        sbe.fit(g, y=labels)
        B_hat = sbe.block_p_
        assert_allclose(B_hat, B, atol=0.01)

    def test_SBM_fit_unsupervised(self):
        np.random.seed(12345)
        n_verts = 1500

        B = np.array([[0.7, 0.1, 0.1], [0.1, 0.9, 0.1], [0.05, 0.1, 0.75]])
        n = np.array([500, 500, 500])
        labels = _n_to_labels(n)
        p_mat = _block_to_full(B, labels, (n_verts, n_verts))
        p_mat -= np.diag(np.diag(p_mat))
        graph = sample_edges(p_mat, directed=True, loops=False)
        sbe = SBMEstimator(directed=True, loops=False)
        sbe.fit(graph)
        assert adjusted_rand_score(labels, sbe.vertex_assignments_) > 0.95
        assert_allclose(p_mat, sbe.p_mat_, atol=0.12)

    def test_SBM_sample(self):
        estimator = self.estimator
        g = self.graph
        p_mat = self.p_mat
        labels = self.labels
        with pytest.raises(NotFittedError):
            estimator.sample()

        estimator.fit(g, y=labels)
        with pytest.raises(ValueError):
            estimator.sample(n_samples=-1)

        with pytest.raises(TypeError):
            estimator.sample(n_samples="nope")

        _test_sample(estimator, p_mat)

    def test_SBM_score(self):
        # tests score() and score_sample()
        B = np.array([[0.75, 0.25], [0.25, 0.75]])
        n_verts = 100
        n = np.array([n_verts, n_verts])
        tau = _n_to_labels(n)
        p_mat = _block_to_full(B, tau, shape=(n_verts * 2, n_verts * 2))
        graph = sample_edges(p_mat, directed=True)
        estimator = SBMEstimator(max_comm=4)
        _test_score(estimator, p_mat, graph)

        with pytest.raises(ValueError):
            estimator.score_samples(graph=graph[1:100, 1:100])

    def test_SBM_nparams(self):
        e = self.estimator.fit(self.graph, y=self.labels)
        assert e._n_parameters() == (4)
        e = SBMEstimator()
        e.fit(self.graph)
        assert e._n_parameters() == (4 + 1)
        e = SBMEstimator(directed=False)
        e.fit(self.graph)
        assert e._n_parameters() == (1 + 3)


class TestDCSBM:
    @classmethod
    def setup_class(cls):
        np.random.seed(8888)
        B = np.array(
            [
                [0.9, 0.2, 0.05, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.2, 0.4, 0.8, 0.5],
                [0.1, 0.2, 0.1, 0.7],
            ]
        )
        n = np.array([1000, 1000, 500, 500])
        dc = np.random.beta(2, 5, size=n.sum())
        labels = _n_to_labels(n)
        p_mat = _block_to_full(B, labels, (n.sum(), n.sum()))
        p_mat = p_mat * np.outer(dc, dc)
        p_mat -= np.diag(np.diag(p_mat))
        g = sample_edges(p_mat, directed=True, loops=False)
        cls.p_mat = p_mat
        cls.labels = labels
        cls.g = g

    def test_DCSBM_score(self):
        p_mat = self.p_mat
        graph = self.g
        estimator = DCSBMEstimator()
        _test_score(estimator, p_mat, graph)

        with pytest.raises(ValueError):
            estimator.score_samples(graph=graph[1:100, 1:100])

    def test_DCSBM_fit_supervised(self):
        p_mat = self.p_mat
        labels = self.labels
        g = self.g
        dcsbe = DCSBMEstimator(directed=True, loops=False)
        dcsbe.fit(g, y=labels)
        assert_allclose(dcsbe.p_mat_, p_mat, atol=0.1)

    def test_DCSBM_inputs(self):
        with pytest.raises(TypeError):
            DCSBMEstimator(directed="hey")

        with pytest.raises(TypeError):
            DCSBMEstimator(loops=6)

        with pytest.raises(TypeError):
            DCSBMEstimator(n_components="XD")

        with pytest.raises(ValueError):
            DCSBMEstimator(n_components=-1)

        with pytest.raises(TypeError):
            DCSBMEstimator(min_comm="1")

        with pytest.raises(ValueError):
            DCSBMEstimator(min_comm=-1)

        with pytest.raises(TypeError):
            DCSBMEstimator(max_comm="ay")

        with pytest.raises(ValueError):
            DCSBMEstimator(max_comm=-1)

        with pytest.raises(ValueError):
            DCSBMEstimator(min_comm=4, max_comm=2)

        graph = er_np(100, 0.5)
        bad_y = np.zeros(99)
        dcsbe = DCSBMEstimator()
        with pytest.raises(ValueError):
            dcsbe.fit(graph, y=bad_y)

        with pytest.raises(ValueError):
            dcsbe.fit(graph[:, :99])

        with pytest.raises(ValueError):
            dcsbe.fit(graph[..., np.newaxis])

        with pytest.raises(TypeError):
            DCSBMEstimator(cluster_kws=1)

        with pytest.raises(TypeError):
            DCSBMEstimator(embed_kws=1)

    def test_DCSBM_fit_unsupervised(self):
        np.random.seed(12345)
        n_verts = 1500

        distances = np.random.beta(4, 1, n_verts)
        B = np.array([[0.7, 0.1, 0.1], [0.1, 0.9, 0.1], [0.05, 0.1, 0.75]])
        n = np.array([500, 500, 500])
        labels = _n_to_labels(n)
        p_mat = _block_to_full(B, labels, (n_verts, n_verts))
        p_mat = p_mat * np.outer(distances, distances)
        p_mat -= np.diag(np.diag(p_mat))
        graph = sample_edges(p_mat, directed=True, loops=False)
        dcsbe = DCSBMEstimator(directed=True, loops=False)
        dcsbe.fit(graph)
        assert adjusted_rand_score(labels, dcsbe.vertex_assignments_) > 0.95
        assert_allclose(p_mat, dcsbe.p_mat_, atol=0.12)

    def test_DCSBM_sample(self):
        np.random.seed(8888)
        estimator = DCSBMEstimator(directed=True, loops=False)
        B = np.array([[0.9, 0.1], [0.1, 0.9]])
        dc = np.random.uniform(0.25, 0.75, size=100)
        labels = _n_to_labels([50, 50])

        p_mat = _block_to_full(B, labels, (100, 100))
        p_mat = p_mat * np.outer(dc, dc)
        p_mat -= np.diag(np.diag(p_mat))
        g = sample_edges(p_mat, directed=True)

        with pytest.raises(NotFittedError):
            estimator.sample()

        estimator.fit(g, y=labels)
        with pytest.raises(ValueError):
            estimator.sample(n_samples=-1)

        with pytest.raises(TypeError):
            estimator.sample(n_samples="nope")
        estimator.p_mat_ = p_mat
        _test_sample(estimator, p_mat, n_samples=1000, atol=0.1)

    def test_DCSBM_nparams(self):
        n_verts = 3000
        n_class = 4
        graph = self.g
        labels = self.labels
        e = DCSBMEstimator(directed=True)
        e.fit(graph)
        assert e._n_parameters() == (n_verts + n_class - 1 + n_class ** 2)

        e = DCSBMEstimator(directed=True)
        e.fit(graph, y=labels)
        assert e._n_parameters() == (n_verts + n_class ** 2)

        e = DCSBMEstimator(directed=True, degree_directed=True)
        e.fit(graph, y=labels)
        assert e._n_parameters() == (2 * n_verts + n_class ** 2)

        e = DCSBMEstimator(directed=False)
        e.fit(graph, y=labels)
        assert e._n_parameters() == (n_verts + 10)


class TestHSBM:
    @classmethod
    def setup_class(cls):
        # a simple 2-level random graph with 2 block
        # each of which contains 2 blocks
        np.random.seed(9)
        n = np.array([100, 100]).astype(int)
        block_labels = _n_to_labels(n)
        n_verts = np.sum(n)
        global_p = 0.01
        prop = np.array([[0.5, 0.5], [0.5, 0.5]])
        B = [np.array([[0.5, 0.1], [0.1, 0.6]]), np.array([[0.8, 0.1], [0.1, 0.8]])]

        graph = er_np(n_verts, global_p)
        for i, n_sub_verts in enumerate(n):
            p = prop[i, :]
            n_vec = (n_sub_verts * p).astype(int)
            subgraph = sbm(n_vec, B[i])
            inds = block_labels == i
            graph[np.ix_(inds, inds)] = subgraph

        labels = np.vstack((np.repeat((0, 1), 100), np.repeat((0, 1, 2, 3), 50))).T
        B_full = np.zeros((4, 4)) + 0.01
        B_full[:2, :2] = B[0]
        B_full[2:4, 2:4] = B[1]
        p_mat_lvl1 = _block_to_full(B_full, labels[:, 0], (200, 200))
        p_mat_lvl1 -= np.diag(np.diag(p_mat_lvl1))
        p_mat_lvl2 = _block_to_full(B_full, labels[:, 1], (200, 200))
        p_mat_lvl2 -= np.diag(np.diag(p_mat_lvl2))
        p_mat = [p_mat_lvl1, p_mat_lvl2]
        cls.graph = graph
        cls.p_mat = p_mat
        cls.labels = labels
        cls.B = B

    def test_HSBM_inputs(self):
        # directed not bool
        with pytest.raises(TypeError):
            HSBMEstimator(directed="1")

        # loops not bool
        with pytest.raises(TypeError):
            HSBMEstimator(loops=6)

        # min_components not integer
        with pytest.raises(TypeError):
            HSBMEstimator(min_components="1")

        # min_components < 1
        with pytest.raises(ValueError):
            HSBMEstimator(min_components=0)

        # max_components not integer
        with pytest.raises(TypeError):
            HSBMEstimator(max_components="1")

        # max_components < min_components
        with pytest.raises(ValueError):
            HSBMEstimator(min_components=1, max_components=0)

        # cluster_method not in ['gmm', 'kmeans']
        with pytest.raises(ValueError):
            HSBMEstimator(cluster_method="graspologic")

        # embed_method not in ['ase', 'lse']
        with pytest.raises(ValueError):
            HSBMEstimator(cluster_method="graspologic")

        # delta_criter negative
        with pytest.raises(ValueError):
            HSBMEstimator(delta_criter=-1)

        # cluster_kws not a dict
        with pytest.raises(TypeError):
            HSBMEstimator(cluster_kws=0)

        # embed_kws not a dict
        with pytest.raises(TypeError):
            HSBMEstimator(embed_kws=0)

        # Generate random graph
        graph = er_np(50, 0.01)
        # max_components > n_verts
        with pytest.raises(ValueError):
            hsbme = HSBMEstimator(max_components=51)
            hsbme.fit(graph)

        # y with invalid shape
        y = np.ones((49, 1))
        with pytest.raises(ValueError):
            hsbme = HSBMEstimator()
            hsbme.fit(graph, y=y)

    def test_HSBM_fit_supervised(self):
        graph = self.graph
        B = self.B
        hsbm = HSBMEstimator(max_level=2)
        labels = self.labels
        hsbm.fit(graph, labels)
        B_hat = hsbm.block_p_[1]
        assert_allclose(B_hat[:2, :2], B[0], atol=0.1)
        assert_allclose(B_hat[2:4, 2:4], B[1], atol=0.1)

    def _test_HSBM_fit_unsupervised(self, **params):
        hsbm = HSBMEstimator(max_level=2, **params)
        graph = self.graph
        labels = self.labels
        p_mat = self.p_mat
        hsbm.fit(graph)
        y = hsbm.vertex_assignments_
        assert adjusted_rand_score(labels[:, 0], y[:, 0]) > 0.95
        assert adjusted_rand_score(labels[:, 1], y[:, 1]) > 0.95
        assert_allclose(p_mat[0], hsbm.p_mat_[0], atol=0.19)
        assert_allclose(p_mat[1], hsbm.p_mat_[1], atol=0.1)

    def test_HSBM_fit_unsupervised_reembed(self):
        self._test_HSBM_fit_unsupervised(reembed=True)

    def test_HSBM_fit_unsupervised_no_reembed(self):
        self._test_HSBM_fit_unsupervised(reembed=False)

    def test_HSBM_sample(self):
        estimator = HSBMEstimator(max_level=2)
        g = self.graph
        p_mat = self.p_mat
        labels = self.labels
        with pytest.raises(NotFittedError):
            estimator.sample()

        estimator.fit(g, y=labels)
        with pytest.raises(ValueError):
            estimator.sample(n_samples=-1)

        with pytest.raises(TypeError):
            estimator.sample(n_samples="1")

        estimator.fit(g)
        np.random.seed(8888)
        estimator.fit(g)
        graphs = estimator.sample(1000)[1]
        graph_mean = graphs.mean(axis=0)

        assert_allclose(graph_mean, p_mat[1], atol=0.1)

    def test_HSBM_score(self):
        # tests score() and score_sample()
        p_mat = self.p_mat[1]
        graph = self.graph
        estimator = HSBMEstimator(max_level=2)
        _test_HSBM_score(estimator, p_mat, graph)

        with pytest.raises(ValueError):
            estimator.score_samples(graph=graph[1:100, 1:100])

    def test_HSBM_nparams(self):
        hsbme = HSBMEstimator(max_level=1)
        hsbme.fit(self.graph, y=self.labels)
        assert hsbme._n_parameters(y=self.labels)[0] == (5)

        hsbme = HSBMEstimator(max_level=1, directed=True)
        hsbme.fit(self.graph)
        assert hsbme._n_parameters()[0] == (7)

        hsbme = HSBMEstimator(max_level=1)
        hsbme.fit(self.graph)
        assert hsbme._n_parameters()[0] == (6)

        hsbme = HSBMEstimator(max_level=1, reembed=False)
        hsbme.fit(self.graph)
        assert hsbme._n_parameters()[0] == (5)


class TestRDPG:
    @classmethod
    def setup_class(cls):
        np.random.seed(8888)
        n_verts = 500
        point1 = np.array([0.1, 0.9])
        point2 = np.array([0.9, 0.1])
        latent1 = np.tile(point1, reps=(n_verts, 1))
        latent2 = np.tile(point2, reps=(n_verts, 1))
        latent = np.concatenate((latent1, latent2), axis=0)
        p_mat = latent @ latent.T
        p_mat -= np.diag(np.diag(p_mat))
        g = sample_edges(p_mat)
        cls.p_mat = p_mat
        cls.graph = g

    def test_RDPG_intputs(self):
        rdpge = RDPGEstimator()

        with pytest.raises(TypeError):
            RDPGEstimator(loops=6)

        with pytest.raises(ValueError):
            rdpge.fit(self.graph[:, :99])

        with pytest.raises(ValueError):
            rdpge.fit(self.graph[..., np.newaxis])

        with pytest.raises(TypeError):
            RDPGEstimator(ase_kws=5)

        with pytest.raises(TypeError):
            RDPGEstimator(diag_aug_weight="f")

        with pytest.raises(ValueError):
            RDPGEstimator(diag_aug_weight=-1)

        with pytest.raises(TypeError):
            RDPGEstimator(plus_c_weight="F")

        with pytest.raises(ValueError):
            RDPGEstimator(plus_c_weight=-1)

    def test_RDPG_fit(self):
        np.random.seed(8888)
        n_points = 2000
        dists = np.random.uniform(0, 1, n_points)
        points = hardy_weinberg(dists)

        p_mat = points @ points.T
        p_mat -= np.diag(np.diag(p_mat))
        g = sample_edges(p_mat)

        estimator = RDPGEstimator(loops=False, n_components=3)
        estimator.fit(g)

        assert_allclose(estimator.p_mat_, p_mat, atol=0.2)

    def test_RDPG_sample(self):
        np.random.seed(8888)
        g = self.graph
        p_mat = self.p_mat
        estimator = RDPGEstimator(n_components=2)
        estimator.fit(g)
        _test_sample(estimator, p_mat, atol=0.2, n_samples=200)

    def test_RDPG_score(self):
        p_mat = self.p_mat
        graph = self.graph
        estimator = RDPGEstimator()
        _test_score(estimator, p_mat, graph)

        with pytest.raises(ValueError):
            estimator.score_samples(graph=graph[1:100, 1:100])

    def test_RDPG_nparams(self):
        n_verts = 1000
        g = self.graph
        e = RDPGEstimator(n_components=2)
        e.fit(g)
        assert e._n_parameters() == n_verts * 2
        g[100:, 50:] = 1
        e = RDPGEstimator(n_components=2)
        e.fit(g)
        assert e._n_parameters() == n_verts * 4


def _n_to_labels(n):
    n = np.array(n)
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels


def _block_to_full(block_mat, inverse, shape):
    """
    "blows up" a k x k matrix, where k is the number of communities,
    into a full n x n probability matrix

    block mat : k x k
    inverse : array like length n,
    """
    block_map = cartesian_product(inverse, inverse).T
    mat_by_edge = block_mat[block_map[0], block_map[1]]
    full_mat = mat_by_edge.reshape(shape)
    return full_mat


def _test_sample(estimator, p_mat, atol=0.1, n_samples=1000):
    np.random.seed(8888)
    graphs = estimator.sample(n_samples)
    graph_mean = graphs.mean(axis=0)

    assert_allclose(graph_mean, p_mat, atol=atol)


def _test_score(estimator, p_mat, graph):
    np.random.seed(8888)
    graph = graph.copy()
    p_mat = p_mat.copy()
    estimator.fit(graph)
    estimator.p_mat_ = p_mat  # hack just for testing likelihood

    if is_symmetric(graph):
        inds = np.triu_indices_from(graph, k=1)
    else:
        xu, yu = np.triu_indices_from(graph, k=1)
        xl, yl = np.tril_indices_from(graph, k=-1)
        x = np.concatenate((xl, xu))
        y = np.concatenate((yl, yu))
        inds = (x, y)

    p_rav = p_mat[inds]
    g_rav = graph[inds]

    lik = np.zeros(g_rav.shape)
    c = 1 / p_mat.size
    for i, (g, p) in enumerate(zip(g_rav, p_rav)):
        if p < c:
            p = c
        if p > 1 - c:
            p = 1 - c
        if g == 1:
            lik[i] = p
        else:
            lik[i] = 1 - p

    # lik = np.reshape(lik_rav, p_mat.shape)
    lik[lik < 1e-10] = 1
    lik = np.log(lik)
    assert_allclose(lik, estimator.score_samples(graph))
    assert np.sum(lik) == estimator.score(graph)


def _test_HSBM_score(estimator, p_mat, graph):
    np.random.seed(8888)
    graph = graph.copy()
    p_mat = p_mat.copy()
    estimator.fit(graph)
    estimator.p_mat_ = [p_mat, p_mat]

    if is_symmetric(graph):
        inds = np.triu_indices_from(graph, k=1)
    else:
        xu, yu = np.triu_indices_from(graph, k=1)
        xl, yl = np.tril_indices_from(graph, k=-1)
        x = np.concatenate((xl, xu))
        y = np.concatenate((yl, yu))
        inds = (x, y)

    p_rav = p_mat[inds]
    g_rav = graph[inds]

    lik = np.zeros(g_rav.shape)
    c = 1 / p_mat.size
    for i, (g, p) in enumerate(zip(g_rav, p_rav)):
        if p < c:
            p = c
        if p > 1 - c:
            p = 1 - c
        if g == 1:
            lik[i] = p
        else:
            lik[i] = 1 - p

    lik[lik < 1e-10] = 1
    lik = np.log(lik)
    assert_allclose(lik, estimator.score_samples(graph)[1])
    assert np.sum(lik) == estimator.score(graph)[1]


def hardy_weinberg(theta):
    """
    Maps a value from [0, 1] to the hardy weinberg curve.
    """
    return np.array([theta ** 2, 2 * theta * (1 - theta), (1 - theta) ** 2]).T
