# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import pytest
import unittest
import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances

from graspologic.embed import AdjacencySpectralEmbed
from graspologic.inference import latent_distribution_test
from graspologic.simulations import er_np, sbm


class TestLatentDistributionTest(unittest.TestCase):
    @classmethod
    def test_ase_works(self):
        np.random.seed(888)
        A1 = er_np(20, 0.3)
        A2 = er_np(20, 0.3)
        tests = {"dcorr": "euclidean", "hsic": "gaussian", "mgc": "euclidean"}
        for test in tests.keys():
            ldt = latent_distribution_test(A1, A2, test, tests[test], n_bootstraps=10)

    def test_workers(self):
        np.random.seed(888)
        A1 = er_np(20, 0.3)
        A2 = er_np(20, 0.3)
        ldt = latent_distribution_test(
            A1, A2, "dcorr", "euclidean", n_bootstraps=4, workers=4
        )

    def test_callable_metric(self):
        np.random.seed(888)
        A1 = er_np(20, 0.3)
        A2 = er_np(20, 0.3)

        def metric_func(X, Y=None, workers=None):
            return pairwise_distances(X, metric="euclidean") * 0.5

        ldt = latent_distribution_test(A1, A2, "dcorr", metric_func, n_bootstraps=10)

    def test_bad_kwargs(self):
        np.random.seed(888)
        A1 = er_np(20, 0.3)
        A2 = er_np(20, 0.3)

        # check test argument
        with pytest.raises(TypeError):
            latent_distribution_test(A1, A2, test=0)
        with pytest.raises(ValueError):
            latent_distribution_test(A1, A2, test="foo")
        # check metric argument
        with pytest.raises(TypeError):
            latent_distribution_test(A1, A2, metric=0)
        with pytest.raises(ValueError):
            latent_distribution_test(A1, A2, metric="some_kind_of_kernel")
        # check n_components argument
        with pytest.raises(TypeError):
            latent_distribution_test(A1, A2, n_components=0.5)
        with pytest.raises(ValueError):
            latent_distribution_test(A1, A2, n_components=-100)
        # check n_bootstraps argument
        with pytest.raises(TypeError):
            latent_distribution_test(A1, A2, n_bootstraps=0.5)
        with pytest.raises(ValueError):
            latent_distribution_test(A1, A2, n_bootstraps=-100)
        # check workers argument
        with pytest.raises(TypeError):
            latent_distribution_test(A1, A2, workers=0.5)
            latent_distribution_test(A1, A2, workers="oops")
        # check size_correction argument
        with pytest.raises(TypeError):
            latent_distribution_test(A1, A2, size_correction=0)
        # check pooled argument
        with pytest.raises(TypeError):
            latent_distribution_test(A1, A2, pooled=0)
        # check align_type argument
        with pytest.raises(ValueError):
            latent_distribution_test(A1, A2, align_type="foo")
        with pytest.raises(TypeError):
            latent_distribution_test(A1, A2, align_type={"not a": "string"})
        # check align_kws argument
        with pytest.raises(TypeError):
            latent_distribution_test(A1, A2, align_kws="foo")
        # check input_graph argument
        with pytest.raises(TypeError):
            latent_distribution_test(A1, A2, input_graph="hello")

    def test_n_bootstraps(self):
        A1 = er_np(20, 0.3)
        A2 = er_np(20, 0.3)

        ldt = latent_distribution_test(A1, A2, n_bootstraps=123)
        assert ldt[2]["null_distribution"].shape[0] == 123

    def test_passing_networkx(self):
        np.random.seed(123)
        A1 = er_np(20, 0.8)
        A2 = er_np(20, 0.8)
        A1_nx = nx.from_numpy_matrix(A1)
        A2_nx = nx.from_numpy_matrix(A2)
        # check passing nx, when exepect embeddings
        with self.assertRaises(TypeError):
            latent_distribution_test(A1_nx, A1, input_graph=False)
        with self.assertRaises(TypeError):
            latent_distribution_test(A1, A2_nx, input_graph=False)
        with self.assertRaises(TypeError):
            latent_distribution_test(A1_nx, A2_nx, input_graph=False)
        # check that the appropriate input works
        latent_distribution_test(A1_nx, A2_nx, input_graph=True)

    def test_passing_embeddings(self):
        np.random.seed(123)
        A1 = er_np(20, 0.8)
        A2 = er_np(20, 0.8)
        ase_1 = AdjacencySpectralEmbed(n_components=2)
        X1 = ase_1.fit_transform(A1)
        ase_2 = AdjacencySpectralEmbed(n_components=2)
        X2 = ase_2.fit_transform(A2)
        ase_3 = AdjacencySpectralEmbed(n_components=1)
        X3 = ase_3.fit_transform(A2)
        # check embeddings having weird ndim
        with self.assertRaises(ValueError):
            latent_distribution_test(X1, X2.reshape(-1, 1, 1), input_graph=False)
        with self.assertRaises(ValueError):
            latent_distribution_test(X1.reshape(-1, 1, 1), X2, input_graph=False)
        # check embeddings having mismatching number of components
        with self.assertRaises(ValueError):
            latent_distribution_test(X1, X3, input_graph=False)
        with self.assertRaises(ValueError):
            latent_distribution_test(X3, X1, input_graph=False)
        # check passing weird stuff as input (caught by us)
        with self.assertRaises(TypeError):
            latent_distribution_test("hello there", X1, input_graph=False)
        with self.assertRaises(TypeError):
            latent_distribution_test(X1, "hello there", input_graph=False)
        with self.assertRaises(TypeError):
            latent_distribution_test({"hello": "there"}, X1, input_graph=False)
        with self.assertRaises(TypeError):
            latent_distribution_test(X1, {"hello": "there"}, input_graph=False)
        # check passing infinite in input (caught by check_array)
        with self.assertRaises(ValueError):
            X1_w_inf = X1.copy()
            X1_w_inf[1, 1] = np.inf
            latent_distribution_test(X1_w_inf, X2, input_graph=False)
        # check that the appropriate input works
        latent_distribution_test(X1, X2, input_graph=False)

    def test_pooled(self):
        np.random.seed(123)
        A1 = er_np(20, 0.3)
        A2 = er_np(100, 0.3)
        latent_distribution_test(A1, A2, pooled=True)

    def test_distances_and_kernels(self):
        np.random.seed(123)
        A1 = er_np(20, 0.3)
        A2 = er_np(100, 0.3)
        # some valid combinations of test and metric
        # # would love to do this, but currently FutureWarning breaks this
        # with pytest.warns(None) as record:
        #     for test in self.tests.keys():
        #         ldt = LatentDistributionTest(test, self.tests[test])
        #         ldt.fit(A1, A2)
        #     ldt = LatentDistributionTest("hsic", "rbf")
        #     ldt.fit(A1, A2)
        # assert len(record) == 0
        latent_distribution_test(A1, A2, test="hsic", metric="rbf")
        # some invalid combinations of test and metric
        with pytest.warns(UserWarning):
            latent_distribution_test(A1, A2, "hsic", "euclidean")
        with pytest.warns(UserWarning):
            latent_distribution_test(A1, A2, "dcorr", "gaussian")
        with pytest.warns(UserWarning):
            latent_distribution_test(A1, A2, "dcorr", "rbf")

    def test_bad_matrix_inputs(self):
        np.random.seed(1234556)
        A1 = er_np(20, 0.3)
        A2 = er_np(20, 0.3)

        bad_matrix = [[1, 2]]
        with self.assertRaises(TypeError):
            latent_distribution_test(bad_matrix, A2, test="dcorr")

    def test_directed_inputs(self):
        np.random.seed(2)
        A = er_np(100, 0.3, directed=True)
        B = er_np(100, 0.3, directed=True)
        C = er_np(100, 0.3, directed=False)

        # two directed graphs is okay
        latent_distribution_test(A, B)

        # an undirected and a direced graph is not okay
        with self.assertRaises(ValueError):
            latent_distribution_test(A, C)
        with self.assertRaises(ValueError):
            latent_distribution_test(C, B)

    def test_SBM_dcorr(self):
        np.random.seed(12345678)
        B1 = np.array([[0.5, 0.2], [0.2, 0.5]])

        B2 = np.array([[0.7, 0.2], [0.2, 0.7]])
        b_size = 200
        A1 = sbm(2 * [b_size], B1)
        A2 = sbm(2 * [b_size], B1)
        A3 = sbm(2 * [b_size], B2)

        # non-parallel test
        ldt_null = latent_distribution_test(A1, A2)
        ldt_alt = latent_distribution_test(A1, A3)
        self.assertTrue(ldt_null[0] > 0.05)
        self.assertTrue(ldt_alt[0] <= 0.05)

        # parallel test
        ldt_null = latent_distribution_test(A1, A2, workers=-1)
        ldt_alt = latent_distribution_test(A1, A3, workers=-1)
        self.assertTrue(ldt_null[0] > 0.05)
        self.assertTrue(ldt_alt[0] <= 0.05)

    def test_different_sizes_null(self):
        np.random.seed(314)

        A1 = er_np(100, 0.8)
        A2 = er_np(1000, 0.8)

        ldt_not_corrected = latent_distribution_test(
            A1,
            A2,
            test="hsic",
            metric="gaussian",
            n_components=2,
            n_bootstraps=100,
            size_correction=False,
        )
        ldt_corrected_1 = latent_distribution_test(
            A1,
            A2,
            test="hsic",
            metric="gaussian",
            n_components=2,
            n_bootstraps=100,
            size_correction=True,
        )
        ldt_corrected_2 = latent_distribution_test(
            A2,
            A1,
            test="hsic",
            metric="gaussian",
            n_components=2,
            n_bootstraps=100,
            size_correction=True,
        )

        self.assertTrue(ldt_not_corrected[0] <= 0.05)
        self.assertTrue(ldt_corrected_1[0] > 0.05)
        self.assertTrue(ldt_corrected_2[0] > 0.05)

    def test_different_sizes_null(self):
        np.random.seed(314)

        A1 = er_np(100, 0.8)
        A2 = er_np(1000, 0.7)

        ldt_corrected_1 = latent_distribution_test(
            A1,
            A2,
            test="hsic",
            metric="gaussian",
            n_components=2,
            n_bootstraps=100,
            size_correction=True,
        )
        ldt_corrected_2 = latent_distribution_test(
            A2,
            A1,
            test="hsic",
            metric="gaussian",
            n_components=2,
            n_bootstraps=100,
            size_correction=True,
        )

        self.assertTrue(ldt_corrected_1[0] <= 0.05)
        self.assertTrue(ldt_corrected_2[0] <= 0.05)

    def test_different_aligners(self):
        np.random.seed(314)
        A1 = er_np(100, 0.8)
        A2 = er_np(100, 0.8)
        ase_1 = AdjacencySpectralEmbed(n_components=2)
        X1 = ase_1.fit_transform(A1)
        ase_2 = AdjacencySpectralEmbed(n_components=2)
        X2 = ase_2.fit_transform(A2)
        X2 = -X2

        ldt_1 = latent_distribution_test(X1, X2, input_graph=False, align_type=None)
        self.assertTrue(ldt_1[0] < 0.05)

        ldt_2 = latent_distribution_test(
            X1, X2, input_graph=False, align_type="sign_flips"
        )
        self.assertTrue(ldt_2[0] >= 0.05)

        # also checking that kws are passed through
        ldt_3 = latent_distribution_test(
            X1,
            X2,
            input_graph=False,
            align_type="seedless_procrustes",
            align_kws={"init": "sign_flips"},
        )
        self.assertTrue(ldt_3[0] >= 0.05)


if __name__ == "__main__":
    unittest.main()
