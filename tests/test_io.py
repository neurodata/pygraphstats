# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import graspologic as gs


class TestImportGraph:
    @classmethod
    def setup_class(cls):
        # simple ERxN graph
        n = 15
        p = 0.5
        cls.A = np.zeros((n, n))
        nedge = int(round(n * n * p))
        np.put(
            cls.A,
            np.random.choice(np.arange(0, n * n), size=nedge, replace=False),
            np.random.normal(size=nedge),
        )

    def test_graphin(self):
        G = nx.from_numpy_array(self.A)
        np.testing.assert_array_equal(nx.to_numpy_array(G), gs.utils.import_graph(G))

    def test_npin(self):
        np.testing.assert_array_equal(self.A, gs.utils.import_graph(self.A))

    def test_wrongtypein(self):
        a = 5
        with pytest.raises(TypeError):
            gs.utils.import_graph(a)
        with pytest.raises(TypeError):
            gs.utils.import_graph(None)

    def test_nonsquare(self):
        non_square = np.hstack((self.A, self.A))
        with pytest.raises(ValueError):
            gs.utils.import_graph(non_square)


class TestImportEdgelist:
    @pytest.fixture(autouse=True)
    def setup_class(self, tmpdir):
        n = 10
        p = 0.5
        wt = np.random.exponential
        wtargs = dict(scale=4)

        np.random.seed(1)

        self.A = gs.simulations.er_np(n, p)
        self.B = gs.simulations.er_np(n, p, wt=wt, wtargs=wtargs)

        G_A = nx.from_numpy_array(self.A)
        G_B = nx.from_numpy_array(self.B)
        G_B = nx.relabel_nodes(G_B, lambda x: x + 10)  # relabel nodes to go from 10-19.

        self.A_path = str(tmpdir / "A_unweighted.edgelist")
        self.B_path = str(tmpdir / "B.edgelist")
        self.root = str(tmpdir)

        nx.write_edgelist(G_A, self.A_path, data=False)
        nx.write_weighted_edgelist(G_B, self.B_path)

    def test_in(self):
        A_from_edgelist = gs.utils.import_edgelist(self.A_path)
        B_from_edgelist = gs.utils.import_edgelist(self.B_path)

        assert np.allclose(A_from_edgelist, self.A)
        assert np.allclose(B_from_edgelist, self.B)

    def test_in_Path_obj(self):
        A_from_edgelist = gs.utils.import_edgelist(Path(self.A_path))
        B_from_edgelist = gs.utils.import_edgelist(Path(self.B_path))

        assert np.allclose(A_from_edgelist, self.A)
        assert np.allclose(B_from_edgelist, self.B)

    def test_multiple_in(self):
        graphs = gs.utils.import_edgelist(self.root)
        A = np.zeros((20, 20))
        A[:10, :10] = self.A

        B = np.zeros((20, 20))
        B[10:, 10:] = self.B

        assert len(graphs) == 2
        assert all(graph.shape == (20, 20) for graph in graphs)
        assert np.allclose(graphs[0], A)
        assert np.allclose(graphs[1], B)

    def test_wrongtypein(self):
        path = 5
        with pytest.raises(TypeError):
            gs.utils.import_edgelist(path)
        with pytest.raises(TypeError):
            gs.utils.import_edgelist(None)

    def test_vertices(self):
        expected_vertices_A = np.arange(0, 10)
        expected_vertices_B = np.arange(10, 20)

        _, A_vertices = gs.utils.import_edgelist(self.A_path, return_vertices=True)
        _, B_vertices = gs.utils.import_edgelist(self.B_path, return_vertices=True)

        assert np.allclose(expected_vertices_A, A_vertices)
        assert np.allclose(expected_vertices_B, B_vertices)

    def test_no_graphs_found(self):
        path = str(self.root + "invalid_edgelist.edgelist")
        with pytest.raises(ValueError):
            gs.utils.import_edgelist(path)

    def test_bad_delimiter(self):
        delimiter = ","
        with pytest.warns(UserWarning):
            graphs = gs.utils.import_edgelist(self.root, delimiter=delimiter)


class TestImportMultiGraph:
    @classmethod
    def setup_class(cls):
        # simple ERxN graph
        n1 = 15
        n2 = 20
        p = 0.5
        cls.A, cls.B = [gs.simulations.er_np(n1, p) for _ in range(2)]
        cls.C = gs.simulations.er_np(n2, p)  # different size

    def test_in(self):
        input_list = [self.A, self.B]
        input_array = np.array([self.A, self.B])

        assert isinstance(gs.utils.import_multigraphs(input_list), list)
        assert isinstance(gs.utils.import_multigraphs(input_array), np.ndarray)

    def test_wrongtypein(self):
        with pytest.raises(TypeError):
            gs.utils.import_multigraphs(5)
        with pytest.raises(TypeError):
            gs.utils.import_multigraphs(None)

    def test_one_graph(self):
        input_list = [self.A]
        input_arr = self.A
        input_tensor = np.array(self.A)

        with pytest.raises(ValueError):
            gs.utils.import_multigraphs(input_list)

        with pytest.raises(ValueError):
            gs.utils.import_multigraphs(input_arr)

        with pytest.raises(ValueError):
            gs.utils.import_multigraphs(input_tensor)

    def test_diff_sizes(self):
        input_list = [self.A, self.B, self.C]
        with pytest.raises(ValueError):
            gs.utils.import_multigraphs(input_list)
