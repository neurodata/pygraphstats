# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import pytest

import numpy as np
from numpy.testing._private.utils import assert_equal
import networkx as nx

from graspologic.inference import affinity_test
from graspologic.simulations.simulations import sbm


def test_inputs():

    X = 5
    with pytest.raises(TypeError):
        affinity_test(X, "homophilic")

    X = nx.Graph()
    with pytest.raises(ValueError):
        affinity_test(X, "homophilic")

    X = np.zeros(
        1,
    )
    with pytest.raises(ValueError):
        affinity_test(X, "homophilic")

    X = np.ones((1, 5))
    with pytest.raises(ValueError):
        affinity_test(X, "homophilic")

    np.random.seed(10)
    n = [50, 50]
    p = [[0.5, 0.2], [0.2, 0.50]]
    X = sbm(n=n, p=p)

    test = 5
    with pytest.raises(TypeError):
        affinity_test(X, test)

    with pytest.raises(ValueError):
        affinity_test(X, "junk")

    X = nx.Graph()
    X.add_node(1)
    X.add_node(2)
    X.add_node(3)
    with pytest.raises(ValueError):
        affinity_test(X, "homophilic")

    n = [50, 50]
    p = [[0.5, 0.2], [0.2, 0.50]]
    X = sbm(n=n, p=p)

    with pytest.raises(TypeError):
        affinity_test(X, "homophilic", comms=5)

    with pytest.raises(ValueError):
        affinity_test(X, "homophilic", comms=np.zeros((1, 2, 3)))

    with pytest.raises(ValueError):
        affinity_test(X, "homophilic", comms=np.zeros((2, 2)))

    with pytest.raises(ValueError):
        affinity_test(X, "homophilic", comms=np.zeros((1,)))

    X = nx.Graph()
    X.add_node(1)
    X.add_node(2)
    with pytest.raises(ValueError):
        affinity_test(X, "homophilic", comms=np.zeros((3,)))

    n = [50, 50]
    p = [[0.5, 0.2], [0.2, 0.50]]
    X = sbm(n=n, p=p)
    with pytest.raises(ValueError):
        affinity_test(X, "homophilic", comms=np.zeros((1000,)))

    with pytest.raises(ValueError):
        affinity_test(X, "homophilic", comms=np.zeros((1000,)))


def test_outputs():
    np.random.seed(10)
    n = [50, 50]
    p = [[0.5, 0.2], [0.2, 0.50]]
    X = sbm(n=n, p=p)

    table, pvalue = affinity_test(X, "homophilic")

    assert type(table) == np.ndarray
    assert type(pvalue) == float

    assert_equal(table.shape, (2, 2))

    for i in table.ravel():
        assert type(i) == np.float64

    table, pvalue = affinity_test(X, "homotopic")

    assert type(table) == np.ndarray
    assert type(pvalue) == float

    assert_equal(table.shape, (2, 2))

    for i in table.ravel():
        assert type(i) == np.float64

    table, pvalue = affinity_test(X, "homophilic", comms=np.asarray([10, 90]))

    assert type(table) == np.ndarray
    assert type(pvalue) == float

    assert_equal(table.shape, (2, 2))

    for i in table.ravel():
        assert type(i) == np.float64

    table, pvalue = affinity_test(X, "homotopic", comms=np.asarray([10, 90]))

    assert type(table) == np.ndarray
    assert type(pvalue) == float

    assert_equal(table.shape, (2, 2))

    for i in table.ravel():
        assert type(i) == np.float64
