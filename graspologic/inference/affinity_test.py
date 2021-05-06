# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import NamedTuple, Union
import networkx as nx
import numpy as np
from scipy.stats import fisher_exact

_VALID_TESTS = ["homophilic", "homotopic"]


class DefinedContingencyTable(NamedTuple):
    """
    Contains the contingency table and p-value for a specific test.
    The table is a 2x2 numpy array and the p-value is a float.
    """

    con_table: np.ndarray
    pvalue: float


def affinity_test(
    graph: Union[nx.Graph, nx.DiGraph, np.ndarray],
    test: str,
    comms: np.ndarray = None,
) -> DefinedContingencyTable:
    """
    Returns a test of homophilic or homotopic affinity in a graph.

    Parameters
    ----------

    graph : Union[nx.Graph, nx.DiGraph, np.ndarray]
        The graph. We assume block communities.
    test : str
        Affinity test, either `homophilic` or `homotopic`.
    comms : np.ndarray
        Array of values indicating the sizes of the communities. If none is
        provided, it is assumed there are two communities, each taking up half
        of the nodes in the graph.

    Returns
    -------
    :class:`DefinedContingencyTable`
        A named tuple that contains the contingency table and the pvalue.

    References
    ----------
    .. [1] Chung, Jaewon, et al. “Statistical Connectomics.”
    OSF Preprints, 12 Aug. 2020. Web.
    """

    if not isinstance(graph, (nx.Graph, nx.DiGraph, np.ndarray)):
        msg = "Graph must be nx.Graph, nx.DiGraph, or np.ndarray, not {}."
        msg = msg.format(type(graph))
        raise TypeError(msg)

    # Catches nxgraphs with zero nodes also
    if isinstance(graph, (nx.Graph, nx.DiGraph)):
        if len(graph.nodes()) < 2:
            msg = "Graph must have more than one node."
            raise ValueError(msg)

    if isinstance(graph, np.ndarray) and graph.shape[0] < 2:
        msg = "Graph must have more than one node."
        raise ValueError(msg)

    if isinstance(graph, np.ndarray) and graph.shape[0] != graph.shape[1]:
        msg = "Graph must be square adjacency matrix. Dimensions not equal."
        raise ValueError(msg)

    if not isinstance(test, str):
        msg = "Test must be a str, not {}.".format(type(test))
        raise TypeError(msg)

    # Amount of tests to be expanded later
    if test not in _VALID_TESTS:
        msg = "Unknown test {}. Valid tests are {}.".format(test, _VALID_TESTS)
        raise ValueError(msg)

    # Assume even number of nodes, two equally sized hemispheres
    if comms is None and isinstance(graph, (nx.Graph, nx.DiGraph)):
        if len(graph.nodes) % 2 != 0:
            msg = (
                "Graph must have even number of nodes if no community array "
                "is provided."
            )
            raise ValueError(msg)

    if comms is not None and not isinstance(comms, np.ndarray):
        msg = "Community array must be np.ndarray, not {}.".format(type(comms))
        raise TypeError(msg)

    if comms is not None and len(comms.shape) > 2:
        msg = "Community array must be 1 or 2-dimensional."
        raise ValueError(msg)

    if comms is not None and len(comms.shape) > 1:
        if comms.shape[0] > 1 and comms.shape[1] > 1:
            msg = "Community array must be vector array, not a matrix."
            raise ValueError(msg)

    if comms is not None and len(np.unique(comms)) < 2:
        msg = "Must have more than one community."
        raise ValueError(msg)

    if comms is not None and isinstance(graph, (nx.Graph, nx.DiGraph)):
        if len(graph.nodes()) < len(comms):
            msg = "Must have at least as many nodes as number of communities."
            raise ValueError(msg)

    if comms is not None and isinstance(graph, np.ndarray):
        if len(graph.ravel()) < len(comms):
            msg = "Must have at least as many nodes as number of communities."
            raise ValueError(msg)

    if isinstance(graph, (nx.Graph, nx.DiGraph)):
        graph = nx.to_numpy_array(graph)

    n = graph.shape[0]
    if comms is None:
        comms = np.asarray([int(0.5 * n), int(0.5 * n)])

    # Contingency Table
    con_table = np.zeros((2, 2))

    con_table[0, 1], con_table[1, 1] = _calc_probs(graph, comms, test)

    con_table[:, 0] = 1 - con_table[:, 1]

    _, pvalue = fisher_exact(con_table)

    return DefinedContingencyTable(con_table=con_table, pvalue=pvalue)


def _calc_probs(graph, comms, test):
    graph_size = len(graph.ravel())

    # Unravel comms for iterating
    comms = comms.ravel()

    # Get indicies
    idx = np.cumsum(comms)

    edge = 0
    total = 0
    for i in range(len(comms) - 1):

        if i == 0:
            c1_0 = 0
        else:
            c1_0 = idx[i - 1]

        if test == "homophilic":

            block = graph[c1_0 : idx[i], c1_0 : idx[i]]
            edge += len(block[block != 0])
            total += comms[i] ** 2

        else:

            for j in range(i + 1, len(comms)):

                c2_0 = idx[j - 1]

                comm_size = int(comms[i])
                further = int(comms[j])

                move = min(comm_size, further)

                c1_c2 = graph[c1_0 : c1_0 + move, c2_0 : c2_0 + move]
                c2_c1 = graph[c2_0 : c2_0 + move, c1_0 : c1_0 + move]

                c1_bilateral = np.diag(c1_c2)
                c2_bilateral = np.diag(c2_c1)

                c1_homotopic_edge = len(c1_bilateral[c1_bilateral != 0])
                c2_homotopic_edge = len(c2_bilateral[c2_bilateral != 0])

                c1_homotopic_total = len(c1_bilateral)
                c2_homotopic_total = len(c2_bilateral)

                edge += c1_homotopic_edge + c2_homotopic_edge
                total += c1_homotopic_total + c2_homotopic_total

    edge_rest = len(graph[graph != 0]) - edge
    total_rest = graph_size - total

    # Convert to probabilities
    prob = edge / total
    prob_rest = edge_rest / total_rest

    return prob, prob_rest
