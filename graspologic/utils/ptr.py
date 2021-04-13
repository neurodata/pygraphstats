# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
from scipy.stats import rankdata

from .utils import import_graph, is_unweighted, is_symmetric, is_loopless, symmetrize, GraphRepresentation


def pass_to_ranks(
    graph: GraphRepresentation,
    method: str = "simple-nonzero"
) -> GraphRepresentation:
    """
    Rescales edge weights of an adjacency matrix based on their relative rank in
    the graph.

    Parameters
    ----------
    graph: array_like or networkx.Graph
        Adjacency matrix

    method: {'simple-nonzero' (default), 'simple-all', 'zero-boost'} string, optional

        - 'simple-nonzero'
            assigns ranks to all non-zero edges, settling ties using
            the average. Ranks are then scaled by
            :math:`\frac{rank(\text{non-zero edges})}{\text{total non-zero edges} + 1}`
        - 'simple-all'
            assigns ranks to all non-zero edges, settling ties using
            the average. Ranks are then scaled by
            :math:`\frac{rank(\text{non-zero edges})}{n^2 + 1}`
            where n is the number of nodes
        - 'zero-boost'
            preserves the edge weight for all 0s, but ranks the other
            edges as if the ranks of all 0 edges has been assigned. If there are
            10 0-valued edges, the lowest non-zero edge gets weight 11 / (number
            of possible edges). Ties settled by the average of the weight that those
            edges would have received. Number of possible edges is determined
            by the type of graph (loopless or looped, directed or undirected).

    See also
    --------
    scipy.stats.rankdata

    Returns
    -------
    graph: numpy.ndarray, shape(n_vertices, n_vertices)
        Adjacency matrix of graph after being passed to ranks
    """

    graph = import_graph(graph)  # just for typechecking

    if is_unweighted(graph):
        return graph

    if graph.min() < 0:
        raise UserWarning(
            "Current pass-to-ranks on graphs with negative"
            + " weights will yield nonsensical results, especially for zero-boost"
        )

    if method == "zero-boost":
        if is_symmetric(graph):
            # start by working with half of the graph, since symmetric
            triu = np.triu(graph)
            non_zeros = triu[triu != 0]
        else:
            non_zeros = graph[graph != 0]
        rank = rankdata(non_zeros)

        if is_symmetric(graph):
            if is_loopless(graph):
                num_zeros = (len(graph[graph == 0]) - graph.shape[0]) / 2
                possible_edges = graph.shape[0] * (graph.shape[0] - 1) / 2
            else:
                num_zeros = (
                    len(triu[triu == 0]) - graph.shape[0] * (graph.shape[0] - 1) / 2
                )
                possible_edges = graph.shape[0] * (graph.shape[0] + 1) / 2
        else:
            if is_loopless(graph):
                # n^2 - num_nonzero - num_diagonal
                num_zeros = graph.size - len(non_zeros) - graph.shape[0]
                # n^2 - num_diagonal
                possible_edges = graph.size - graph.shape[0]
            else:
                num_zeros = graph.size - len(non_zeros)
                possible_edges = graph.size

        # shift up by the number of zeros
        rank = rank + num_zeros
        # normalize by the number of possible edges for this kind of graph
        rank = rank / possible_edges
        # put back into matrix form (and reflect over the diagonal if necessary)
        if is_symmetric(graph):
            triu[triu != 0] = rank
            graph = symmetrize(triu, method="triu")
        else:
            graph[graph != 0] = rank
        return graph
    elif method in ["simple-all", "simple-nonzero"]:
        non_zeros = graph[graph != 0]
        rank = rankdata(non_zeros)
        if method == "simple-all":
            normalizer = graph.size
        elif method == "simple-nonzero":
            normalizer = rank.shape[0]
        rank = rank / (normalizer + 1)
        graph[graph != 0] = rank
        return graph
    else:
        raise ValueError("Unsuported pass-to-ranks method")
