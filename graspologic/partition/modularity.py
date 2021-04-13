# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import networkx as nx
from collections import defaultdict
from typing import Any, Dict


def _modularity_component(
    intra_community_degree: float,
    total_community_degree: float,
    network_degree_sum: float,
    resolution: float,
) -> float:
    community_degree_ratio = math.pow(total_community_degree, 2.0) / (
        2.0 * network_degree_sum
    )
    return (intra_community_degree - resolution * community_degree_ratio) / (
        2.0 * network_degree_sum
    )


def _assertions(
    graph: nx.Graph,
    partitions: Dict[Any, int],
    weight_attribute: str,
    resolution: float,
) -> None:
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a networkx undirected graph")
    if graph.is_directed():
        raise ValueError("The graph must be an undirected graph")
    if graph.is_multigraph():
        raise ValueError(
            "Multigraphs must be provided in the form of a non multigraph."
        )
    if not nx.is_weighted(graph, weight=weight_attribute):
        raise ValueError(
            f"weight_attribute {weight_attribute} not found on every edge in the provided graph"
        )
    if not isinstance(partitions, dict):
        raise TypeError("partitions must be a dictionary")
    if not isinstance(resolution, float):
        raise TypeError("resolution must be a float")


def modularity(
    graph: nx.Graph,
    partitions: Dict[Any, int],
    weight_attribute: str = "weight",
    resolution: float = 1.0,
) -> float:
    """
    Given an undirected graph and a dictionary of vertices to community ids, calculate
    the modularity.

    Parameters
    ----------
    graph : nx.Graph
        An undirected graph
    partitions : Dict[Any, int]
        A dictionary representing a community partitioning scheme with the keys being
        the vertex and the value being a community id.
    weight_attribute : str
        The edge data attribute on the graph that contains a float weight for the edge.
    resolution : float
        The resolution to use when calculating the modularity.

    Returns
    -------
    float
                The sum of the modularity of each of the communities.

    Raises
    ------
    TypeError
        If ``graph`` is not a networkx Graph or
        If ``partitions`` is not a dictionary or
        If ``resolution`` is not a float
    ValueError
        If ``graph`` is unweighted
        If ``graph`` is directed
        If ``graph`` is a multigraph

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Modularity_(networks)
    """
    _assertions(graph, partitions, weight_attribute, resolution)

    components = modularity_components(graph, partitions, weight_attribute, resolution)

    return sum(components.values())


def modularity_components(
    graph: nx.Graph,
    partitions: Dict[Any, int],
    weight_attribute: str = "weight",
    resolution: float = 1.0,
) -> Dict[int, float]:
    """
    Given an undirected, weighted graph and a community partition dictionary,
    calculates a modularity quantum for each community ID. The sum of these quanta
    is the modularity of the graph and partitions provided.

    Parameters
    ----------
    graph : nx.Graph
        An undirected graph
    partitions : Dict[Any, int]
        A dictionary representing a community partitioning scheme with the keys being
        the vertex and the value being a community id.
    weight_attribute : str
        The edge data attribute on the graph that contains a float weight for the edge.
    resolution : float
        The resolution to use when calculating the modularity.

    Returns
    -------
    Dict[int, float]
        A dictionary of the community id to the modularity component of that community

    Raises
    ------
    TypeError
        If ``graph`` is not a networkx Graph or
        If ``partitions`` is not a dictionary or
        If ``resolution`` is not a float
    ValueError
        If ``graph`` is unweighted
        If ``graph`` is directed
        If ``graph`` is a multigraph
    """
    _assertions(graph, partitions, weight_attribute, resolution)

    total_edge_weight = 0.0

    communities = set(partitions.values())

    degree_sums_within_community: Dict[int, float] = defaultdict(lambda: 0.0)
    degree_sums_for_community: Dict[int, float] = defaultdict(lambda: 0.0)
    for vertex, neighbor_vertex, weight in graph.edges(data=weight_attribute):
        vertex_community = partitions[vertex]
        neighbor_community = partitions[neighbor_vertex]
        if vertex_community == neighbor_community:
            if vertex == neighbor_vertex:
                degree_sums_within_community[vertex_community] += weight
            else:
                degree_sums_within_community[vertex_community] += weight * 2.0
        degree_sums_for_community[vertex_community] += weight
        degree_sums_for_community[neighbor_community] += weight
        total_edge_weight += weight

    return {
        comm: _modularity_component(
            degree_sums_within_community[comm],
            degree_sums_for_community[comm],
            total_edge_weight,
            resolution,
        )
        for comm in communities
    }
