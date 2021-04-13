# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import networkx as nx
import numpy as np
import scipy
from .. import utils

import graspologic_native as gn


def _validate_and_build_edge_list(
    graph: Union[
        List[Tuple[Any, Any, Union[int, float]]],
        nx.Graph,
        np.ndarray,
        scipy.sparse.csr.csr_matrix,
    ],
    is_weighted: Optional[bool],
    weight_attribute: str,
    check_directed: bool,
    weight_default: float,
) -> Tuple[Dict[str, Any], List[Tuple[str, str, float]]]:
    if isinstance(graph, (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError("directed or multigraphs are not supported in these functions")
    if (
        isinstance(graph, (np.ndarray, scipy.sparse.csr.csr_matrix))
        and check_directed is True
        and not utils.is_almost_symmetric(graph)
    ):
        raise ValueError(
            "leiden only supports undirected graphs and the adjacency matrix provided "
            "was found to be directed"
        )

        # if we're already an edge list of str, str, float, return it
    if weight_default is not None and not isinstance(weight_default, (float, int)):
        raise TypeError("weight default must be a float or int")

    if isinstance(graph, list):
        if len(graph) == 0:
            return {}, []
        if not isinstance(graph[0], tuple) or len(graph[0]) != 3:
            raise TypeError(
                "If the provided graph is a list, it must be a list of tuples with 3 "
                "values in the form of Tuple[Any, Any, Union[int, float]], you provided"
                f"{type(graph[0])}, {repr(graph[0])}"
            )

        new_to_old = {}
        stringified_new = []

        for source, target, weight in graph:
            source_str = str(source)
            target_str = str(target)
            weight = float(weight)
            if (source_str in new_to_old and new_to_old[source_str] != source) or (
                target_str in new_to_old and new_to_old[target_str] != target
            ):
                raise ValueError(
                    "str() representation collision in dataset. Please ensure that "
                    "str(node_id) cannot result in multiple node_ids being turned "
                    "into the same string. This exception is unlikely but would "
                    "result if a non primitive node ID of some sort had a "
                    "barebones __str__() definition for it."
                )
            new_to_old[source_str] = source
            new_to_old[target_str] = target
            stringified_new.append((source_str, target_str, weight))

        return new_to_old, stringified_new

    if isinstance(graph, nx.Graph):
        # will catch all networkx graph types
        try:
            new_to_old = {}
            stringified_new = []
            for source, target, data in graph.edges(data=True):
                source_str = str(source)
                target_str = str(target)
                weight = float(data.get(weight_attribute, weight_default))
                if (source_str in new_to_old and new_to_old[source_str] != source) or (
                    target_str in new_to_old and new_to_old[target_str] != target
                ):
                    raise ValueError(
                        "str() representation collision in dataset. Please ensure that "
                        "str(node_id) cannot result in multiple node_ids being turned "
                        "into the same string. This exception is unlikely but would "
                        "result if a non primitive node ID of some sort had a "
                        "barebones __str__() definition for it."
                    )
                new_to_old[source_str] = source
                new_to_old[target_str] = target
                stringified_new.append((source_str, target_str, weight))
            return new_to_old, stringified_new
        except TypeError:
            # None is returned for the weight if it doesn't exist and a weight_default
            # is not set, which results in a TypeError when you call float(None)
            raise ValueError(
                f"The networkx graph provided did not contain a {weight_attribute} that"
                " could be cast to float in one of the edges"
            )

    if isinstance(graph, (np.ndarray, scipy.sparse.csr.csr_matrix)):
        shape = graph.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(
                "graphs of type np.ndarray or csr.sparse.csr.csr_matrix should be "
                "adjacency matrices with n x n shape"
            )

        if is_weighted is None:
            is_weighted = not utils.is_unweighted(graph)

        if not is_weighted and weight_default is None:
            raise ValueError(
                "the adjacency matrix provided is not weighted and a default weight has"
                " not been set"
            )

        edges = []
        new_to_old = {}
        if isinstance(graph, np.ndarray):
            for i in range(0, shape[0]):
                new_to_old[str(i)] = i
                start = i
                for j in range(start, shape[1]):
                    weight = graph[i][j]
                    if weight != 0:
                        if not is_weighted and weight == 1:
                            weight = weight_default
                        edges.append((str(i), str(j), float(weight)))
        else:
            rows, columns = graph.nonzero()
            for i in range(0, len(rows)):
                row = rows[i]
                column = columns[i]
                new_to_old[str(row)] = row
                new_to_old[str(column)] = column
                if row <= column:
                    edges.append((str(row), str(column), float(graph[row, column])))

        return new_to_old, edges

    raise TypeError(
        f"The type of graph provided {type(graph)} is not a list of 3-tuples, networkx "
        f"graph, numpy.ndarray, or scipy.sparse.csr_matrix"
    )


def _validate_common_arguments(
    starting_communities: Optional[Dict[str, int]] = None,
    extra_forced_iterations: int = 0,
    resolution: float = 1.0,
    randomness: float = 0.001,
    use_modularity: bool = True,
    random_seed: Optional[int] = None,
    is_weighted: Optional[bool] = None,
    weight_default: float = 1.0,
    check_directed: bool = True,
) -> None:
    if starting_communities is not None and not isinstance(starting_communities, dict):
        raise TypeError("starting_communities must be a dictionary")
    if not isinstance(extra_forced_iterations, int):
        raise TypeError("iterations must be an int")
    if not isinstance(resolution, (int, float)):
        raise TypeError("resolution must be a float")
    if not isinstance(randomness, (int, float)):
        raise TypeError("randomness must be a float")
    if not isinstance(use_modularity, bool):
        raise TypeError("use_modularity must be a bool")
    if random_seed is not None and not isinstance(random_seed, int):
        raise TypeError("random_seed must either be an int or None")
    if is_weighted is not None and not isinstance(is_weighted, bool):
        raise TypeError("is_weighted must either be a bool or None")
    if not isinstance(weight_default, (int, float)):
        raise TypeError("weight_default must be a float")
    if not isinstance(check_directed, bool):
        raise TypeError("check_directed must be a bool")

    if extra_forced_iterations < 0:
        raise ValueError("iterations must be a non negative integer")
    if resolution <= 0:
        raise ValueError("resolution must be a positive float")
    if randomness <= 0:
        raise ValueError("randomness must be a positive float")
    if random_seed is not None and random_seed <= 0:
        raise ValueError(
            "random_seed must be a positive integer (the native PRNG implementation is"
            " an unsigned 64 bit integer)"
        )


def leiden(
    graph: Union[
        List[Tuple[Any, Any, Union[int, float]]],
        nx.Graph,
        np.ndarray,
        scipy.sparse.csr.csr_matrix,
    ],
    starting_communities: Optional[Dict[str, int]] = None,
    extra_forced_iterations: int = 0,
    resolution: float = 1.0,
    randomness: float = 0.001,
    use_modularity: bool = True,
    random_seed: Optional[int] = None,
    weight_attribute: str = "weight",
    is_weighted: Optional[bool] = None,
    weight_default: float = 1.0,
    check_directed: bool = True,
) -> Dict[str, int]:
    """
    Leiden is a global network partitioning algorithm. Given a graph, it will iterate
    through the network node by node, and test for an improvement in our quality
    maximization function by speculatively joining partitions of each neighboring node.

    This process continues until no moves are made that increases the partitioning
    quality.

    Parameters
    ----------
    graph : Union[List[Tuple[Any, Any, Union[int, float]]], nx.Graph, np.ndarray, scipy.sparse.csr.csr_matrix]
        A graph representation, whether a weighted edge list referencing an undirected
        graph, an undirected networkx graph, or an undirected adjacency matrix in either
        numpy.ndarray or scipy.sparse.csr.csr_matrix form.
    starting_communities : Optional[Dict[str, int]]
        Default is ``None``. An optional community mapping dictionary that contains the
        string representation of the node and the community it belongs to. Note this map
        must contain precisely the same nodes as the graph and every node must have a
        positive community id. This mapping forms the starting point of Leiden
        clustering and can be very useful in saving the state of a previous partition
        schema from a previous graph and then adjusting the graph based on new temporal
        data (additions, removals, weight changes, connectivity changes, etc). New nodes
        should get their own unique community positive integer, but the original
        partition can be very useful to speed up future runs of leiden. If no community
        map is provided, the default behavior is to create a node community identity
        map, where every node is in their own community.
    extra_forced_iterations : int
        Default is ``0``. Leiden will run until a maximum quality score has been found
        for the node clustering and no nodes are moved to a new cluster in another
        iteration. As there is an element of randomness to the Leiden algorithm, it is
        sometimes useful to set ``extra_forced_iterations`` to a number larger than 0
        where the entire process is forced to attempt further refinement.
    resolution : float
        Default is ``1.0``. Higher resolution values lead to more communities and lower
        resolution values leads to fewer communities. Must be greater than 0.
    randomness : float
        Default is ``0.001``. The larger the randomness value, the more exploration of
        the partition space is possible. This is a major difference from the Louvain
        algorithm, which is purely greedy in the partition exploration.
    use_modularity : bool
        Default is ``True``. If ``False``, will use a Constant Potts Model (CPM).
    random_seed : Optional[int]
        Default is ``None``. Can provide an optional seed to the PRNG used in Leiden for
        deterministic output.
    weight_attribute : str
        Default is ``weight``. Only used when creating a weighed edge list of tuples
        when the source graph is a networkx graph. This attribute corresponds to the
        edge data dict key.
    is_weighted : Optional[bool]
        Default is ``None``. Only used when creating a weighted edge list of tuples
        when the source graph is an adjacency matrix. The
        :func:`graspologic.utils.to_weighted_edge_list` function will scan these
        matrices and attempt to determine whether it is weighted or not. This flag can
        short circuit this test and the values in the adjacency matrix will be treated
        as weights.
    weight_default : float
        Default is ``1.0``. If the graph is a networkx graph and the graph does not have
        a fully weighted sequence of edges, this default will be used. If the adjacency
        matrix is found or specified to be unweighted, this weight_default will be used
        for every edge.
    check_directed : bool
        Default is ``True``. If the graph is an adjacency matrix, we will attempt to
        ascertain whether it is directed or undirected. As our leiden implementation is
        only known to work with an undirected graph, this function will raise an error
        if it is found to be a directed graph. If you know it is undirected and wish to
        avoid this scan, you can set this value to ``False`` and only the lower triangle
        of the adjacency matrix will be used to generate the weighted edge list.

    Returns
    -------
    Dict[str, int]
        The results of running leiden over the provided graph, a dictionary containing
        mappings of node -> community id. The keys in the dictionary are the string
        representations of the nodes.

    Raises
    ------
    ValueError
    TypeError

    See Also
    --------
    graspologic.utils.to_weighted_edge_list

    References
    ----------
    .. [1] Traag, V.A.; Waltman, L.; Van, Eck N.J. "From Louvain to Leiden:
         guaranteeing well-connected communities", Scientific Reports, Vol. 9, 2019
    .. [2] https://github.com/microsoft/graspologic-native

    Notes
    -----
    This function is implemented in the `graspologic-native` Python module, a module
    written in Rust for Python.

    """
    _validate_common_arguments(
        starting_communities,
        extra_forced_iterations,
        resolution,
        randomness,
        use_modularity,
        random_seed,
        is_weighted,
        weight_default,
        check_directed,
    )
    node_id_mapping, graph = _validate_and_build_edge_list(
        graph, is_weighted, weight_attribute, check_directed, weight_default
    )

    _improved, _modularity, native_partitions = gn.leiden(
        edges=graph,
        starting_communities=starting_communities,
        resolution=resolution,
        randomness=randomness,
        iterations=extra_forced_iterations + 1,
        use_modularity=use_modularity,
        seed=random_seed,
    )
    
    # to be fixed in 738, should be Dict[Any, int]
    partitions: Dict[str, int] = native_partitions

    proper_partitions = {
        node_id_mapping[key]: value for key, value in partitions.items()
    }

    return proper_partitions


class HierarchicalCluster(NamedTuple):
    node: Any
    cluster: int
    parent_cluster: Optional[int]
    level: int
    is_final_cluster: bool

    @staticmethod
    def final_hierarchical_clustering(
        hierarchical_clusters: List[
            Union["HierarchicalCluster", gn.HierarchicalCluster]
        ],
    ) -> Dict[str, int]:
        if not isinstance(hierarchical_clusters, list):
            raise TypeError(
                "This static method requires a list of hierarchical clusters"
            )
        final_clusters = (
            cluster for cluster in hierarchical_clusters if cluster.is_final_cluster
        )
        return {cluster.node: cluster.cluster for cluster in final_clusters}


def _from_native(
    native_cluster: gn.HierarchicalCluster,
    node_id_map: Dict[str, Any],
) -> HierarchicalCluster:
    if not isinstance(native_cluster, gn.HierarchicalCluster):
        raise TypeError(
            "This class method is only valid for graspologic_native.HierarchicalCluster"
        )
    node_id = node_id_map[native_cluster.node]
    return HierarchicalCluster(
        node=node_id,
        cluster=native_cluster.cluster,
        parent_cluster=native_cluster.parent_cluster,
        level=native_cluster.level,
        is_final_cluster=native_cluster.is_final_cluster,
    )


def hierarchical_leiden(
    graph: Union[List[Tuple[str, str, float]], nx.Graph],
    max_cluster_size: int = 1000,
    starting_communities: Optional[Dict[str, int]] = None,
    extra_forced_iterations: int = 0,
    resolution: float = 1.0,
    randomness: float = 0.001,
    use_modularity: bool = True,
    random_seed: Optional[int] = None,
    weight_attribute: str = "weight",
    is_weighted: Optional[bool] = None,
    weight_default: float = 1.0,
    check_directed: bool = True,
) -> List[HierarchicalCluster]:
    """

    Leiden is a global network partitioning algorithm. Given a graph, it will iterate
    through the network node by node, and test for an improvement in our quality
    maximization function by speculatively joining partitions of each neighboring node.

    This process continues until no moves are made that increases the partitioning
    quality.

    Unlike the function :func:`graspologic.partition.leiden`, this function does not
    stop after maximization has been achieved. On some large graphs, it's useful to
    identify particularly large communities whose membership counts exceed
    ``max_cluster_size`` and induce a subnetwork solely out of that community. This
    subnetwork is then treated as a wholly separate entity, leiden is run over it, and
    the new, smaller communities are then mapped into the original community map space.

    The results also differ substantially; the returned List[HierarchicalCluster] is
    more of a log of state at each level. All HierarchicalClusters at level 0 should be
    considered to be the results of running :func:`graspologic.partition.leiden`. Every
    community whose membership is greater than ``max_cluster_size`` will then
    also have entries where level == 1, and so on until no communities are greater in
    population than ``max_cluster_size`` OR we are unable to break them down any
    further.

    Once a node's membership registration in a community cannot be changed any further,
    it is marked with the flag
    ``graspologic.partition.HierarchicalCluster.is_final_cluster = 1``.

    Parameters
    ----------
    graph : Union[List[Tuple[Any, Any, Union[int, float]]], nx.Graph, np.ndarray, scipy.sparse.csr.csr_matrix]
        A graph representation, whether a weighted edge list referencing an undirected
        graph, an undirected networkx graph, or an undirected adjacency matrix in
        either numpy.ndarray or scipy.sparse.csr.csr_matrix form.
    max_cluster_size : int
        Default is ``1000``. Any partition or cluster with
        membership >= ``max_cluster_size`` will be isolated into a subnetwork. This
        subnetwork will be used for a new leiden global partition mapping, which will
        then be remapped back into the global space after completion. Once all
        clusters with membership >= ``max_cluster_size`` have been completed, the level
        increases and the partition scheme is scanned again for any new clusters with
        membership >= ``max_cluster_size`` and the process continues until every
        cluster's membership is < ``max_cluster_size`` or if they cannot be broken into
        more than one new community.
    starting_communities : Optional[Dict[str, int]]
        Default is ``None``. An optional community mapping dictionary that contains the
        string representation of the node and the community it belongs to. Note this
        map must contain precisely the same nodes as the graph and every node must
        have a positive community id. This mapping forms the starting point of Leiden
        clustering and can be very useful in saving the state of a previous partition
        schema from a previous graph and then adjusting the graph based on new temporal
        data (additions, removals, weight changes, connectivity changes, etc). New
        nodes should get their own unique community positive integer, but the original
        partition can be very useful to speed up future runs of leiden. If no community
        map is provided, the default behavior is to create a node community identity
        map, where every node is in their own community.
    extra_forced_iterations : int
        Default is ``0``. Leiden will run until a maximum quality score has been found
        for the node clustering and no nodes are moved to a new cluster in another
        iteration. As there is an element of randomness to the Leiden algorithm, it is
        sometimes useful to set ``extra_forced_iterations`` to a number larger than 0
        where the entire process is forced to attempt further refinement.
    resolution : float
        Default is ``1.0``. Higher resolution values lead to more communities and lower
        resolution values leads to fewer communities. Must be greater than 0.
    randomness : float
        Default is ``0.001``. The larger the randomness value, the more exploration of
        the partition space is possible. This is a major difference from the Louvain
        algorithm, which is purely greedy in the partition exploration.
    use_modularity : bool
        Default is ``True``. If ``False``, will use a Constant Potts Model (CPM).
    random_seed : Optional[int]
        Default is ``None``. Can provide an optional seed to the PRNG used in Leiden
        for deterministic output.
    weight_attribute : str
        Default is ``weight``. Only used when creating a weighed edge list of tuples
        when the source graph is a networkx graph. This attribute corresponds to the
        edge data dict key.
    is_weighted : Optional[bool]
        Default is ``None``. Only used when creating a weighted edge list of tuples
        when the source graph is an adjacency matrix. The
        :func:`graspologic.utils.to_weighted_edge_list` function will scan these
        matrices and attempt to determine whether it is weighted or not. This flag can
        short circuit this test and the values in the adjacency matrix will be treated
        as weights.
    weight_default : float
        Default is ``1.0``. If the graph is a networkx graph and the graph does not
        have a fully weighted sequence of edges, this default will be used. If the
        adjacency matrix is found or specified to be unweighted, this weight_default
        will be used for every edge.
    check_directed : bool
        Default is ``True``. If the graph is an adjacency matrix, we will attempt to
        ascertain whether it is directed or undirected. As our leiden implementation is
        only known to work with an undirected graph, this function will raise an error
        if it is found to be a directed graph. If you know it is undirected and wish to
        avoid this scan, you can set this value to ``False`` and only the lower triangle
        of the adjacency matrix will be used to generate the weighted edge list.

    Returns
    -------
    List[HierarchicalCluster]
        The results of running hierarchical leiden over the provided graph, a list of
        HierarchicalClusters identifying the state of every node and cluster at each
        level. The function
        :func:`graspologic.partition.HierarchicalCluster.final_hierarchical_clustering`
        can be used to create a dictionary mapping of node -> cluster ID

    Raises
    ------
    ValueError
    TypeError

    See Also
    --------
    graspologic.utils.to_weighted_edge_list

    References
    ----------
    .. [1] Traag, V.A.; Waltman, L.; Van, Eck N.J. "From Louvain to Leiden:
        guaranteeing well-connected communities",Scientific Reports, Vol. 9, 2019
    .. [2] https://github.com/microsoft/graspologic-native

    Notes
    -----
    This function is implemented in the `graspologic-native` Python module, a module
    written in Rust for Python.

    """
    _validate_common_arguments(
        starting_communities,
        extra_forced_iterations,
        resolution,
        randomness,
        use_modularity,
        random_seed,
        is_weighted,
        weight_default,
        check_directed,
    )
    if not isinstance(max_cluster_size, int):
        raise TypeError("max_cluster_size must be an int")
    if max_cluster_size <= 0:
        raise ValueError("max_cluster_size must be a positive int")

    node_id_mapping, graph = _validate_and_build_edge_list(
        graph, is_weighted, weight_attribute, check_directed, weight_default
    )
    hierarchical_clusters_native = gn.hierarchical_leiden(
        edges=graph,
        starting_communities=starting_communities,
        resolution=resolution,
        randomness=randomness,
        iterations=extra_forced_iterations + 1,
        use_modularity=use_modularity,
        max_cluster_size=max_cluster_size,
        seed=random_seed,
    )

    return [
        _from_native(entry, node_id_mapping) for entry in hierarchical_clusters_native
    ]
