# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import umap
from sklearn.manifold import TSNE

from ..embed import node2vec_embed
from ..partition import leiden
from ..preprocessing import cut_edges_by_weight, histogram_edge_weight
from .classes import NodePosition
from .nooverlap import remove_overlaps

logger = logging.getLogger(__name__)


# automatically generates a layout by running node2vec over the graph and down
# projecting the embedding into 2d space via umap or tsne
def layout_tsne(
    graph: nx.Graph,
    perplexity: int,
    n_iter: int,
    max_edges: int = 10000000,
    random_seed: Optional[int] = None,
    adjust_overlaps: bool = True,
) -> Tuple[nx.Graph, List[NodePosition]]:
    """
    Automatic graph layout generation by creating a generalized node2vec embedding,
    then using t-SNE for dimensionality reduction to 2d space.

    By default, this function automatically attempts to prune each graph to a maximum
    of 10,000,000 edges by removing the lowest weight edges. This pruning is approximate
    and will leave your graph with at most ``max_edges``, but is not guaranteed to be
    precisely ``max_edges``.

    In addition to pruning edges by weight, this function also only operates over the
    largest connected component in the graph.

    After dimensionality reduction, sizes are generated for each node based upon
    their degree centrality, and these sizes and positions are further refined by an
    overlap removal phase. Lastly, a global partitioning algorithm
    (:func:`graspologic.partition.leiden`) is executed for the largest connected
    component and the partition ID is included with each node position.

    Parameters
    ----------
    graph : :class:`networkx.Graph`
        The graph to generate a layout for. This graph may have edges pruned if the
        count is too high and only the largest connected component will be used to
        automatically generate a layout.
    perplexity : int
        The perplexity is related to the number of nearest neighbors that is used in
        other manifold learning algorithms. Larger datasets usually require a larger
        perplexity. Consider selecting a value between 4 and 100. Different values can
        result in significanlty different results.
    n_iter : int
        Maximum number of iterations for the optimization. We have found in practice
        that larger graphs require more iterations. We hope to eventually have more
        guidance on the number of iterations based on the size of the graph and the
        density of the edge connections.
    max_edges : int
        The maximum number of edges to use when generating the embedding.  Default is
        ``10000000``. The edges with the lowest weights will be pruned until at most
        ``max_edges`` exist. Warning: this pruning is approximate and more edges than
        are necessary may be pruned. Running in 32 bit enviornment you will most
        likely need to reduce this number or you will out of memory.
    random_seed : int
        Seed to be used for reproducible results. Default is None and will produce
        a new random state. Specifying a random state will provide consistent results
        between runs. In addition the environment variable ``PYTHONHASHSEED`` must be
        set to control hash randomization.
    adjust_overlaps : bool
        Make room for overlapping nodes while maintaining some semblance of the
        2d spatial characteristics of each node. Default is ``True``

    Returns
    -------
    Tuple[nx.Graph, List[NodePosition]]
        The largest connected component and a list of NodePositions for each node in
        the largest connected component. The NodePosition object contains:
        - node_id
        - x coordinate
        - y coordinate
        - size
        - community

    References
    ----------
    .. [1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data Using
        t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.
    """
    lcc_graph, tensors, labels = _node2vec_for_layout(graph, max_edges, random_seed)
    points = TSNE(
        perplexity=perplexity, n_iter=n_iter, random_state=random_seed
    ).fit_transform(tensors)
    positions = _node_positions_from(
        lcc_graph,
        labels,
        points,
        random_seed=random_seed,
        adjust_overlaps=adjust_overlaps,
    )
    return lcc_graph, positions


def layout_umap(
    graph: nx.Graph,
    min_dist: float = 0.75,
    n_neighbors: int = 25,
    max_edges: int = 10000000,
    random_seed: Optional[int] = None,
    adjust_overlaps: bool = True,
) -> Tuple[nx.Graph, List[NodePosition]]:
    """
    Automatic graph layout generation by creating a generalized node2vec embedding,
    then using UMAP for dimensionality reduction to 2d space.

    By default, this function automatically attempts to prune each graph to a maximum
    of 10,000,000 edges by removing the lowest weight edges. This pruning is approximate
    and will leave your graph with at most ``max_edges``, but is not guaranteed to be
    precisely ``max_edges``.

    In addition to pruning edges by weight, this function also only operates over the
    largest connected component in the graph.

    After dimensionality reduction, sizes are generated for each node based upon
    their degree centrality, and these sizes and positions are further refined by an
    overlap removal phase. Lastly, a global partitioning algorithm
    (:func:`graspologic.partition.leiden`) is executed for the largest connected
    component and the partition ID is included with each node position.

    Parameters
    ----------
    graph : :class:`networkx.Graph`
        The graph to generate a layout for. This graph may have edges pruned if the
        count is too high and only the largest connected component will be used to
        automatically generate a layout.
    min_dist : float
        The effective minimum distance between embedded points. Default is ``0.75``.
        Smaller values will result in a more clustered/clumped embedding where nearby
        points on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set relative to
        the ``spread`` value, which determines the scale at which embedded points will
        be spread out.
    n_neighbors : int
        The size of local neighborhood (in terms of number of neighboring sample points)
        used for manifold approximation. Default is ``25``. Larger values result in
        more global views of the manifold, while smaller values result in more local
        data being preserved.
    max_edges : int
        The maximum number of edges to use when generating the embedding.  Default is
        ``10000000``. The edges with the lowest weights will be pruned until at most
        ``max_edges`` exist. Warning: this pruning is approximate and more edges than
        are necessary may be pruned. Running in 32 bit environment you will most
        likely need to reduce this number or you will out of memory.
    random_seed : int
        Seed to be used for reproducible results. Default is None and will produce
        random results.
    adjust_overlaps : bool
        Make room for overlapping nodes while maintaining some semblance of the
        2d spatial characteristics of each node. Default is ``True``

    Returns
    -------
    Tuple[nx.Graph, List[NodePosition]]
        The largest connected component and a list of NodePositions for each node in
        the largest connected component. The NodePosition object contains:
        - node_id
        - x coordinate
        - y coordinate
        - size
        - community

    References
    ----------
    .. [1] McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection
        for Dimension Reduction, ArXiv e-prints 1802.03426, 2018
    .. [2] Böhm, Jan Niklas; Berens, Philipp; Kobak, Dmitry. A Unifying Perspective
        on Neighbor Embeddings along the Attraction-Repulsion Spectrum. ArXiv e-prints
        2007.08902v1, 17 Jul 2020.
    """

    lcc_graph, tensors, labels = _node2vec_for_layout(graph, max_edges, random_seed)
    points = umap.UMAP(
        min_dist=min_dist, n_neighbors=n_neighbors, random_state=random_seed
    ).fit_transform(tensors)
    positions = _node_positions_from(
        lcc_graph,
        labels,
        points,
        random_seed=random_seed,
        adjust_overlaps=adjust_overlaps,
    )
    return lcc_graph, positions


def _largest_connected_component(graph: nx.Graph) -> nx.Graph:
    largest_component = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_component).copy()


def _approximate_prune(graph: nx.Graph, max_edges_to_keep: int = 1000000) -> None:
    num_edges = len(graph.edges())
    logger.info(f"num edges: {num_edges}")

    if num_edges > max_edges_to_keep:
        histogram, bins = histogram_edge_weight(graph, bin_directive=100)
        counts = 0
        bin_edge_for_maximum_weight = bins[0]
        for i, count in reversed(list(enumerate(histogram))):
            counts += count
            if counts >= max_edges_to_keep:
                bin_edge_for_maximum_weight = bins[i + 1]
                break
        threshold = bins[bin_edge_for_maximum_weight]
        graph = cut_edges_by_weight(
            graph, cut_threshold=threshold, cut_process="smaller_than_inclusive"
        )
        logger.debug(f"after cut num edges: {len(graph.edges())}")

    return graph


def _node2vec_for_layout(
    graph: nx.Graph,
    max_edges: int = 10000000,
    random_seed: Optional[int] = None,
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    graph = _approximate_prune(graph, max_edges)
    graph = _largest_connected_component(graph)

    start = time.time()
    tensors, labels = node2vec_embed(
        graph=graph,
        dimensions=128,
        num_walks=10,
        window_size=2,
        iterations=3,
        random_seed=random_seed,
    )
    embedding_time = time.time() - start
    logger.info(f"embedding completed in {embedding_time} seconds")
    return graph, tensors, labels


def _node_positions_from(
    graph: nx.Graph,
    labels: np.ndarray,
    down_projection_2d: np.ndarray,
    random_seed: Optional[int] = None,
    adjust_overlaps: bool = True,
) -> List[NodePosition]:
    degree = graph.degree()
    sizes = _compute_sizes(degree)
    covered_area = _covered_size(sizes)
    scaled_points = _scale_points(down_projection_2d, covered_area)
    partitions = leiden(graph, random_seed=random_seed)
    positions = [
        NodePosition(
            node_id=key,
            x=scaled_points[index][0],
            y=scaled_points[index][1],
            size=sizes[key],
            community=partitions[key],
        )
        for index, key in enumerate(labels)
    ]
    if adjust_overlaps is True:
        positions = remove_overlaps(positions)
    return positions


def _find_min_max_degree(
    degrees: nx.classes.reportviews.DegreeView,
) -> Tuple[float, float]:
    min_degree = math.inf
    max_degree = -math.inf
    for _, degree in degrees:
        min_degree = min(min_degree, degree)
        max_degree = max(max_degree, degree)
    return min_degree, max_degree


def _compute_sizes(
    degrees: nx.classes.reportviews.DegreeView,
    min_size: float = 5.0,
    max_size: float = 150.0,
) -> Dict[Any, float]:
    min_degree, max_degree = _find_min_max_degree(degrees)
    sizes = {}
    for node_id, degree in degrees:
        if max_degree == min_degree:
            size = min_size
        else:
            normalized = (degree - min_degree) / (max_degree - min_degree)
            size = normalized * (max_size - min_size) + min_size
        sizes[node_id] = size
    return sizes


def _covered_size(sizes: Dict[Any, float]) -> float:
    total = sum(math.pow(value, 2) * math.pi for value in sizes.values())
    return total


def _get_bounds(points: np.ndarray) -> Tuple[float, float, float, float]:
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    return min_x, min_y, max_x, max_y


def _center_points_on_origin(points: np.ndarray) -> np.ndarray:
    min_x, min_y, max_x, max_y = _get_bounds(points)
    x_range = max_x - min_x
    y_range = max_y - min_y
    center_x = x_range / 2 + min_x
    center_y = y_range / 2 + min_y
    points[:, 0] -= center_x
    points[:, 1] -= center_y
    return points


def _scale_to_unit_square(points: np.ndarray) -> np.ndarray:
    _, _, max_x, max_y = _get_bounds(points)
    points[:, 0] /= max_x
    points[:, 1] /= max_y
    return points


def _scale_to_new_bounds(points: np.ndarray, max_x: float, max_y: float) -> np.ndarray:
    points[:, 0] *= max_x
    points[:, 1] *= max_y
    return points


def _new_bounds(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    covered_area: float,
    target_ratio: float,
) -> Tuple[float, float, float, float]:
    range_x = max_x - min_x
    mid_x = min_x + range_x / 2
    range_y = max_y - min_y
    mid_y = min_y + range_y / 2
    range_ratio = range_x / range_y

    new_area = covered_area / target_ratio
    new_range_y = math.sqrt(new_area / range_ratio)
    new_range_x = new_area / new_range_y
    new_min_x, new_min_y, new_max_x, new_max_y = (
        mid_x - new_range_x / 2,
        mid_y - new_range_y / 2,
        mid_x + new_range_x / 2,
        mid_y + new_range_y / 2,
    )

    return new_min_x, new_min_y, new_max_x, new_max_y


def _scale_points(
    points: np.ndarray, covered_area: float, target_ratio: float = 0.14
) -> np.ndarray:
    # through lots of experiments we have found 14% to be a good ratio
    # of color to whitespace
    moved_points = _center_points_on_origin(points)
    min_x, min_y, max_x, max_y = _get_bounds(moved_points)

    _, _, new_max_x, new_max_y = _new_bounds(
        min_x, min_y, max_x, max_y, covered_area, target_ratio
    )

    scaled_points = _scale_to_unit_square(moved_points)

    final_locs = _scale_to_new_bounds(scaled_points, new_max_x, new_max_y)

    return final_locs
