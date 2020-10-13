from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import networkx as nx
import numpy as np
from scipy.special import gamma

from ..embed import MultipleASE, OmnibusEmbed
from ..utils import import_multigraphs, is_almost_symmetric


class AnomalyResult(NamedTuple):
    graph_anomaly_indices: np.ndarray
    graph_anomaly_dict: Dict[str, np.ndarray]
    vertex_anomaly_indices: Dict[int, np.ndarray]
    vertex_anomaly_dict: Dict[str, np.ndarray]


def _compute_statistics(embeddings: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    For graph anomaly detection, computes spectral norm between each pair-wise
    embeddings. For vertex anomaly detection, computes L2 norm between each
    pair-wise embeddings for each vertex.

    Parameters
    ----------
    embeddings : list of ndarray
        Sequential pair-wise embeddings.

    Returns
    -------
    graph_stats : ndarray, shape (n_graphs - 1, )
        Test statistics based on spectral norms.

    vertex_stats : ndarray, shape (n_graphs - 1, n_vertices)
        Test statistics based on L2 norms.
    """
    # This is \tilde{y}^t
    graph_stats = np.array(
        [
            np.linalg.norm(embeddings[i][0] - embeddings[i][1], ord=2)
            for i in range(len(embeddings))
        ]
    )

    # This is \tilde{y}_i^t
    vertex_stats = np.array(
        [
            np.linalg.norm(embeddings[i][0] - embeddings[i][1], ord=None, axis=1)
            for i in range(len(embeddings))
        ]
    )

    return graph_stats, vertex_stats


def _compute_control_chart(
    graph_stats: np.ndarray, vertex_stats: np.ndarray, time_window: int
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Computes the control limits using 3 standard deviations.

    Parameters
    ----------
    graph_stats : ndarray, shape (n_graphs - 1, )
        Test statistics based on spectral norms.

    vertex_stats : ndarray, shape (n_graphs - 1, n_vertices)
        Test statistics based on L2 norms.

    time_window : int
        The number of graphs in time window in estimating the moving mean and
        moving standard deviation.

    Returns
    -------
    graph_means : ndarray, shape (n_graphs - time_window, )
        Moving average of graph-wise test statistics.

    graph_stds : ndarray, shape (n_graphs - time_window, )
        Moving standard deviation of graph-wise test statistics.

    graph_upper_central_line : ndarray, shape (n_graphs - time_window, )
        Upper threshold of graph-wise test statistics.

    graph_lower_central_line : ndarray, shape (n_graphs - time_window, )
        Lower threshold of graph-wise test statistics.

    vertex_means : ndarray, shape (n_graphs - time_window, )
        Moving average of vertex-wise test statistics.

    vertex_stds : ndarray, shape (n_graphs - time_window, )
        UnWeighted AVErage of subgroup estimates based on subgroup Standard
        Deviations (UWAVE-SD) of vertex-wise test statistics.

    vertex_upper_central_line : ndarray, shape (n_graphs - time_window, )
        Upper threshold of vertex-wise test statistics.

    vertex_lower_central_line : ndarray, shape (n_graphs - time_window, )
        Lower threshold of vertex-wise test statistics.
    """
    el = time_window
    m = graph_stats.size + 1  # num graphs
    n = vertex_stats.shape[-1]  # num verts

    # Compute moving means
    graph_means, vertex_means = np.empty((2, m - el))
    for i in range(m - el):
        graph_means[i] = graph_stats[i : i + el - 1].sum() / (el - 1)
        vertex_means[i] = vertex_stats[i : i + el - 1].sum() / (n * (el - 1))

    # Sequential differences
    graph_diffs = np.abs(np.diff(graph_stats))

    # Compute vertex standard deviation
    vertex_sample_stds = np.std(vertex_stats, axis=1, ddof=1)
    constant = gamma(n / 2) * np.sqrt(2 / (n - 1)) / gamma((n - 1) / 2)

    # Compute moving standard deviations
    graph_stds, vertex_stds = np.empty((2, m - el))
    for i in range(m - el):
        graph_stds[i] = graph_diffs[i : i + el - 2].sum() / (1.128 * (el - 2))
        vertex_stds[i] = vertex_sample_stds[i : i + el - 1].sum() / (
            constant * (el - 1)
        )

    graph_upper_central_line = graph_means + 3 * graph_stds
    graph_lower_central_line = graph_means - 3 * graph_stds
    vertex_upper_central_line = vertex_means + 3 * vertex_stds
    vertex_lower_central_line = vertex_means - 3 * vertex_stds

    return (
        graph_means,
        graph_stds,
        graph_upper_central_line,
        graph_lower_central_line,
        vertex_means,
        vertex_stds,
        vertex_upper_central_line,
        vertex_lower_central_line,
    )


def anomaly_detection(
    graphs: Union[List[nx.Graph], List[np.ndarray], np.ndarray],
    method: str = "omni",
    time_window: int = 3,
    use_lower_line: bool = False,
    diag_aug: bool = True,
    scaled: bool = True,
    n_components: Optional[int] = None,
    n_elbows: int = 2,
    algorithm: str = "randomized",
    n_iter: int = 5,
) -> AnomalyResult:
    """
    Function for computing anomalous graphs and vertices given time series of
    graphs. Anomaly is defined as graphs or vertices at a specific time that
    changes significantly from the previous time point.

    Specifically, anomaly detection is a five step process:

        1. Estimate latent positions of graphs for each sequential pair of
        graphs using either :class:`~graspy.embed.OmnibusEmbed` or
        :class:`~graspy.embed.MultipleASE`.

        2. Compute test statistics, or the magnitude of change, for each pair
        of latent positions via spectral norm for graph anomaly detection
        or L2 norm for vertex anomaly detection.

        3. Compute moving means and standard deviations of the test statistics
        given a time window.

        4. Compute the upper and lower threshold for test statistics.

        5. Compute graphs or vertices with test statistics that are above the
        upper threshold or below the lower threshold.

    Parameters
    ----------
    graphs : list of nx.Graph or ndarray, or ndarray
        If list of nx.Graph, each Graph must contain same number of vertices.
        If list of ndarray, each array must have shape (n_vertices, n_vertices).
        If ndarray, then array must have shape (n_graphs, n_vertices, n_vertices).

    method : {"omni" (default), "mase"}
        Embedding method for estimating latent positions for each sequential
        pair of graphs.

    time_window : int, default=3
        The number of graphs in time window in estimating the moving mean and
        moving standard deviation. Must be greater than or equal to 3 and less
        than number of input graphs.

    use_lower_line : bool, default=False
        Use the lower control line to compute anomalous graphs or vertices.

    diag_aug : bool, default=True
        Augment the diagonals of each input graph prior to embeddings.

    scaled : bool, default=True
        Only used when ``method == "mase"``. Whether to scale individual eigenvectors
        with eigenvalues in first MASE embedding stage.

    n_components : int or None, default=None
        Desired dimensionality of the embeddings. If "full",
        n_components must be <= min(X.shape). Otherwise, n_components must be
        < min(X.shape). If None, then optimal dimensions will be chosen by
        :func:`~graspy.embed.select_dimension` using ``n_elbows`` argument.

    n_elbows : int, optional, default=2
        If ``n_components=None``, then compute the optimal embedding dimension using
        :func:`~graspy.embed.select_dimension`. Otherwise, ignored.

    algorithm : {'randomized' (default), 'full', 'truncated'}, optional
        SVD solver to use:

        - 'randomized'
            Computes randomized svd using
            :func:`sklearn.utils.extmath.randomized_svd`
        - 'full'
            Computes full svd using :func:`scipy.linalg.svd`
        - 'truncated'
            Computes truncated svd using :func:`scipy.sparse.linalg.svds`

    n_iter : int, optional (default = 5)
        Number of iterations for randomized SVD solver. Not used by 'full' or
        'truncated'. The default is larger than the default in randomized_svd
        to handle sparse matrices that may have large slowly decaying spectrum.

    Returns
    -------
    graph_anomaly_indices : ndarray
        Indices of the input graph that are anomalous.

    vertex_anomaly_indices : dict
        Indices of vertices that are anomalous. Key is the index of the input
        graph, and value are the indices of anomalous vertices for that particular
        graph.

    graph_anomaly_dict, vertex_anomaly_dict : dict
        Contains additional useful additional returns containing the following
        keys:

            - statistics : ndarray
                Graph-wise test statistics with shape (n_graphs - time_window, )
                or vertex-wise test statistics with shape (n_graphs - time_window,
                n_vertices).
            - means : ndarray, shape (n_graphs - time_window, )
                Moving average of test statistics.
            - stds : ndarray, shape (n_graphs - time_window, )
                Moving standard deviation of test statistics.
            - upper_central_line : ndarray, shape
                Upper threshold of test statistics.
            - lower_central_line : ndarray, shape
                Lower threshold of test statistics.

    References
    ----------
    .. [1] Chen, G., Arroyo, J., Athreya, A., Cape, J., Vogelstein, J.T., Park, Y.,
        White, C., Larson, J., Yang, W. and Priebe, C.E., (2020). "Multiple Network
        Embedding for Anomaly Detection in Time Series of Graphs."
        arXiv:2008.10055.
    """
    if not isinstance(method, str):
        raise TypeError(f"method must be a string.")
    elif method.lower() not in ["omni", "mase"]:
        raise ValueError(f"method must be one of {'omni', 'mase'}, not {method}.")
    else:
        embed_kwargs = dict(
            n_components=n_components,
            n_elbows=n_elbows,
            algorithm=algorithm,
            n_iter=n_iter,
            diag_aug=diag_aug,
        )
        if method.lower() == "omni":
            embedder = OmnibusEmbed(
                **embed_kwargs,
                check_lcc=False,
            )
        elif method.lower() == "mase":
            embedder = MultipleASE(
                **embed_kwargs,
                scaled=scaled,
            )

    if isinstance(time_window, int):
        if time_window < 3 or time_window >= len(graphs):
            msg = f"time_window must be within [3, {len(graphs)})."
            raise ValueError(msg)
    else:
        msg = f"time_window must be an integer, not {type(time_window)}."
        raise TypeError(msg)

    if not isinstance(use_lower_line, bool):
        raise TypeError(f"use_lower_line must be a bool, not {type(use_lower_line)}.")

    graphs = import_multigraphs(graphs)
    n_graphs = len(graphs)

    # Anomaly detection is not defined for directed graphs
    if not all([is_almost_symmetric(g) for g in graphs]):
        raise ValueError("All input graphs must be undirected.")

    # Sequential pair-wise embeddings
    # Create list of ndarrays with shape (2, n_verts, n_components)
    if method.lower() == "omni":  # Do omni
        embeddings = [
            embedder.fit_transform(graphs[i : i + 2]) for i in range(n_graphs - 1)
        ]
    else:
        # Do mase. Vhat needs to be rescaled by |Rhat|^1/2
        embeddings = []
        for i in range(n_graphs - 1):
            Vhat = embedder.fit_transform(graphs[i : i + 2])
            U, D, V = np.linalg.svd(embedder.scores_)
            root_scores = U @ np.stack([np.diag(np.sqrt(diag)) for diag in D]) @ V
            embeddings.append(embedder.latent_left_ @ root_scores)

    graph_stats, vertex_stats = _compute_statistics(embeddings)

    (
        graph_means,
        graph_stds,
        graph_upper_central_line,
        graph_lower_central_line,
        vertex_means,
        vertex_stds,
        vertex_upper_central_line,
        vertex_lower_central_line,
    ) = _compute_control_chart(graph_stats, vertex_stats, time_window)

    if use_lower_line:
        graph_idx = (
            np.where(
                (graph_stats[time_window - 1 :] > graph_upper_central_line)
                | (graph_stats[time_window - 1 :] < graph_lower_central_line)
            )[0]
            + time_window
        )

        g_idx, v_idx = np.where(
            (vertex_stats[time_window - 1 :] > vertex_upper_central_line.reshape(-1, 1))
            | (
                vertex_stats[time_window - 1 :]
                < vertex_lower_central_line.reshape(-1, 1)
            )
        )
        vertex_idx = {i + time_window: v_idx[g_idx == i] for i in np.unique(g_idx)}
    else:
        graph_idx = (
            np.where(graph_stats[time_window - 1 :] > graph_upper_central_line)[0]
            + time_window
        )

        g_idx, v_idx = np.where(
            vertex_stats[time_window - 1 :] > vertex_upper_central_line.reshape(-1, 1)
        )
        vertex_idx = {i + time_window: v_idx[g_idx == i] for i in np.unique(g_idx)}

    graph_dict = dict(
        statistics=graph_stats[time_window - 1 :],
        means=graph_means,
        stds=graph_stds,
        upper_central_line=graph_upper_central_line,
        lower_central_line=graph_lower_central_line,
    )
    vertex_dict = dict(
        statistics=vertex_stats[time_window - 1 :],
        means=vertex_means,
        stds=vertex_stds,
        upper_central_line=vertex_upper_central_line,
        lower_central_line=vertex_lower_central_line,
    )

    return AnomalyResult(graph_idx, graph_dict, vertex_idx, vertex_dict)
