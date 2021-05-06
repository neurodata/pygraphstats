# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from anytree import NodeMixin

from ..embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from ..cluster import DivisiveCluster
from ..cluster.divisive_cluster import _check_common_inputs
from .sbm import _calculate_block_p, _block_to_full, _get_block_indices
from .base import BaseGraphEstimator, _check_n_samples
from ..simulations import sample_edges
from ..utils import (
    augment_diagonal,
    pass_to_ranks,
    import_graph,
    remove_loops,
    symmetrize,
    is_unweighted,
)


class _DivisiveGraphCluster(NodeMixin, BaseEstimator):
    """
    Recursively clusters a graph based on a chosen clustering algorithm.
    This algorithm implements a "divisive" or "top-down" approach.

    Parameters
    ----------
    cluster_method : str {"gmm", "kmeans"}, defaults to "gmm".
        The underlying clustering method to apply. If "gmm" will use
        :class:`~graspologic.cluster.AutoGMMCluster`. If "kmeans", will use
        :class:`~graspologic.cluster.KMeansCluster`.
    embed_method : str {"ase", "lse"}, defaults to "ase".
        The embedding method to apply. If "ase" will use
        :class:`~graspologic.embed.AdjacencySpectralEmbed`. If "lse", will use
        :class:`~graspologic.embed.LaplacianSpectralEmbed`.
    min_components : int, defaults to 1.
        The minimum number of mixture components/clusters to consider
        for the first split if "gmm" is selected as ``cluster_method``;
        and is set to 1 for later splits.
        If ``cluster_method`` is "kmeans", it is set to 2 for all splits.
    max_components : int, defaults to 2.
        The maximum number of mixture components/clusters to consider
        at each split.
    min_split : int, defaults to 1.
        The minimum size of a cluster for it to be considered to be split again.
    max_level : int, defaults to 4.
        The maximum number of times to recursively cluster the data.
    delta_criter : float, non-negative, defaults to 0.
        The smallest difference between selection criterion values of a new
        model and the current model that is required to accept the new model.
        Applicable only if ``cluster_method`` is "gmm".
    cluster_kws : dict, defaults to {}
        Keyword arguments (except ``min_components`` and ``max_components``) for chosen
        clustering method.
    embed_kws : dict, defaults to {}
        Keyword arguments for chosen embedding method.
    loops : boolean, optional (default=False)
        Whether to allow entries on the diagonal of the adjacency matrix,
        i.e. loops in the graph where a node connects to itself.

    Attributes
    ----------
    model_ : GaussianMixture or KMeans object
        Fitted clustering object based on which ``cluster_method`` was used.

    See Also
    --------
    graspologic.cluster.divisive_cluster
    anytree.node.nodemixin.NodeMixin

    Notes
    -----
    This class inherits from :class:`anytree.node.nodemixin.NodeMixin`, a lightweight
    class for doing various simple operations on trees.

    This algorithm was strongly inspired by maggotcluster, a divisive
    clustering algorithm in https://github.com/neurodata/maggot_models and the
    algorithm for estimating a hierarchical stochastic block model presented in [2]_.

    References
    ----------
    .. [1]  Athey, T. L., & Vogelstein, J. T. (2019).
            AutoGMM: Automatic Gaussian Mixture Modeling in Python.
            arXiv preprint arXiv:1909.02688.
    .. [2]  Lyzinski, V., Tang, M., Athreya, A., Park, Y., & Priebe, C. E
            (2016). Community detection and classification in hierarchical
            stochastic blockmodels. IEEE Transactions on Network Science and
            Engineering, 4(1), 13-26.
    """

    def __init__(
        self,
        cluster_method="gmm",
        embed_method="ase",
        min_components=1,
        max_components=2,
        cluster_kws={},
        embed_kws={},
        min_split=1,
        max_level=4,
        delta_criter=0,
        loops=False,
    ):
        _check_common_inputs(min_components, max_components, cluster_kws, embed_kws)

        if embed_method not in ["ase", "lse"]:
            msg = "clustering method must be one of {ase, lse}"
            raise ValueError(msg)

        if cluster_method not in ["gmm", "kmeans"]:
            msg = "clustering method must be one of {gmm, kmeans}"
            raise ValueError(msg)

        if delta_criter < 0:
            raise ValueError("delta_criter must be non-negative")

        self.parent = None
        self.min_components = min_components
        self.max_components = max_components
        self.cluster_method = cluster_method
        self.embed_method = embed_method
        self.cluster_kws = cluster_kws
        self.embed_kws = embed_kws
        self.min_split = min_split
        self.max_level = max_level
        self.delta_criter = delta_criter
        self.loops = loops

    def fit(self, graph):
        """
        Fits clustering models to the graph as well as resulting subgraphs

        Parameters
        ----------
        graph : array_like or networkx.Graph
            Input graph to fit

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self.fit_predict(graph)
        return self

    def fit_predict(self, graph):
        """
        Fits clustering models to the graph as well as resulting subgraphs
        and using fitted models to predict a hierarchy of labels

        Parameters
        ----------
        graph : array_like or networkx.Graph
            Input graph to fit

        Returns
        -------
        labels : array_label, shape (n_verts, n_levels)
        """
        if self.max_components > graph.shape[0]:
            msg = "max_components must be >= n_samples, but max_components = "
            msg += "{}, n_samples = {}".format(self.max_components, len(graph))
            raise ValueError(msg)

        graph = import_graph(graph)
        if not self.loops:
            graph = remove_loops(graph)

        graph = augment_diagonal(graph)
        embed_graph = pass_to_ranks(graph)
        embed = self._embed(embed_graph)

        labels = self._fit_graph(embed_graph, embed)
        # delete the last column if predictions at the last level
        # are all zero vectors
        if (labels.shape[1] > 1) and (np.max(labels[:, -1]) == 0):
            labels = labels[:, :-1]
        labels = DivisiveCluster._relabel(self, labels)

        return labels

    def _fit_graph(self, graph, embed):
        pred = DivisiveCluster._cluster_and_decide(self, embed)
        self.children = []

        uni_labels = np.unique(pred)
        labels = pred.reshape((-1, 1)).copy()
        if len(uni_labels) > 1:
            for ul in uni_labels:
                inds = pred == ul
                new_graph = graph[np.ix_(inds, inds)]
                dc = _DivisiveGraphCluster(
                    cluster_method=self.cluster_method,
                    embed_method=self.embed_method,
                    max_components=self.max_components,
                    min_split=self.min_split,
                    max_level=self.max_level,
                    cluster_kws=self.cluster_kws,
                    embed_kws=self.embed_kws,
                    delta_criter=self.delta_criter,
                    loops=self.loops,
                )
                dc.parent = self
                if (
                    len(new_graph) > self.max_components
                    and len(new_graph) >= self.min_split
                    and self.depth + 1 < self.max_level
                ):
                    embed = dc._embed(new_graph)
                    child_labels = dc._fit_graph(new_graph, embed)
                    while labels.shape[1] <= child_labels.shape[1]:
                        labels = np.column_stack(
                            (labels, np.zeros((len(graph), 1), dtype=int))
                        )
                    labels[inds, 1 : child_labels.shape[1] + 1] = child_labels

        return labels

    def _embed(self, graph):
        if self.embed_method == "ase":
            embedder = AdjacencySpectralEmbed(**self.embed_kws)
            embed = embedder.fit_transform(graph)
        elif self.embed_method == "lse":
            embedder = LaplacianSpectralEmbed(**self.embed_kws)
            embed = embedder.fit_transform(graph)

        if isinstance(embed, tuple):
            embed = np.concatenate(embed, axis=1)

        return embed


class _HierarchicalBaseGraphEstimator(BaseGraphEstimator):
    def __init__(self, directed=True, loops=False):
        BaseGraphEstimator.__init__(self, directed=directed, loops=loops)

    def bic(self, graph):
        bic_per_level = []
        y = self.vertex_assignments_
        clustered_graphs = self.get_clustered_graphs(graph)
        for lvl in range(y.shape[1]):
            bic = BaseGraphEstimator.bic(self, clustered_graphs[lvl])
            bic_per_level.append(bic)

        return bic_per_level

    def mse(self, graph):
        check_is_fitted(self, "p_mat_")
        mse_per_level = []
        y = self.vertex_assignments_
        for lvl in range(y.shape[1]):
            mse_per_level.append(np.linalg.norm(graph - self.p_mat_[lvl]) ** 2)

        return mse_per_level

    def score_samples(self, graph, clip=None):
        check_is_fitted(self, "p_mat_")
        ori_graph = import_graph(graph)
        y = self.vertex_assignments_
        if not is_unweighted(graph):
            raise ValueError("Model only implemented for unweighted graphs")

        log_lik_per_level = []
        for lvl in range(y.shape[1]):
            p_mat = self.p_mat_[lvl].copy()
            if np.shape(p_mat) != np.shape(ori_graph):
                raise ValueError("Input graph size must be the same size as P matrix")

            inds = None
            if not self.directed and self.loops:
                inds = np.triu_indices_from(ori_graph)
            elif not self.directed and not self.loops:
                inds = np.triu_indices_from(ori_graph, k=1)
            elif self.directed and not self.loops:
                xu, yu = np.triu_indices_from(ori_graph, k=1)
                xl, yl = np.tril_indices_from(ori_graph, k=-1)
                x = np.concatenate((xl, xu))
                y = np.concatenate((yl, yu))
                inds = (x, y)
            if inds is not None:
                p_mat = p_mat[inds]
                graph = ori_graph[inds]

            if clip is not None:
                p_mat[p_mat < clip] = clip
                p_mat[p_mat > 1 - clip] = 1 - clip

            successes = np.multiply(p_mat, graph)
            failures = np.multiply((1 - p_mat), (1 - graph))
            likelihood = successes + failures
            log_lik_per_level.append(np.log(likelihood))

        return log_lik_per_level

    def score(self, graph):
        check_is_fitted(self, "p_mat_")
        score_samples = self.score_samples(graph)
        score_per_level = [np.sum(s) for s in score_samples]

        return score_per_level

    def sample(self, n_samples=1):
        check_is_fitted(self, "p_mat_")
        _check_n_samples(n_samples)
        graphs_by_level = []
        y = self.vertex_assignments_
        for lvl in range(y.shape[1]):
            n_verts = self.p_mat_[lvl].shape[0]
            graphs = np.zeros((n_samples, n_verts, n_verts))
            p_mat = self.p_mat_[lvl].copy()
            p_mat[p_mat > 1] = 1
            p_mat[p_mat < 0] = 0
            for i in range(n_samples):
                graphs[i, :, :] = sample_edges(
                    p_mat, directed=self.directed, loops=self.loops
                )
            graphs_by_level.append(graphs)
        return graphs_by_level


class HSBMEstimator(
    DivisiveCluster, _DivisiveGraphCluster, _HierarchicalBaseGraphEstimator
):
    r"""
    Hierarchical Stochastic Block Model

    The hierarchical stochastic block model (HSBM) represents each node as
    belonging to a block (or community) at each level of a hierarchy. For
    a given potential edge between node :math:`i` and :math:`j`, the probability
    of an edge existing is specified by the block that nodes :math:`i`
    and :math:`j` belong to:

    :math:`P_{ij} = B_{\tau_i \tau_j}`

    where :math:`B \in \mathbb{[0, 1]}^{K x K}` and :math:`\tau` is an `n\_nodes`
    length vector specifying which block each node belongs to.

    Parameters
    ----------
    min_components : int, defaults to 1.
        The minimum number of mixture components/clusters to consider
        for the first split if "gmm" is selected as ``cluster_method``;
        and is set to 1 for later splits.
        If ``cluster_method`` is "kmeans", it is set to 2 for all splits.
    max_components : int, defaults to 2.
        The maximum number of mixture components/clusters to consider
        at each split.
    cluster_method : str {"gmm", "kmeans"}, defaults to "gmm".
        The underlying clustering method to apply. If "gmm" will use
        :class:`~graspologic.cluster.AutoGMMCluster`. If "kmeans", will use
        :class:`~graspologic.cluster.KMeansCluster`.
    embed_method : str {"ase", "lse"}, defaults to "ase".
        The embedding method to apply. If "ase" will use
        :class:`~graspologic.embed.AdjacencySpectralEmbed`. If "lse", will use
        :class:`~graspologic.embed.LaplacianSpectralEmbed`.
    cluster_kws : dict, optional (default={})
        Additional kwargs passed down to
        :class:`~graspologic.cluster.GaussianCluster`
    embed_kws : dict, optional (default={})
        Additional kwargs passed down to
        :class:`~graspologic.embed.AdjacencySpectralEmbed`
    reembed : boolean, optional (default=True)
        Whether to perform reembedding before clustering on subgraphs.
    directed : boolean, optional (default=True)
        Whether to treat the input graph as directed. Even if a directed graph
        is inupt, this determines whether to force symmetry upon the block
        probability matrix fit for the SBM. It will also determine whether
        graphs sampled from the model are directed.
    loops : boolean, optional (default=False)
        Whether to allow entries on the diagonal of the adjacency matrix,
        i.e. loops in the graph where a node connects to itself.
    min_split : int, defaults to 1.
        During clustering, the minimum size of a cluster for it to be considered
        to be split again.
    max_level : int, defaults to 4.
        During clustering, the maximum number of times to recursively cluster the data.
    delta_criter : float, non-negative, defaults to 0.
        During clustering, the smallest difference between selection criterion values
        of a new model and the current model that is required to accept the new model.
        Applicable only if ``cluster_method`` is "gmm".

    Attributes
    ----------
    block_p_ : list of np.ndarray, length n_level, each item has shape
        (n_blocks, n_blocks)
        The block probability matrix :math:`B`, where the element :math:`B_{i, j}`
        represents the probability of an edge between block :math:`i` and block
        :math:`j` for each level.
    p_mat_ : list of np.ndarray, length n_level, each item has shape (n_verts, n_verts)
        Probability matrix :math:`P` for the fit model for each level,
        from which graphs could be sampled.
    vertex_assignments_ : np.ndarray, shape (n_verts, n_levels)
        An array of integer labels corresponding to the predicted block that each node
        belongs to at each level if ``y`` was not passed during the call
        to :func:`~graspologic.models.SBMEstimator.fit`.
    block_weights_ : list of np.ndarray, length n_level, each item has shape (n_blocks)
        Contains the proportion of nodes that belong to each block in the fit model
        for each level.

    See also
    --------
    graspologic.cluster.DivisiveCluster
    graspologic.simulations.sbm

    References
    ----------
    .. [1]  Lyzinski, V., Tang, M., Athreya, A., Park, Y., & Priebe, C. E
            (2016). Community detection and classification in hierarchical
            stochastic blockmodels. IEEE Transactions on Network Science and
            Engineering, 4(1), 13-26.
    """

    def __init__(
        self,
        cluster_method="gmm",
        embed_method="ase",
        min_components=1,
        max_components=2,
        cluster_kws={},
        embed_kws={},
        min_split=1,
        max_level=4,
        delta_criter=0,
        directed=False,
        loops=False,
        reembed=True,
    ):
        _DivisiveGraphCluster.__init__(
            self,
            cluster_method=cluster_method,
            embed_method=embed_method,
            min_components=min_components,
            max_components=max_components,
            cluster_kws=cluster_kws,
            embed_kws=embed_kws,
            min_split=min_split,
            max_level=max_level,
            delta_criter=delta_criter,
        )

        _HierarchicalBaseGraphEstimator.__init__(self, directed=directed, loops=loops)

        if not isinstance(reembed, bool):
            raise TypeError("`reembed` must be of type bool")
        self.reembed = reembed

    def fit(self, graph, y=None):
        """
        Fit the HSBM to a graph, optionally with known block labels

        If y is `None`, the block assignments for each vertex will first be
        estimated.

        Parameters
        ----------
        graph : array_like or networkx.Graph
            Input graph to fit

        y : array_like, shape (n_verts, n_levels), optional
            Categorical labels for the block assignments of the graph

        """
        if y is None:
            if self.reembed:
                y = _DivisiveGraphCluster.fit_predict(self, graph)
            else:
                y = DivisiveCluster.fit_predict(self, graph, fcluster=True)
            self.vertex_assignments_ = y
        else:
            _, y = check_X_y(graph, y, multi_output=True)

        n_levels = y.shape[1]
        self.block_weights_ = np.empty(n_levels, dtype=object)
        self.block_p_ = np.empty(n_levels, dtype=object)
        self.p_mat_ = np.empty(n_levels, dtype=object)

        for i in range(n_levels):
            single_label = y[:, i]
            _, counts = np.unique(single_label, return_counts=True)
            self.block_weights_[i] = counts / graph.shape[0]
            block_vert_inds, block_inds, block_inv = _get_block_indices(single_label)
            block_p = _calculate_block_p(graph, block_inds, block_vert_inds)

            if not self.directed:
                block_p = symmetrize(block_p)
            self.block_p_[i] = block_p

            p_mat = _block_to_full(block_p, block_inv, graph.shape)
            if not self.loops:
                p_mat = remove_loops(p_mat)
            self.p_mat_[i] = p_mat

        return self

    def _n_parameters(self, y=None):
        if hasattr(self, "vertex_assignments_"):
            n_levels = self.vertex_assignments_.shape[1]
        elif y is not None:
            n_levels = y.shape[1]
        n_parameters_per_level = np.empty(n_levels, dtype=object)

        for i in range(n_levels):
            n_blocks = self.block_p_[i].shape[0]
            n_parameters = 0
            if self.directed:
                n_parameters += n_blocks ** 2
            else:
                n_parameters += n_blocks * (n_blocks + 1) / 2
            if hasattr(self, "vertex_assignments_"):
                n_parameters += n_blocks - 1
            n_parameters += 1  # level index
            if self.reembed:
                n_parameters += 1

            n_parameters_per_level[i] = n_parameters
        return n_parameters_per_level
