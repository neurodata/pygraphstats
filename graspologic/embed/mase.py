# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..utils import import_graph, import_multigraphs, is_almost_symmetric
from .base import BaseEmbedMulti
from .svd import select_dimension, selectSVD


class MultipleASE(BaseEmbedMulti):
    r"""
    Multiple Adjacency Spectral Embedding (MASE) embeds arbitrary number of input
    graphs with matched vertex sets.

    For a population of undirected graphs, MASE assumes that the population of graphs
    is sampled from :math:`VR^{(i)}V^T` where :math:`V \in \mathbb{R}^{n\times d}` and
    :math:`R^{(i)} \in \mathbb{R}^{d\times d}`. Score matrices, :math:`R^{(i)}`, are
    allowed to vary for each graph, but are symmetric. All graphs share a common a
    latent position matrix :math:`V`.

    For a population of directed graphs, MASE assumes that the population is sampled
    from :math:`UR^{(i)}V^T` where :math:`U \in \mathbb{R}^{n\times d_1}`,
    :math:`V \in \mathbb{R}^{n\times d_2}`, and
    :math:`R^{(i)} \in \mathbb{R}^{d_1\times d_2}`. In this case, score matrices
    :math:`R^{(i)}` can be assymetric and non-square, but all graphs still share a
    common latent position matrices :math:`U` and :math:`V`.

    Parameters
    ----------
    n_components : int or None, default = None
        Desired dimensionality of output data. If "full",
        n_components must be <= min(X.shape). Otherwise, n_components must be
        < min(X.shape). If None, then optimal dimensions will be chosen by
        :func:`~graspologic.embed.select_dimension` using ``n_elbows`` argument.

    n_elbows : int, optional, default: 2
        If ``n_components=None``, then compute the optimal embedding dimension using
        :func:`~graspologic.embed.select_dimension`. Otherwise, ignored.

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

    scaled : bool, optional (default=True)
        Whether to scale individual eigenvectors with eigenvalues in first embedding
        stage.

    diag_aug : bool, optional (default = True)
        Whether to replace the main diagonal of each adjacency matrices with
        a vector corresponding to the degree (or sum of edge weights for a
        weighted network) before embedding.

    concat : bool, optional (default False)
        If graph(s) are directed, whether to concatenate each graph's left and right (out and in) latent positions
        along axis 1.


    Attributes
    ----------
    n_graphs_ : int
        Number of graphs

    n_vertices_ : int
        Number of vertices in each graph

    latent_left_ : array, shape (n_samples, n_components)
        Estimated left latent positions of the graph.

    latent_right_ : array, shape (n_samples, n_components), or None
        Estimated right latent positions of the graph. Only computed when the an input
        graph is directed, or adjacency matrix is assymetric. Otherwise, None.

    scores_ : array, shape (n_samples, n_components, n_components)
        Estimated :math:`\hat{R}` matrices for each input graph.


    Notes
    -----
    When an input graph is directed, `n_components` of `latent_left_` may not be equal
    to `n_components` of `latent_right_`.
    """

    def __init__(
        self,
        n_components=None,
        n_elbows=2,
        algorithm="randomized",
        n_iter=5,
        scaled=True,
        diag_aug=True,
        concat=False,
    ):
        if not isinstance(scaled, bool):
            msg = "scaled must be a boolean, not {}".format(scaled)
            raise TypeError(msg)

        super().__init__(
            n_components=n_components,
            n_elbows=n_elbows,
            algorithm=algorithm,
            n_iter=n_iter,
            diag_aug=diag_aug,
            concat=concat,
        )
        self.scaled = scaled

    def _reduce_dim(self, graphs):
        # first embed into log2(n_vertices) for each graph
        n_components = int(np.ceil(np.log2(np.min(self.n_vertices_))))

        # embed individual graphs
        embeddings = [
            selectSVD(
                graph,
                n_components=n_components,
                algorithm=self.algorithm,
                n_iter=self.n_iter,
            )
            for graph in graphs
        ]
        Us, Ds, Vs = zip(*embeddings)

        # Choose the best embedding dimension for each graphs
        if self.n_components is None:
            embedding_dimensions = []
            for D in Ds:
                elbows, _ = select_dimension(D, n_elbows=self.n_elbows)
                embedding_dimensions.append(elbows[-1])

            # Choose the max of all of best embedding dimension of all graphs
            best_dimension = int(np.ceil(np.max(embedding_dimensions)))
        else:
            best_dimension = self.n_components

        if not self.scaled:
            Us = np.hstack([U[:, :best_dimension] for U in Us])
            Vs = np.hstack([V.T[:, :best_dimension] for V in Vs])
        else:
            # Equivalent to ASE
            Us = np.hstack(
                [
                    U[:, :best_dimension] @ np.diag(np.sqrt(D[:best_dimension]))
                    for U, D in zip(Us, Ds)
                ]
            )
            Vs = np.hstack(
                [
                    V.T[:, :best_dimension] @ np.diag(np.sqrt(D[:best_dimension]))
                    for V, D in zip(Vs, Ds)
                ]
            )

        # Second SVD for vertices
        # The notation is slightly different than the paper
        Uhat, _, _ = selectSVD(
            Us,
            n_components=self.n_components,
            n_elbows=self.n_elbows,
            algorithm=self.algorithm,
            n_iter=self.n_iter,
        )

        Vhat, _, _ = selectSVD(
            Vs,
            n_components=self.n_components,
            n_elbows=self.n_elbows,
            algorithm=self.algorithm,
            n_iter=self.n_iter,
        )
        return Uhat, Vhat

    def fit(self, graphs, y=None):
        """
        Fit the model with graphs.

        Parameters
        ----------
        graphs : list of nx.Graph or ndarray, or ndarray
            If list of nx.Graph, each Graph must contain same number of nodes.
            If list of ndarray, each array must have shape (n_vertices, n_vertices).
            If ndarray, then array must have shape (n_graphs, n_vertices, n_vertices).

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        graphs = import_multigraphs(graphs)
        self.n_graphs_ = len(graphs)
        self.n_vertices_ = graphs[0].shape[0]

        # Check if undirected
        undirected = all(is_almost_symmetric(g) for g in graphs)

        # Diag augment
        if self.diag_aug:
            graphs = self._diag_aug(graphs)

        # embed
        Uhat, Vhat = self._reduce_dim(graphs)
        self.latent_left_ = Uhat
        if not undirected:
            self.latent_right_ = Vhat
            self.scores_ = Uhat.T @ graphs @ Vhat
        else:
            self.latent_right_ = None
            self.scores_ = Uhat.T @ graphs @ Uhat

        return self

    def fit_transform(self, graphs, y=None):
        """
        Fit the model with graphs and apply the embedding on graphs.
        n_components is either automatically determined or based on user input.

        Parameters
        ----------
        graphs : list of nx.Graph or ndarray, or ndarray
            If list of nx.Graph, each Graph must contain same number of nodes.
            If list of ndarray, each array must have shape (n_vertices, n_vertices).
            If ndarray, then array must have shape (n_graphs, n_vertices, n_vertices).

        Returns
        -------
        out : np.ndarray or length 2 tuple of np.ndarray.
            If input graphs were symmetric shape (n_vertices, n_components).
            If graphs were directed and ``concat`` is False, returns tuple of two arrays (same shape as above).
            The first corresponds to the left latent positions, and the second to the right latent positions.
            When ``concat`` is True left and right (out and in) latent positions are concatenated along axis 1.
        """
        return self._fit_transform(graphs)
