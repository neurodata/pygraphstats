# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import warnings
import numpy as np

from .base import BaseSpectralEmbed
from ..utils import (
    import_graph,
    is_fully_connected,
    augment_diagonal,
    pass_to_ranks,
    is_unweighted,
)


class AdjacencySpectralEmbed(BaseSpectralEmbed):
    r"""
    Class for computing the adjacency spectral embedding of a graph.

    The adjacency spectral embedding (ASE) is a k-dimensional Euclidean representation
    of the graph based on its adjacency matrix. It relies on an SVD to reduce
    the dimensionality to the specified k, or if k is unspecified, can find a number of
    dimensions automatically (see :class:`~graspologic.embed.selectSVD`).

    Read more in the `Adjacency Spectral Embedding Tutorial
    <https://microsoft.github.io/graspologic/tutorials/embedding/AdjacencySpectralEmbed.html>`_

    Parameters
    ----------
    n_components : int or None, default = None
        Desired dimensionality of output data. If "full",
        ``n_components`` must be ``<= min(X.shape)``. Otherwise, ``n_components`` must be
        ``< min(X.shape)``. If None, then optimal dimensions will be chosen by
        :func:`~graspologic.embed.select_dimension` using ``n_elbows`` argument.

    n_elbows : int, optional, default: 2
        If ``n_components`` is None, then compute the optimal embedding dimension using
        :func:`~graspologic.embed.select_dimension`. Otherwise, ignored.

    algorithm : {'randomized' (default), 'full', 'truncated'}, optional
        SVD solver to use:

        - 'randomized'
            Computes randomized svd using
            :func:`sklearn.utils.extmath.randomized_svd`
        - 'full'
            Computes full svd using :func:`scipy.linalg.svd`
            Does not support ``graph`` input of type scipy.sparse.csr_matrix
        - 'truncated'
            Computes truncated svd using :func:`scipy.sparse.linalg.svds`

    n_iter : int, optional (default = 5)
        Number of iterations for randomized SVD solver. Not used by 'full' or
        'truncated'. The default is larger than the default in randomized_svd
        to handle sparse matrices that may have large slowly decaying spectrum.

    check_lcc : bool , optional (default = True)
        Whether to check if input graph is connected. May result in non-optimal
        results if the graph is unconnected. If True and input is unconnected,
        a UserWarning is thrown. Not checking for connectedness may result in
        faster computation.

    diag_aug : bool, optional (default = True)
        Whether to replace the main diagonal of the adjacency matrix with a vector
        corresponding to the degree (or sum of edge weights for a weighted network)
        before embedding. Empirically, this produces latent position estimates closer
        to the ground truth.

    concat : bool, optional (default False)
        If graph is directed, whether to concatenate left and right (out and in) latent positions along axis 1.



    Attributes
    ----------
    n_features_in_: int
        Number of features passed to the :func:`~graspologic.embed.AdjacencySpectralEmbed.fit` method.
    latent_left_ : array, shape (n_samples, n_components)
        Estimated left latent positions of the graph.
    latent_right_ : array, shape (n_samples, n_components), or None
        Only computed when the graph is directed, or adjacency matrix is assymetric.
        Estimated right latent positions of the graph. Otherwise, None.
    singular_values_ : array, shape (n_components)
        Singular values associated with the latent position matrices.

    See Also
    --------
    graspologic.embed.selectSVD
    graspologic.embed.select_dimension

    Notes
    -----
    The singular value decomposition:

    .. math:: A = U \Sigma V^T

    is used to find an orthonormal basis for a matrix, which in our case is the
    adjacency matrix of the graph. These basis vectors (in the matrices U or V) are
    ordered according to the amount of variance they explain in the original matrix.
    By selecting a subset of these basis vectors (through our choice of dimensionality
    reduction) we can find a lower dimensional space in which to represent the graph.

    References
    ----------
    .. [1] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
       Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
       Journal of the American Statistical Association, Vol. 107(499), 2012

    .. [2] Levin, K., Roosta-Khorasani, F., Mahoney, M. W., & Priebe, C. E. (2018).
        Out-of-sample extension of graph adjacency spectral embedding. PMLR: Proceedings
        of Machine Learning Research, 80, 2975-2984.
    """

    def __init__(
        self,
        n_components=None,
        n_elbows=2,
        algorithm="randomized",
        n_iter=5,
        check_lcc=True,
        diag_aug=True,
        concat=False,
    ):
        super().__init__(
            n_components=n_components,
            n_elbows=n_elbows,
            algorithm=algorithm,
            n_iter=n_iter,
            check_lcc=check_lcc,
            concat=concat,
        )

        if not isinstance(diag_aug, bool):
            raise TypeError("`diag_aug` must be of type bool")
        self.diag_aug = diag_aug
        self.is_fitted_ = False

    def fit(self, graph, y=None):
        """
        Fit ASE model to input graph

        Parameters
        ----------
        graph : array-like, scipy.sparse.csr_matrix, or networkx.Graph
            Input graph to embed.

        y: Ignored

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        A = self._fit(graph)

        if self.diag_aug:
            A = augment_diagonal(A)

        self._reduce_dim(A)

        self.is_fitted_ = True

        return self

    def _compute_oos_prediction(self, X, directed):
        """
        Computes the out-of-sample latent position estimation.
        Parameters
        ----------
        X: np.ndarray
            Input to do oos embedding on.
        directed: bool
            Indication if graph is directed or undirected
        Returns
        -------
        out : array_like or tuple, shape
        """
        if not directed:
            return X @ self._pinv_left
        elif directed:
            return X[1] @ self._pinv_right, X[0] @ self._pinv_left
