# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import warnings
import numpy as np
from typing import List, Optional, Union
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_is_fitted
import networkx as nx

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
        n_components: Optional[int] = None,
        n_elbows: int = 2,
        algorithm: str = "randomized",
        n_iter: int = 5,
        check_lcc: bool = True,
        diag_aug: bool = True,
        concat: bool = False,
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

    def fit(self, graph: Union[ArrayLike, nx.Graph], y: None = None) -> "AdjacencySpectralEmbed":
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
        A = import_graph(graph)

        if self.check_lcc:
            if not is_fully_connected(A):
                msg = (
                    "Input graph is not fully connected. Results may not"
                    + "be optimal. You can compute the largest connected component by"
                    + "using ``graspologic.utils.largest_connected_component``."
                )
                warnings.warn(msg, UserWarning)

        if self.diag_aug:
            A = augment_diagonal(A)

        self.n_features_in_ = A.shape[0]
        self._reduce_dim(A)

        # for out-of-sample
        inv_eigs = np.diag(1 / self.singular_values_)
        self._pinv_left = self.latent_left_ @ inv_eigs
        if self.latent_right_ is not None:
            self._pinv_right = self.latent_right_ @ inv_eigs

        self.is_fitted_ = True
        return self

    def transform(
        self, X: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Obtain latent positions from an adjacency matrix or matrix of out-of-sample
        vertices. For more details on transforming out-of-sample vertices, see the
        :ref:`tutorials <embed_tutorials>`. For mathematical background, see [2].

        Parameters
        ----------
        X : array-like or tuple, original shape or (n_oos_vertices, n_vertices).

            The original fitted matrix ("graph" in fit) or new out-of-sample data.
            If ``X`` is the original fitted matrix, returns a matrix close to
            ``self.fit_transform(X)``.

            If ``X`` is an out-of-sample matrix, n_oos_vertices is the number
            of new vertices, and n_vertices is the number of vertices in the
            original graph. If tuple, graph is directed and ``X[0]`` contains
            edges from out-of-sample vertices to in-sample vertices.

        Returns
        -------
        array_like or tuple, shape (n_oos_vertices, n_components)
        or (n_vertices, n_components).

            Array of latent positions. Transforms the fitted matrix if it was passed
            in.

            If ``X`` is an array or tuple containing adjacency vectors corresponding to
            new nodes, returns the estimated latent positions for the new out-of-sample
            adjacency vectors.
            If undirected, returns array.
            If directed, returns ``(X_out, X_in)``, where ``X_out`` contains
            latent positions corresponding to nodes with edges from out-of-sample
            vertices to in-sample vertices.

        Notes
        -----
        If the matrix was diagonally augmented (e.g., ``self.diag_aug`` was True), ``fit``
        followed by ``transform`` will produce a slightly different matrix than
        ``fit_transform``.

        To get the original embedding, using ``fit_transform`` is recommended. In the
        directed case, if A is the original in-sample adjacency matrix, the tuple
        (A.T, A) will need to be passed to ``transform`` if you do not wish to use
        ``fit_transform``.
        """

        # checks
        check_is_fitted(self, "is_fitted_")
        if isinstance(X, nx.classes.graph.Graph):
            X = import_graph(X)
        directed = self.latent_right_ is not None

        # correct types?
        if directed and not isinstance(X, tuple):
            if X.shape[0] == X.shape[1]:  # in case original matrix was passed
                msg = """A square matrix A was passed to ``transform`` in the directed case. 
                If this was the original in-sample matrix, either use ``fit_transform`` 
                or pass a tuple (A.T, A). If this was an out-of-sample matrix, directed
                graphs require a tuple (X_out, X_in)."""
                raise TypeError(msg)
            else:
                msg = "Directed graphs require a tuple (X_out, X_in) for out-of-sample transforms."
                raise TypeError(msg)
        if not directed and not isinstance(X, np.ndarray):
            raise TypeError("Undirected graphs require array input")

        # correct shape in y?
        latent_rows = self.latent_left_.shape[0]
        _X = X[0] if directed else X
        X_cols = _X.shape[-1]
        if _X.ndim > 2:
            raise ValueError("out-of-sample vertex must be 1d or 2d")
        if latent_rows != X_cols:
            msg = "out-of-sample vertex must be shape (n_oos_vertices, n_vertices)"
            raise ValueError(msg)

        # workhorse code
        if not directed:
            return X @ self._pinv_left
        elif directed:
            return X[1] @ self._pinv_right, X[0] @ self._pinv_left
