# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import warnings

from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ..utils import (
    augment_diagonal,
    import_graph,
    is_almost_symmetric,
    is_fully_connected,
)
from .svd import selectSVD

import networkx as nx


class BaseSpectralEmbed(BaseEstimator):
    """
    A base class for embedding a graph.

    Parameters
    ----------
    n_components : int or None, default = None
        Desired dimensionality of output data. If "full",
        n_components must be <= min(X.shape). Otherwise, n_components must be
        < min(X.shape). If None, then optimal dimensions will be chosen by
        ``select_dimension`` using ``n_elbows`` argument.
    n_elbows : int, optional, default: 2
        If `n_compoents=None`, then compute the optimal embedding dimension using
        `select_dimension`. Otherwise, ignored.
    algorithm : {'full', 'truncated' (default), 'randomized'}, optional
        SVD solver to use:

        - 'full'
            Computes full svd using ``scipy.linalg.svd``
        - 'truncated'
            Computes truncated svd using ``scipy.sparse.linalg.svd``
        - 'randomized'
            Computes randomized svd using
            ``sklearn.utils.extmath.randomized_svd``
    n_iter : int, optional (default = 5)
        Number of iterations for randomized SVD solver. Not used by 'full' or
        'truncated'. The default is larger than the default in randomized_svd
        to handle sparse matrices that may have large slowly decaying spectrum.
    check_lcc : bool , optional (defult =True)
        Whether to check if input graph is connected. May result in non-optimal
        results if the graph is unconnected. Not checking for connectedness may
        result in faster computation.
    concat : bool, optional (default = False)
        If graph(s) are directed, whether to concatenate each graph's left and right (out and in) latent positions
        along axis 1.

    Attributes
    ----------
    n_components_ : int
        Dimensionality of the embedded space.
    n_features_in_: int
        Number of features passed to the fit method.

    See Also
    --------
    graspologic.embed.selectSVD, graspologic.embed.select_dimension
    """

    def __init__(
        self,
        n_components=None,
        n_elbows=2,
        algorithm="randomized",
        n_iter=5,
        check_lcc=True,
        concat=False,
    ):
        self.n_components = n_components
        self.n_elbows = n_elbows
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.check_lcc = check_lcc
        if not isinstance(concat, bool):
            msg = "Parameter `concat` is expected to be type bool"
            raise TypeError(msg)
        self.concat = concat

    def _reduce_dim(self, A):
        """
        A function that reduces the dimensionality of an adjacency matrix
        using the desired embedding method.

        Parameters
        ----------
        A: array-like, shape (n_vertices, n_vertices)
            Adjacency matrix to embed.
        """
        U, D, V = selectSVD(
            A,
            n_components=self.n_components,
            n_elbows=self.n_elbows,
            algorithm=self.algorithm,
            n_iter=self.n_iter,
        )

        self.n_components_ = D.size
        self.singular_values_ = D
        self.latent_left_ = U @ np.diag(np.sqrt(D))
        if not is_almost_symmetric(A):
            self.latent_right_ = V.T @ np.diag(np.sqrt(D))
        else:
            self.latent_right_ = None

    @property
    def _pairwise(self):
        """This is for sklearn compliance."""
        return True

    @abstractmethod
    def fit(self, graph, y=None):
        """
        A method for embedding.
        Parameters
        ----------
        graph: np.ndarray or networkx.Graph
        y : Ignored
        Returns
        -------
        lpm : LatentPosition object
            Contains X (the estimated latent positions), Y (same as X if input is
            undirected graph, or right estimated positions if directed graph), and d.
        See Also
        --------
        import_graph, LatentPosition
        """
        # call self._reduce_dim(A) from your respective embedding technique.
        # import graph(s) to an adjacency matrix using import_graph function
        # here

        return self

    def _fit(self, graph, y=None):
        """
        A method for embedding.

        Parameters
        ----------
        graph: np.ndarray or networkx.Graph

        y : Ignored

        Returns
        -------
        A : array-like, shape (n_vertices, n_vertices)
            A graph

        See Also
        --------
        import_graph, LatentPosition
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

        self.n_features_in_ = A.shape[0]
        return A

    def _fit_transform(self, graph):
        "Fits the model and returns the estimated latent positions."
        self.fit(graph)

        if self.latent_right_ is None:
            return self.latent_left_
        else:
            if self.concat:
                return np.concatenate((self.latent_left_, self.latent_right_), axis=1)
            else:
                return self.latent_left_, self.latent_right_

    def fit_transform(self, graph, y=None):
        """
        Fit the model with graphs and apply the transformation.

        n_dimension is either automatically determined or based on user input.

        Parameters
        ----------
        graph: np.ndarray or networkx.Graph
            Input graph to embed.

        Returns
        -------
        out : np.ndarray OR length 2 tuple of np.ndarray.
            If undirected then returns single np.ndarray of latent position, shape(n_vertices, n_components).
            If directed, ``concat`` is True then concatenate latent matrices on axis 1, shape(n_vertices, 2*n_components).
            If directed, ``concat`` is False then tuple of the latent matrices. Each of shape (n_vertices, n_components).
        """
        return self._fit_transform(graph)

    def transform(self, X):
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
        out : np.ndarray OR length 2 tuple of np.ndarray

            Array of latent positions, shape (n_oos_vertices, n_components) or (n_vertices, n_components). Transforms the fitted matrix if it was passed
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

        # for oos prediction
        inv_eigs = np.diag(1 / self.singular_values_)

        self._pinv_left = self.latent_left_ @ inv_eigs
        if self.latent_right_ is not None:
            self._pinv_right = self.latent_right_ @ inv_eigs

        # correct shape in y?
        latent_rows = self.latent_left_.shape[0]
        _X = X[0] if directed else X
        X_cols = _X.shape[-1]
        if _X.ndim > 2:
            raise ValueError("out-of-sample vertex must be 1d or 2d")
        if latent_rows != X_cols:
            msg = "out-of-sample vertex must be shape (n_oos_vertices, n_vertices)"
            raise ValueError(msg)

        return self._compute_oos_prediction(X, directed)

    @abstractmethod
    def _compute_oos_prediction(self, X, directed):
        """
        Computes the oos class specific estimation given in an input array and if the graph is directed.

        Parameters
        ----------
        X: np.ndarray
            Input to do oos embedding on.

        directed: bool
            Indication if graph is directed or undirected

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
        """

        pass


class BaseEmbedMulti(BaseSpectralEmbed):
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

    def _check_input_graphs(self, graphs):
        """
        Checks if all graphs in list have same shapes.

        Raises an ValueError if there are more than one shape in the input list,
        or if the list is empty or has one element.

        Parameters
        ----------
        graphs : list of nx.Graph or ndarray, or ndarray
            If list of nx.Graph, each Graph must contain same number of nodes.
            If list of ndarray, each array must have shape (n_vertices, n_vertices).
            If ndarray, then array must have shape (n_graphs, n_vertices, n_vertices).

        Returns
        -------
        out : ndarray, shape (n_graphs, n_vertices, n_vertices)

        Raises
        ------
        ValueError
            If all graphs do not have same shape, or input list is empty or has
            one element.
        """
        # Convert input to np.arrays
        # This check is needed because np.stack will always duplicate array in memory.
        if isinstance(graphs, (list, tuple)):
            if len(graphs) <= 1:
                msg = "Input {} must have at least 2 graphs, not {}.".format(
                    type(graphs), len(graphs)
                )
                raise ValueError(msg)
            out = [import_graph(g, copy=False) for g in graphs]
        elif isinstance(graphs, np.ndarray):
            if graphs.ndim != 3:
                msg = "Input tensor must be 3-dimensional, not {}-dimensional.".format(
                    graphs.ndim
                )
                raise ValueError(msg)
            elif graphs.shape[0] <= 1:
                msg = "Input tensor must have at least 2 elements, not {}.".format(
                    graphs.shape[0]
                )
                raise ValueError(msg)
            out = import_graph(graphs, copy=False)
        else:
            msg = "Input must be a list or ndarray, not {}.".format(type(graphs))
            raise TypeError(msg)

        # Save attributes
        self.n_graphs_ = len(out)
        self.n_vertices_ = out[0].shape[0]

        return out

    def _diag_aug(self, graphs):
        """
        Augments the diagonal off each input graph. Returns the original
        input object type.

        Parameters
        ----------
        graphs : list of nx.Graph or ndarray, or ndarray
            If list of nx.Graph, each Graph must contain same number of nodes.
            If list of ndarray, each array must have shape (n_vertices, n_vertices).
            If ndarray, then array must have shape (n_graphs, n_vertices, n_vertices).


        Returns
        -------
        out : list of ndarray, or ndarray
            If input is list of ndarray, then list is returned.
            If input is ndarray, then ndarray is returned.
        """
        if isinstance(graphs, list):
            out = [augment_diagonal(g) for g in graphs]
        elif isinstance(graphs, np.ndarray):
            # Copying is necessary to not overwrite input array
            out = np.array([augment_diagonal(graphs[i]) for i in range(self.n_graphs_)])

        return out
