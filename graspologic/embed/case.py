import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.preprocessing import normalize, scale
from typing import Tuple, Callable, Optional

from graspologic.utils import import_graph, to_laplacian, is_almost_symmetric
from graspologic.embed.base import BaseSpectralEmbed


class CovariateAssistedEmbed(BaseSpectralEmbed):
    """
    The Covariate-Assisted Spectral Embedding is a tabular representation of a graph
    which has extra covariate information for each node. It returns an n x d matrix,
    similarly to Adjacency Spectral Embedding or Laplacian Spectral Embedding.

    Parameters
    ----------
    alpha : float, optional, default = None
        Tuning parameter. This is a value which lets you change how much information
        the covariates contribute to the embedding. Higher values mean the covariates
        contribute more information.
            -  None: Use a weight which causes the covariates and the graph to contribute
               the same amount of information to the embedding.
            - float: Manually set the tuning parameter value.

    assortative : bool, default = True
        Embedding algorithm to use. An assortative network is any network where the
        within-group probabilities are greater than the between-group probabilities.
        Here, L is the regularized Laplacian, Y is the covariate matrix, and a is the
        tuning parameter alpha:
            - True: Embed ``L + a*Y@Y.T``. Better for assortative graphs.
            - False: Embed ``L@L + a*Y@Y.T``. Better for non-assortative graphs.

    center_covariates: bool, default = True
        Whether or not to center the columns of the covariates to have mean 0.

    scale_covariates: bool, default = True
        Whether or not to scale the columns of the covariates to have unit L2-norm.

    n_components : int or None, default = None
        Desired dimensionality of output data. If "full",
        n_components must be <= min(Y.shape). Otherwise, n_components must be
        < min(Y.shape). If None, then optimal dimensions will be chosen by
        ``select_dimension`` using ``n_elbows`` argument.

    n_elbows : int, optional, default: 2
        If `n_components=None`, then compute the optimal embedding dimension using
        `select_dimension`. Otherwise, ignored.

    check_lcc : bool , optional, defult = True
        Whether to check if input graph is connected. May result in non-optimal
        results if the graph is unconnected. Not checking for connectedness may
        result in faster computation.

    Notes
    -----
    Depending on the embedding algorithm, we embed

        .. math:: L_ = LL + \alpha YY^T
        .. math:: L_ = L + \alpha YY^T

        where :math:`\alpha` is the tuning parameter which makes the leading eigenvalues
        of the two summands the same. Here, :math:`L` is the regularized
        graph Laplacian, and :math:`Y` is a matrix of covariates for each node.

    For more information, see [1].

    References
    ---------
    .. [1] Binkiewicz, N., Vogelstein, J. T., & Rohe, K. (2017). Covariate-assisted
    spectral clustering. Biometrika, 104(2), 361-377.
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        assortative: bool = True,
        center_covariates: bool = True,
        scale_covariates: bool = True,
        n_components: Optional[int] = None,
        n_elbows: int = 2,
        check_lcc: bool = False,
    ):
        super().__init__(
            n_components=n_components,
            n_elbows=n_elbows,
            check_lcc=check_lcc,
            concat=False,
            algorithm="eigsh",
        )

        if not isinstance(assortative, bool):
            msg = "embedding_alg must be a boolean value."
            raise ValueError(msg)
        self.assortative = assortative

        if not isinstance(alpha, (float, int, type(None))):
            msg = "alpha's type must be in {None, float, int}."
            raise TypeError(msg)

        self.alpha = alpha
        self.latent_right_ = None
        self._centered = center_covariates
        self._scaled = scale_covariates
        self.is_fitted_ = False

    def fit(
        self, graph: np.ndarray, covariates: np.ndarray, y: None = None
    ) -> "CovariateAssistedEmbed":
        """
        Fit a CASE model to an input graph, along with its covariates.

        Parameters
        ----------
        graph : array-like or networkx.Graph
            Input graph to embed. See graspologic.utils.import_graph

        covariates : array-like, shape (n_vertices, n_covariates)
            Covariate matrix. Each node of the graph is associated with a set of
            `d` covariates. Row `i` of the covariates matrix corresponds to node
            `i`, and the number of columns are the number of covariates.

        y: Ignored

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # setup
        A = import_graph(graph)
        directed = is_almost_symmetric(A)
        if directed:
            raise NotImplementedError("CASE currently only works with directed graphs.")

        # Create regularized Laplacian, scale covariates to unit norm
        L = to_laplacian(A, form="R-DAD")
        Y = covariates
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        if self._centered:
            Y = scale(Y, with_std=False, axis=0)
        if self._scaled:
            Y = normalize(Y, axis=0)

        # Use ratio of the two leading eigenvalues if alpha is None
        self._get_tuning_parameter(L, Y)

        # get embedding matrix as a LinearOperator (for computational efficiency)
        n = A.shape[0]
        mv, rmv = self._matvec(L, Y, a=self.alpha_, assortative=self.assortative)
        L_ = LinearOperator((n, n), matvec=mv, rmatvec=rmv)

        # Dimensionality reduction with SVD
        self._reduce_dim(L_, directed=False)

        self.is_fitted_ = True
        return self

    def fit_transform(self, graph: np.ndarray, covariates: np.ndarray):
        # Allows `for self.fit_transform(graph, covariates)` without needing keyword arguments.
        return self._fit_transform(graph, covariates=covariates)

    def _get_tuning_parameter(
        self, L: np.ndarray, Y: np.ndarray
    ) -> "CovariateAssistedEmbed":
        """
        Find the alpha which causes the leading eigenspace of LL and YYt to be the same.
        Saves that alpha as the class attribute ``self.alpha_``.

        Parameters
        ----------
        L : array, n x n
            The regularized graph Laplacian, where ``n`` is the number of nodes.
        Y : array, n x d
            The covariate matrix, where ``n`` is the number of nodes and ``d`` is the
             number of covariates.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # setup
        if self.alpha is not None:
            self.alpha_ = self.alpha
            return self

        # Laplacian leading eigenvector
        (L_top,) = eigsh(L, return_eigenvectors=False, k=1)
        if self.assortative:
            L_top **= 2

        # YY^T leading eigenvector
        n = Y.shape[0]
        YO = LinearOperator((n, n), matvec=lambda v: Y @ (Y.T @ v))
        (YYt_top,) = eigsh(YO, return_eigenvectors=False, k=1)

        # just use the ratio of the leading eigenvalues for the
        # tuning parameter, or the closest value in its possible range.
        self.alpha_ = np.float(L_top / YYt_top)

        return self

    @staticmethod
    def _matvec(
        L: np.ndarray, Y: np.ndarray, a: float, assortative: bool = True
    ) -> Tuple[Callable, Callable]:
        """
        Defines matrix multiplication and matrix multiplication by transpose for the
        LinearOperator object.
        """
        if assortative:
            mv = lambda v: (L @ v) + a * (Y @ (Y.T @ v))
            rmv = lambda v: (v.T @ L) + a * ((v.T @ Y) @ Y.T)
        else:
            mv = lambda v: (L @ (L @ v)) + a * (Y @ (Y.T @ v))
            rmv = lambda v: (v.T @ L) @ L + a * ((v.T @ Y) @ Y.T)
        return mv, rmv
