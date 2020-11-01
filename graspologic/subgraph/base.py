# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from scipy.stats import multiscale_graphcorr

from hyppo.independence import Dcorr, RV, CCA


class BaseSubgraph(BaseEstimator):
    """
    Base class for estimating the signal subgraph (ss).

    Parameters
    ----------
    stat : string
        Desired test statistic to use on data if alg is screen or itscreen.
        Must be either "mgc, "dcorr", "rv", or "cca". Defaulted to None.

    See Also
    --------
    graspy.subgraph.screen, graspy.subgraph.itscreen
    graspy.subgraph.coherence, graspy.subgraph.parse
    """

    def __init__(self, stat=None) -> None:
        stats = [None, "mgc", "dcorr", "rv", "cca"]

        if stat not in stats:
            msg = 'stat must be either None, "mgc", "dcorr", "rv", or "cca".'
            raise ValueError(msg)
        else:
            self.stat = stat

    def _screen(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Performs non-iterative screening on graphs.

        Parameters
        ----------
        X: np.ndarray, shape ``(n_graphs, n_vertices, n_vertices)``
            Tensor of adjacency matrices.
        y: np.ndarray, shape ``(n_graphs, 1)``
            Vector of ground truth labels.

        Attributes
        ----------
        n_graphs: int
            Number of graphs in X
        n_vertices: int
            Dimension of each graph in X

        Returns
        -------
        corrs: np.ndarray, shape ``(n_vertices, 1)``
            Vector of correlation values for each node.

        References
        ----------
        .. [1] S. Wang, C. Chen, A. Badea, Priebe, C.E., Vogelstein, J.T.
        "Signal Subgraph Estimation Via Vertex Screening" arXiv: 1801.07683
        [stat.ME], 2018
        """
        if type(X) is not np.ndarray:
            msg = "X must be numpy.ndarray"
            raise TypeError(msg)
        if type(y) is not np.ndarray:
            msg = "y must be numpy.ndarray"
            raise TypeError(msg)

        check_array(X, dtype=int, ensure_2d=False, allow_nd=True)
        check_array(y, dtype=int)

        self.n_graphs = X.shape[0]
        self.n_verts = X.shape[-1]

        if len(X.shape) != 3:
            msg = "X must be a tensor"
            raise ValueError(msg)
        if X.shape[1] != X.shape[2]:
            msg = "Entries in X must be square matrices"
            raise ValueError(msg)

        if y.shape != (self.n_graphs, 1):
            msg = "y must have shape (n_graphs, 1)"
            raise ValueError(msg)

        corrs = np.zeros((self.n_verts, 1))
        for i in range(self.n_verts):

            # Stacks the ith row of each matrix, creates a
            # matrix with dimension n_graphs by n_vertices
            mat = X[:, i]

            # Statistical measurement chosen by the user
            # Finds correlation between mat and labels
            if self.stat == "mgc":
                c_u, p_value, mgc_dict = multiscale_graphcorr(mat, y, reps=1)

            else:
                if self.stat == "dcorr":
                    test = Dcorr()
                elif self.stat == "rv":
                    test = RV()
                else:
                    test = CCA()
                c_u = test._statistic(mat, y)

            corrs[i][0] = c_u

        return corrs

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Estimate the signal subgraph.

        Parameters
        ----------
        X: np.ndarray
        y: np.ndarray

        Returns
        -------
        self : returns an instance of self.
        """
        pass
