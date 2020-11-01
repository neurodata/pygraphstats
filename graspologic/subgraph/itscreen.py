# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import numbers

from .base import BaseSubgraph
from . import Screen


class ItScreen(BaseSubgraph):
    """
    Class to estimate signal subgraph (ss) for a covariate using iterative
    vertex screening.

    Iterative screening uses the rows of the adjacency matricies as feature
    vectors for each node. This algorithm then calculates the correlation
    between the features and the label vector. Any node whose correlation
    value is in a designated quantile of the correlation values is kept
    and the non-iterative algorithm is run on the remaining nodes.
    Read more in the :ref:`tutorials <subgraph_tutorials>`

    Parameters
    ----------
    stat : string
        Desired test statistic to use on data. If "mgc",
        mulstiscale graph correlation will be used. Otherwise, must be
        either "dcorr", "rv", or "cca".
    delta : float or int
        Desired quantile to be used for iterative screening. Must be between
        0 and 1.
    sg_n_verts : int
        Number of vertices for the produed subgraph. Must be between 0 and n
        vertices.

    See Also
    --------
    graspy.subgraph.nonitscreen

    References
    ----------
    .. [1] S. Wang, C. Chen, A. Badea, Priebe, C.E., Vogelstein, J.T. "Signal
       Subgraph Estimation Via Vertex Screening" arXiv: 1801.07683 [stat.ME],
       2018
    """

    def __init__(self, stat: str, delta: float, sg_n_verts: int) -> None:
        super().__init__(stat=stat)

        if not isinstance(delta, numbers.Real):
            msg = "delta must be float or int"
            raise ValueError(msg)
        if delta < 0 or delta > 1:
            msg = "delta must be in the interval (0,1)"
            raise ValueError(msg)

        self.delta = delta

        if not isinstance(sg_n_verts, int):
            msg = "sg_n_verts must be an int"
            raise ValueError(msg)

        self.sg_n_verts = sg_n_verts

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Performs iterative screening on graphs to estimate signal subgraph.

        Parameters
        ----------
        X : np.ndarray, shape ``(n_graphs, n_vertices, n_vertices)``
            Tensor of adjacency matrices
        y : np.ndarray, shape ``(n_graphs, 1)``
            Vector of ground truth labels

        Returns
        -------
        self : returns an instance of self.
        """

        self._screen(X, y)

        if self.sg_n_verts < 0 or self.sg_n_verts > self.n_verts:
            msg = "sg_n_vertices must be between 0 and n_vertices"
            raise ValueError(msg)

        # Create empty array to store correlation values
        cors = np.zeros((self.n_verts, 1))
        sg_inds = np.arange(self.n_verts, dtype="int64")
        itr = 1

        tmpq = float("-inf")
        Atmp = X

        val = 1 - self.delta
        while (val ** itr) * self.n_verts > self.sg_n_verts:

            # Creating new temporary set of subgraphs
            screen = Screen(self.stat, tmpq)
            Atmp = screen.fit_transform(Atmp, y)
            screen = screen.fit(Atmp, y)

            # Correlation values
            tmpcors = screen.corrs

            # Specified quantile of correlation values
            tmpq = np.quantile(tmpcors, self.delta)

            # Add weights to the remaining correlation values
            cors[sg_inds] = tmpcors + itr

            # Iterate
            ind = tmpcors > tmpq
            ind = ind.reshape(1, len(ind))
            sg_inds = sg_inds[ind[0]]
            itr += 1

        self.corrs = cors
        return self

    def _fit_transform(self, X, y):
        "Finds the signal subgraph from the correlation values"
        self.fit(X, y)

        # Returns the indicies of the sg_n_verts largest values
        ind = self.corrs.reshape(1, self.n_verts)
        val = self.sg_n_verts
        ind = ind[0].argsort()[-val:][::-1]
        ind = np.sort(ind)
        sg_inds = ind.reshape(1, len(ind))[0]
        self.sg_inds = sg_inds

        n = len(sg_inds)
        S_hat = np.zeros((self.n_graphs, n, n))
        for i in range(self.n_graphs):
            S_hat[i] = X[i][sg_inds][:, sg_inds]

        return S_hat

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Apply screening to graph set X to find nodes with
        high enough correlation values.

        Parameters
        ----------
        X : np.ndarray, shape ``(n_graphs, n_vertices, n_vertices)``
        y : np.ndarray, shape ``(n_graphs, 1)``

        Returns
        -------
        out : np.ndarray, shape ``(n_graphs, sg_n_inds, sg_n_inds)``
        """

        return self._fit_transform(X, y)
