# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import List
from typing import Tuple

import numpy as np
from scipy.stats import fisher_exact

from .base import BaseSubgraph

Vector = List[float]


class Coherence(BaseSubgraph):
    """
    Class to estimate signal subgraph (ss) for a covariate using either an
    incoherent estimator, which is constrained by the number of edges, or a
    coherent estimator, which is constrained by the number of edges and by the
    number of vertices that the edges in the ss may be incident to.
    Read more in the :ref:`tutorials <subgraph_tutorials>`

    Attributes
    ----------
    contmat_ : np.ndarray, shape ``(n_vertices, n_vertices, 2, 2)``
        An array that stores the 2-by-2 contingency matrix for each point in
        the graph samples.
    sigsub_ : tuple, shape ``(2, n_edges)``
        A tuple of a row index array and column index array, where n_edges is
        determined by constraints.
    mask_ : np.ndarray, shape ``(n_vertices, n_vertices)``
        An array of boolean values. Entries are true for edges that are in the
        ss.

    References
    ----------
    .. [1] J. T. Vogelstein, W. R. Gray, R. J. Vogelstein, and C. E. Priebe,
       "Graph Classification using Signal-Subgraphs: Applications in
       Statistical Connectomics," arXiv:1108.1427v2 [stat.AP], 2012.
    """

    def __init__(self) -> None:
        pass

    def fit(
        self,
        X: np.ndarray,
        y: Tuple[np.ndarray, Vector],
        constraints: Tuple[int, np.ndarray, Vector],
    ) -> None:
        """
        Fit the signal-subgraph estimator according to the constraints given.

        Parameters
        ----------
        X : np.ndarray, shape ``(n_graphs, n_vertices, n_vertices)``
            A series of labeled ``(n_vertices, n_vertices)`` unweighted graph
            samples. If undirected, the upper or lower triangle matrices
            should be used.
        y : vector, length ``(n_graphs)``
            A vector of class labels. There must be a maximum of two classes.
        constraints : int or vector
            The constraints that will be imposed onto the estimated ss.

            If constraints is an int, constraints is the number of edges in
            the ss.

            If constraints is a vector, then its first element is the number of
            edges in the ss, and its second element is the number of vertices
            that the ss must be incident to.

        Returns
        -------
        self : returns an instance of self
        """
        if not isinstance(constraints, int):
            if (not isinstance(constraints, np.ndarray)) and len(constraints) != 2:
                msg = "constraints must be an int or vector with length 2."
                raise TypeError(msg)

        if not isinstance(X, np.ndarray):
            msg = "X must be np.ndarray, not {}.".format(type(X))
            raise TypeError(msg)
        if not isinstance(y, (list, np.ndarray)):
            msg = "y must be list or np.ndarray, not {}.".format(type(y))
            raise TypeError(msg)

        shape = np.shape(X)
        if len(shape) != 3:
            msg = "X must be 3d with shape (n_vertices, n_vertices, n_graphs)."
            raise ValueError(msg)
        if shape[0] != shape[1]:
            msg = "X must have matching number of vertices."
            raise ValueError(msg)

        if len(np.shape(y)) != 1:
            msg = "y must be 1-dimensional."
            raise ValueError(msg)
        if len(np.unique(y)) > 2:
            msg = "y must have a maximum of two classes, not {}.".format(
                len(np.unique(y))
            )
            raise ValueError(msg)
        if len(y) != shape[2]:
            msg = "y length must match the number of graph samples."
            raise ValueError(msg)
        else:
            self.X = X
            self.y = y

        nverts = np.shape(self.X)[0]
        out = np.zeros((nverts, nverts, 2, 2))
        rowsum1 = sum(self.y)
        rowsum0 = len(self.y) - rowsum1
        for i in range(nverts):
            for j in range(nverts):
                a = sum(self.X[i, j, self.y == 0])
                b = sum(self.X[i, j, :]) - a
                out[i, j, :, :] = [[a, rowsum0 - a], [b, rowsum1 - b]]
        self.contmat_ = out

        verts = np.shape(self.X)[0]
        sigmat = np.array(
            [
                [fisher_exact(self.contmat_[i, j, :, :])[1] for j in range(verts)]
                for i in range(verts)
            ]
        )

        # incoherent
        if isinstance(constraints, int):
            nedges = constraints
            sigsub = np.dstack(
                np.unravel_index(np.argsort(sigmat.ravel()), np.shape(sigmat))
            )
            sigsub = sigsub[0, :nedges, :]
            sigsub = tuple(np.transpose(sigsub))

        # coherent
        else:
            nedges = constraints[0]
            nverts = constraints[1]

            wset = np.unique(sigmat, axis=None)
            wcounter = 0
            wconv = 0

            while wconv == 0:
                w = wset[wcounter]
                blank = sigmat
                blank = blank > w

                score = 2 * verts - (np.sum(blank, axis=1) + np.sum(blank, axis=0))
                vscore = np.sort(score)[::-1]
                vstars = np.argsort(score)[::-1]

                if (vscore[:nverts].sum()) >= nedges:
                    blank = np.ones(np.shape(sigmat))
                    nstars = np.amin([len(vscore[vscore > 0]), nverts])
                    vstars = vstars[:nstars]

                    blank[vstars, :] = sigmat[vstars, :]
                    blank[:, vstars] = sigmat[:, vstars]

                    indsp = np.dstack(
                        np.unravel_index(np.argsort(blank.ravel()), np.shape(blank))
                    )
                    sigsub = indsp[0, :nedges, :]
                    sigsub = tuple(np.transpose(sigsub))
                    wconv = 1
                else:
                    wcounter = wcounter + 1
                    if wcounter > len(wset):
                        sigsub = []
                        wconv = 1

        self.sigsub_ = sigsub
        return self

    def fit_transform(
        self,
        X: np.ndarray,
        y: Tuple[np.ndarray, Vector],
        constraints: Tuple[int, np.ndarray, Vector],
    ) -> None:
        """
        A function to return the indices of the signal subgraph.

        Parameters
        ----------
        X : array-like, shape ``(n_vertices, n_vertices, n_graphs)``
            A series of labeled ``(n_vertices, n_vertices)`` unweighted graph
            samples. If undirected, the upper or lower triangle matrices should
            be used.
        y : vector, length ``(n_graphs)``
            A vector of class labels. There must be a maximum of two classes.
        constraints : int or vector
            The constraints that will be imposed onto the estimated ss.

            If constraints is an int, constraints is the number of edges in
            the ss.

            If constraints is a vector, then its first element is the number of
            edges in the ss, and its second element is the number of vertices
            that the ss must be incident to.

        Returns
        -------
        sigsub : tuple, shape ``(2, n_edges)``
        """
        self.fit(X, y, constraints)
        verts = np.shape(self.X)[0]
        mask = np.full((verts, verts), False)
        mask[self.sigsub_] = True
        self.mask_ = mask
        return self.sigsub_
