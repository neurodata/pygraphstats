import numpy as np
from scipy.stats import fisher_exact


class SignalSubgraph:
    """
    Estimate the signal-subgraph of a set of labeled graph samples.

    The incoherent estimator finds the signal-subgraph, constrained by the
    number of edges. The coherent estimator finds the signal-subgraph,
    constrained by the number of edges and by the number of vertices that the
    edges in the signal-subgraph may be incident to.

    Parameters
    ----------
    graphs: array-like, shape (n_vertices, n_vertices, s_samples)
        A series of labeled (n_vertices, n_vertices) unweighted graph samples.
        If undirected, the upper or lower triangle matrices should be used.      
    labels: vector, length (s_samples)
        A vector of class labels. There must be a maximum of two classes.

    Attributes
    ----------
    contmat_: array-like, shape (n_vertices, n_vertices, 2, 2)
        An array that stores the 2-by-2 contingency matrix for each point in
        the graph samples.
    sigsub_: tuple, shape (2, n_edges)
        A tuple of a row index array and column index array, where n_edges is
        the size of the signal-subgraph determined by *constraints*.
    mask_: array-like, shape (n_vertices, n_vertices)
        An array of boolean values. Entries are true for edges that are in the
        signal subgraph.

    References
    ----------
    .. [1] J. T. Vogelstein, W. R. Gray, R. J. Vogelstein, and C. E. Priebe, "Graph Classification using Signal-Subgraphs: Applications in Statistical Connectomics," arXiv:1108.1427v2 [stat.AP], 2012.

    """

    def __construct_contingency(self):
        nverts = np.shape(self.graphs)[0]
        out = np.zeros((nverts, nverts, 2, 2))
        rowsum1 = sum(self.labels)
        rowsum0 = len(self.labels) - rowsum1
        for i in range(nverts):
            for j in range(nverts):
                a = sum(self.graphs[i, j, self.labels == 0])
                b = sum(self.graphs[i, j, :]) - a
                out[i, j, :, :] = [[a, rowsum0 - a], [b, rowsum1 - b]]
        self.contmat_ = out

    def fit(self, graphs, labels, constraints):
        """
        Fit the signal-subgraph estimator according to the constraints given.

        Parameters
        ----------
        graphs: array-like, shape (n_vertices, n_vertices, s_samples)
            A series of labeled (n_vertices, n_vertices) unweighted graph
            samples. If undirected, the upper or lower triangle matrices
            should be used.   
        labels: vector, length (s_samples)
            A vector of class labels. There must be a maximum of two classes.
        constraints: int or vector
            The constraints that will be imposed onto the estimated
            signal-subgraph.

            If *constraints* is an int, *constraints* is the number of edges
            in the signal-subgraph.
            If *constraints* is a vector, the first element of *constraints*
            is the number of edges in the signal-subgraph, and the second
            element of *constraints* is the number of vertices that the
            signal-subgraph must be incident to.

        Returns
        -------
        self: returns an instance of self
        """
        if not isinstance(graphs, np.ndarray):
            msg = "Input array 'graphs' must be np.ndarray, not {}.".format(
                type(graphs)
            )
            raise TypeError(msg)
        if not isinstance(labels, (list, np.ndarray)):
            msg = "Input vector 'labels' must be list or np.ndarray, not {}.".format(
                type(labels)
            )
            raise TypeError(msg)

        shape = np.shape(graphs)
        if len(shape) != 3:
            msg = "Input array 'graphs' must be 3-dimensional with shape (n_vertices, n_vertices, s_samples)."
            raise ValueError(msg)
        if shape[0] != shape[1]:
            msg = "Input array 'graphs' must have matching number of vertices."
            raise ValueError(msg)

        if len(np.shape(labels)) != 1:
            msg = "Input vector 'labels' must be 1-dimensional."
            raise ValueError(msg)
        if len(np.unique(labels)) > 2:
            msg = "Input arrays must have a maximum of two classes, not {}.".format(
                len(np.unique(labels))
            )
            raise ValueError(msg)
        if len(labels) != shape[2]:
            msg = "Input vector length must match the number of graph samples."
            raise ValueError(msg)
        else:
            self.graphs = graphs
            self.labels = labels

        self.__construct_contingency()
        verts = np.shape(self.graphs)[0]
        sigmat = np.array(
            [
                [fisher_exact(self.contmat_[i, j, :, :])[1] for j in range(verts)]
                for i in range(verts)
            ]
        )

        if isinstance(constraints, (int)):  # incoherent
            nedges = constraints
            sigsub = np.dstack(
                np.unravel_index(np.argsort(sigmat.ravel()), np.shape(sigmat))
            )
            sigsub = sigsub[0, :nedges, :]
            sigsub = tuple(np.transpose(sigsub))

        elif len(constraints) == 2:  # coherent
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
        else:
            msg = "Input constraints must be an int for the incoherent signal-subgraph estimator, or a vector of length 2 for the coherent subgraph estimator."
            raise TypeError(msg)
        self.sigsub_ = sigsub
        return self

    def fit_transform(self, graphs, labels, constraints):
        """
        A function to return the indices of the signal-subgraph.
        If *return_mask* is True, also returns a mask for the signal-subgraph.

        Parameters
        ----------
        graphs: array-like, shape (n_vertices, n_vertices, s_samples)
            A series of labeled (n_vertices, n_vertices) unweighted graph
            samples. If undirected, the upper or lower triangle matrices
            \should be used.     
        labels: vector, length (s_samples)
            A vector of class labels. There must be a maximum of two classes.
        constraints: int or vector
            The constraints that will be imposed onto the estimated signal-subgraph.

            If *constraints* is an int, *constraints* is the number of edges
            in the signal-subgraph.

            If *constraints* is a vector, the first element of *constraints*
            is the number of edges in the signal-subgraph, and the second
            element of *constraints* is the number of vertices that the
            signal-subgraph must be incident to.

        Returns
        -------
        sigsub: tuple
            Contains an array of row indices and an array of column indices.
        """
        self.fit(graphs, labels, constraints)
        verts = np.shape(self.graphs)[0]
        mask = np.full((verts, verts), False)
        mask[self.sigsub_] = True
        self.mask_ = mask
        return self.sigsub_

    def _estimate_p_uv_y(self, u, v, class_label):
        """
        Calculate the smoothed class-conditional probability of an edge
        occuring between vertecies u and v.
        """

        # Calculate class-conditional probability
        n = len(self.labels)
        eta = 1 / (10 * n)
        n_edge, n_no_edge = self.contmat_[u, v, :, class_label]
        p_uv_y = n_edge / (n_edge + n_no_edge)

        # If necessary, smooth estimate
        if p_uv_y == 0:
            return eta
        elif p_uv_y == 1:
            return 1 - eta
        else:
            return p_uv_y

    def _estimate_pi(self):
        """Calculate the posterior distribution of class labels."""
        n = len(self.labels)
        self.pi_ = dict()
        for class_label in np.unique(self.labels):
            pi_i = sum(self.labels == class_label) / n
            self.pi_[class_label] = pi_i

    def _predict_class_proba(self, graph, class_label):
        """Calculate the probability of a graph belonging to a specific class."""
        pi_i = self.pi_[class_label]
        proba = 1
        for u, v in zip(*self.sigsub_):
            a_uv = graph[u, v]
            p_uv_y = self._estimate_p_uv_y(u, v, class_label)
            proba *= (p_uv_y ** a_uv) * (1 - p_uv_y ** (1 - a_uv))
        return proba * pi_i

    def predict_proba(self, graph):
        """Predict the probability of a graph belonging to all classes."""

        # Estimate pi
        self._estimate_pi()

        return {
            class_label: self._predict_class_proba(graph, class_label)
            for class_label in np.unique(self.labels)
        }

    def predict(self, graph):
        """Predict most likely class for a given graph."""
        proba = self.predict_proba(graph)
        return max(proba, key=proba.get)
