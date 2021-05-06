# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np

from ..utils import symmetrize, cartesian_product
from sklearn.utils import check_array, check_scalar
import warnings


def _n_to_labels(n):
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels


def sample_edges(P, directed=False, loops=False, rng=None):
    """
    Gemerates a binary random graph based on the P matrix provided

    Each element in P represents the probability of a connection between
    a vertex indexed by the row i and the column j.

    Parameters
    ----------
    P: np.ndarray, shape (n_vertices, n_vertices)
        Matrix of probabilities (between 0 and 1) for a random graph
    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.
    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.
    rng: numpy.random.Generator, optional (default = None)
        :class:`numpy.random.Generator` object to sample from distributions.
        If None, :class:`numpy.random.RandomState` is utilized.

    Returns
    -------
    A: ndarray (n_vertices, n_vertices)
        Binary adjacency matrix the same size as P representing a random
        graph

    References
    ----------
    .. [1] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
       Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
       Journal of the American Statistical Association, Vol. 107(499), 2012
    """
    if type(P) is not np.ndarray:
        raise TypeError("P must be numpy.ndarray")
    if len(P.shape) != 2:
        raise ValueError("P must have dimension 2 (n_vertices, n_dimensions)")
    if P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix")
    if rng is None:
        rng = np.random
    elif not isinstance(rng, (np.random._generator.Generator)):
        msg = "rng must be <class 'numpy.random._generator.Generator'> not {}.".format(
            type(rng)
        )
        raise TypeError(msg)
    if not directed:
        # can cut down on sampling by ~half
        triu_inds = np.triu_indices(P.shape[0])
        samples = rng.binomial(1, P[triu_inds])
        A = np.zeros_like(P)
        A[triu_inds] = samples
        A = symmetrize(A, method="triu")
    else:
        A = rng.binomial(1, P)

    if loops:
        return A
    else:
        return A - np.diag(np.diag(A))


def er_np(n, p, directed=False, loops=False, wt=1, wtargs=None, dc=None, dc_kws={}):
    r"""
    Samples a Erdos Renyi (n, p) graph with specified edge probability.

    Erdos Renyi (n, p) graph is a simple graph with n vertices and a probability
    p of edges being connected.

    Read more in the `Erdos-Renyi (ER) Model Tutorial
    <https://microsoft.github.io/graspologic/tutorials/simulations/erdos_renyi.html>`_

    Parameters
    ----------
    n: int
       Number of vertices

    p: float
        Probability of an edge existing between two vertices, between 0 and 1.

    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.

    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.

    wt: object, optional (default=1)
        Weight function for each of the edges, taking only a size argument.
        This weight function will be randomly assigned for selected edges.
        If 1, graph produced is binary.

    wtargs: dictionary, optional (default=None)
        Optional arguments for parameters that can be passed
        to weight function ``wt``.

    dc: function or array-like, shape (n_vertices)
        ``dc`` is used to generate a degree-corrected Erdos Renyi Model in
        which each node in the graph has a parameter to specify its expected degree
        relative to other nodes.

        - function:
            should generate a non-negative number to be used as a degree correction to
            create a heterogenous degree distribution. A weight will be generated for
            each vertex, normalized so that the sum of weights is 1.
        - array-like of scalars, shape (n_vertices):
            The weights should sum to 1; otherwise, they will be
            normalized and a warning will be thrown. The scalar associated with each
            vertex is the node's relative expected degree.

    dc_kws: dictionary
        Ignored if ``dc`` is none or array of scalar.
        If ``dc`` is a function, ``dc_kws`` corresponds to its named arguments.
        If not specified, in either case all functions will assume their default
        parameters.

    Returns
    -------
    A : ndarray, shape (n, n)
        Sampled adjacency matrix

    Examples
    --------
    >>> np.random.seed(1)
    >>> n = 4
    >>> p = 0.25

    To sample a binary Erdos Renyi (n, p) graph:

    >>> er_np(n, p)
    array([[0., 0., 1., 0.],
           [0., 0., 1., 0.],
           [1., 1., 0., 0.],
           [0., 0., 0., 0.]])

    To sample a weighted Erdos Renyi (n, p) graph with Uniform(0, 1) distribution:

    >>> wt = np.random.uniform
    >>> wtargs = dict(low=0, high=1)
    >>> er_np(n, p, wt=wt, wtargs=wtargs)
    array([[0.        , 0.        , 0.95788953, 0.53316528],
           [0.        , 0.        , 0.        , 0.        ],
           [0.95788953, 0.        , 0.        , 0.31551563],
           [0.53316528, 0.        , 0.31551563, 0.        ]])
    """
    if isinstance(dc, (list, np.ndarray)) and all(callable(f) for f in dc):
        raise TypeError("dc is not of type function or array-like of scalars")
    if not np.issubdtype(type(n), np.integer):
        raise TypeError("n is not of type int.")
    if not np.issubdtype(type(p), np.floating):
        raise TypeError("p is not of type float.")
    if type(loops) is not bool:
        raise TypeError("loops is not of type bool.")
    if type(directed) is not bool:
        raise TypeError("directed is not of type bool.")
    n_sbm = np.array([n])
    p_sbm = np.array([[p]])
    g = sbm(n_sbm, p_sbm, directed, loops, wt, wtargs, dc, dc_kws)
    return g


def er_nm(n, m, directed=False, loops=False, wt=1, wtargs=None):
    r"""
    Samples an Erdos Renyi (n, m) graph with specified number of edges.

    Erdos Renyi (n, m) graph is a simple graph with n vertices and exactly m
    number of total edges.

    Read more in the `Erdos-Renyi (ER) Model Tutorial
    <https://microsoft.github.io/graspologic/tutorials/simulations/erdos_renyi.html>`_

    Parameters
    ----------
    n: int
        Number of vertices

    m: int
        Number of edges, a value between 1 and :math:`n^2`.

    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.

    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.

    wt: object, optional (default=1)
        Weight function for each of the edges, taking only a size argument.
        This weight function will be randomly assigned for selected edges.
        If 1, graph produced is binary.

    wtargs: dictionary, optional (default=None)
        Optional arguments for parameters that can be passed
        to weight function ``wt``.

    Returns
    -------
    A: ndarray, shape (n, n)
        Sampled adjacency matrix

    Examples
    --------
    >>> np.random.seed(1)
    >>> n = 4
    >>> m = 4

    To sample a binary Erdos Renyi (n, m) graph:

    >>> er_nm(n, m)
    array([[0., 1., 1., 1.],
           [1., 0., 0., 1.],
           [1., 0., 0., 0.],
           [1., 1., 0., 0.]])

    To sample a weighted Erdos Renyi (n, m) graph with Uniform(0, 1) distribution:

    >>> wt = np.random.uniform
    >>> wtargs = dict(low=0, high=1)
    >>> er_nm(n, m, wt=wt, wtargs=wtargs)
    array([[0.        , 0.66974604, 0.        , 0.38791074],
           [0.66974604, 0.        , 0.        , 0.39658073],
           [0.        , 0.        , 0.        , 0.93553907],
           [0.38791074, 0.39658073, 0.93553907, 0.        ]])
    """
    if not np.issubdtype(type(m), np.integer):
        raise TypeError("m is not of type int.")
    elif m <= 0:
        msg = "m must be > 0."
        raise ValueError(msg)
    if not np.issubdtype(type(n), np.integer):
        raise TypeError("n is not of type int.")
    elif n <= 0:
        msg = "n must be > 0."
        raise ValueError(msg)
    if type(directed) is not bool:
        raise TypeError("directed is not of type bool.")
    if type(loops) is not bool:
        raise TypeError("loops is not of type bool.")

    # check weight function
    if not np.issubdtype(type(wt), np.integer):
        if not callable(wt):
            raise TypeError("You have not passed a function for wt.")

    # compute max number of edges to sample
    if loops:
        if directed:
            max_edges = n ** 2
            msg = "n^2"
        else:
            max_edges = n * (n + 1) // 2
            msg = "n(n+1)/2"
    else:
        if directed:
            max_edges = n * (n - 1)
            msg = "n(n-1)"
        else:
            max_edges = n * (n - 1) // 2
            msg = "n(n-1)/2"
    if m > max_edges:
        msg = "You have passed a number of edges, {}, exceeding {}, {}."
        msg = msg.format(m, msg, max_edges)
        raise ValueError(msg)

    A = np.zeros((n, n))
    # check if directedness is desired
    if directed:
        if loops:
            # use all of the indices
            idx = np.where(np.logical_not(A))
        else:
            # use only the off-diagonal indices
            idx = np.where(~np.eye(n, dtype=bool))
    else:
        # use upper-triangle indices, and ignore diagonal according
        # to loops argument
        idx = np.triu_indices(n, k=int(loops is False))

    # get idx in 1d coordinates by ravelling
    triu = np.ravel_multi_index(idx, A.shape)
    # choose M of them
    triu = np.random.choice(triu, size=m, replace=False)
    # unravel back
    triu = np.unravel_index(triu, A.shape)
    # check weight function
    if not np.issubdtype(type(wt), np.number):
        wt = wt(size=m, **wtargs)
    A[triu] = wt

    if not directed:
        A = symmetrize(A, method="triu")

    return A


def sbm(
    n,
    p,
    directed=False,
    loops=False,
    wt=1,
    wtargs=None,
    dc=None,
    dc_kws={},
    return_labels=False,
):
    """
    Samples a graph from the stochastic block model (SBM).

    SBM produces a graph with specified communities, in which each community can
    have different sizes and edge probabilities.

    Read more in the `Stochastic Block Model (SBM) Tutorial
    <https://microsoft.github.io/graspologic/tutorials/simulations/sbm.html>`_

    Parameters
    ----------
    n: list of int, shape (n_communities)
        Number of vertices in each community. Communities are assigned n[0], n[1], ...

    p: array-like, shape (n_communities, n_communities)
        Probability of an edge between each of the communities, where ``p[i, j]`` indicates
        the probability of a connection between edges in communities ``[i, j]``.
        ``0 < p[i, j] < 1`` for all ``i, j``.

    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.

    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.

    wt: object or array-like, shape (n_communities, n_communities)
        if ``wt`` is an object, a weight function to use globally over
        the sbm for assigning weights. 1 indicates to produce a binary
        graph. If ``wt`` is an array-like, a weight function for each of
        the edge communities. ``wt[i, j]`` corresponds to the weight function
        between communities i and j. If the entry is a function, should
        accept an argument for size. An entry of ``wt[i, j] = 1`` will produce a
        binary subgraph over the i, j community.

    wtargs: dictionary or array-like, shape (n_communities, n_communities)
        if ``wt`` is an object, ``wtargs`` corresponds to the trailing arguments
        to pass to the weight function. If Wt is an array-like, ``wtargs[i, j]``
        corresponds to trailing arguments to pass to ``wt[i, j]``.

    dc: function or array-like, shape (n_vertices) or (n_communities), optional
        ``dc`` is used to generate a degree-corrected stochastic block model [1] in
        which each node in the graph has a parameter to specify its expected degree
        relative to other nodes within its community.

        - function:
            should generate a non-negative number to be used as a degree correction to
            create a heterogenous degree distribution. A weight will be generated for
            each vertex, normalized so that the sum of weights in each block is 1.
        - array-like of functions, shape (n_communities):
            Each function will generate the degree distribution for its respective
            community.
        - array-like of scalars, shape (n_vertices):
            The weights in each block should sum to 1; otherwise, they will be
            normalized and a warning will be thrown. The scalar associated with each
            vertex is the node's relative expected degree within its community.

    dc_kws: dictionary or array-like, shape (n_communities), optional
        Ignored if ``dc`` is none or array of scalar.
        If ``dc`` is a function, ``dc_kws`` corresponds to its named arguments.
        If ``dc`` is an array-like of functions, ``dc_kws`` should be an array-like, shape
        (n_communities), of dictionary. Each dictionary is the named arguments
        for the corresponding function for that community.
        If not specified, in either case all functions will assume their default
        parameters.

    return_labels: boolean, optional (default=False)
        If False, only output is adjacency matrix. Otherwise, an additional output will
        be an array with length equal to the number of vertices in the graph, where each
        entry in the array labels which block a vertex in the graph is in.

    References
    ----------
    .. [1] Tai Qin and Karl Rohe. "Regularized spectral clustering under the
        Degree-Corrected Stochastic Blockmodel," Advances in Neural Information
        Processing Systems 26, 2013

    Returns
    -------
    A: ndarray, shape (sum(n), sum(n))
        Sampled adjacency matrix
    labels: ndarray, shape (sum(n))
        Label vector

    Examples
    --------
    >>> np.random.seed(1)
    >>> n = [3, 3]
    >>> p = [[0.5, 0.1], [0.1, 0.5]]

    To sample a binary 2-block SBM graph:

    >>> sbm(n, p)
    array([[0., 0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0., 1.],
           [1., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 1., 0., 0.],
           [0., 1., 0., 0., 0., 0.]])

    To sample a weighted 2-block SBM graph with Poisson(2) distribution:

    >>> wt = np.random.poisson
    >>> wtargs = dict(lam=2)
    >>> sbm(n, p, wt=wt, wtargs=wtargs)
    array([[0., 4., 0., 1., 0., 0.],
           [4., 0., 0., 0., 0., 2.],
           [0., 0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 2., 0., 0., 0., 0.]])
    """
    # Check n
    if not isinstance(n, (list, np.ndarray)):
        msg = "n must be a list or np.array, not {}.".format(type(n))
        raise TypeError(msg)
    else:
        n = np.array(n)
        if not np.issubdtype(n.dtype, np.integer):
            msg = "There are non-integer elements in n"
            raise ValueError(msg)

    # Check p
    if not isinstance(p, (list, np.ndarray)):
        msg = "p must be a list or np.array, not {}.".format(type(p))
        raise TypeError(msg)
    else:
        p = np.array(p)
        if not np.issubdtype(p.dtype, np.number):
            msg = "There are non-numeric elements in p"
            raise ValueError(msg)
        elif p.shape != (n.size, n.size):
            msg = "p is must have shape len(n) x len(n), not {}".format(p.shape)
            raise ValueError(msg)
        elif np.any(p < 0) or np.any(p > 1):
            msg = "Values in p must be in between 0 and 1."
            raise ValueError(msg)

    # Check wt and wtargs
    if not np.issubdtype(type(wt), np.number) and not callable(wt):
        if not isinstance(wt, (list, np.ndarray)):
            msg = "wt must be a numeric, list, or np.array, not {}".format(type(wt))
            raise TypeError(msg)
        if not isinstance(wtargs, (list, np.ndarray)):
            msg = "wtargs must be a numeric, list, or np.array, not {}".format(
                type(wtargs)
            )
            raise TypeError(msg)

        wt = np.array(wt, dtype=object)
        wtargs = np.array(wtargs, dtype=object)
        # if not number, check dimensions
        if wt.shape != (n.size, n.size):
            msg = "wt must have size len(n) x len(n), not {}".format(wt.shape)
            raise ValueError(msg)
        if wtargs.shape != (n.size, n.size):
            msg = "wtargs must have size len(n) x len(n), not {}".format(wtargs.shape)
            raise ValueError(msg)
        # check if each element is a function
        for element in wt.ravel():
            if not callable(element):
                msg = "{} is not a callable function.".format(element)
                raise TypeError(msg)
    else:
        wt = np.full(p.shape, wt, dtype=object)
        wtargs = np.full(p.shape, wtargs, dtype=object)

    # Check directed
    if not directed:
        if np.any(p != p.T):
            raise ValueError("Specified undirected, but P is directed.")
        if np.any(wt != wt.T):
            raise ValueError("Specified undirected, but Wt is directed.")
        if np.any(wtargs != wtargs.T):
            raise ValueError("Specified undirected, but Wtargs is directed.")

    K = len(n)  # the number of communities
    counter = 0
    # get a list of community indices
    cmties = []
    for i in range(0, K):
        cmties.append(range(counter, counter + n[i]))
        counter += n[i]

    # Check degree-corrected input parameters
    if callable(dc):
        # Check that the paramters are a dict
        if not isinstance(dc_kws, dict):
            msg = "dc_kws must be of type dict not{}".format(type(dc_kws))
            raise TypeError(msg)
        # Create the probability matrix for each vertex
        dcProbs = np.array([dc(**dc_kws) for _ in range(0, sum(n))], dtype="float")
        for indices in cmties:
            dcProbs[indices] /= sum(dcProbs[indices])
    elif isinstance(dc, (list, np.ndarray)) and np.issubdtype(
        np.array(dc).dtype, np.number
    ):
        dcProbs = np.array(dc, dtype=float)
        # Check size and element types
        if not np.issubdtype(dcProbs.dtype, np.number):
            msg = "There are non-numeric elements in dc, {}".format(dcProbs.dtype)
            raise ValueError(msg)
        elif dcProbs.shape != (sum(n),):
            msg = "dc must have size equal to the number of"
            msg += " vertices {0}, not {1}".format(sum(n), dcProbs.shape)
            raise ValueError(msg)
        elif np.any(dcProbs < 0):
            msg = "Values in dc cannot be negative."
            raise ValueError(msg)
        # Check that probabilities sum to 1 in each block
        for i in range(0, K):
            if not np.isclose(sum(dcProbs[cmties[i]]), 1, atol=1.0e-8):
                msg = "Block {} probabilities should sum to 1, normalizing...".format(i)
                warnings.warn(msg, UserWarning)
                dcProbs[cmties[i]] /= sum(dcProbs[cmties[i]])
    elif isinstance(dc, (list, np.ndarray)) and all(callable(f) for f in dc):
        dcFuncs = np.array(dc)
        if dcFuncs.shape != (len(n),):
            msg = "dc must have size equal to the number of blocks {0}, not {1}".format(
                len(n), dcFuncs.shape
            )
            raise ValueError(msg)
        # Check that the parameters type, length, and type
        if not isinstance(dc_kws, (list, np.ndarray)):
            # Allows for nonspecification of default parameters for all functions
            if dc_kws == {}:
                dc_kws = [{} for _ in range(0, len(n))]
            else:
                msg = "dc_kws must be of type list or np.ndarray, not {}".format(
                    type(dc_kws)
                )
                raise TypeError(msg)
        elif not len(dc_kws) == len(n):
            msg = "dc_kws must have size equal to"
            msg += " the number of blocks {0}, not {1}".format(len(n), len(dc_kws))
            raise ValueError(msg)
        elif not all(type(kw) == dict for kw in dc_kws):
            msg = "dc_kws elements must all be of type dict"
            raise TypeError(msg)
        # Create the probability matrix for each vertex
        dcProbs = np.array(
            [
                dcFunc(**kws)
                for dcFunc, kws, size in zip(dcFuncs, dc_kws, n)
                for _ in range(0, size)
            ],
            dtype="float",
        )
        # dcProbs = dcProbs.astype(float)
        for indices in cmties:
            dcProbs[indices] /= sum(dcProbs[indices])
            # dcProbs[indices] = dcProbs / dcProbs[indices].sum()
    elif dc is not None:
        msg = "dc must be a function or a list or np.array of numbers or callable"
        msg += " functions, not {}".format(type(dc))
        raise ValueError(msg)

    # End Checks, begin simulation
    A = np.zeros((sum(n), sum(n)))

    for i in range(0, K):
        if directed:
            jrange = range(0, K)
        else:
            jrange = range(i, K)
        for j in jrange:
            block_wt = wt[i, j]
            block_wtargs = wtargs[i, j]
            block_p = p[i, j]
            # identify submatrix for community i, j
            # cartesian product to identify edges for community i,j pair
            cprod = cartesian_product(cmties[i], cmties[j])
            # get idx in 1d coordinates by ravelling
            triu = np.ravel_multi_index((cprod[:, 0], cprod[:, 1]), A.shape)
            pchoice = np.random.uniform(size=len(triu))
            if dc is not None:
                # (v1,v2) connected with probability p*k_i*k_j*dcP[v1]*dcP[v2]
                num_edges = sum(pchoice < block_p)
                edge_dist = dcProbs[cprod[:, 0]] * dcProbs[cprod[:, 1]]
                # If n_edges greater than support of dc distribution, pick fewer edges
                if num_edges > sum(edge_dist > 0):
                    msg = "More edges sampled than nonzero pairwise dc entries."
                    msg += " Picking fewer edges"
                    warnings.warn(msg, UserWarning)
                    num_edges = sum(edge_dist > 0)
                triu = np.random.choice(
                    triu, size=num_edges, replace=False, p=edge_dist
                )
            else:
                # connected with probability p
                triu = triu[pchoice < block_p]
            if type(block_wt) is not int:
                block_wt = block_wt(size=len(triu), **block_wtargs)
            triu = np.unravel_index(triu, A.shape)
            A[triu] = block_wt

    if not loops:
        A = A - np.diag(np.diag(A))
    if not directed:
        A = symmetrize(A, method="triu")
    if return_labels:
        labels = _n_to_labels(n)
        return A, labels
    return A


def rdpg(X, Y=None, rescale=False, directed=False, loops=False, wt=1, wtargs=None):
    r"""
    Samples a random graph based on the latent positions in X (and
    optionally in Y)

    If only X :math:`\in\mathbb{R}^{n\times d}` is given, the P matrix is calculated as
    :math:`P = XX^T`. If X, Y :math:`\in\mathbb{R}^{n\times d}` is given, then
    :math:`P = XY^T`. These operations correspond to the dot products between a set of
    latent positions, so each row in X or Y represents the latent positions in
    :math:`\mathbb{R}^{d}` for a single vertex in the random graph
    Note that this function may also rescale or clip the resulting P
    matrix to get probabilities between 0 and 1, or remove loops.
    A binary random graph is then sampled from the P matrix described
    by X (and possibly Y).

    Read more in the `Random Dot Product Graph (RDPG) Model Tutorial
    <https://microsoft.github.io/graspologic/tutorials/simulations/rdpg.html>`_

    Parameters
    ----------
    X: np.ndarray, shape (n_vertices, n_dimensions)
        latent position from which to generate a P matrix
        if Y is given, interpreted as the left latent position

    Y: np.ndarray, shape (n_vertices, n_dimensions) or None, optional
        right latent position from which to generate a P matrix

    rescale: boolean, optional (default=False)
        when ``rescale`` is True, will subtract the minimum value in
        P (if it is below 0) and divide by the maximum (if it is
        above 1) to ensure that P has entries between 0 and 1. If
        False, elements of P outside of [0, 1] will be clipped

    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.

    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Diagonal elements in P
        matrix are removed prior to rescaling (see above) which may affect behavior.
        Otherwise, edges are sampled in the diagonal.

    wt: object, optional (default=1)
        Weight function for each of the edges, taking only a size argument.
        This weight function will be randomly assigned for selected edges.
        If 1, graph produced is binary.

    wtargs: dictionary, optional (default=None)
        Optional arguments for parameters that can be passed
        to weight function ``wt``.

    Returns
    -------
    A: ndarray (n_vertices, n_vertices)
        A matrix representing the probabilities of connections between
        vertices in a random graph based on their latent positions

    References
    ----------
    .. [1] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
       Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
       Journal of the American Statistical Association, Vol. 107(499), 2012

    Examples
    --------
    >>> np.random.seed(1)

    Generate random latent positions using 2-dimensional Dirichlet distribution.

    >>> X = np.random.dirichlet([1, 1], size=5)

    Sample a binary RDPG using sampled latent positions.

    >>> rdpg(X, loops=False)
    array([[0., 1., 0., 0., 1.],
           [1., 0., 0., 1., 1.],
           [0., 0., 0., 1., 1.],
           [0., 1., 1., 0., 0.],
           [1., 1., 1., 0., 0.]])

    Sample a weighted RDPG with Poisson(2) weight distribution

    >>> wt = np.random.poisson
    >>> wtargs = dict(lam=2)
    >>> rdpg(X, loops=False, wt=wt, wtargs=wtargs)
    array([[0., 4., 0., 2., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 2.],
           [1., 0., 0., 0., 1.],
           [0., 2., 2., 0., 0.]])
    """
    P = p_from_latent(X, Y, rescale=rescale, loops=loops)
    A = sample_edges(P, directed=directed, loops=loops)

    # check weight function
    if (not np.issubdtype(type(wt), np.integer)) and (
        not np.issubdtype(type(wt), np.floating)
    ):
        if not callable(wt):
            raise TypeError("You have not passed a function for wt.")

    if not np.issubdtype(type(wt), np.number):
        wts = wt(size=(np.count_nonzero(A)), **wtargs)
        A[A > 0] = wts
    else:
        A *= wt
    return A


def p_from_latent(X, Y=None, rescale=False, loops=True):
    r"""
    Gemerates a matrix of connection probabilities for a random graph
    based on a set of latent positions

    If only X is given, the P matrix is calculated as :math:`P = XX^T`
    If X and Y is given, then :math:`P = XY^T`
    These operations correspond to the dot products between a set of latent
    positions, so each row in X or Y represents the latent positions in
    :math:`\mathbb{R}^{num-columns}` for a single vertex in the random graph
    Note that this function may also rescale or clip the resulting P
    matrix to get probabilities between 0 and 1, or remove loops

    Parameters
    ----------
    X: np.ndarray, shape (n_vertices, n_dimensions)
        latent position from which to generate a P matrix
        if Y is given, interpreted as the left latent position

    Y: np.ndarray, shape (n_vertices, n_dimensions) or None, optional
        right latent position from which to generate a P matrix

    rescale: boolean, optional (default=False)
        when rescale is True, will subtract the minimum value in
        P (if it is below 0) and divide by the maximum (if it is
        above 1) to ensure that P has entries between 0 and 1. If
        False, elements of P outside of [0, 1] will be clipped

    loops: boolean, optional (default=True)
        whether to allow elements on the diagonal (corresponding
        to self connections in a graph) in the returned P matrix.
        If loops is False, these elements are removed prior to
        rescaling (see above) which may affect behavior

    Returns
    -------
    P: ndarray (n_vertices, n_vertices)
        A matrix representing the probabilities of connections between
        vertices in a random graph based on their latent positions

    References
    ----------
    .. [1] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
       Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
       Journal of the American Statistical Association, Vol. 107(499), 2012

    """
    if Y is None:
        Y = X
    if type(X) is not np.ndarray or type(Y) is not np.ndarray:
        raise TypeError("Latent positions must be numpy.ndarray")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(
            "Latent positions must have dimension 2 (n_vertices, n_dimensions)"
        )
    if X.shape != Y.shape:
        raise ValueError("Dimensions of latent positions X and Y must be the same")
    P = X @ Y.T
    # should this be before or after the rescaling, could give diff answers
    if not loops:
        P = P - np.diag(np.diag(P))
    if rescale:
        if P.min() < 0:
            P = P - P.min()
        if P.max() > 1:
            P = P / P.max()
    else:
        P[P < 0] = 0
        P[P > 1] = 1
    return P


def mmsbm(
    n,
    p,
    alpha=None,
    rng=None,
    directed=False,
    loops=False,
    return_labels=False,
):
    r"""
    Samples a graph from Mixed Membership Stochastic Block Model (MMSBM).

    MMSBM produces a graph given the specified block connectivity matrix B,
    which indicates the probability of connection between nodes based upon
    their community membership.
    Each node is assigned a membership vector drawn from Dirichlet distribution
    with parameter :math:`\vec{\alpha}`. The entries of this vector indicate the
    probabilities for that node of pertaining to each community when interacting with
    another node. Each node's membership is determined according to those probabilities.
    Finally, interactions are sampled according to the assigned memberships.

    Read more in the `Mixed Membership Stochastic Blockmodel (MMSBM) Tutorial
    <https://microsoft.github.io/graspologic/tutorials/simulations/mmsbm.html>`_

    Parameters
    ----------
    n: int
        Number of vertices of the graph.

    p: array-like, shape (n_communities, n_communities)
        Probability of an edge between each of the communities, where ``p[i, j]``
        indicates the probability of a connection between edges in communities
        :math:`(i, j)`.
        0 < ``p[i, j]`` < 1 for all :math:`i, j`.

    alpha: array-like, shape (n_communities,)
        Parameter alpha of the Dirichlet distribution used
        to sample the mixed-membership vectors for each node.
        ``alpha[i]`` > 0 for all :math:`i`.

    rng: numpy.random.Generator, optional (default = None)
        :class:`numpy.random.Generator` object to sample from distributions.
        If None, the random number generator is the Generator object constructed
        by ``np.random.default_rng()``.

    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.

    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.

    return_labels: boolean, optional (default=False)
        If False, the only output is the adjacency matrix.
        If True, output is a tuple. The first element of the tuple is the adjacency
        matrix. The second element is a matrix in which the :math:`(i^{th}, j^{th})`
        entry indicates the membership assigned to node i when interacting with node j.
        Community 1 is labeled with a 0, community 2 with 1, etc.
        -1 indicates that no community was assigned for that interaction.

    References
    ----------
    .. [1] Airoldi, Edoardo, et al. “Mixed Membership Stochastic Blockmodels.”
       Journal of Machine Learning Research, vol. 9, 2008, pp. 1981–2014.

    Returns
    -------
    A: ndarray, shape (n, n)
        Sampled adjacency matrix
    labels: ndarray, shape (n, n), optional
        Array containing the membership assigned to each node when interacting with
        another node.

    Examples
    --------
    >>> rng = np.random.default_rng(1)
    >>> n = 6
    >>> p = [[0.5, 0], [0, 1]]

    To sample a binary MMSBM in which very likely all nodes pertain to community two:

    >>> alpha = [0.05, 1000]
    >>> mmsbm(n, p, alpha, rng = rng)
    array([[0., 1., 1., 1., 1., 1.],
           [1., 0., 1., 1., 1., 1.],
           [1., 1., 0., 1., 1., 1.],
           [1., 1., 1., 0., 1., 1.],
           [1., 1., 1., 1., 0., 1.],
           [1., 1., 1., 1., 1., 0.]])

    To sample a binary MMSBM similar to 2-block SBM with connectivity matrix B:

    >>> rng = np.random.default_rng(1)
    >>> alpha = [0.05, 0.05]
    >>> mmsbm(n, p, alpha, rng = rng)
    array([[0., 1., 1., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1., 1.],
           [0., 0., 0., 1., 0., 1.],
           [0., 0., 0., 1., 1., 0.]])

    """

    check_scalar(x=n, name="n", target_type=int, min_val=1)

    p = check_array(p, ensure_2d=True)
    nx, ny = p.shape
    if nx != ny:
        msg = "p must be a square matrix, not {}".format(p.shape)
        raise ValueError(msg)
    if not np.issubdtype(p.dtype, np.number):
        msg = "There are non-numeric elements in p"
        raise ValueError(msg)
    if np.any(p < 0) or np.any(p > 1):
        msg = "Values in p must be in between 0 and 1."
        raise ValueError(msg)

    alpha = check_array(alpha, ensure_2d=False, ensure_min_features=1)
    if not np.issubdtype(alpha.dtype, np.number):
        msg = "There are non-numeric elements in alpha"
        raise ValueError(msg)
    if np.any(alpha <= 0):
        msg = "Alpha entries must be > 0."
        raise ValueError(msg)
    if alpha.shape != (len(p),):
        msg = "alpha must be a list or np.array of shape {c}, not {w}.".format(
            c=(len(p),), w=alpha.shape
        )
        raise ValueError(msg)

    if not isinstance(rng, (np.random._generator.Generator)):
        msg = "rng must be <class 'numpy.random._generator.Generator'> not {}.".format(
            type(rng)
        )
        raise TypeError(msg)
    elif rng == None:
        rng = np.random.default_rng()

    if type(loops) is not bool:
        raise TypeError("loops is not of type bool.")
    if type(directed) is not bool:
        raise TypeError("directed is not of type bool.")
    if type(return_labels) is not bool:
        raise TypeError("return_labels is not of type bool.")

    if not directed:
        if np.any(p != p.T):
            raise ValueError("Specified undirected, but P is directed.")

    # Naming convention follows paper listed in references.
    mm_vectors = rng.dirichlet(alpha, n)

    mm_vectors = np.array(sorted(mm_vectors, key=lambda x: np.argmax(x)))

    # labels:(n,n) matrix with all membership indicators for initiators and receivers
    # instead of storing the indicator vector, argmax is directly computed
    # check docstrings for more info.
    labels = np.apply_along_axis(
        lambda p_vector: np.argmax(
            rng.multinomial(n=1, pvals=p_vector, size=n), axis=1
        ),
        axis=1,
        arr=mm_vectors,
    )

    P = p[(labels, labels.T)]

    A = sample_edges(P, directed=directed, loops=loops, rng=rng)

    if not loops:
        np.fill_diagonal(labels, -1)

    if return_labels:
        return (A, labels)

    return A
