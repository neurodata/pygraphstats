# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from ..utils import symmetrize, cartprod
import warnings
from scipy.stats import bernoulli


def _n_to_labels(n):
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels


def sample_edges(P, directed=False, loops=False):
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
    if not directed:
        # can cut down on sampling by ~half
        triu_inds = np.triu_indices(P.shape[0])
        samples = np.random.binomial(1, P[triu_inds])
        A = np.zeros_like(P)
        A[triu_inds] = samples
        A = symmetrize(A, method="triu")
    else:
        A = np.random.binomial(1, P)

    if loops:
        return A
    else:
        return A - np.diag(np.diag(A))


def er_np(n, p, directed=False, loops=False, wt=1, wtargs=None, dc=None, dc_kws={}):
    r"""
    Samples a Erdos Renyi (n, p) graph with specified edge probability.

    Erdos Renyi (n, p) graph is a simple graph with n vertices and a probability
    p of edges being connected.

    Read more in the :ref:`tutorials <simulations_tutorials>`

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
        `dc` is used to generate a degree-corrected Erdos Renyi Model in
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
        Ignored if `dc` is none or array of scalar.
        If `dc` is a function, `dc_kws` corresponds to its named arguments.
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

    Read more in the :ref:`tutorials <simulations_tutorials>`

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


def siem(
    n,
    p,
    edge_comm,
    directed=False,
    loops=False,
    wt=None,
    wtargs=None,
    return_labels=False):
    """
    Samples a graph from the structured independent edge model (SIEM) 
    SIEM produces a graph with specified communities, in which each community can
    have different sizes and edge probabilities. 
    Read more in the :ref:`tutorials <simulations_tutorials>`
    Parameters
    ----------
    n: int
        Number of vertices
    p: float or list of floats of length K (k_communities)
        Probability of an edge existing within the corresponding communities.
        If a float, a probability, or a float greater than or equal to zero and less than or equal to 1.
        It is assumed that the probability is constant over all communities within the graph.
        If a list of floats of length K, each entry p[i] should be a float greater than or equal to zero
        and less than or equal to 1, where p[i] indicates the probability of an edge existing in the ith edge
        community.
    edge_comm: array-like shape (n, n)
        a square 2d numpy array or square numpy matrix of the edge community each edge is assigned to.
        All edges should be assigned a single community, taking values in the integers 1:K
        where K is the total number of unique communities. Note that edge_comm is expected to respect succeeding
        options passed in; particularly, directedness and loopiness. If loops is False, the entire diagonal of
        edge_comm should be 0.
    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.
    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.
    wt: object or list of K objects
        if Wt is an object, a weight function to use globally over
        the siem for assigning weights. If Wt is a list, a weight function for each of
        the edge communities to use for connection strengths Wt[i] corresponds to the weight function
        for edge community i. Default of None results in a binary graph
    wtargs: dictionary or array-like, shape
        if Wt is an object, Wtargs corresponds to the trailing arguments
        to pass to the weight function. If Wt is an array-like, Wtargs[i, j] 
        corresponds to trailing arguments to pass to Wt[i, j].
    return_labels: boolean, optional (default=True)
        whether to return the community labels of each edge.
    Returns
    -------
    A: ndarray, shape (n, n)
        Sampled adjacency matrix
    labels: ndarray, shape (n, n)
        Square numpy array of labels for each of the edges. Returned if return_labels is True.
    """
    # check booleans
    if not isinstance(loops, bool):
        raise TypeError("`loops` should be a boolean. You passed %s.".format(type(loops)))
    if not isinstance(directed, bool):
        raise TypeError("`directed` should be a boolean. You passed %s.".format(type(directed)))
    # Check n
    if not isinstance(n, (int)):
        msg = "n must be a int, not {}.".format(type(n))
        raise TypeError(msg)
    # Check edge_comm
    if not isinstance(edge_comm, np.ndarray):
        msg = "edge_comm must be a square numpy array or matrix."
        raise TypeError(msg)
    try:
        if np.any(edge_comm != edge_comm.astype(int)):
            msg = "edge_comm must contain only natural numbers. Contains non-integers."
            raise ValueError(msg)
    except ValueError as err:
        err.message = "edge_comm must contain only natural numbers. Contains non-numerics."
        raise
    edge_comm = edge_comm.astype(int)
    K = edge_comm.max()  # number of communities
    if loops:
        if edge_comm.min() != 1:
            msg = "`edge_comm` should all be numbered sequentially from 1:K. The minimum is not 1."
            raise ValueError(msg)
        if len(np.unique(edge_comm)) != K:
            msg = "`edge_comm` should be numbered sequentially from 1:K. The sequence is not consecutive."
            raise ValueError(msg)
    elif not loops:
        if (edge_comm[~np.eye(edge_comm.shape[0], dtype=bool)].min() != 1):
            msg = """Since your graph has no loops, all off-diagonal elements of`edge_comm`
            should have a minimum of 1. The minimum is not 1."""
            raise ValueError(msg)
        if np.any(np.diagonal(edge_comm) != 0):
            msg = """You requested a loopless graph, but assigned a diagonal element to a 
            non-zero community. All diagonal elements of `edge_comm` should be zero if
            `loops` is False."""
            raise ValueError(msg)
        if len(np.unique(edge_comm)) != K + 1:
            msg = """`edge_comm` should be numbered sequentially from 1:K for off-diagonals,
            and 0s on the diagonal. The sequence is not consecutive."""
            raise ValueError(msg)
        
    n = edge_comm.shape[0]
    if (edge_comm.shape[0] != edge_comm.shape[1]):
        msg = "`edge_comm` should be square. `edge_comm` has dimensions [%d, %d]"
        raise ValueError(msg.format(edge_comm.shape[0], edge_comm.shape[1]))
    if (len(edge_comm.shape) != 2):
        msg = "`edge_comm` should be a 2d array or a matrix, but `edge_comm` has %d dimensions."
        raise ValueError(msg.format(len(edge_comm.shape)))
    if (not directed) and np.any(edge_comm != edge_comm.T):
        msg = "You requested an undirected SIEM, but `edge_comm` is directed."
    
    # Check p
    if isinstance(p, float) or isinstance(p, int):
        p = p*np.ones(K)
    if not isinstance(p, (list, np.ndarray)):
        msg = "p must be a list or np.array, not {}.".format(type(p))
        raise TypeError(msg)
    else:
        p = np.array(p)
        if len(p.shape) > 1:
            raise ValueError("p should be a float or a vector/list of length K.")
        if not np.issubdtype(p.dtype, np.number):
            msg = "There are non-numeric elements in p."
            raise ValueError(msg)
        elif np.any(p < 0) or np.any(p > 1):
            msg = "Values in p must be in between 0 and 1."
            raise ValueError(msg)
        elif len(p) != K:
            msg = "# of Probabilities in `p` and # of Communities in `edge_comm` Don't match up."
            raise ValueError(msg)
    # Check wt and wtargs
    if (wt is not None) and (wtargs is None):
        raise TypeError("wtargs should be a dictionary or a list of dictionaries. It is of type None.")
    if (wt is not None) and (wtargs is not None): 
        if callable(wt):
            #extend the function to size of K
            wt = np.full(K, wt, dtype=object)
            if isinstance(wtargs, dict):
                wtargs = np.full(K, wtargs, dtype=object)
            else:        
                for wtarg in wtargs:
                    if not isinstance(wtarg, dict):
                        raise TypeError("wtarg should be a dictionary or a list of dictionaries.")
        elif isinstance(wt, list):
            if all(callable(x) for x in wt): 
                # if not object, check dimensions
                if not isinstance(wtargs, list):
                    raise TypeError("Since wt is a list, wtargs should be a list of dictionaries.")
                if len(wt) != K:
                    msg = "wt must have size K, not {}".format(len(wt))
                    raise ValueError(msg)
                if len(wtargs) != K:
                    msg = "wtargs must have size K, not {}".format(len(wtargs))
                    raise ValueError(msg)
            else: 
                msg = "wt must contain all callable objects."
                raise TypeError(msg)
        else:
            msg = "wt must be a callable object or list of callable objects"
            raise TypeError(msg)


    # End Checks, begin simulation
    A = np.zeros((n,n))
    for i in range(1, K+1):
        edge_comm_i = (edge_comm == i)
        A[np.where(edge_comm_i)] = bernoulli.rvs(p[i-1], size=edge_comm_i.sum())

        if (wt is not None):
            for k, l in zip(*np.where(edge_comm_i)):
                A[k,l] = A[k,l]*wt[i-1](**wtargs[i-1])
    # if not directed, just look at upper triangle and duplicate
    if not directed:
        A = symmetrize(A, method="triu")
    if (return_labels):
        return (A, edge_comm)
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

    Read more in the :ref:`tutorials <simulations_tutorials>`

    Parameters
    ----------
    n: list of int, shape (n_communities)
        Number of vertices in each community. Communities are assigned n[0], n[1], ...

    p: array-like, shape (n_communities, n_communities)
        Probability of an edge between each of the communities, where p[i, j] indicates 
        the probability of a connection between edges in communities [i, j]. 
        0 < p[i, j] < 1 for all i, j.

    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.

    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.

    wt: object or array-like, shape (n_communities, n_communities)
        if Wt is an object, a weight function to use globally over
        the sbm for assigning weights. 1 indicates to produce a binary
        graph. If Wt is an array-like, a weight function for each of
        the edge communities. Wt[i, j] corresponds to the weight function
        between communities i and j. If the entry is a function, should
        accept an argument for size. An entry of Wt[i, j] = 1 will produce a
        binary subgraph over the i, j community.

    wtargs: dictionary or array-like, shape (n_communities, n_communities)
        if Wt is an object, Wtargs corresponds to the trailing arguments
        to pass to the weight function. If Wt is an array-like, Wtargs[i, j] 
        corresponds to trailing arguments to pass to Wt[i, j].

    dc: function or array-like, shape (n_vertices) or (n_communities), optional
        `dc` is used to generate a degree-corrected stochastic block model [1] in
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
        Ignored if `dc` is none or array of scalar.
        If `dc` is a function, `dc_kws` corresponds to its named arguments.
        If `dc` is an array-like of functions, `dc_kws` should be an array-like, shape
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
            cprod = cartprod(cmties[i], cmties[j])
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

    Read more in the :ref:`tutorials <simulations_tutorials>`

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
