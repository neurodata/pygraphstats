# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import warnings
from collections import Iterable
from functools import reduce
from pathlib import Path
from typing import List, Union

import networkx as nx
import numpy as np
from sklearn.utils import check_array


def import_graph(graph, copy=True):
    """
    A function for reading a graph and returning a shared data type.

    Parameters
    ----------
    graph: object
        Either array-like, shape (n_vertices, n_vertices) numpy array,
        or an object of type networkx.Graph.

    copy: bool, (default=True)
        Whether to return a copied version of array. If False and input is np.array,
        the output returns the original input.

    Returns
    -------
    out: array-like, shape (n_vertices, n_vertices)
        A graph.

    See Also
    --------
    networkx.Graph, numpy.array
    """
    if isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        out = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes), dtype=np.float)
    elif isinstance(graph, (np.ndarray, np.memmap)):
        shape = graph.shape
        if len(shape) > 3:
            msg = "Input tensor must have at most 3 dimensions, not {}.".format(
                len(shape)
            )
            raise ValueError(msg)
        elif len(shape) == 3:
            if shape[1] != shape[2]:
                msg = "Input tensor must have same number of vertices."
                raise ValueError(msg)
            min_features = shape[1]
            min_samples = 2
        else:
            min_features = np.max(shape)
            min_samples = min_features
        out = check_array(
            graph,
            dtype=[np.float64, np.float32],
            ensure_2d=True,
            allow_nd=True,  # For omni tensor input
            ensure_min_features=min_features,
            ensure_min_samples=min_samples,
            copy=copy,
        )
    else:
        msg = "Input must be networkx.Graph or np.array, not {}.".format(type(graph))
        raise TypeError(msg)
    return out


def import_edgelist(
    path, extension="edgelist", delimiter=None, nodetype=int, return_vertices=False
):
    """
    Function for reading a single or multiple edgelists. When importing multiple
    edgelists, the union of vertices from all graphs is computed so that each output
    graph have matched vertex set. The order of nodes are sorted by node values.

    Parameters
    ----------
    path : str, Path object, or iterable
        If ``path`` is a directory, then the importing order will be sorted in
        alphabetical order.

    extension : str, optional
        If ``path`` is a directory, then the function will convert all files
        with matching extension.

    delimiter : str or None, default=None, optional
        Delimiter of edgelist. If None, the delimiter is whitespace.

    nodetype : int (default), float, str, Python type, optional
       Convert node data from strings to specified type.

    return_vertices : bool, default=False, optional
        Returns the union of all ind

    Returns
    -------
    out : list of array-like, or array-like, shape (n_vertices, n_vertices)
        If ``path`` is a directory, a list of arrays is returned. If ``path`` is a file,
        an array is returned.

    vertices : array-like, shape (n_vertices, )
        If ``return_vertices`` == True, then returns an array of all vertices that were
        included in the output graphs.
    """
    # p = Path(path)
    if not isinstance(path, (str, Path, Iterable)):
        msg = "path must be a string or Iterable, not {}".format(type(path))
        raise TypeError(msg)

    # get a list of files to import
    if isinstance(path, (str, Path)):
        p = Path(path)
        if p.is_dir():
            files = sorted(p.glob("*" + extension))
        elif p.is_file():
            files = [p]
        else:
            raise ValueError("No graphs founds to import.")
    else:  # path is an iterable
        files = [Path(f) for f in path]

    if len(files) == 0:
        msg = "No files found with '{}' extension found.".format(extension)
        raise ValueError(msg)

    graphs = [
        nx.read_weighted_edgelist(f, nodetype=nodetype, delimiter=delimiter)
        for f in files
    ]

    if all(len(G.nodes) == 0 for G in graphs):
        msg = (
            "All graphs have 0 vertices. Please double check if proper "
            + "'delimiter' is given."
        )
        warnings.warn(msg, UserWarning)

    # Compute union of all vertices
    vertices = np.sort(reduce(np.union1d, [G.nodes for G in graphs]))
    out = [nx.to_numpy_array(G, nodelist=vertices, dtype=np.float) for G in graphs]

    # only return adjacency matrix if input is only 1 graph
    if len(out) == 1:
        out = out[0]

    if return_vertices:
        return out, vertices
    else:
        return out


def import_multigraphs(
    graphs: Union[List[nx.Graph], List[np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    A function for reading multiple graph and returning the original input data
    type. Ensures all graphs have the same shape.

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

        # Check if all arrays have same shape
        if np.unique([x.shape for x in out]).size != 1:
            msg = f"All input graphs must have same number of vertices."
            raise ValueError(msg)
    elif isinstance(graphs, np.ndarray):
        if graphs.ndim != 3:
            msg = f"Input tensor must be 3-dimensional, not {graphs.ndim}-dimensional."
            raise ValueError(msg)
        elif graphs.shape[0] <= 1:
            msg = f"Input tensor must have at least 2 elements, not {graphs.shape[0]}."
            raise ValueError(msg)
        out = import_graph(graphs, copy=False)
    else:
        msg = f"Input must be a list or ndarray, not {type(graphs)}."
        raise TypeError(msg)

    return out


def is_symmetric(X):
    return np.array_equal(X, X.T)


def is_loopless(X):
    return not np.any(np.diag(X) != 0)


def is_unweighted(X):
    return ((X == 0) | (X == 1)).all()


def is_almost_symmetric(X, atol=1e-15):
    return np.allclose(X, X.T, atol=atol)


def symmetrize(graph, method="avg"):
    """
    A function for forcing symmetry upon a graph.

    Parameters
    ----------
    graph: object
        Either array-like, (n_vertices, n_vertices) numpy matrix,
        or an object of type networkx.Graph.

    method: {'avg' (default), 'triu', 'tril',}, optional
        An option indicating which half of the edges to
        retain when symmetrizing.

            - 'avg'
                Retain the average weight between the upper and lower
                right triangle, of the adjacency matrix.
            - 'triu'
                Retain the upper right triangle.
            - 'tril'
                Retain the lower left triangle.

    Returns
    -------
    graph: array-like, shape (n_vertices, n_vertices)
        Graph with asymmetries removed.

    Examples
    --------
    >>> a = np.array([
    ...    [0, 1, 1],
    ...    [0, 0, 1],
    ...    [0, 0, 1]])
    >>> symmetrize(a, method="triu")
    array([[0, 1, 1],
           [1, 0, 1],
           [1, 1, 1]])
    """
    # graph = import_graph(graph)
    if method == "triu":
        graph = np.triu(graph)
    elif method == "tril":
        graph = np.tril(graph)
    elif method == "avg":
        graph = (np.triu(graph) + np.tril(graph)) / 2
    else:
        msg = "You have not passed a valid parameter for the method."
        raise ValueError(msg)
    # A = A + A' - diag(A)
    graph = graph + graph.T - np.diag(np.diag(graph))
    return graph


def remove_loops(graph):
    """
    A function to remove loops from a graph.

    Parameters
    ----------
    graph: object
        Either array-like, (n_vertices, n_vertices) numpy matrix,
        or an object of type networkx.Graph.

    Returns
    -------
    graph: array-like, shape(n_vertices, n_vertices)
        the graph with self-loops (edges between the same node) removed.
    """
    graph = import_graph(graph)
    graph = graph - np.diag(np.diag(graph))

    return graph


def to_laplace(graph, form="DAD", regularizer=None):
    r"""
    A function to convert graph adjacency matrix to graph Laplacian.

    Currently supports I-DAD, DAD, and R-DAD Laplacians, where D is the diagonal
    matrix of degrees of each node raised to the -1/2 power, I is the
    identity matrix, and A is the adjacency matrix.

    R-DAD is regularized Laplacian: where :math:`D_t = D + regularizer*I`.

    Parameters
    ----------
    graph: object
        Either array-like, (n_vertices, n_vertices) numpy array,
        or an object of type networkx.Graph.

    form: {'I-DAD' (default), 'DAD', 'R-DAD'}, string, optional

        - 'I-DAD'
            Computes :math:`L = I - D_i*A*D_i`
        - 'DAD'
            Computes :math:`L = D_o*A*D_i`
        - 'R-DAD'
            Computes :math:`L = D_o^r*A*D_i^r`
            where :math:`D_o^r = D_o + regularizer * I` and likewise for :math:`D_i`

    regularizer: int, float or None, optional (default=None)
        Constant to add to the degree vector(s). If None, average node degree is added.
        If int or float, must be >= 0. Only used when ``form`` == 'R-DAD'.

    Returns
    -------
    L : numpy.ndarray
        2D (n_vertices, n_vertices) array representing graph
        Laplacian of specified form

    References
    ----------
    .. [1] Qin, Tai, and Karl Rohe. "Regularized spectral clustering
           under the degree-corrected stochastic blockmodel." In Advances
           in Neural Information Processing Systems, pp. 3120-3128. 2013

    .. [2] Rohe, Karl, Tai Qin, and Bin Yu. "Co-clustering directed graphs to discover
           asymmetries and directional communities." Proceedings of the National Academy
           of Sciences 113.45 (2016): 12679-12684.

    Examples
    --------
    >>> a = np.array([
    ...    [0, 1, 1],
    ...    [1, 0, 0],
    ...    [1, 0, 0]])
    >>> to_laplace(a, "DAD")
    array([[0.        , 0.70710678, 0.70710678],
           [0.70710678, 0.        , 0.        ],
           [0.70710678, 0.        , 0.        ]])

    """
    valid_inputs = ["I-DAD", "DAD", "R-DAD"]
    if form not in valid_inputs:
        raise TypeError("Unsuported Laplacian normalization")

    A = import_graph(graph)

    in_degree = np.sum(A, axis=0)
    out_degree = np.sum(A, axis=1)

    # regularize laplacian with parameter
    # set to average degree
    if form == "R-DAD":
        if regularizer is None:
            regularizer = np.mean(out_degree)
        elif not isinstance(regularizer, (int, float)):
            raise TypeError(
                "Regularizer must be a int or float, not {}".format(type(regularizer))
            )
        elif regularizer < 0:
            raise ValueError("Regularizer must be greater than or equal to 0")

        in_degree += regularizer
        out_degree += regularizer

    with np.errstate(divide="ignore"):
        in_root = 1 / np.sqrt(in_degree)  # this is 10x faster than ** -0.5
        out_root = 1 / np.sqrt(out_degree)

    in_root[np.isinf(in_root)] = 0
    out_root[np.isinf(out_root)] = 0

    in_root = np.diag(in_root)  # just change to sparse diag for sparse support
    out_root = np.diag(out_root)

    if form == "I-DAD":
        L = np.diag(in_degree) - A
        L = in_root @ L @ in_root
    elif form == "DAD" or form == "R-DAD":
        L = out_root @ A @ in_root
    if is_symmetric(A):
        return symmetrize(
            L, method="avg"
        )  # sometimes machine prec. makes this necessary
    return L


def is_fully_connected(graph):
    r"""
    Checks whether the input graph is fully connected in the undirected case
    or weakly connected in the directed case.

    Connected means one can get from any vertex u to vertex v by traversing
    the graph. For a directed graph, weakly connected means that the graph
    is connected after it is converted to an unweighted graph (ignore the
    direction of each edge)

    Parameters
    ----------
    graph: nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
        Input graph in any of the above specified formats. If np.ndarray,
        interpreted as an :math:`n \times n` adjacency matrix

    Returns
    -------
    boolean: True if the entire input graph is connected

    References
    ----------
    http://mathworld.wolfram.com/ConnectedGraph.html
    http://mathworld.wolfram.com/WeaklyConnectedDigraph.html

    Examples
    --------
    >>> a = np.array([
    ...    [0, 1, 0],
    ...    [1, 0, 0],
    ...    [0, 0, 0]])
    >>> is_fully_connected(a)
    False
    """
    if type(graph) is np.ndarray:
        if is_symmetric(graph):
            g_object = nx.Graph()
        else:
            g_object = nx.DiGraph()
        graph = nx.from_numpy_array(graph, create_using=g_object)
    if type(graph) in [nx.Graph, nx.MultiGraph]:
        return nx.is_connected(graph)
    elif type(graph) in [nx.DiGraph, nx.MultiDiGraph]:
        return nx.is_weakly_connected(graph)


def get_lcc(graph, return_inds=False):
    r"""
    Finds the largest connected component for the input graph.

    The largest connected component is the fully connected subgraph
    which has the most nodes.

    Parameters
    ----------
    graph: nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
        Input graph in any of the above specified formats. If np.ndarray,
        interpreted as an :math:`n \times n` adjacency matrix

    return_inds: boolean, default: False
        Whether to return a np.ndarray containing the indices in the original
        adjacency matrix that were kept and are now in the returned graph.
        Ignored when input is networkx object

    Returns
    -------
    graph: nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
        New graph of the largest connected component of the input parameter.

    inds: (optional)
        Indices from the original adjacency matrix that were kept after taking
        the largest connected component.
    """
    input_ndarray = False
    if type(graph) is np.ndarray:
        input_ndarray = True
        if is_symmetric(graph):
            g_object = nx.Graph()
        else:
            g_object = nx.DiGraph()
        graph = nx.from_numpy_array(graph, create_using=g_object)
    if type(graph) in [nx.Graph, nx.MultiGraph]:
        lcc_nodes = max(nx.connected_components(graph), key=len)
    elif type(graph) in [nx.DiGraph, nx.MultiDiGraph]:
        lcc_nodes = max(nx.weakly_connected_components(graph), key=len)
    lcc = graph.subgraph(lcc_nodes).copy()
    lcc.remove_nodes_from([n for n in lcc if n not in lcc_nodes])
    if return_inds:
        nodelist = np.array(list(lcc_nodes))
    if input_ndarray:
        lcc = nx.to_numpy_array(lcc)
    if return_inds:
        return lcc, nodelist
    return lcc


def get_multigraph_union_lcc(graphs, return_inds=False):
    r"""
    Finds the union of all multiple graphs, then compute the largest connected
    component.

    Parameters
    ----------
    graphs: list or np.ndarray
        List of array-like, (n_vertices, n_vertices), or list of np.ndarray
        nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph.

    return_inds: boolean, default: False
        Whether to return a np.ndarray containing the indices in the original
        adjacency matrix that were kept and are now in the returned graph.
        Ignored when input is networkx object

    Returns
    -------
    out : list or np.ndarray
        If input was a list
    """
    if isinstance(graphs, list):
        if not isinstance(graphs[0], np.ndarray):
            raise NotImplementedError

        out = [import_graph(g) for g in graphs]
        if len(set(map(np.shape, out))) != 1:
            msg = "All input graphs must have the same size"
            raise ValueError(msg)
        bar = np.stack(out).mean(axis=0)
    elif isinstance(graphs, np.ndarray):
        shape = graphs.shape
        if shape[1] != shape[2]:
            msg = "Input graphs must be square"
            raise ValueError(msg)
        bar = graphs.mean(axis=0)
    else:
        msg = "Expected list or np.ndarray, but got {} instead.".format(type(graphs))
        raise ValueError(msg)

    _, idx = get_lcc(bar, return_inds=True)
    idx = np.array(idx)

    if isinstance(graphs, np.ndarray):
        graphs[:, idx[:, None], idx]
    elif isinstance(graphs, list):
        if isinstance(graphs[0], np.ndarray):
            graphs = [g[idx[:, None], idx] for g in graphs]
    if return_inds:
        return graphs, idx
    return graphs


def get_multigraph_intersect_lcc(graphs, return_inds=False):
    r"""
    Finds the intersection of multiple graphs's largest connected components.

    Computes the largest connected component for each graph that was input, and
    takes the intersection over all of these resulting graphs. Note that this
    does not guarantee finding the largest graph where every node is shared among
    all of the input graphs.

    Parameters
    ----------
    graphs: list or np.ndarray
        if list, each element must be an :math:`n \times n` np.ndarray adjacency matrix

    return_inds: boolean, default: False
        Whether to return a np.ndarray containing the indices in the original
        adjacency matrix that were kept and are now in the returned graph.
        Ignored when input is networkx object

    Returns
    -------
    graph: nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
        New graph of the largest connected component of the input parameter.

    inds: (optional)
        Indices from the original adjacency matrix that were kept after taking
        the largest connected component
    """
    lcc_by_graph = []
    inds_by_graph = []
    for graph in graphs:
        lcc, inds = get_lcc(graph, return_inds=True)
        lcc_by_graph.append(lcc)
        inds_by_graph.append(inds)
    inds_intersection = reduce(np.intersect1d, inds_by_graph)
    new_graphs = []
    for graph in graphs:
        if type(graph) is np.ndarray:
            lcc = graph[inds_intersection, :][:, inds_intersection]
        else:
            lcc = graph.subgraph(inds_intersection).copy()
            lcc.remove_nodes_from([n for n in lcc if n not in inds_intersection])
        new_graphs.append(lcc)
    # this is not guaranteed be connected after one iteration because taking the
    # intersection of nodes among graphs can cause some components to become
    # disconnected, so, we check for this and run again if necessary
    recurse = False
    for new_graph in new_graphs:
        if not is_fully_connected(new_graph):
            recurse = True
            break
    if recurse:
        new_graphs, new_inds_intersection = get_multigraph_intersect_lcc(
            new_graphs, return_inds=True
        )
        # new inds intersection are the indices of new_graph that were kept on recurse
        # need to do this because indices could have shifted during recursion
        if type(graphs[0]) is np.ndarray:
            inds_intersection = inds_intersection[new_inds_intersection]
        else:
            inds_intersection = new_inds_intersection
    if type(graphs) != list:
        new_graphs = np.stack(new_graphs)
    if return_inds:
        return new_graphs, inds_intersection
    else:
        return new_graphs


def augment_diagonal(graph, weight=1):
    r"""
    Replaces the diagonal of an adjacency matrix with :math:`\frac{d}{nverts - 1}` where
    :math:`d` is the degree vector for an unweighted graph and the sum of magnitude of
    edge weights for each node for a weighted graph. For a directed graph the in/out
    :math:`d` is averaged.

    Parameters
    ----------
    graph: nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
        Input graph in any of the above specified formats. If np.ndarray,
        interpreted as an :math:`n \times n` adjacency matrix
    weight: float/int
        scalar value to multiply the new diagonal vector by

    Returns
    -------
    graph : np.array
        Adjacency matrix with average degrees added to the diagonal.

    Examples
    --------
    >>> a = np.array([
    ...    [0, 1, 1],
    ...    [1, 0, 0],
    ...    [1, 0, 0]])
    >>> augment_diagonal(a)
    array([[1. , 1. , 1. ],
           [1. , 0.5, 0. ],
           [1. , 0. , 0.5]])
    """
    graph = import_graph(graph)
    graph = remove_loops(graph)

    divisor = graph.shape[0] - 1

    in_degrees = np.sum(np.abs(graph), axis=0)
    out_degrees = np.sum(np.abs(graph), axis=1)
    degrees = (in_degrees + out_degrees) / 2

    diag = weight * degrees / divisor
    graph += np.diag(diag)

    return graph


def binarize(graph):
    """
    Binarize the input adjacency matrix.

    Parameters
    ----------
    graph: nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
        Input graph in any of the above specified formats. If np.ndarray,
        interpreted as an :math:`n \times n` adjacency matrix

    Returns
    -------
    graph : np.array
        Adjacency matrix with all nonzero values transformed to one.

    Examples
    --------
    >>> a = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    >>> binarize(a)
    array([[0., 1., 1.],
           [1., 0., 1.],
           [1., 1., 0.]])
    """
    graph = import_graph(graph)
    graph[graph != 0] = 1
    return graph


def cartprod(*arrays):
    """
    Compute the cartesian product of multiple arrays
    """
    N = len(arrays)
    return np.transpose(
        np.meshgrid(*arrays, indexing="ij"), np.roll(np.arange(N + 1), -1)
    ).reshape(-1, N)


def fit_plug_in_variance_estimator(X):
    """
    Takes in ASE of a graph and returns a function that estimates
    the variance-covariance matrix at a given point using the
    plug-in estimator from the RDPG Central Limit Theorem.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        adjacency spectral embedding of a graph

    Returns
    -------
    plug_in_variance_estimtor: functions
        a function that estimates variance (see below)
    """

    n = len(X)

    # precompute the Delta and the middle term matrix part
    delta = 1 / (n) * (X.T @ X)
    delta_inverse = np.linalg.inv(delta)
    middle_term_matrix = np.einsum("bi,bo->bio", X, X)

    def plug_in_variance_estimator(x):
        """
        Takes in a point of a matrix of points in R^d and returns an
        estimated covariance matrix for each of the points

        Parameters:
        -----------
        x: np.ndarray, shape (n, d)
            points to estimate variance at
            if 1-dimensional - reshaped to (1, d)

        Returns:
        -------
        covariances: np.ndarray, shape (n, d, d)
            n estimated variance-covariance matrices of the points provided
        """
        if x.ndim < 2:
            x = x.reshape(1, -1)
        # the following two lines are a properly vectorized version of
        # middle_term = 0
        # for i in range(n):
        #     middle_term += np.multiply.outer((x @ X[i] - (x @ X[i]) ** 2),
        #                                      np.outer(X[i], X[i]))
        # where the matrix part does not involve x and has been computed above
        middle_term_scalar = x @ X.T - (x @ X.T) ** 2
        middle_term = np.tensordot(middle_term_scalar, middle_term_matrix, axes=1)
        covariances = delta_inverse @ (middle_term / n) @ delta_inverse
        return covariances

    return plug_in_variance_estimator
