# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import warnings
from collections import Iterable
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, diags, isspmatrix_csr
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_array, check_consistent_length, column_or_1d
from sklearn.utils.multiclass import type_of_target, unique_labels


def import_graph(graph, copy=True):
    """
    A function for reading a graph and returning a shared data type.

    Parameters
    ----------
    graph: object
        Either array-like, shape (n_vertices, n_vertices) numpy array,
        a scipy.sparse.csr_matrix, or an object of type networkx.Graph.

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
    elif isinstance(graph, (np.ndarray, np.memmap, csr_matrix)):
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
            accept_sparse=True,
            ensure_2d=True,
            allow_nd=True,  # For omni tensor input
            ensure_min_features=min_features,
            ensure_min_samples=min_samples,
            copy=copy,
        )
    else:
        msg = "Input must be networkx.Graph, np.array, or scipy.sparse.csr_matrix,\
        not {}.".format(
            type(graph)
        )
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
        Returns the union of all individual edgelists.

    Returns
    -------
    out : list of array-like, or array-like, shape (n_vertices, n_vertices)
        If ``path`` is a directory, a list of arrays is returned. If ``path`` is a file,
        an array is returned.

    vertices : array-like, shape (n_vertices, )
        If ``return_vertices``` is True, then returns an array of all vertices that were
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


def is_symmetric(X):
    return abs(X - X.T).sum() == 0


def is_loopless(X):
    return not np.any(np.diag(X) != 0)


def is_unweighted(
    graph: Union[
        np.ndarray,
        scipy.sparse.csr_matrix,
        nx.Graph,
        nx.DiGraph,
        nx.MultiGraph,
        nx.MultiDiGraph,
    ],
    weight_attribute: Any = "weight",
):
    """
    Attempts to determine if the provided graph is weighted.

    Parameters
    ----------
    graph : Union[np.ndarray, scipy.sparse.csr_matrix, nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDigraph]
        The graph to test for weightedness. If a networkx graph, we can just ask it directly by querying the weight
        attribute specified on every edge. It's possible an individual edge can be weighted but the full graph is not.
        If an adjacency matrix defined by a numpy.ndarray or scipy.sparse.csr_matrix, we check every value; if
        they are only 0 and 1, we claim the graph is unweighted.
    weight_attribute : Any
        Default is ``weight``. Only used for networkx, and used on the edge data dictionary as a key to look up the
        weight.

    Returns
    -------
    bool
        True if unweighted, False if weighted

    Raises
    ------
    TypeError
        If the provided graph is not a numpy.ndarray, scipy.sparse.csr_matrix, or nx.Graph
    """
    if isinstance(graph, np.ndarray):
        return ((graph == 0) | (graph == 1)).all()
    elif isinstance(graph, csr_matrix):
        # brute force.  if anyone has a better way, please PR
        rows, columns = graph.nonzero()
        for i in range(0, len(rows)):
            if graph[rows[i], columns[i]] != 1 or graph[rows[i], columns[i]] != 0:
                return False
        return True
    elif isinstance(graph, nx.Graph):
        return nx.is_weighted(graph, weight=weight_attribute)
    else:
        raise TypeError(
            "This function only works on numpy.ndarray or scipy.sparse.csr_matrix instances"
        )


def is_almost_symmetric(X, atol=1e-15):
    return abs(X - X.T).max() <= atol


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
    sparse = isspmatrix_csr(graph)
    pac = scipy.sparse if sparse else np

    if method == "triu":
        graph = pac.triu(graph)
    elif method == "tril":
        graph = pac.tril(graph)
    elif method == "avg":
        graph = (pac.triu(graph) + pac.tril(graph)) / 2
    else:
        msg = "You have not passed a valid parameter for the method."
        raise ValueError(msg)

    dia = diags(graph.diagonal()) if sparse else np.diag(np.diag(graph))
    graph = graph + graph.T - dia
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

    dia = diags(graph.diagonal()) if isspmatrix_csr(graph) else np.diag(np.diag(graph))

    graph = graph - dia

    return graph


def to_laplacian(graph, form="DAD", regularizer=None):
    r"""
    A function to convert graph adjacency matrix to graph Laplacian.

    Currently supports I-DAD, DAD, and R-DAD Laplacians, where D is the diagonal
    matrix of degrees of each node raised to the -1/2 power, I is the
    identity matrix, and A is the adjacency matrix.

    R-DAD is regularized Laplacian: where :math:`D_t = D + regularizer \times I`.

    Parameters
    ----------
    graph: object
        Either array-like, (n_vertices, n_vertices) numpy array,
        scipy.sparse.csr_matrix, or an object of type networkx.Graph.

    form: {'I-DAD' (default), 'DAD', 'R-DAD'}, string, optional

        - 'I-DAD'
            Computes :math:`L = I - D_i A D_i`
        - 'DAD'
            Computes :math:`L = D_o A D_i`
        - 'R-DAD'
            Computes :math:`L = D_o^r A D_i^r`
            where :math:`D_o^r = D_o + regularizer \times I` and likewise for :math:`D_i`

    regularizer: int, float or None, optional (default=None)
        Constant to add to the degree vector(s). If None, average node degree is added.
        If int or float, must be >= 0. Only used when ``form`` is 'R-DAD'.

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
    >>> to_laplacian(a, "DAD")
    array([[0.        , 0.70710678, 0.70710678],
           [0.70710678, 0.        , 0.        ],
           [0.70710678, 0.        , 0.        ]])

    """

    valid_inputs = ["I-DAD", "DAD", "R-DAD"]
    if form not in valid_inputs:
        raise TypeError("Unsuported Laplacian normalization")

    A = import_graph(graph)

    in_degree = np.reshape(np.asarray(A.sum(axis=0)), (-1,))
    out_degree = np.reshape(np.asarray(A.sum(axis=1)), (-1,))

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

    diag = diags if isspmatrix_csr(graph) else np.diag

    in_root[np.isinf(in_root)] = 0
    out_root[np.isinf(out_root)] = 0

    in_root = diag(in_root)  # just change to sparse diag for sparse support
    out_root = diag(out_root)

    if form == "I-DAD":
        L = diag(in_degree) - A
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

    Connected means one can get from any vertex :math:`u` to vertex :math:`v` by traversing
    the graph. For a directed graph, weakly connected means that the graph
    is connected after it is converted to an unweighted graph (ignore the
    direction of each edge)

    Parameters
    ----------
    graph: nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph,
        scipy.sparse.csr_matrix, np.ndarray
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

    if isinstance(graph, (np.ndarray, csr_matrix)):
        directed = not is_symmetric(graph)

        n_components = connected_components(
            csgraph=graph, directed=directed, connection="weak", return_labels=False
        )
        return n_components == 1

    else:
        if type(graph) in [nx.Graph, nx.MultiGraph]:
            return nx.is_connected(graph)
        elif type(graph) in [nx.DiGraph, nx.MultiDiGraph]:
            return nx.is_weakly_connected(graph)


def largest_connected_component(graph, return_inds=False):
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


def multigraph_lcc_union(graphs, return_inds=False):
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

    _, idx = largest_connected_component(bar, return_inds=True)
    idx = np.array(idx)

    if isinstance(graphs, np.ndarray):
        graphs[:, idx[:, None], idx]
    elif isinstance(graphs, list):
        if isinstance(graphs[0], np.ndarray):
            graphs = [g[idx[:, None], idx] for g in graphs]
    if return_inds:
        return graphs, idx
    return graphs


def multigraph_lcc_intersection(graphs, return_inds=False):
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
        lcc, inds = largest_connected_component(graph, return_inds=True)
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
        new_graphs, new_inds_intersection = multigraph_lcc_intersection(
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
    graph: nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray,
        scipy.scr_matrix.
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

    in_degrees = np.squeeze(np.asarray(abs(graph).sum(axis=0)))
    out_degrees = np.squeeze(np.asarray(abs(graph).sum(axis=1)))

    degrees = (in_degrees + out_degrees) / 2
    diag = weight * degrees / divisor

    graph += diags(diag) if isspmatrix_csr(graph) else np.diag(diag)

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


def cartesian_product(*arrays):
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


def remove_vertices(graph, indices, return_removed=False):
    """
    Remove a subgraph of adjacency vectors from an adjacency matrix, giving back the
    truncated matrix and optionally the removed subgraph. Here, an adjacency vector
    is the set of edge weights for a particular vertex.

    Parameters
    ----------
    graph: networkx.Graph or array-like, shape (n, n)
        The adjacency matrix for some graph.
    indices: int or array-like, length m
        Index/indices of the adjacency vector(s) to be removed.
    return_removed: bool, by default False (optional)
        Whether to return the tuple ``(A, V)``,
        where ``A`` is the truncated adjacency matrix,
        ``V`` is an array representing the removed subgraph.

    Returns
    -------
    truncated_graph: np.ndarray
        The truncated matrix.
        This is a copy of `graph` of shape (k, k), with ``k=n-m``, without the ``m``
        adjacency vectors given by `indices`.

    removed_subgraph: np.ndarray or tuple, shape (m, k) (optional)
        Array of removed adjacency vectors without edges to each other.
        If directed, this is a tuple ``(V_1, V_2)``,
        with ``V_1`` being an array of adjacency vectors from the removed subgraph to the truncated graph,
        and ``V_2`` being an array of adjacency vectors from the truncated graph to the removed subgraph.

    Examples
    --------
    # Undirected
    >>> A = np.array([[0, 1, 2],
                      [1, 0, 3],
                      [2, 3, 0]])
    >>> remove_vertices(A, 0)
    array([[0., 3.],
           [3., 0.]]))
    >>> remove_vertices(A, 0, return_removed=True)
    (array([[0., 3.],
            [3., 0.]]),
     array([1., 2.]))

    # Directed
    >>> B = np.array([[0, 1, 2, 3],
                      [4, 0, 5, 6],
                      [7, 8, 0, 9],
                      [10, 11, 12, 0]])
    >>> remove_vertices(B, 0, return_removed=True)
    (array([[ 0.,  5.,  6.],
            [ 8.,  0.,  9.],
            [11., 12.,  0.]]),
    (array([ 4.,  7., 10.]), array([1., 2., 3.])))
    >>> remove_vertices(B, [0, -1], return_removed=True)
    (array([[0., 5.],
            [8., 0.]]),
    (array([[4., 7.],
            [6., 9.]]),
    array([[ 1.,  2.],
            [11., 12.]])))
    """
    graph = import_graph(graph)
    if isinstance(indices, list) and len(indices) >= len(graph):
        raise IndexError("You must pass in fewer vertex indices than vertices.")
    directed = not is_almost_symmetric(graph)

    # truncate graph
    mask = np.ones(graph.shape[0], dtype=bool)
    mask[indices] = 0
    A = graph[mask, :][:, mask]

    if return_removed:
        rows = graph[mask]
        vertices = rows[:, indices].T
        if directed:
            cols = graph[:, mask]
            vertices_right = cols[indices]
            return A, (vertices, vertices_right)
        return A, vertices
    return A


def remap_labels(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    return_map: bool = False,
) -> np.ndarray:
    """
    Remaps a categorical labeling (such as one predicted by a clustering algorithm) to
    match the labels used by another similar labeling.

    Given two :math:`n`-length vectors describing a categorical labeling of :math:`n`
    samples, this method reorders the labels of the second vector (`y_pred`) so that as
    many samples as possible from the two label vectors are in the same category.


    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels, or, labels to map to.
    y_pred : array-like of shape (n_samples,)
        Labels to remap to match the categorical labeling of `y_true`. The categorical
        labeling of `y_pred` will be preserved exactly, but the labels used to
        denote the categories will be changed to best match the categories used in
        `y_true`.
    return_map : bool, optional
        Whether to return a dictionary where the keys are the original category labels
        from `y_pred` and the values are the new category labels that they were mapped
        to.

    Returns
    -------
    remapped_y_pred : np.ndarray of shape (n_samples,)
        Same categorical labeling as that of `y_pred`, but with the category labels
        permuted to best match those of `y_true`.
    label_map : dict
        Mapping from the original labels of `y_pred` to the new labels which best
        resemble those of `y_true`. Only returned if `return_map` was True.

    Examples
    --------
    >>> y_true = np.array([0,0,1,1,2,2])
    >>> y_pred = np.array([2,2,1,1,0,0])
    >>> remap_labels(y_true, y_pred)
    array([0, 0, 1, 1, 2, 2])

    Notes
    -----
    This method will work well when the label vectors describe a somewhat similar
    categorization of the data (as measured by metrics such as
    :func:`sklearn.metrics.adjusted_rand_score`, for example). When the categorizations
    are not similar, the remapping may not make sense (as such a remapping does not
    exist).

    For example, consider when one category in `y_true` is exactly split in half into
    two categories in `y_pred`. If this is the case, it is impossible to say which of
    the categories in `y_pred` match that original category from `y_true`.
    """
    check_consistent_length(y_true, y_pred)
    true_type = type_of_target(y_true)
    pred_type = type_of_target(y_pred)

    valid_target_types = {"binary", "multiclass"}
    if (true_type not in valid_target_types) or (pred_type not in valid_target_types):
        msg = "Elements of `y_true` and `y_pred` must represent a valid binary or "
        msg += "multiclass labeling, see "
        msg += "https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html"
        msg += " for more information."
        raise ValueError(msg)

    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)

    if not isinstance(return_map, bool):
        raise TypeError("return_map must be of type bool.")

    labels = unique_labels(y_true, y_pred)
    confusion_mat = confusion_matrix(y_true, y_pred, labels=labels)
    row_inds, col_inds = linear_sum_assignment(confusion_mat, maximize=True)
    label_map = dict(zip(labels[col_inds], labels[row_inds]))

    remapped_y_pred = np.vectorize(label_map.get)(y_pred)
    if return_map:
        return remapped_y_pred, label_map
    else:
        return remapped_y_pred


def remap_node_ids(
    graph: nx.Graph, weight_attribute: str = "weight"
) -> Tuple[nx.Graph, Dict[Any, str]]:
    """
    Given a graph with arbitrarily types node ids, return a new graph that contains the exact same edgelist
    except the node ids are remapped to a string representation.

    Parameters
    ----------
    graph : nx.Graph
        A graph that has node ids of arbitrary types.
    weight_attribute : str,
        Default is ``weight``. An optional attribute to specify which column in your graph contains the weight value.

    Returns
    -------
    Tuple[nx.Graph, Dict[Any, str]]
        A new graph that contains the same edges except the node ids are remapped to strings. The keys in
        the dictionary are the old node ids and the values are the newly remapped node ids.

    Raises
    ------
    TypeError
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be of type nx.Graph")

    node_id_dict = dict()
    graph_remapped = type(graph)()

    for source, target, weight in graph.edges(data=weight_attribute):
        if source not in node_id_dict:
            node_id_dict[source] = str(len(node_id_dict.keys()))

        if target not in node_id_dict:
            node_id_dict[target] = str(len(node_id_dict.keys()))

        graph_remapped.add_edge(node_id_dict[source], node_id_dict[target])

        graph_remapped[node_id_dict[source]][node_id_dict[target]][
            weight_attribute
        ] = weight

    return graph_remapped, node_id_dict


def suppress_common_warnings():
    """
    Suppresses common warnings that occur when using graspologic.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.simplefilter("always", category=UserWarning)
