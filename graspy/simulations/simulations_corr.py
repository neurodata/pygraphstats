import numpy as np
from graspy.simulations import sample_edges


def sample_edges_corr(P, R, directed, loops):
    """
    Generate a pair of correlated graphs with Bernoulli distribution.
    Both G1 and G2 are binary matrices. 

    Parameters
    ----------
    P: np.ndarray, shape (n_vertices, n_vertices)
        Matrix of probabilities (between 0 and 1) for a random graph
    R: np.ndarray, shape (n_vertices, n_vertices)
        Matrix to definite the correlation between graph1 and graph2
    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.
    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.

    References
    ----------
    .. [1] Vince Lyzinski, et al. "Seeded Graph Matching for Correlated Erdos-Renyi Graphs", 
       Journal of Machine Learning Research 15, 2014
        
    Returns
    -------
    G1: ndarray (n_vertices, n_vertices)
        Adjacency matrix the same size as P representing a random graph,

    G2: ndarray (n_vertices, n_vertices)
        Adjacency matrix the same size as P representing a random graph,

    Examples
    --------
    >>> np.random.seed(1)
    >>> p = 0.5
    >>> rho = 0.3
    >>> P = p * np.ones((5,5))
    >>> Rho = rho * np.ones((5,5))

    To sample a correlated graph pair based on P and Rho matrices:

    >>> sample_edges_corr(P,Rho,directed = False, loops = False)
    (array([[0., 1., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 1.],
            [0., 0., 1., 1., 0.]]), array([[0., 1., 0., 0., 0.],
            [1., 0., 1., 0., 1.],
            [0., 1., 0., 1., 1.],
            [0., 0., 1., 0., 1.],
            [0., 1., 1., 1., 0.]]))
    """
    G1 = sample_edges(P, directed=directed, loops=loops)
    P2 = G1.copy()
    P2 = np.where(P2 == 1, P + R * (1 - P), P * (1 - R))
    G2 = sample_edges(P2, directed=directed, loops=loops)
    return G1, G2
