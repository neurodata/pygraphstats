# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import scipy
import sklearn
from scipy.stats import norm
from scipy.sparse import isspmatrix_csr
from graspologic.utils import is_almost_symmetric


def _compute_likelihood(arr):
    """
    Computes the log likelihoods based on normal distribution given
    a 1d-array of sorted values. If the input has no variance,
    the likelihood will be nan.
    """
    n_elements = len(arr)
    likelihoods = np.zeros(n_elements)

    for idx in range(1, n_elements + 1):
        # split into two samples
        s1 = arr[:idx]
        s2 = arr[idx:]

        # deal with when input only has 2 elements
        if (s1.size == 1) & (s2.size == 1):
            likelihoods[idx - 1] = -np.inf
            continue

        # compute means
        mu1 = np.mean(s1)
        if s2.size != 0:
            mu2 = np.mean(s2)
        else:
            # Prevent numpy warning for taking mean of empty array
            mu2 = -np.inf

        # compute pooled variance
        variance = ((np.sum((s1 - mu1) ** 2) + np.sum((s2 - mu2) ** 2))) / (
            n_elements - 1 - (idx < n_elements)
        )
        std = np.sqrt(variance)

        # compute log likelihoods
        likelihoods[idx - 1] = np.sum(norm.logpdf(s1, loc=mu1, scale=std)) + np.sum(
            norm.logpdf(s2, loc=mu2, scale=std)
        )

    return likelihoods


def select_dimension(
    X, n_components=None, n_elbows=2, threshold=None, return_likelihoods=False
):
    """
    Generates profile likelihood from array based on Zhu and Godsie method.
    Elbows correspond to the optimal embedding dimension.

    Parameters
    ----------
    X : 1d or 2d array-like
        Input array generate profile likelihoods for. If 1d-array, it should be
        sorted in decreasing order. If 2d-array, shape should be
        (n_samples, n_features).
    n_components : int, optional, default: None.
        Number of components to embed. If None, ``n_components =
        floor(log2(min(n_samples, n_features)))``. Ignored if ``X`` is 1d-array.
    n_elbows : int, optional, default: 2.
        Number of likelihood elbows to return. Must be ``> 1``.
    threshold : float, int, optional, default: None
        If given, only consider the singular values that are ``> threshold``. Must
        be ``>= 0``.
    return_likelihoods : bool, optional, default: False
        If True, returns the all likelihoods associated with each elbow.

    Returns
    -------
    elbows : list
        Elbows indicate subsequent optimal embedding dimensions. Number of
        elbows may be less than ``n_elbows`` if there are not enough singular
        values.
    sing_vals : list
        The singular values associated with each elbow.
    likelihoods : list of array-like
        Array of likelihoods of the corresponding to each elbow. Only returned
        if ``return_likelihoods`` is True.

    References
    ----------
    .. [1] Zhu, M. and Ghodsi, A. (2006).
        Automatic dimensionality selection from the scree plot via the use of
        profile likelihood. Computational Statistics & Data Analysis, 51(2),
        pp.918-930.
    """
    # Handle input data
    if not isinstance(X, np.ndarray) and not isspmatrix_csr(X):
        msg = "X must be a numpy array or scipy.sparse.csr_matrix, not {}.".format(
            type(X)
        )
        raise ValueError(msg)
    if X.ndim > 2:
        msg = "X must be a 1d or 2d-array, not {}d array.".format(X.ndim)
        raise ValueError(msg)
    elif np.min(X.shape) <= 1:
        msg = "X must have more than 1 samples or 1 features."
        raise ValueError(msg)

    # Handle n_elbows
    if not isinstance(n_elbows, int):
        msg = "n_elbows must be an integer, not {}.".format(type(n_elbows))
        raise ValueError(msg)
    elif n_elbows < 1:
        msg = f"number of elbows should be an integer > 1, not {n_elbows}."
        raise ValueError(msg)

    # Handle threshold
    if threshold is not None:
        if not isinstance(threshold, (int, float)):
            msg = "threshold must be an integer or a float, not {}.".format(
                type(threshold)
            )
            raise ValueError(msg)
        elif threshold < 0:
            msg = "threshold must be >= 0, not {}.".format(threshold)
            raise ValueError(msg)

    # Handle n_components
    if n_components is None:
        # per recommendation by Zhu & Godsie
        k = int(np.ceil(np.log2(np.min(X.shape))))
    elif not isinstance(n_components, int):
        msg = "n_components must be an integer, not {}.".format(type(n_components))
        raise ValueError(msg)
    else:
        k = n_components

    # Check to see if svd is needed
    if X.ndim == 1:
        D = np.sort(X)[::-1]
    elif X.ndim == 2:
        # Singular values in decreasing order
        D = scipy.sparse.linalg.svds(A=X, k=k, return_singular_vectors=False)
        D = np.sort(D)[::-1]
        # U, D, V = sklearn.utils.extmath.randomized_svd()

    if threshold is not None:
        D = D[D > threshold]

        if len(D) == 0:
            msg = "No values greater than threshold {}."
            raise IndexError(msg.format(threshold))

    idx = 0
    elbows = []
    values = []
    likelihoods = []
    for _ in range(n_elbows):
        arr = D[idx:]
        if arr.size <= 1:  # Cant compute likelihoods with 1 numbers
            break
        lq = _compute_likelihood(arr)
        idx += np.argmax(lq) + 1
        elbows.append(idx)
        values.append(D[idx - 1])
        likelihoods.append(lq)

    if return_likelihoods:
        return elbows, values, likelihoods
    else:
        return elbows, values


def selectSVD(X, n_components=None, n_elbows=2, algorithm="randomized", n_iter=5):
    r"""
    Dimensionality reduction using SVD.

    Performs linear dimensionality reduction by using either full singular
    value decomposition (SVD) or truncated SVD. Full SVD is performed using
    SciPy's wrapper for ARPACK, while truncated SVD is performed using either
    SciPy's wrapper for LAPACK or Sklearn's implementation of randomized SVD.

    It also performs optimal dimensionality selection using Zhu & Godsie algorithm
    if number of target dimension is not specified.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to perform svd on.
    n_components : int or None, default = None
        Desired dimensionality of output data. If "full",
        ``n_components`` must be ``<= min(X.shape)``. Otherwise, ``n_components`` must be
        ``< min(X.shape)``. If None, then optimal dimensions will be chosen by
        :func:`~graspologic.embed.select_dimension` using ``n_elbows`` argument.
    n_elbows : int, optional, default: 2
        If ``n_components`` is None, then compute the optimal embedding dimension using
        :func:`~graspologic.embed.select_dimension`. Otherwise, ignored.
    algorithm : {'randomized' (default), 'full', 'truncated'}, optional
        SVD solver to use:

        - 'randomized'
            Computes randomized svd using
            :func:`sklearn.utils.extmath.randomized_svd`
        - 'full'
            Computes full svd using :func:`scipy.linalg.svd`
            Does not support ``graph`` input of type scipy.sparse.csr_matrix
        - 'truncated'
            Computes truncated svd using :func:`scipy.sparse.linalg.svds`
        - 'eigsh'
            Computes svd of a real, symmetric square matrix using
            :func:`scipy.sparse.linalg.eigsh`. Extremely fast for these types of
            matrices.
    n_iter : int, optional (default = 5)
        Number of iterations for randomized SVD solver. Not used by 'full' or
        'truncated'. The default is larger than the default in randomized_svd
        to handle sparse matrices that may have large slowly decaying spectrum.

    Returns
    -------
    U : array-like, shape (n_samples, n_components)
        Left singular vectors corresponding to singular values.
    D : array-like, shape (n_components)
        Singular values in decreasing order, as a 1d array.
    V : array-like, shape (n_components, n_samples)
        Right singular vectors corresponding to singular values.

    References
    ----------
    .. [1] Zhu, M. and Ghodsi, A. (2006).
        Automatic dimensionality selection from the scree plot via the use of
        profile likelihood. Computational Statistics & Data Analysis, 51(2),
        pp.918-930.
    """
    # Added in order to pass check estimator, must include words "one sample"
    if X.shape[0] == 1:
        msg = "Input data has only one sample (node)"
        raise ValueError(msg)

    # Deal with algorithms
    if algorithm not in ["full", "truncated", "randomized", "eigsh"]:
        msg = "algorithm must be one of {full, truncated, randomized, eigsh}."
        raise ValueError(msg)

    if algorithm == "full" and isspmatrix_csr(X):
        msg = "'full' agorithm does not support scipy.sparse.csr_matrix inputs."
        raise TypeError(msg)

    if n_components is None:
        elbows, _ = select_dimension(X, n_elbows=n_elbows, threshold=None)
        n_components = elbows[-1]

    # Check
    if (algorithm == "full") & (n_components > min(X.shape)):
        msg = "n_components must be <= min(X.shape)."
        raise ValueError(msg)

    if (algorithm in ["truncated", "randomized"]) & (n_components >= min(X.shape)):
        msg = "n_components must be strictly < min(X.shape)."
        raise ValueError(msg)

    if algorithm == "full":
        U, D, V = scipy.linalg.svd(X)
        U = U[:, :n_components]
        D = D[:n_components]
        V = V[:n_components, :]

    elif algorithm == "truncated":
        U, D, V = scipy.sparse.linalg.svds(X, k=n_components)
        idx = np.argsort(D)[::-1]  # sort in decreasing order
        D = D[idx]
        U = U[:, idx]
        V = V[idx, :]

    elif algorithm == "eigsh":
        D, U = scipy.sparse.linalg.eigsh(X, k=n_components)
        # singular values of a real symmetric matrix are the absolute values of its
        # eigenvalues, so need to take np.abs
        D = np.abs(D)
        V = U.T

        # sort in decreasing order
        idx = np.argsort(D)[::-1]
        D = D[idx]
        U = U[:, idx]
        V = V[idx, :]

    elif algorithm == "randomized":
        U, D, V = sklearn.utils.extmath.randomized_svd(X, n_components, n_iter=n_iter)

    else:
        raise ValueError(
            "algorithm must be in {'full', 'truncated', 'randomized', 'eigsh'}"
        )

    return U, D, V
