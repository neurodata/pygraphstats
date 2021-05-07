# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy.spatial import procrustes

from graspologic.embed.svd import selectSVD
from graspologic.simulations.simulations import er_np
from graspologic.utils import symmetrize


def test_bad_inputs():
    X = np.random.normal(size=(100, 5))
    with pytest.raises(ValueError):
        bad_algo = "ROFLMAO"
        selectSVD(X, algorithm=bad_algo)

    with pytest.raises(ValueError):
        algorithm = "full"
        bad_components = 1000
        selectSVD(X, n_components=bad_components, algorithm=algorithm)

    with pytest.raises(ValueError):
        algorithm = "truncated"
        bad_components = 1000
        selectSVD(X, n_components=bad_components, algorithm=algorithm)


def test_outputs():
    np.random.seed(123)
    X = np.vstack(
        [
            np.repeat([[0.2, 0.2, 0.2]], 50, axis=0),
            np.repeat([[0.5, 0.5, 0.5]], 50, axis=0),
        ]
    )
    P = X @ X.T
    A = np.random.binomial(1, P).astype(np.float)

    n_components = 3

    # Full SVD
    U_full, D_full, V_full = selectSVD(A, n_components=n_components, algorithm="full")
    X_full = U_full @ np.diag(np.sqrt(D_full))
    _, _, norm_full = procrustes(X, X_full)

    # Truncated SVD
    U_trunc, D_trunc, V_trunc = selectSVD(
        A, n_components=n_components, algorithm="truncated"
    )
    X_trunc = U_trunc @ np.diag(np.sqrt(D_trunc))
    _, _, norm_trunc = procrustes(X, X_trunc)

    # Randomized SVD
    U_rand, D_rand, V_rand = selectSVD(
        A, n_components=n_components, algorithm="randomized", n_iter=10
    )
    X_rand = U_rand @ np.diag(np.sqrt(D_rand))
    _, _, norm_rand = procrustes(X, X_rand)

    rtol = 1e-4
    atol = 1e-4
    assert_allclose(norm_full, norm_trunc, rtol, atol)
    assert_allclose(norm_full, norm_rand, rtol, atol)


def test_eigsh():
    np.random.seed(123)
    X = np.vstack(
        [
            np.repeat([[0.2, 0.2, 0.2]], 50, axis=0),
            np.repeat([[0.5, 0.5, 0.5]], 50, axis=0),
        ]
    )
    P = X @ X.T
    A = np.random.binomial(1, P).astype(np.float)
    A = symmetrize(A, method="triu")
    n_components = 3

    # Full SVD
    U_full, D_full, V_full = selectSVD(A, n_components=n_components, algorithm="full")
    X_full = U_full @ np.diag(np.sqrt(D_full))
    _, _, norm_full = procrustes(X, X_full)

    # eigsh SVD
    U_square, D_square, V_square = selectSVD(
        A, n_components=n_components, algorithm="eigsh", n_iter=10
    )
    X_square = U_square @ np.diag(np.sqrt(D_square))
    _, _, norm_square = procrustes(X, X_square)

    rtol = 1e-4
    atol = 1e-4
    assert_allclose(norm_full, norm_square, rtol, atol)
