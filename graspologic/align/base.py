# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import Tuple, TypeVar

import numpy as np
from abc import abstractmethod
from sklearn.utils import check_array
from sklearn.base import BaseEstimator

Self = TypeVar("Self", bound="BaseAlign")


class BaseAlign(BaseEstimator):
    """
    Base class for align tasks such as sign flipping, procrustes and seedless
    procrustes.

    Attributes
    ----------
    Q_ : array, size (d, d)
            Final orthogonal matrix, used to modify ``X`` passed to transform

    """

    def __init__(self) -> None:
        pass

    def _check_datasets(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensures that the datasets are numpy, 2d, finite, and have the same
        number of components. Does not check for same number of vertices.
        Returns copies of these datasets.
        """
        # check for numpy-ness
        if not isinstance(X, np.ndarray):
            msg = f"First dataset is a {type(X)}, not an np.ndarray! "
            raise TypeError(msg)
        if not isinstance(Y, np.ndarray):
            msg = f"Second dataset is a {type(Y)}, not an np.ndarray! "
            raise TypeError(msg)
        # check for 2-dness and finiteness
        X = check_array(X, copy=True)
        Y = check_array(Y, copy=True)
        # check for equal components
        if X.shape[1] != Y.shape[1]:
            msg = "Two datasets have different number of components!"
            raise ValueError(msg)
        return X, Y

    @abstractmethod
    def fit(self: Self, X: np.ndarray, Y: np.ndarray) -> Self:
        """
        Uses the two datasets to learn the matrix :attr:`~graspologic.align.BaseAlign.Q_` that aligns the
        first dataset with the second.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Dataset to be mapped to ``Y``, must have same number of dimensions
            (axis 1) as ``Y``.

        Y : np.ndarray, shape (m, d)
            Target dataset, must have same number of dimensions (axis 1) as ``X``.

        Returns
        -------
        self : returns an instance of self
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the dataset ``X`` using the learned matrix :attr:`~graspologic.align.BaseAlign.Q_`. This may
        be the same as the first dataset as in :func:`~graspologic.align.BaseAlign.fit`, or a new dataset.
        For example, additional samples from the same dataset.

        Parameters
        ----------
        X : np.ndarray, shape(m, d)
            Dataset to be transformed, must have same number of dimensions
            (axis 1) as ``X`` and ``Y`` that were passed to fit.

        Returns
        -------
        X_prime : np.ndarray, shape (n, d)
            First dataset of vectors, aligned to second. Equal to
            ``X`` @ :attr:`~graspologic.align.BaseAlign.Q_`.
        """
        if not isinstance(X, np.ndarray):
            msg = f"Dataset is a {type(X)}, not an np.ndarray! "
            raise TypeError(msg)
        X = check_array(X)
        if not X.shape[1] == self.Q_.shape[0]:
            msg = (
                "Dataset needs to have the same number of dimensions, d, "
                "as datasets X and Y used in fit. Currently, vectors in "
                f"the dataset to transform have {X.shape[1]} dimensions, "
                f"while vectors in fit had {self.Q_.shape[0]} dimensions."
            )
            raise ValueError(msg)
        result: np.ndarray = X @ self.Q_
        return result

    def fit_transform(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Uses the two datasets to learn the matrix :attr:`~graspologic.align.BaseAlign.Q_` that aligns the
        first dataset with the second. Then, transforms the first dataset ``X``
        using the learned matrix :attr:`~graspologic.align.BaseAlign.Q_`.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Dataset to be mapped to ``Y``, must have same number of dimensions
            (axis 1) as ``Y``.

        Y : np.ndarray, shape (m, d)
            Target dataset, must have same number of dimensions (axis 1) as ``X``.

        Returns
        -------
        X_prime : np.ndarray, shape (n, d)
            First dataset of vectors, aligned to second. Equal to
            ``X`` @ :attr:`~graspologic.align.BaseAlign.Q_`.
        """
        self.fit(X, Y)
        return self.transform(X)
