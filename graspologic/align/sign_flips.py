# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np

from .base import BaseAlign


class SignFlips(BaseAlign):
    """
    Flips the signs of all entries in one dataset, ``X`` along some of the
    dimensions. In particular, it does so in a way that brings this dataset to
    the same orthant as the second dataset, ``Y``, according to some criterion,
    computed along each dimension. The two critera currently available are the
    median and the maximum (in magnitude) value along each dimension.

    This module can also be used to bring the dataset to the first orthant
    (i.e. with all criteras being positive) by providing the identity matrix as
    the second dataset.

    Parameters
    ----------
    criterion : string, {'median' (default), 'max'}, optional
        String describing the criterion used to choose whether to flip signs.
        Two options are currently supported:

        - 'median'
            Uses the median along each dimension
        - 'max'
            Uses the maximum (in magintude) along each dimension

    Attributes
    ----------
    Q_ : array, size (d, d)
        Final orthogonal matrix, used to modify ``X``.

    """

    def __init__(
        self,
        criterion: str = "median",
    ):
        # checking criterion argument
        if type(criterion) is not str:
            raise TypeError("Criterion must be str")
        if criterion not in ["median", "max"]:
            raise ValueError(f"{criterion} is not a valid criterion.")

        super().__init__()

        self.criterion = criterion

    def set_criterion_function(self) -> None:
        # perform a check, in case it was modified directly
        if self.criterion not in ["median", "max"]:
            raise ValueError(f"{self.criterion} is not a valid criterion")

        if self.criterion == "median":

            def median_criterion(X: np.ndarray) -> np.ndarray:
                result: np.ndarray = np.median(X, axis=0)
                return result

            self.criterion_function_ = median_criterion
        if self.criterion == "max":

            def max_criterion(X: np.ndarray) -> np.ndarray:
                result: np.ndarray = X[np.argmax(np.abs(X), axis=0), np.arange(X.shape[1])]
                return result

            self.criterion_function_ = max_criterion

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "SignFlips":
        """
        Uses the two datasets to learn the matrix :attr:`~graspologic.align.SignFlips.Q_` that aligns the
        first dataset with the second.

        In sign flips, :attr:`~graspologic.align.SignFlips.Q_` is an diagonal orthogonal matrices (i.e. a
        matrix with 1 or -1 in each entry on diagonal and 0 everywhere else)
        picked such that all dimensions of ``X`` @ :attr:`~graspologic.align.SignFlips.Q_`
        and ``Y`` are in the same orthant using some critera (median or max magnitude).

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
        X, Y = self._check_datasets(X, Y)
        _, d = X.shape

        self.set_criterion_function()
        X_criterias = self.criterion_function_(X)
        Y_criterias = self.criterion_function_(Y)

        val = np.multiply(X_criterias, Y_criterias)
        t_X = (val >= 0) * 2 - 1

        self.Q_ = np.diag(t_X)
        return self
