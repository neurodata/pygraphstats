# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from abc import ABC, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.validation import check_is_fitted


class BaseCluster(ABC, BaseEstimator, ClusterMixin):
    """
    Base clustering class.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y=None):
        """
        Compute clusters based on given method.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : array-like, shape (n_samples,), optional (default=None)
            List of labels for X if available. Used to compute
            ARI scores.

        Returns
        -------
        self
        """

    def predict(self, X, y=None):  # pragma: no cover
        """
        Predict clusters based on best model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : array-like, shape (n_samples, ), optional (default=None)
            List of labels for X if available. Used to compute
            ARI scores.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        # Check if fit is already called
        check_is_fitted(self, ["model_"], all_or_any=all)
        labels = self.model_.predict(X)

        return labels

    def fit_predict(self, X, y=None):  # pragma: no cover
        """
        Fit the models and predict clusters based on best model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : array-like, shape (n_samples,), optional (default=None)
            List of labels for X if available. Used to compute
            ARI scores.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        self.fit(X, y)

        labels = self.predict(X, y)
        return labels
