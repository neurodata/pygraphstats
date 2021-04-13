# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import Optional, Tuple, Union

import ot
import numpy as np
from sklearn.utils import check_array

from .base import BaseAlign
from .sign_flips import SignFlips
from .orthogonal_procrustes import OrthogonalProcrustes


class SeedlessProcrustes(BaseAlign):
    """
    Matches two datasets using an orthogonal matrix. Unlike
    :class:`~graspologic.align.OrthogonalProcrustes`, this does not require a
    matching between entries. It can even be used in the settings where the two
    datasets do not have the same number of entries.

    In graph setting, it is used to align the embeddings of two different
    graphs, when it requires some simultaneous inference task and no 1-1
    matching between the vertices of the two graphs can be established, for
    example, inside of the test for the equivalence of the latent distributions
    (see: :class:`~graspologic.inference.LatentDistributionTest`).

    Parameters
    ----------
    optimal_transport_lambda : float (default=0.1), optional
        Regularization term of the Sinkhorn optimal transport algorithm.

    optimal_transport_eps : float (default=0.01), optional
        Tolerance parameter for the each Sinkhorn optimal transport algorithm.
        I.e. tolerance for each "E-step".

    optimal_transport_num_reps : int (default=1000), optional
        Number of repetitions in each iteration of the iterative optimal
        transport problem. I.e. maximum number of repetitions in each "E-step".

    iterative_num_reps : int (default=100), optional
        Number of reps in each iteration of the iterative optimal transport
        problem. I.e. maxumum number of total iterations the whole "EM"
        algorithm.

    init : string, {'2d' (default), 'sign_flips', 'custom'}, optional

        - '2d'
            Uses :math:`2^d` different restarts, where :math:`d` is the
            dimension of the datasets. In particular, tries all matrices that
            are simultaneously diagonal and orthogonal. In other words, these
            are diagonal matrices with all entries on the diagonal being either
            +1 or -1. This is motivated by the fact that spectral graph
            embeddings have two types of orthogonal non-identifiability, one of
            which is captured by the orthogonal diagonal matrices. The final
            result is picked based on the final values of the objective
            function. For more on this, see [2]_.
        - 'sign_flips'
            Initial alignment done by making the median value in each dimension
            have the same sign. The motivation is similar to that in '2d',
            except this is a heuristic that can save time, but can sometimes
            yield suboptimal results.
        - 'custom'
            Expects either an initial guess for :attr:`Q_` or an initial guess
            for :attr:`P_`, but not both. See ``initial_Q`` and ``initial_P``,
            respectively. If neither is provided, initializes ``initial_Q`` to an
            identity with an appropriate number of dimensions.

    initial_Q : np.ndarray, shape (d, d) or None, optional (default=None)
        An initial guess for the alignment matrix, :attr:`Q_`, if such exists.
        Only one of ``initial_Q``, ``initial_P`` can be provided at the same time,
        and only if ``init`` argument is set to 'custom'. If None, and
        ``initial_P`` is also None - initializes ``initial_Q`` to identity matrix.
        Must be an orthogonal matrix, if provided.

    initial_P : np.ndarray, shape (n, m) or None, optional (default=None)
        Initial guess for the optimal transport matrix, :attr:`P_`, if such
        exists. Only one of ``initial_Q``, ``initial_P`` can be provided at the
        same time, and only if ``init`` argument is set to 'custom'. If None, and
        ``initial_Q`` is also None - initializes ``initial_Q`` to identity matrix.
        Must be a soft assignment matrix if provided (rows sum up to 1/n, cols
        sum up to 1/m.)

    Attributes
    ----------
    Q_ : array, size (d, d)
        Final orthogonal matrix, used to modify ``X``.

    P_ : array, size (n, m) where n and m are the sizes of two datasets
        Final matrix of optimal transports, represent soft matching weights
        from points in one dataset to the other, normalized such that all rows
        sum to 1/n and all columns sum to 1/m.

    score_ : float
        Final value of the objective function: :math:`|| X Q - P Y ||_F`
        Lower means the datasets have been matched together better.

    selected_initial_Q_ : array, size (d, d)
        Initial orthogonal matrix which was used as the initialization.
        If ``init`` was set to '2d' or 'sign_flips', then it is the adaptively
        selected matrix.
        If ``init`` was set to 'custom', and ``initial_Q`` was provided, then equal
        to that. If it was not provided, but ``initial_P`` was, then it is the
        matrix after the first procrustes performed. If neither was provided,
        then it is the identity matrix.

    References
    ----------

    .. [1] Agterberg, J., Tang, M., Priebe., C. E. (2020).
        "Nonparametric Two-Sample Hypothesis Testing for Random Graphs with Negative and Repeated Eigenvalues"
        arXiv:2012.09828

    .. [2] Agterberg, J., Tang, M., Priebe., C. E. (2020).
        "On Two Distinct Sources of Nonidentifiability in Latent Position Random Graph Models"
        arXiv:2003.14250

    Notes
    -----
    In essence, the goal of this procedure is to simultaneously obtain a, not
    necessarily 1-to-1, correspondence between the vertices of the two data
    sets, and an orthogonal alignment between two datasets. If the two datasets
    are represented with matrices :math:`X \in M_{n, d}` and
    :math:`Y \in M_{m, d}`, then the correspondence is a matrix
    :math:`P \in M_{n, m}` that is soft assignment matrix (that is, its rows
    sum to :math:`1/n`, and columns sum to :math:`1/m`) and the orthogonal
    alignment is an orthogonal matrix :math:`Q \in M_{d, d}` (an orthogonal
    matrix is any matrix that satisfies :math:`Q^T Q = Q Q^T = I`). The global
    objective function is :math:`|| X Q - P Y ||_F`.

    Note that both :math:`X` and :math:`PY` are matrices in :math:`M_{n, d}`.
    Thus, if one knew :math:`P`, it would be simple to obtain an estimate for
    :math:`Q`, using the regular orthogonal procrustes. On the other hand, if
    :math:`Q` was known, then :math:`XQ` and :math:`Y` could be thought of
    distributions over a finite number of masses, each with weight :math:`1/n`
    or :math:`1/m`, respectively. These distributions could be "matched" via
    solving an optimal transport problem.

    However, both :math:`Q` and :math:`P` are simultaneously unknown here. So
    the algorithm performs a sequence of alternating steps, obtaining
    iteratively improving estimates of :math:`Q` and :math:`P`, similarly to an
    expectation-maximization (EM) procedure. It is not known whether this
    procedure is formally an EM, but the analogy can be drawn as follows: after
    obtaining an initial guess of of :math:`\hat{Q}_{0}`, obtaining an
    assignment matrix :math:`\hat{P}_{i+1} | \hat{Q}_{i}` ("E-step") is done by
    solving an optimal transport problem via Sinkhorn algorithm, whereas
    obtaining an orthogonal alignment matrix :math:`Q_{i+1} | P_{i}` ("M-step")
    is done via regular orthogonal procurstes. These alternating steps are
    performed until ``iterative_num_reps`` is reached.

    For more on how the initial guess can be performed, see ``init``.

    """

    def __init__(
        self,
        optimal_transport_lambda: float = 0.1,
        optimal_transport_eps: float = 0.01,
        optimal_transport_num_reps: int = 1000,
        iterative_num_reps:int = 100,
        init: str = "2d",
        initial_Q: Optional[np.ndarray] = None,
        initial_P: Optional[np.ndarray] = None,
    ):
        # check optimal_transport_lambda argument
        if type(optimal_transport_lambda) is not float:
            msg = "Optimal_transport_lambda must be a float, not {}".format(
                type(optimal_transport_lambda)
            )
            raise TypeError(msg)
        if optimal_transport_lambda < 0:
            msg = "{} is an invalud value of the optimal_transport_lambda, must be non-negative".format(
                optimal_transport_lambda
            )
            raise ValueError(msg)
        # check optimal_transport_lambda argument
        if type(optimal_transport_eps) is not float:
            msg = "Optimal_transport_eps must be a float, not {}".format(
                type(optimal_transport_eps)
            )
            raise TypeError(msg)
        if optimal_transport_eps <= 0:
            msg = "{} is an invalid value of the optimal transport eps, must be postitive".format(
                optimal_transport_eps
            )
            raise ValueError(msg)
        # check optimal_transport_num_reps argument
        if type(optimal_transport_num_reps) is not int:
            msg = "Optimal_transport_num_reps must be a int, not {}".format(
                type(optimal_transport_num_reps)
            )
            raise TypeError(msg)
        if optimal_transport_num_reps < 1:
            msg = "{} is an invalid number of repetitions, must be non-negative".format(
                iterative_num_reps
            )
            raise ValueError(msg)
        # check iterative_num_reps argument
        if type(iterative_num_reps) is not int:
            msg = "Iterative_num_reps must be a int, not {}".format(
                type(iterative_num_reps)
            )
            raise TypeError(msg)
        if iterative_num_reps < 0:
            msg = "{} is an invalid number of repetitions, must be non-negative".format(
                iterative_num_reps
            )
            raise ValueError(msg)
        # check init argument
        if type(init) is not str:
            msg = "Init must be a str, not {}".format(type(init))
            raise TypeError(msg)
        inits_supported = ["2d", "sign_flips", "custom"]
        if init not in inits_supported:
            msg = "Supported inits are {}".format(inits_supported)
            raise ValueError(msg)
        # check that initial_Q and intial_P aren't provided when shouldn't be
        if initial_Q is not None and init != "custom":
            msg = "Initial_Q can only be provided if init is set to custom"
            raise ValueError(msg)
        if initial_P is not None and init != "custom":
            msg = "Initial_P can only be provided if init is set to custom"
            raise ValueError(msg)
        if initial_Q is not None and initial_P is not None:
            msg = "Initial_Q and initial_P cannot be provided simultaneously"
            raise ValueError(msg)
        # check initial_Q argument
        if initial_Q is not None:
            if not isinstance(initial_Q, np.ndarray):
                msg = f"Initial_Q must be np.ndarray or None, not {type(initial_Q)}"
                raise TypeError(msg)
            initial_Q = check_array(initial_Q, copy=True)
            if initial_Q.shape[0] != initial_Q.shape[1]:
                msg = "Initial_Q must be a square orthogonal matrix"
                raise ValueError(msg)
            if not np.allclose(initial_Q.T @ initial_Q, np.eye(initial_Q.shape[0])):
                msg = "Initial_Q must be a square orthogonal matrix"
                raise ValueError(msg)
        # check initial_P argument
        if initial_P is not None:
            if not isinstance(initial_P, np.ndarray):
                msg = f"Initial_P must be np.ndarray or None, not {type(initial_P)}"
                raise TypeError(msg)
            initial_P = check_array(initial_P, copy=True)
            n, m = initial_P.shape
            if not (
                np.allclose(initial_P.sum(axis=0), np.ones(m) / m)
                and np.allclose(initial_P.sum(axis=1), np.ones(n) / n)
            ):
                msg = (
                    "Initial_P must be a soft assignment matrix "
                    "(rows add up to (1/number of cols) "
                    "and columns add up to (1/number of rows))"
                )
                raise ValueError(msg)

        super().__init__()

        self.optimal_transport_eps = optimal_transport_eps
        self.optimal_transport_num_reps = optimal_transport_num_reps
        self.optimal_transport_lambda = optimal_transport_lambda
        self.iterative_num_reps = iterative_num_reps
        self.init = init
        self.initial_Q = initial_Q
        self.initial_P = initial_P

    def _optimal_transport(self, X: np.ndarray, Y: np.ndarray, Q: np.ndarray) -> np.ndarray:
        # "E step" of the SeedlessProcrustes.
        n, d = X.shape
        m, _ = Y.shape
        # initialize probability mass arrays & the cost matrix ; run sinkhorn
        probability_mass_X = np.ones(n) / n
        probability_mass_Y = np.ones(m) / m
        cost_matrix = (
            np.linalg.norm((X @ Q).reshape(n, 1, d) - Y.reshape(1, m, d), axis=2) ** 2
        )
        # Note: the type info on ot.sinkhorn can support a return type of Tuple[np.ndarray, Dict[str, Any]], but only
        # if this function is called with log=True, which we aren't doing.
        P: np.ndarray = ot.sinkhorn(
            a=probability_mass_X,
            b=probability_mass_Y,
            M=cost_matrix,
            reg=self.optimal_transport_lambda,
            numItermax=self.optimal_transport_num_reps,
            stopThr=self.optimal_transport_eps,
        )
        return P

    def _procrustes(self, X: np.ndarray, Y: np.ndarray, P: np.ndarray) -> np.ndarray:
        # "M step" of the SeedlessProcurstes.
        aligner = OrthogonalProcrustes()
        Q: np.ndarray = aligner.fit(X, P @ Y).Q_
        return Q

    def _iterative_ot(self, X: np.ndarray, Y: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # this P is not used. it is set to default in case numreps=0
        P: np.ndarray = np.ones((X.shape[0], Y.shape[0])) / (X.shape[0] * Y.shape[0])
        for i in range(self.iterative_num_reps):
            P = self._optimal_transport(X, Y, Q)
            Q = self._procrustes(X, Y, P)
        return P, Q

    def _compute_objective(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            Q: Optional[np.ndarray] = None,
            P: Optional[np.ndarray] = None
    ) -> Union[float, np.ndarray]:
        if Q is None:
            Q = self.Q_
        if P is None:
            P = self.P_
        return np.linalg.norm(X @ Q - P @ Y, ord="fro")

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "SeedlessProcrustes":
        """
        Uses the two datasets to learn the matrix `self.Q_` that aligns the
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
        X, Y = self._check_datasets(X, Y)
        n, d = X.shape
        m, _ = Y.shape

        if self.init == "2d":
            P_matrices = np.zeros((2 ** d, n, m))
            Q_matrices = np.zeros((2 ** d, d, d))
            objectives = np.zeros(2 ** d)
            # try 2^d different initializations
            for i in range(2 ** d):
                initial_Q = _sign_flip_matrix_from_int(i, d)
                P_matrices[i], Q_matrices[i] = P, Q = self._iterative_ot(
                    X, Y, initial_Q
                )
                objectives[i] = self._compute_objective(X, Y, Q, P)
            # pick the best one, using the objective function value
            best = np.argmin(objectives)
            self.selected_initial_Q_ = _sign_flip_matrix_from_int(best, d)
            self.P_, self.Q_ = P_matrices[best], Q_matrices[best]
        elif self.init == "sign_flips":
            aligner = SignFlips()
            self.selected_initial_Q_ = aligner.fit(X, Y).Q_
            self.P_, self.Q_ = self._iterative_ot(X, Y, self.selected_initial_Q_)
        else:
            # determine initial Q if "custom"
            if self.initial_Q is not None:
                self.selected_initial_Q_ = self.initial_Q
            elif self.initial_P is not None:
                # use initial P, if provided
                self.selected_initial_Q_ = self._procrustes(X, Y, self.initial_P)
            else:
                # set to initial Q to identity if neither Q nor P provided
                self.selected_initial_Q_ = np.eye(d)
            self.P_, self.Q_ = self._iterative_ot(X, Y, self.selected_initial_Q_)
        self.score_ = self._compute_objective(X, Y)

        return self


def _sign_flip_matrix_from_int(val_int: int, d: int) -> np.ndarray:
    val_bin = bin(val_int)[2:]
    val_bin = "0" * (d - len(val_bin)) + val_bin
    return np.diag(np.array([(float(i) - 0.5) * -2 for i in val_bin]))
