# Originally released by:
# Authors: Yuto Yamaguchi <yamaguchi.yuto@aist.go.jp>
# Lisence: MIT
# At: https://github.com/yamaguchiyuto/label_propagation

# Updated by Ryan Hausen
# License: MIT
# At: https://github.com/sciserver/label_propagation

"""
Graph-Based Semi-Supervised Learning (GBSSL) implementation.
"""

# Authors: Yuto Yamaguchi <yamaguchi.yuto@aist.go.jp>
# Lisence: MIT

import logging
import warnings
from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin


class Base(BaseEstimator, ClassifierMixin):
    __metaclass__ = ABCMeta

    def __init__(self, graph, max_iter=30):
        self.max_iter = max_iter
        self.graph = graph

    @abstractmethod
    def _build_propagation_matrix(self):
        raise NotImplementedError(
            "Propagation matrix construction must be implemented to fit a model."
        )

    @abstractmethod
    def _build_base_matrix(self):
        raise NotImplementedError(
            "Base matrix construction must be implemented to fit a model."
        )

    def _init_label_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max() + 1
        return np.zeros((n_samples, n_classes))

    def _arrange_params(self):
        """Do nothing by default"""
        pass

    def fit(self, x, y):
        """Fit a graph-based semi-supervised learning model

        All the input data is provided array X (labeled samples only)
        and corresponding label array y.

        Parameters
        ----------
        x : array_like, shape = [n_labeled_samples]
            Node IDs of labeled samples
        y : array_like, shape = [n_labeled_samples]
            Label IDs of labeled samples

        Returns
        -------
        self : returns an instance of self.
        """
        self.x_ = x
        self.y_ = y

        self._arrange_params()

        self.F_ = self._init_label_matrix()

        self.P_ = self._build_propagation_matrix()
        self.B_ = self._build_base_matrix()

        remaining_iter = self.max_iter
        while remaining_iter > 0:
            self.F_ = self._propagate()
            remaining_iter -= 1

        return self

    def _propagate(self):
        return self.P_.dot(self.F_) + self.B_

    def predict(self, x):
        """Performs prediction based on the fitted model

        Parameters
        ----------
        x : array_like, shape = [n_samples]
            Node IDs

        Returns
        -------
        y : array_like, shape = [n_samples]
            Predictions for input node IDs
        """
        probas = self.predict_proba(x)
        return np.argmax(probas, axis=1)

    def predict_proba(self, x):
        """Predict probability for each possible label

        Parameters
        ----------
        x : array_like, shape = [n_samples]
            Node IDs

        Returns
        -------
        probabilities : array_like, shape = [n_samples, n_classes]
            Probability distributions across class labels
        """
        z = np.sum(self.F_[x], axis=1)
        z[z == 0] += 1  # Avoid division by 0
        return (self.F_[x].T / z).T


class LGC(Base):
    """Local and Global Consistency (LGC) for GBSSL

    Parameters
    ----------
    alpha : float
      clamping factor
    max_iter : float
      maximum number of iterations allowed

    Attributes
    ----------
    x_ : array, shape = [n_samples]
        Input array of node IDs.

    References
    ----------
    Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004).
    Learning with local and global consistency.
    Advances in neural information processing systems, 16(16), 321-328.
    """

    def __init__(self, graph=None, alpha=0.99, max_iter=30):
        super(LGC, self).__init__(graph, max_iter=30)
        self.alpha = alpha

    def _build_propagation_matrix(self):
        """LGC computes the normalized Laplacian as its propagation matrix"""
        degrees = np.asarray(self.graph.sum(axis=0))
        degrees[degrees == 0] += 1  # Avoid division by 0
        D2 = np.sqrt(sparse.diags((1.0 / degrees), offsets=0))
        S = D2.dot(self.graph).dot(D2)
        return self.alpha * S

    def _build_base_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max() + 1
        B = np.zeros((n_samples, n_classes))
        B[self.x_, self.y_] = 1
        return (1 - self.alpha) * B


class HMN(Base):
    """Harmonic funcsion (HMN) for GBSSL

    Parameters
    ----------
    max_iter : float
      maximum number of iterations allowed

    Attributes
    ----------
    x_ : array, shape = [n_samples]
        Input array of node IDs.

    References
    ----------
    Zhu, X., Ghahramani, Z., & Lafferty, J. (2003, August).
    Semi-supervised learning using gaussian fields and harmonic functions.
    In ICML (Vol. 3, pp. 912-919).
    """

    def __init__(self, graph=None, max_iter=30):
        super(HMN, self).__init__(graph, max_iter=30)

    def _build_propagation_matrix(self):
        degrees = np.asarray(self.graph.sum(axis=0))
        degrees[degrees == 0] += 1  # Avoid division by 0
        D = sparse.diags((1.0 / degrees), offsets=0)
        P = D.dot(self.graph).tolil()
        P[self.x_] = 0
        return P.tocsr()

    def _build_base_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max() + 1
        B = np.zeros((n_samples, n_classes))
        B[self.x_, self.y_] = 1
        return B


class PARW(Base):
    """Partially Absorbing Random Walk (PARW) for GBSSL

    Parameters
    ----------
    lamb: float (default=0.001)
      Absorbing parameter
    max_iter : float
      maximum number of iterations allowed

    Attributes
    ----------
    x_ : array, shape = [n_samples]
        Input array of node IDs.

    References
    ----------
    Wu, X. M., Li, Z., So, A. M., Wright, J., & Chang, S. F. (2012).
    Learning with partially absorbing random walks.
    In Advances in Neural Information Processing Systems (pp. 3077-3085).
    """

    def __init__(self, graph=None, lamb=1.0, max_iter=30):
        super(PARW, self).__init__(graph, max_iter=30)
        self.lamb = lamb

    def _build_propagation_matrix(self):
        d = np.asarray(self.graph.sum(axis=1).T)
        Z = sparse.diags(1.0 / (d + self.lamb), offsets=0)
        P = Z.dot(self.graph)
        return P

    def _build_base_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max() + 1
        B = np.zeros((n_samples, n_classes))
        B[self.x_, self.y_] = 1
        d = np.array(self.graph.sum(axis=1).T)
        Z = sparse.diags(1.0 / (d + self.lamb), offsets=0)
        Lamb = sparse.diags(self.lamb * np.ones(n_samples), offsets=0)
        return Z.dot(Lamb).dot(B)


class OMNI(Base):
    """OMNI-Prop for GBSSL

    Parameters
    ----------
    lamb : float > 0 (default = 1.0)
      Define importance between prior and evidence from neighbors
    max_iter : float
      maximum number of iterations allowed

    Attributes
    ----------
    x_ : array, shape = [n_samples]
        Input array of node IDs.

    References
    ----------
    Yamaguchi, Y., Faloutsos, C., & Kitagawa, H. (2015, February).
    OMNI-Prop: Seamless Node Classification on Arbitrary Label Correlation.
    In Twenty-Ninth AAAI Conference on Artificial Intelligence.
    """

    def __init__(self, graph=None, lamb=1.0, max_iter=30):
        super(OMNI, self).__init__(graph, max_iter)
        self.lamb = lamb

    def _build_propagation_matrix(self):
        d = np.asarray(self.graph.sum(axis=0))
        dT = np.asarray(self.graph.sum(axis=1).T)
        Q = (
            (sparse.diags(1.0 / (d + self.lamb), offsets=0).dot(self.graph))
            .dot(sparse.diags(1.0 / (dT + self.lamb), offsets=0).dot(self.graph.T))
            .tolil()
        )
        Q[self.x_] = 0
        return Q

    def _build_base_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max() + 1
        unlabeled = np.setdiff1d(np.arange(n_samples), self.x_)

        dU = np.asarray(self.graph[unlabeled].sum(axis=1).T)
        dT = np.asarray(self.graph.sum(axis=0))
        n_samples = self.graph.shape[0]
        r = sparse.diags(1.0 / (dU + self.lamb), offsets=0).dot(
            self.lamb
            * self.graph[unlabeled]
            .dot(sparse.diags(1.0 / (dT + self.lamb), offsets=0))
            .dot(np.ones(n_samples))
            + self.lamb
        )

        b = np.ones(n_classes) / float(n_classes)

        B = np.zeros((n_samples, n_classes))
        B[unlabeled] = np.outer(r, b)
        B[self.x_, self.y_] = 1
        return B


class CAMLP(Base):
    """Confidence-Aware Modulated Label Propagation (CAMLP) for GBSSL

    Parameters
    ----------
    beta : float > 0 (default = 0.1)
      Define importance between prior and evidence from neighbors
    H : array_like, shape = [n_classes, n_classes]
      Define affinities between labels
      if None, identity matrix is set
    max_iter : float
      maximum number of iterations allowed

    Attributes
    ----------
    x_ : array, shape = [n_samples]
        Input array of node IDs.

    References
    ----------
    Yamaguchi, Y., Faloutsos, C., & Kitagawa, H. (2016, May).
    CAMLP: Confidence-Aware Modulated Label Propagation.
    In SIAM International Conference on Data Mining.
    """

    def __init__(
        self,
        graph: Union[np.ndarray, sparse.spmatrix] = None,
        beta: float = 0.1,
        H: Optional[np.ndarray] = None,
        max_iter: int = 30,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ):
        super(CAMLP, self).__init__(graph, max_iter)
        self.beta = beta
        self.H = H

        rtol_provided = rtol is not None
        atol_provided = atol is not None

        if rtol_provided ^ atol_provided:
            warnings.warn(
                "You need to provide both rtol and atol if you want to use tolerance"
            )
            self.check_tol = False
        elif rtol_provided and atol_provided:
            self.check_tol = True
        else:
            self.check_tol = False

    def _arrange_params(self):
        if self.H is None:
            n_classes = self.y_.max() + 1
            self.H = np.identity(n_classes)
        self.Z = self._build_normalization_term()

    def _propagate(self):
        return self.P_.dot(self.F_).dot(self.H) + self.B_

    def _build_normalization_term(self):
        d = np.asarray(self.graph.sum(axis=1))
        if len(d.shape) == 2:
            d = d.flatten()
        print(self.graph.shape, d.shape)
        return sparse.diags(1.0 / (1.0 + d * self.beta), offsets=0)

    def _build_propagation_matrix(self):
        return self.Z.dot(self.beta * self.graph)

    def _build_base_matrix(
        self,
        prior_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if prior_b is not None:
            assert len(prior_b.shape) == 1, "prior array should be 1d"

        if prior_b is None:
            n_samples = self.graph.shape[0]
            n_classes = self.y_.max() + 1
            B = np.ones((n_samples, n_classes)) / float(n_classes)
            # we don't use this. if we don't get anything, assume uniform prior
            # B[self.x_] = 0
            # B[self.x_, self.y_] = 1
        else:
            # this assumes just two class classification.
            # the negative class is the first column and
            # the positive class is the second
            B = np.stack([1 - prior_b, prior_b], axis=1)

        return self.Z.dot(B)

    def fit(self, x, y):
        raise NotImplementedError(
            "Run `fit_predict_graph`, we don't care about maintaining self"
        )

    def fit_predict_graph(
        self,
        A: Union[np.ndarray, sparse.spmatrix],
        prior_f: np.ndarray,
    ) -> np.ndarray:
        """Fit a graph-based semi-supervised learning model.

        This function is designed specifically for the two class case
        and for the TIP project, assumptions made here are not
        generic.

        Parameters
        ----------
        A : adjacency matrix
        prior_f : the prior scores for the samples

        Returns
        -------
        An array of length y that has the propogated labels
        """
        self.graph = A
        # self.x_ = x
        self.y_ = np.arange(2)

        self._arrange_params()

        self.F_ = self._init_label_matrix()  # [n, n_classes]

        self.P_ = self._build_propagation_matrix()
        self.B_ = self._build_base_matrix(prior_f)

        remaining_iter = self.max_iter
        while remaining_iter > 0:
            F_new = self._propagate()

            if self.check_tol and np.allclose(self.F_, F_new):
                break

            self.F_ = F_new
            remaining_iter -= 1

        return self.F_[:, 1]
