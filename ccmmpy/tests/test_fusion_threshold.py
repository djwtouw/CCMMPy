"""Tests for the C++ median used by the fusion threshold.

Mirror of the R package's tests/testthat/test-fusion-threshold.R. The two cases
exercise both branches of the C++ median(): the average of the two middle values
(even number of pairwise distances) and the single middle value (odd number).
"""
import numpy as np

from ccmmpy._ccmmpy import _fusion_threshold


def _pairwise_distances(X):
    # Euclidean distances between all unordered pairs of rows of X.
    n = X.shape[0]
    return np.array([np.linalg.norm(X[i] - X[j])
                     for i in range(n) for j in range(i)])


def test_fusion_threshold_even_count():
    # n = 4 observations -> 6 pairwise distances (even), exercising the
    # average-of-two-middle-values branch of the C++ median.
    X = np.array([[0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 3.0],
                  [4.0, 4.0]])

    # _fusion_threshold expects the data with observations in the columns.
    got = _fusion_threshold(X.T, 1.0)
    want = np.median(_pairwise_distances(X))

    assert abs(got - want) < 1e-8


def test_fusion_threshold_odd_count():
    # n = 3 observations -> 3 pairwise distances (odd), exercising the
    # single-middle-value branch of the C++ median.
    X = np.array([[0.0, 0.0],
                  [2.0, 0.0],
                  [0.0, 5.0]])

    got = _fusion_threshold(X.T, 0.5)
    want = 0.5 * np.median(_pairwise_distances(X))

    assert abs(got - want) < 1e-8
