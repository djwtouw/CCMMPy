"""Regression tests pinning the numeric output of the main user-facing classes.

Mirror of the R package's tests/testthat/test-regression.R, using the exact same
`two_half_moons` data set (shipped as tests/data/two_half_moons.csv). The data is
fully deterministic (the nearest-neighbor search and the optimizer introduce no
randomness), so the reference values below are identical to the R ones and guard
against unintended changes when the internals are modified.
"""
import os

import numpy as np
import pytest

from ccmmpy import CCMM, SparseWeights
from ccmmpy._ccmmpy import _fusion_threshold

tol = 1e-5
tol_lambda = 1e-4

_DATA = os.path.join(os.path.dirname(__file__), "data", "two_half_moons.csv")


def get_X():
    # Same data as the R package's bundled `two_half_moons`. The third column is
    # the (unused) true label; it is dropped to match the R tests' `[, -3]`.
    return np.genfromtxt(_DATA, delimiter=",")[:, :2]


def test_sparse_weights_sc_stable():
    X = get_X()
    W = SparseWeights(X, k=5, phi=8.0)

    assert isinstance(W, SparseWeights)
    assert W.indices().shape[0] == 1204
    assert W.values().sum() == pytest.approx(862.06142430432635, abs=tol)
    # With n = 150 the scaling denominator does not overflow, so all Gaussian
    # weights must stay within (0, 1].
    assert np.all((W.values() > 0) & (W.values() <= 1))
    np.testing.assert_allclose(
        W.values()[:4],
        [8.8720292688782368e-05, 0.85733295288816891,
         0.93169786827839174, 0.90642893685425141],
        atol=tol)


def test_sparse_weights_mst_stable():
    X = get_X()
    W = SparseWeights(X, k=5, phi=8.0, connection_type="MST")

    assert W.indices().shape[0] == 914
    assert W.values().sum() == pytest.approx(839.82276458875947, abs=tol)
    assert np.all((W.values() > 0) & (W.values() <= 1))


def test_to_dense_matches_values():
    # to_dense() must place the computed weights into the matrix unchanged; it
    # should agree with values() exactly (guards the double-exponential bug).
    X = get_X()
    W = SparseWeights(X, k=5, phi=8.0)

    dense = W.to_dense()
    keys = W.indices()
    reconstructed = np.zeros_like(dense)
    reconstructed[keys[:, 0], keys[:, 1]] = W.values()

    np.testing.assert_allclose(dense, reconstructed, atol=1e-12)
    assert np.allclose(dense, dense.T)


def test_convex_clusterpath_stable():
    X = get_X()
    W = SparseWeights(X, k=5, phi=8.0)
    model = CCMM(X, W).convex_clusterpath(np.arange(0, 50.0001, 10),
                                          save_clusterpath=True)

    assert isinstance(model, CCMM)
    assert model.info["clusters"].astype(int).tolist() == [150, 27, 15, 11, 9, 8]
    np.testing.assert_allclose(
        model.info["loss"].to_numpy(),
        [0, 0.046961942259711051, 0.075587177658330923,
         0.098547234068701681, 0.11808814825028721, 0.13604511781387246],
        atol=tol)
    # The fusion threshold the path was computed with (CCMM uses tau=1e-3).
    assert _fusion_threshold(X.T, 1e-3) == pytest.approx(
        0.0011715742205747801, abs=tol)
    assert model.merge.shape[0] == 142
    assert list(model.coordinates.shape) == [900, 2]


def test_convex_clustering_stable():
    X = get_X()
    W = SparseWeights(X, k=5, phi=8.0)
    model = CCMM(X, W).convex_clustering(target_low=2, target_high=5)

    assert isinstance(model, CCMM)
    assert model.info["clusters"].astype(int).tolist() == [5, 4, 3, 2]
    np.testing.assert_allclose(
        model.info["loss"].to_numpy(),
        [0.43836641086492378, 0.44636998233975644,
         0.45759817321919571, 0.49604066148526821],
        atol=tol)
    np.testing.assert_allclose(
        model.info["lambda"].to_numpy(),
        [365.62663322573178, 386.69676619810804,
         421.63549877541982, 633.74427116468769],
        atol=tol_lambda)
    # Two balanced clusters of 75 observations each.
    counts = np.bincount(model.clusters(2))
    assert sorted(counts.tolist()) == [75, 75]
