import numpy as np
from sklearn.neighbors import KDTree
from ._ccmmpy import _sparse_weights
from ._input_checks import (_check_array, _check_scalar, _check_boolean,
                            _check_int)


class SparseWeights:
    """A sparse weight matrix class.

    Stores sparse weight matrices used for convex clustering in a dictionary of
    keys format.


    Methods
    -------
    indices()
        Return the indices of the nonzero weights.
    values()
        Return the values of the nonzero weights.
    to_dense()
        Transform the sparse weight matrix into a dense weight matrix.

    """

    def __init__(self, X, k, phi, connected=True, scale=True):
        """Construct a sparse weight matrix.

        Constrcuct a sparse weight matrix in a dictionary-of-keys format. Each
        nonzero weight is computed as exp(-phi * ||x_i - x_j||^2), where the
        squared Euclidean distance may be scaled by the average squared
        Euclidean distance, depending on the argument scale.

        Parameters
        ----------
        X : numpy 2D array
            The n x p data matrix for which a sparse weight matrix should be
            constructed. Assumes that each row of X represents an object in the
            data.
        k : int
            The number of nearest neighbors to use when determining which
            weights should be nonzero.
        phi : float
            Tuning parameter of the Gaussian weights. Input should be a
            nonnegative value.
        connected : bool, optional
            If True, guarantee a connected structure of the weight matrix by
            using a symmetric circulant matrix to add nonzero weights. This
            ensures that groups of observations that would not be connected
            through weights that are based only on the k nearest neighbors are
            (indirectly) connected anyway. The default is True.
        scale : bool, optional
            If True, scale each squared l2-norm by the mean squared l2-norm to
            ensure scale invariance of the weights. The default is True.

        Returns
        -------
        None.

        """
        # Input checks
        _check_array(X, 2, "X")
        _check_int(k, True, "k")
        _check_scalar(phi, False, "phi")
        _check_boolean(connected, "connected")
        _check_boolean(scale, "scale")

        # Preliminaries
        n = X.shape[0]

        # Construct KD tree
        kdt = KDTree(X, leaf_size=30, metric="euclidean")

        # Query for knn
        distances, indices = kdt.query(X, k=k+1, return_distance=True)

        # Transform the indices of the k-nn into a dictionary of keys sparse
        # matrix
        res = _sparse_weights(X.T, indices.T, distances.T, phi, k, connected,
                              scale)
        keys = res["keys"].T
        values = res["values"]

        # Unique keys
        keys, u_idx = np.unique(keys, axis=0, return_index=True)
        values = values[u_idx]

        # Swap the columns of the keys to switch to col-major order
        keys = keys[:, ::-1]

        # Store variables
        self.__keys = keys
        self.__values = values
        self.__n_obs = n
        self.__phi = phi

    @property
    def phi(self):
        """Get or set the value for phi. Setting phi to a new value
        automatically computes the new values for the weights."""
        return self.__phi

    @phi.setter
    def phi(self, val):
        # Input checks
        _check_scalar(val, False, "phi")

        # Set the new value
        self.__phi = val

    def indices(self):
        """Return the indices of the nonzero weights.

        Returns the indices of all nonzero weights, including the ones that are
        equivalent due to symmetry.

        Returns
        -------
        indices : numpy 2D array
            The matrix of indices.

        """
        return self.__keys.copy()

    def values(self):
        """Return the values of the nonzero weights.

        Returns the values of all nonzero weights, value[i] corresponds to the
        pair of objects on row i of the matrix returned by indices().


        Returns
        -------
        values : numpy 1D array
            The vector of weights.

        """
        return np.exp(-self.__phi * self.__values)

    def to_dense(self):
        """Transform the sparse weight matrix into a dense weight matrix.

        Construct the dense version of the weight matrix.


        Returns
        -------
        weights : numpy 2D array
            Dense weight n x n weight matrix, where n is the number of objects
            in the data on which the weight matrix is based.

        """
        # Intialize dense weight matrix
        result = np.zeros((self.__n_obs, self.__n_obs))

        # Fill nonzero entries of the matrix
        for idx, w in zip(self.__keys, self.__values):
            result[idx[0], idx[1]] = np.exp(-self.__phi * w)

        return result
