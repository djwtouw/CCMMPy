import numpy as np
from sklearn.neighbors import KDTree
from ._ccmmpy import _sparse_weights
from ._input_checks import (_check_array, _check_scalar, _check_boolean,
                            _check_int)


class SparseWeights:
    """A sparse weight matrix class.

    Stores sparse weight matrices used for convex clustering in a dictionary of
    keys format. For help on the constructor, see help(SparseWeights.__init__).

    Attributes
    ----------
    k : int
        The number of nearest neighbors to use when determining which
        weights should be nonzero.
    phi : float
        Tuning parameter of the Gaussian weights. Input should be a
        nonnegative value.
    connected : bool
        If True, guarantee a connected structure of the weight matrix by
        using a symmetric circulant matrix to add nonzero weights. This
        ensures that groups of observations that would not be connected
        through weights that are based only on the k nearest neighbors are
        (indirectly) connected anyway. The default is True.

    Methods
    -------
    indices() : numpy.ndarray of shape (n_nonzero, 2)
        Return the indices of the nonzero weights.
    values() : numpy.ndarray of shape (n_nonzero,)
        Return the values of the nonzero weights.
    to_dense() : numpy.ndarray (n_samples, n_samples)
        Convert the sparse weight matrix into a dense weight matrix.

    Notes
    -----
    The weight matrix is computed using the Euclidean distance measure and an
    exponentially decaying kernel function: 
        w_ij = exp(-phi * ||x_i - x_j||^2).

    """

    def __init__(self, X, k, phi, connected=True, scale=True):
        """Construct a sparse weight matrix.

        Constrcuct a sparse weight matrix in a dictionary-of-keys format. Each
        nonzero weight is computed as exp(-phi * ||x_i - x_j||^2), where the
        squared Euclidean distance may be scaled by the average squared
        Euclidean distance, depending on the argument scale.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The 2D data matrix for which a sparse weight matrix should be
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
        self.__kdt = KDTree(X, leaf_size=30, metric="euclidean")

        # Store variables required for the computation of the weights
        self.__phi = phi
        self.__k = k
        self.__connected = connected
        self.__scale = scale

        # Compute keys, squared distances, and values
        self.__compute_weights()

        # Store number of observations, used to convert to a dense weight
        # matrix
        self.__n_obs = n

    def __compute_weights(self):
        # Query for knn
        distances, indices = self.__kdt.query(self.__kdt.data, k=self.__k+1,
                                              return_distance=True)

        # Transform the indices of the k-nn into a dictionary of keys sparse
        # matrix
        res = _sparse_weights(np.array(self.__kdt.data).T, indices.T,
                              distances.T, self.__phi, self.__k,
                              self.__connected, self.__scale)
        keys = res["keys"].T
        sqdists = res["values"]

        # Unique keys
        keys, u_idx = np.unique(keys, axis=0, return_index=True)
        sqdists = sqdists[u_idx]

        # Swap the columns of the keys to switch to col-major order
        keys = keys[:, ::-1]

        # Store keys, squared distances, and values
        self.__keys = keys
        self.__sqdists = sqdists
        self.__values = np.exp(-self.__phi * sqdists)

        return None

    @property
    def phi(self):
        """Get or set the value for phi. Setting phi to a new value
        automatically computes the new values for the weights."""
        return self.__phi

    @phi.setter
    def phi(self, val):
        # If no change is made, instantly return
        if val == self.__phi:
            return

        # Input checks
        _check_scalar(val, False, "phi")

        # Set the new value and the new weights
        self.__phi = val
        self.__values = np.exp(-val * self.__sqdists)

    @property
    def k(self):
        """Get or set the value for k. Setting k to a new value automatically
        computes the new values for the weights."""
        return self.__k

    @k.setter
    def k(self, val):
        # If no change is made, instantly return
        if val == self.__k:
            return

        # Input checks
        _check_int(val, True, "k")

        # Set the new value
        self.__k = val

        # Compute new keys, squared distances, and values
        self.__compute_weights()

    @property
    def connected(self):
        """Get or set the value for connected. Setting connected to a new value
        automatically computes the new values for the weights."""
        return self.__connected

    @connected.setter
    def connected(self, val):
        # If no change is made, instantly return
        if val == self.__connected:
            return

        # Input checks
        _check_boolean(val, "connected")

        # Set the new value
        self.__connected = val

        # Compute new keys, squared distances, and values
        self.__compute_weights()

    def indices(self):
        """Return the indices of the nonzero weights.

        Returns the indices of all nonzero weights, including the ones that are
        equivalent due to symmetry.

        Returns
        -------
        indices : numpy.ndarray of shape (n_nonzero, 2)
            The matrix of indices.

        """
        return self.__keys.copy()

    def values(self):
        """Return the values of the nonzero weights.

        Returns the values of all nonzero weights, value[i] corresponds to the
        pair of objects on row i of the matrix returned by indices().


        Returns
        -------
        values : numpy.ndarray of shape (n_nonzero,)
            The vector of weights.

        """
        return self.__values.copy()

    def to_dense(self):
        """Convert the sparse weight matrix into a dense weight matrix.

        Construct the dense version of the weight matrix.


        Returns
        -------
        weights : numpy.ndarray of shape (n_samples, n_samples)
            Dense weight n x n weight matrix, where n is the number of objects
            in the data on which the weight matrix is based.

        """
        # Intialize dense weight matrix
        result = np.zeros((self.__n_obs, self.__n_obs))

        # Fill nonzero entries of the matrix
        for idx, w in zip(self.__keys, self.__values):
            result[idx[0], idx[1]] = np.exp(-self.__phi * w)

        return result
