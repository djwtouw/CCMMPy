import numpy as np
from sklearn.neighbors import KDTree
from ._ccmmpy import _sparse_weights, _find_mst, _find_subgraphs
from ._input_checks import (_check_array, _check_scalar, _check_boolean,
                            _check_int, _check_string_value)


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

    def __init__(self, X, k, phi, connected=True, scale=True,
                 connection_type="SC"):
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
            If True, guarantee a connected structure of the weight matrix
            This ensures that groups of observations that would not be
            connected through weights that are based only on the k nearest
            neighbors are (indirectly) connected anyway. The method is
            determined by the argument connection_type. Default is True.
        scale : bool, optional
            If True, scale each squared l2-norm by the mean squared l2-norm to
            ensure scale invariance of the weights. The default is True.
        connection_type : string, optional
            Determines the method to ensure a connected weight matrix if 
            connected is True. Should be one of ["SC", "MST"]. SC stands for
            the method using a symmetric circulant matrix, connecting objects i
            with objects i+1 (and n with 1). MST stands for minimum spanning
            tree. The graph that results from the nonzero weights determined by
            the k nearest neighbors is divided into c subgraphs and a minimum
            spanning tree algorithm is used to add c-1 nonzero weights to
            ensure that all objects are indirectly connected. Default is "SC".

        Returns
        -------
        None.

        """
        # Input checks
        _check_array(X, 2, "X")
        _check_int(k, True, "k")
        _check_scalar(phi, False, "phi")
        _check_boolean(connected, "connected")
        _check_string_value(connection_type, "connection_type", ["SC", "MST"])
        _check_boolean(scale, "scale")

        # Preliminaries
        n = X.shape[0]

        # Construct KD tree
        self.__kdt = KDTree(X, leaf_size=30, metric="euclidean")

        # Store variables required for the computation of the weights
        self.__phi = phi
        self.__k = k
        self.__connected = connected
        self.__connection_type = connection_type
        self.__scale = scale

        # Store number of observations, used to convert to a dense weight
        # matrix
        self.__n_obs = n

        # Compute keys, squared distances, and values
        self.__compute_weights()

    def __compute_weights(self):
        # Query for knn
        distances, indices = self.__kdt.query(self.__kdt.data, k=self.__k+1,
                                              return_distance=True)

        # Transform the indices of the k-nn into a dictionary of keys sparse
        # matrix
        res = _sparse_weights(
            np.array(self.__kdt.data).T, indices.T, distances.T, self.__phi,
            self.__k, (self.__connection_type == "SC") and self.__connected,
            self.__scale
        )
        keys = res["keys"].T
        sqdists = res["values"]

        if self.__connection_type == "MST" and self.__connected:
            # Use the keys of the sparse weight matrix to find clusters in the
            # data
            ids = _find_subgraphs(keys.T, self.__n_obs)

            # The number of disconnected parts of the graph
            n_clusters = max(ids) + 1

            if n_clusters > 1:
                # Array to keep track of which objects are responsible for
                # those distances
                closest_objects = np.zeros(
                    (n_clusters * (n_clusters - 1) // 2, 2), dtype=int
                )

                # Matrix to keep track of the shortest distance between the
                # clusters
                D_between = np.zeros((n_clusters, n_clusters))

                for a in range(n_clusters):
                    kdt_a = KDTree(np.array(self.__kdt.data)[ids == a, :],
                                   leaf_size=30, metric="euclidean")

                    for b in range(a):
                        # Find out which member of cluster a is closest to each
                        # of the members of cluster b
                        nn_between = kdt_a.query(
                            np.array(self.__kdt.data)[ids == b, :], k=1,
                            return_distance=True
                        )

                        # Get the indices of the objects with respect to their
                        # cluster
                        b_idx = np.argmin(nn_between[0])
                        a_idx = nn_between[1][b_idx, 0]

                        # Save the distance
                        D_between[a, b] = nn_between[0][b_idx]
                        D_between[b, a] = nn_between[0][b_idx]

                        # Get the original indices of the objects
                        a_idx = np.where(ids == a)[0][a_idx]
                        b_idx = np.where(ids == b)[0][b_idx]

                        # Store the objects
                        idx = a * (a - 1) // 2 + b
                        closest_objects[idx, :] = (a_idx, b_idx)

                # Find minimum spanning tree for D_between
                mst_keys = _find_mst(D_between)

                # Array for the weights
                mst_values = np.empty(mst_keys.shape[0])

                # Get the true object indices from the closest_objects list
                for i in range(mst_keys.shape[0]):
                    # Get the index for the closest_objects list
                    ii = min(mst_keys[i, 0], mst_keys[i, 1])
                    jj = max(mst_keys[i, 0], mst_keys[i, 1])
                    idx = jj * (jj - 1) // 2 + ii

                    # Fill in the distances in the values vector
                    mst_values[i] = D_between[mst_keys[i, 0], mst_keys[i, 1]]

                    # Replace the cluster ids by the object ids
                    mst_keys[i, :] = closest_objects[idx, :]

                # Because both the upper and lower part of the weight matrix
                # are stored, the key and value pairs are duplicated
                mst_keys = np.r_[mst_keys, mst_keys[:, ::-1]]
                mst_values = np.r_[mst_values, mst_values]

                # Scale the squared distances
                if self.__scale:
                    mst_values = np.square(mst_values) / res["msd"]

                # Append everything to the existing keys and values, ensuring a
                # connected weight matrix based on a minimum spanning tree
                keys = np.r_[keys, mst_keys]
                sqdists = np.r_[sqdists, mst_values]

        # Unique keys, also sorts
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

    @property
    def connection_type(self):
        """Get or set the value for connection_type. Setting connection_type to
        a new value automatically computes the new values for the weights if
        connected is true."""
        return self.__connection_type

    @connection_type.setter
    def connection_type(self, val):
        # If no change is made, instantly return
        if val == self.__connection_type:
            return

        # Input checks
        _check_string_value(val, "connection_type", ["SC", "MST"])

        # Set the new value
        self.__connection_type = val

        # Compute new keys, squared distances, and values if connected is true
        if self.__connected:
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
