import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

from sklearn.decomposition import PCA
from ._ccmmpy import _convex_clusterpath, _fusion_threshold, _convex_clustering
from ._input_checks import (_check_array, _check_scalar, _check_boolean,
                            _check_weights, _check_iterable, _check_string,
                            _check_int, _check_cluster_targets, _check_lambdas)


class NotFittedError(Exception):
    """Class for throwing not fitted errors."""

    pass


class CCMM:
    """Convex clustering through majorization minimization (CCMM).

    Convex clustering is a method that applies shrinkage to cluster centroids
    that represent the objects in the data. More shrinkage is applied if the
    penalty parameter lambda is larger. When cluster centroids are equal, the
    objects they represent belong to the same cluster.

    Methods
    -------
    convex_clustering()
        Perform convex clustering with a target number of clusters.
    convex_clusterpath()
        Minimize the convex clustering loss function.
    plot_clusterpath()
        Plot the clusterpath.
    plot_dendrogram()
        Plot the clusterpath dendrogram.
    scatter()
        Plot a scatterplot.

    """

    def __init__(self, X, weights, tau=1e-3, center=True, scale=True,
                 eps_conv=1e-6, burnin_iter=25, max_iter_conv=5000):
        """Initialize CCMM object.

        Initialize the CCMM solver with the data and weights for which convex
        clustering should be performed.


        Parameters
        ----------
        X : numpy 2D array
            The n x p data matrix for which a sparse weight matrix should be
            constructed. Assumes that each row of X represents an object in the
            data. Should be the same matrix that is used in SparseWeights().
        weights : SparseWeights
            The sparse weights used in the penalty term that applies shrinkage.
        tau : double, optional
            Parameter to compute the threshold to fuse clusters. The default is
            1e-3.
        center : bool, optional
            If True, center X so that each column has mean zero. The default is
            True.
        scale : bool, optional
            If True, scale the loss function to ensure that the cluster
            solution is invariant to the scale of X. Not recommended to set to
            False unless comparing to algorithms that minimize the unscaled
            convex clustering loss function. The default is True.
        eps_conv : double, optional
            Parameter for determining convergence of the minimization. The
            default is 1e-6.
        burnin_iter : int, optional
            Number of updates of the loss function that are done without step
            doubling. The default is 25.
        max_iter_conv : int, optional
            Maximum number of iterations for minimizing the loss function. The
            default is 5000.

        Returns
        -------
        None.

        """
        # Input checks
        _check_array(X, 2, "X")
        _check_weights(weights, "weights")
        _check_scalar(tau, True, "tau", upper_bound=1.0)
        _check_boolean(center, "center")
        _check_boolean(scale, "scale")
        _check_scalar(eps_conv, True, "eps_conv", upper_bound=1.0)
        _check_int(burnin_iter, False, "burnin_iter")
        _check_int(max_iter_conv, False, "max_iter_conv")

        # Input for the algorithm
        self.__X = X.copy().T
        self.__W_idx = weights.indices().T
        self.__W_val = weights.values()
        self.__eps_fusions = _fusion_threshold(X.T, tau)
        self.__scale = scale
        self.__eps_conv = eps_conv
        self.__burnin_iter = burnin_iter
        self.__max_iter_conv = max_iter_conv

        # Set the mean of each variable to zero
        if center:
            rowmeans = self.__X.mean(axis=1)
            for row, m in zip(self.__X, rowmeans):
                row -= m

        self.__clustering = False
        self.__clusterpath = False

    def convex_clustering(self, target_low, target_high, max_iter_phase_1=2000,
                          max_iter_phase_2=20, verbose=0, lambda_init=0.01,
                          factor=0.025, save_clusterpath=False):
        """Perform convex clustering with a target number of clusters.

        Automatically searches for a result with the specified number of
        clusters. The penalty parameter lambda is incremented in small steps
        until the number of clusters is equal or smaller than target_high, then
        a refinement stage is entered in which each number of clusters between
        target_low and target_high is attempted to be attained. If an increment
        in lambda causes the number of clusters to go down by more than one, a
        smaller increment is chosen. It is recommended to specify a range
        around the desired number of clusters, as not each number of clusters
        between 1 and n may be attainable due to numerical inaccuracies.

        Parameters
        ----------
        target_low : int
            Lower bound on the number of clusters that should be searched for.
        target_high : int
            Upper bound on the number of clusters that should be searched for.
            If target_high is equal to target_low, only a single clustering of
            the data is searched for. It is always recommended to specify a
            range larger than 1.
        max_iter_phase_1 : int, optional
            Maximum number of iterations to find an upper and lower bound for
            the value for lambda for which the desired number of clusters is
            attained. The default is 2000.
        max_iter_phase_2 : int, optional
            Maximum number of iterations to to refine the upper and lower
            bounds for lambda. The default is 20.
        lambda_init : float, optional
            The first value for lambda other than 0 to use for convex
            clustering. The default is 0.01.
        factor : float, optional
            The percentage by which to increase lambda in each step. The
            default is 0.025.
        save_clusterpath : bool, optional
            If True, store the solution that minimized the loss function for
            each lambda. Is required for drawing the clusterpath. To store the
            clusterpath coordinates, n x p x lambdas.size values have to be
            stored, this may require too much memory for large data sets. The
            default is False.

        Returns
        -------
        self : CCMM
            A reference to the object itself.

        """
        # Input checks
        _check_int(target_high, True, "target_high")
        _check_int(target_low, True, "target_low")
        _check_cluster_targets(target_low, target_high, self.__X.shape[1])
        _check_int(max_iter_phase_1, False, "max_iter_phase_1")
        _check_int(max_iter_phase_2, False, "max_iter_phase_2")
        _check_scalar(lambda_init, True, "lambda_init")
        _check_scalar(factor, True, "factor")
        _check_boolean(save_clusterpath, "save_clusterpath")

        # Compute the clusterpath
        t1 = time.perf_counter_ns()
        clust = _convex_clustering(self.__X, self.__W_idx, self.__W_val,
                                   self.__eps_conv, self.__eps_fusions,
                                   self.__scale, save_clusterpath,
                                   self.__burnin_iter, self.__max_iter_conv,
                                   target_low, target_high, max_iter_phase_1,
                                   max_iter_phase_2, verbose, lambda_init,
                                   factor)
        t2 = time.perf_counter_ns()

        # Store whether clusterpath is saved
        self.__save_clusterpath = save_clusterpath

        # General information about each instance
        info = pd.DataFrame(columns=["lambda", "clusters", "loss"])
        info["lambda"] = clust["info_d"][0, :]
        info["clusters"] = clust["info_i"][1, :]
        info["loss"] = clust["info_d"][1, :]
        self.info = info.iloc[:clust["targets_found"], :]

        # Vector of possible cluster counts
        self.num_clusters = np.unique(self.info["clusters"].values)

        # Elapsed time
        self.elapsed_time = (t2 - t1) / 1e9

        # Clusterpath coordinates
        self.coordinates = clust["clusterpath"].T

        # Variables for cluster membership and dendrogram
        self.merge = clust["merge"].T
        self.height = clust["height"]

        # Number of instances solved
        self.phase_1_instances = clust["phase_1_instances"]
        self.phase_2_instances = clust["phase_2_instances"]

        # Determine order of the observations for a dendrogram
        # Start with an entry in a hashmap for each observation
        d = dict()
        for i in range(self.__X.shape[1]):
            d[-(i + 1)] = [i]

        # Work through the merge table to make sure that everything that is
        # merged is next to each other
        for i, entry in enumerate(self.merge):
            d[i + 1] = d.pop(entry[0]) + d.pop(entry[1])

        # Finally, create a list with the order of the observations
        self.order = []
        keys = d.keys()
        for key in keys:
            self.order += d[key]

        # Store which type of convexing clustering is used last
        self.__clustering = True
        self.__clusterpath = False

        return self

    def convex_clusterpath(self, lambdas, save_clusterpath=True):
        """Minimize the convex clustering loss function.

        Minimizes the convex clustering loss function for a given set of values
        for lambda


        Parameters
        ----------
        lambdas : numpy 1D array
            A vector containing the values for the penalty parameter.
        save_clusterpath : bool, optional
            If True, store the solution that minimized the loss function for
            each lambda. Is required for drawing the clusterpath. To store the
            clusterpath coordinates, n x p x lambdas.size values have to be
            stored, this may require too much memory for large data sets. The
            default is False.

        Returns
        -------
        self : CCMM
            A reference to the object itself.

        """
        # Input checks
        _check_lambdas(lambdas)
        _check_boolean(save_clusterpath, "save_clusterpath")

        # Store whether clusterpath is saved
        self.__save_clusterpath = save_clusterpath

        # Compute the clusterpath
        t1 = time.perf_counter_ns()
        cpath = _convex_clusterpath(self.__X, self.__W_idx, self.__W_val,
                                    lambdas, self.__eps_conv,
                                    self.__eps_fusions, self.__scale,
                                    save_clusterpath, self.__burnin_iter,
                                    self.__max_iter_conv)
        t2 = time.perf_counter_ns()

        # General information about each instance
        info = pd.DataFrame(columns=["lambda", "iterations", "clusters",
                                     "loss"])
        info["lambda"] = cpath["info_d"][0, :]
        info["iterations"] = cpath["info_i"][0, :]
        info["clusters"] = cpath["info_i"][1, :]
        info["loss"] = cpath["info_d"][1, :]
        self.info = info

        # Vector of possible cluster counts
        self.num_clusters = np.unique(self.info["clusters"].values)

        # Elapsed time
        self.elapsed_time = (t2 - t1) / 1e9

        # Clusterpath coordinates
        self.coordinates = cpath["clusterpath"].T

        # Variables for cluster membership and dendrogram
        self.merge = cpath["merge"].T
        self.height = cpath["height"]

        # Determine order of the observations for a dendrogram
        # Start with an entry in a hashmap for each observation
        d = dict()
        for i in range(self.__X.shape[1]):
            d[-(i + 1)] = [i]

        # Work through the merge table to make sure that everything that is
        # merged is next to each other
        for i, entry in enumerate(self.merge):
            d[i + 1] = d.pop(entry[0]) + d.pop(entry[1])

        # Finally, create a list with the order of the observations
        self.order = []
        keys = d.keys()
        for key in keys:
            self.order += d[key]

        # Add the lambdas for which the clusterpath was computed
        self.lambdas = self.info["lambda"].values

        # Store which type of convexing clustering is used last
        self.__clustering = False
        self.__clusterpath = True

        return self

    def plot_clusterpath(self, n_clusters=None, cluster_colors=None,
                         color_palette="husl", draw_nz_weights=False,
                         default_labels=False):
        """Plot the clusterpath.

        Plots the clusterpath that resulted from minimizing the convex
        clustering loss function for a sequence of values for lambda.

        Can only be called if convex_clusterpath() was called with
        save_clusterpath=True. If a call was made to convex_clustering()
        after a call to convex_clusterpath(), convex_clusterpath() must be
        called again (with save_clusterpath=True) as a call to either method
        overwrites all previous results.

        If the data for which a clusterpath was computed had one variable, a
        column of zeroes is added to create a 2D plot. If there were more than
        two variables, dimension reduction is applied by means of PCA.

        Parameters
        ----------
        n_clusters : int, optional
            The number of clusters the coloring of the individual observations
            is based on. The default is None, which gives all observations the
            same color.
        cluster_colors : iterable object, optional
            The colors used for each of the clusters. The cluster label
            (ranging from 0 to n_clusters-1) is used to index the colors. The
            default is None, which means that a seaborn color palette is used.
        color_palette : str, optional
            If cluster_colors is None, the seaborn color palette that should be
            used to assign colors to the different clusters. The default is
            "husl".
        draw_nz_weights : bool, optional
            If True, the nonzero weights that were used to compute the
            clusterpath are drawn as dashed lines between the observations. The
            default is False.
        default_labels : bool, optional
            If True, the observations will be labeled based on their index+1 in
            the data matrix X. The default is False.

        Returns
        -------
        None.

        """
        if not (self.__clusterpath and self.__save_clusterpath):
            raise NotFittedError(
                "This CCMM instance is not fitted yet. Call "
                "convex_clusterpath() with save_clusterpath=True."
            )

        # Input checks
        if n_clusters is not None:
            _check_int(n_clusters, True, "n_clusters")
        if cluster_colors is not None:
            _check_iterable(cluster_colors, "cluster_colors")
        _check_string(color_palette, "color_palette")
        _check_boolean(draw_nz_weights, "draw_nz_weights")
        _check_boolean(default_labels, "default_labels")

        # Arguments for plotting
        path_kwargs = {"c": "grey", "linewidth": 0.75, "zorder": 0,
                       "alpha": 0.75}
        edge_kwargs = {"c": "grey", "linewidth": 0.50, "zorder": -1,
                       "alpha": 0.35, "ls": "--"}
        point_kwargs = {"c": "cornflowerblue", "s": 10, "zorder": 1}
        label_kwargs = {"ha": "center", "va": "bottom", "fontsize": 8,
                        "zorder": 2}

        # Use the clusters to color the observations
        if n_clusters is not None:
            # Check default colors
            if cluster_colors is None:
                cluster_colors = sns.color_palette(color_palette,
                                                   n_colors=n_clusters)

            # Assign each observation its color based on the cluster it belongs
            # to
            colors = self.clusters(n_clusters)
            colors = [cluster_colors[c] for c in colors]
            point_kwargs["c"] = colors

        # If the number of variables is larger than two, use PCA for dimension
        # reduction. If the number of variables is exactly one, add a column
        # of zeros
        if self.coordinates.shape[1] > 2:
            pca = PCA(n_components=2).fit(self.coordinates.T)
            coordinates = pca.components_.T
        elif self.coordinates.shape[1] == 1:
            coordinates = np.c_[self.coordinates,
                                np.ones((self.coordinates.shape[0], 1))]
        else:
            coordinates = self.coordinates.copy()

        # Draw the paths that the cluster centroids follow as grey lines
        # Use the merge table to find points for which the path is drawn
        # partially
        merge_draw = self.merge.copy()
        for i, entry in enumerate(merge_draw):
            if entry[0] < 0:
                entry[0] = abs(entry[0])
            else:
                entry[0] = merge_draw[entry[0] - 1, 0]

            if entry[1] < 0:
                entry[1] = abs(entry[1])
            else:
                entry[1] = merge_draw[entry[1] - 1, 0]
        merge_draw = merge_draw[:, 1] - 1

        # Remaining points for which the full path is drawn
        n = self.__X.shape[1]
        full_draw = set([i for i in range(n)])
        full_draw = np.array(list(full_draw - set(merge_draw)))

        # Draw the partial paths for the points selected from the merge table
        for draw_i, lambda_i in zip(merge_draw, self.height):
            # Length of the segment
            n_points = (self.lambdas <= lambda_i).sum()

            # Select relevant points
            draw_idx = [draw_i + n * j for j in range(n_points)]
            points = coordinates[draw_idx, :]
            plt.plot(points[:, 0], points[:, 1], **path_kwargs)

        # Draw the full paths for the remaining points
        n_points = self.lambdas.size
        for draw_i in full_draw:
            # Select relevant points
            draw_idx = [draw_i + n * j for j in range(n_points)]
            points = coordinates[draw_idx, :]
            plt.plot(points[:, 0], points[:, 1], **path_kwargs)

        # Draw the starting points of the clusterpath
        plt.scatter(coordinates[:n, 0], coordinates[:n, 1], **point_kwargs)

        # Draw the eind points of the clusterpath
        point_kwargs["c"] = "black"
        point_kwargs["s"] = point_kwargs["s"] / 2
        for draw_i in full_draw:
            point = coordinates[draw_i + (n_points - 1) * n, :]
            plt.scatter(point[0], point[1], **point_kwargs)

        # Draw the nonzero weights as dashed lines between the observations
        if draw_nz_weights:
            nnz_weights = self.__W_idx.shape[1]
            for i in range(nnz_weights):
                if self.__W_idx[0, i] < self.__W_idx[1, i]:
                    edge_x = coordinates[self.__W_idx[:, i], 0]
                    edge_y = coordinates[self.__W_idx[:, i], 1]
                    plt.plot(edge_x, edge_y, **edge_kwargs)

        # Draw default labels for the observations
        if default_labels:
            # Extend the ylimit of the plot to create room for the labels
            ylim = plt.gca().get_ylim()
            yrange = ylim[1] - ylim[0]
            ylim = (ylim[0], ylim[1] + 0.012 * yrange)
            plt.ylim(ylim)

            # Draw the labels directly above the observations
            for i, row in enumerate(coordinates[:n, :]):
                plt.text(row[0], row[1] + 0.01 * yrange, f"{i + 1}",
                         **label_kwargs)

        # Finish the axes
        plt.xticks([])
        plt.xlabel("$x_1$")
        plt.yticks([])
        plt.ylabel("$x_2$")
        plt.gca().set_aspect("equal")
        plt.show()

        return None

    def plot_dendrogram(self, height_transformation="log"):
        """Plot the clusterpath dendrogram.

        Plots the clusterpath dendrogram that resulted from minimizing the
        convex clustering loss function using convex_clustering() or
        convex_clusterpath(). The height at which cluster fusions occur in the
        dendrogram is determined by the value of the penalty parameter lambda
        at which the fusion actually occurred.


        Parameters
        ----------
        height_transformation : string, optional
            The method to transform the height at which clusters fuse. There
            are three options: ln(lambda + 1), sqrt(lambda), and no
            transformation. The default is "log".

        Returns
        -------
        None.

        """
        if not (self.__clusterpath or self.__clustering):
            raise NotFittedError(
                "This CCMM instance is not fitted yet. Call "
                "convex_clustering() or convex_clusterpath() with the "
                "appropriate arguments."
            )

        # Input checks
        _check_string(height_transformation, "height_transformation")

        # Preliminaries
        n_merges = self.merge.shape[0]
        n = len(self.order)

        # Apply transformation to the height
        if height_transformation == "log":
            height_used = np.log(1 + self.height)
        elif height_transformation == "sqrt":
            height_used = np.sqrt(self.height)
        else:
            height_used = self.height

        # Plot settings
        plot_kwargs = {"linewidth": 1, "c": "black"}

        # Initialize dictionary with horizontal and vertical positions of the
        # lines
        positions = dict()
        for i in range(n):
            positions[-(self.order[i] + 1)] = [i, 0]

        for merge_idx in range(n_merges):
            # Get the merge candidates, the height at which it occurs and the
            # previous height
            entry = self.merge[merge_idx, :]

            # Draw leaves
            for m in entry:
                # Get height and previous height
                h0 = positions[m][1]
                h1 = height_used[merge_idx]

                # Get x-coordinate
                x = positions[m][0]

                # Draw vertical line
                plt.plot([x, x], [h0, h1], **plot_kwargs)

            # Draw horizontal line
            x0 = positions[entry[0]][0]
            x1 = positions[entry[1]][0]
            plt.plot([x0, x1], [h1, h1], **plot_kwargs)

            # Update positions
            x_new = (positions.pop(entry[0])[0] + positions.pop(entry[1])[0])
            positions[merge_idx + 1] = [x_new / 2, h1]

        # Draw leaves for unmerged observations
        for key in positions:
            if key < 0:
                # Get height and previous height
                h0 = 0
                h1 = height_used[0]

                # Get x-coordinate
                x = positions[key][0]

                # Draw vertical line
                plt.plot([x, x], [h0, h1], **plot_kwargs)

        # Draw labels
        for i in range(n):
            plt.text(i, -0.02 * height_used[-1], f"{self.order[i] + 1}",
                     ha="center", va="top")

        # Finish the axes
        plt.xticks([])
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.ylim([0, height_used[-1] * 1.05])

        # Select correct ylabel
        if height_transformation == "log":
            plt.ylabel(r"$\ln(1 + \lambda)$")
        elif height_transformation == "sqrt":
            plt.ylabel(r"$\sqrt{\lambda}$")
        else:
            plt.ylabel(r"$\lambda$")
        plt.show()

        return None

    def scatter(self, n_clusters, cluster_colors=None, color_palette="husl",
                draw_nz_weights=False, default_labels=False):
        """Plot a scatterplot.

        Visualizes a clustering of the data by drawing a 2D scatterplot in
        which the observations are colored based on their cluster label.

        Can only be called after either convex_clusterpath() or
        convex_clustering() has been called.

        If the clustered data had one variable, a column of zeroes is added to
        create a 2D plot. If there were more than two variables, dimension
        reduction is applied by means of PCA.

        Parameters
        ----------
        n_clusters : int
            The number of clusters the coloring of the individual observations
            is based on.
        cluster_colors : iterable object, optional
            The colors used for each of the clusters. The cluster label
            (ranging from 0 to n_clusters-1) is used to index the colors. The
            default is None, which means that a seaborn color palette is used.
        color_palette : str, optional
            If cluster_colors is None, the seaborn color palette that should be
            used to assign colors to the different clusters. The default is
            "husl".
        draw_nz_weights : bool, optional
            If True, the nonzero weights that were used to compute the
            clusterpath are drawn as dashed lines between the observations. The
            default is False.
        default_labels : bool, optional
            If True, the observations will be labeled based on their index+1 in
            the data matrix X. The default is False.

        Returns
        -------
        None.

        """
        if not (self.__clusterpath or self.__clustering):
            raise NotFittedError(
                "This CCMM instance is not fitted yet. Call "
                "convex_clustering() or convex_clusterpath() with the "
                "appropriate arguments."
            )

        # Input checks
        _check_int(n_clusters, True, "n_clusters")
        if cluster_colors is not None:
            _check_iterable(cluster_colors, "cluster_colors")
        _check_string(color_palette, "color_palette")
        _check_boolean(draw_nz_weights, "draw_nz_weights")
        _check_boolean(default_labels, "default_labels")

        # Arguments for plotting
        edge_kwargs = {"c": "grey", "linewidth": 0.50, "zorder": -1,
                       "alpha": 0.35, "ls": "--"}
        point_kwargs = {"c": "cornflowerblue", "s": 10, "zorder": 1}
        label_kwargs = {"ha": "center", "va": "bottom", "fontsize": 8,
                        "zorder": 2}

        # Check default colors
        if cluster_colors is None:
            cluster_colors = sns.color_palette(color_palette,
                                               n_colors=n_clusters)

        # Assign each observation its color based on the cluster it belongs to
        colors = self.clusters(n_clusters)
        colors = [cluster_colors[c] for c in colors]
        point_kwargs["c"] = colors

        # If the number of variables is larger than two, use PCA for dimension
        # reduction. If the number of variables is exactly one, add a column
        # of zeros
        coordinates = self.__X.T
        if coordinates.shape[1] > 2:
            pca = PCA(n_components=2).fit(coordinates.T)
            coordinates = pca.components_.T
        elif coordinates.shape[1] == 1:
            coordinates = np.c_[coordinates,
                                np.ones((coordinates.shape[0], 1))]

        # Draw the starting points of the clusterpath
        plt.scatter(coordinates[:, 0], coordinates[:, 1], **point_kwargs)

        # Draw the nonzero weights as dashed lines between the observations
        if draw_nz_weights:
            nnz_weights = self.__W_idx.shape[1]
            for i in range(nnz_weights):
                if self.__W_idx[0, i] < self.__W_idx[1, i]:
                    edge_x = coordinates[self.__W_idx[:, i], 0]
                    edge_y = coordinates[self.__W_idx[:, i], 1]
                    plt.plot(edge_x, edge_y, **edge_kwargs)

        # Draw default labels for the observations
        if default_labels:
            # Extend the ylimit of the plot to create room for the labels
            ylim = plt.gca().get_ylim()
            yrange = ylim[1] - ylim[0]
            ylim = (ylim[0], ylim[1] + 0.012 * yrange)
            plt.ylim(ylim)

            # Draw the labels directly above the observations
            for i, row in enumerate(coordinates):
                plt.text(row[0], row[1] + 0.01 * yrange, f"{i + 1}",
                         **label_kwargs)

        # Finish the axes
        plt.xticks([])
        plt.xlabel("$x_1$")
        plt.yticks([])
        plt.ylabel("$x_2$")
        plt.gca().set_aspect("equal")
        plt.show()

        return None

    def clusters(self, n_clusters, return_alternative=False):
        """Return a clustering of the data.

        Obtain a vector of labels for the data using the solution of the CCMM
        algorithm.

        It may not be possible to get the desired number of clusters due to
        various reasons. The CCMM algorithm approximates a minimum of the
        convex clustering loss function. As such, if cluster centroids are
        "close enough" they are fused together. This means that sometimes
        multiple clusters are fused when going from the solution for one value
        for lambda to the next. Another reason can be disconnectedness of the
        weight matrix. If SparseWeights() is called with connected=False, there
        may be groups of objects that are not (in)directly connected via
        nonzero weights, which means that the distance between the cluster
        centroids that represent these objects is not penalized.

        Parameters
        ----------
        n_clusters : int
            The number of clusters.
        return_alternative : bool, optional
            If true, this function returns the closest number of clusters that
            is possible to return, from below and above. The lower bound is 0
            if no fewer clusters can be returned, and np.inf if not more
            clusters can be returned. The default is False.

        Returns
        -------
        clusters : numpy 1D array
            A vector of cluster membership information. Observations that
            belong to the same cluster, receive the same label.

        """
        if not (self.__clustering or self.__clusterpath):
            raise NotFittedError(
                "This CCMM instance is not fitted yet. Call "
                "convex_clustering() or convex_clusterpath() with the "
                "appropriate arguments."
            )

        # Input checks
        _check_int(n_clusters, True, "n_clusters")
        _check_boolean(return_alternative, "return_alternative")

        if n_clusters not in set(self.info["clusters"]):
            # Check which number of clusters is attainable
            cluster_counts = np.array(list(set(self.info["clusters"])))
            cluster_counts.sort()
            idx = np.searchsorted(cluster_counts, n_clusters)

            # Print information
            if idx > 0 and idx < cluster_counts.size - 1:
                # Find closest cluster counts
                lb = cluster_counts[idx - 1]
                ub = cluster_counts[idx]
                message = (f"Closest cluster counts: {lb}, {ub}")
            elif idx == 0:
                # Find closest cluster counts
                lb = 0
                ub = cluster_counts[idx]
                message = f"Closest cluster count: {ub}"
            else:
                # Find closest cluster counts
                lb = cluster_counts[idx - 1]
                ub = np.inf
                message = f"Closest cluster count: {lb}"

            # Return bounds around requested number of clusters or raise an
            # exception
            if return_alternative:
                warnings.warn(message)
                return lb, ub
            else:
                message = ("Unable to retrieve a clustering for the requested "
                           "number of clusters. " + message)
                raise Exception(message)

        # Preliminaries
        n = self.__X.shape[1]

        # Start with an entry in a hashmap for each observation
        d = dict()
        for i in range(n):
            d[-(i + 1)] = [i]

        # Work through the merge table to reduce the number of clusters until
        # the desired number is reached
        for i, entry in enumerate(self.merge):
            if len(d) == n_clusters:
                break
            d[i + 1] = d.pop(entry[0]) + d.pop(entry[1])

        # Create cluster labels
        result = np.empty(n, dtype=int)
        for i, key in enumerate(d.keys()):
            result[d[key]] = i

        return result
