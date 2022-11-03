#include <pybind11/eigen.h>
#include <algorithm>

#include "eps_fusions.h"


double median(std::vector<double>& vec)
{
    // Preliminaries
    double result;
    int nth = vec.size() / 2;

    // Partial sort vector
    std::nth_element(vec.begin(), vec.begin() + nth, vec.end());

    // Compute median
    if (vec.size() % 2 == 1) {
        result = vec[nth];
    } else {
        double max = *std::max_element(vec.begin(), vec.begin() + nth - 1);
        result = 0.5 * (max + vec[nth]);
    }

    return result;
}


double partial_median_dist(const Eigen::MatrixXd& X, int start, int stop)
{
    int n = stop - start;
    int n_dists = (n * (n - 1)) >> 1;
    std::vector<double> dists(n_dists);

    int idx = 0;

    for (int i = start; i < stop; i++) {
        for (int j = start; j < i; j++) {
            dists[idx] = (X.col(i) - X.col(j)).norm();
            idx++;
        }
    }

    // Compute median
    double result = median(dists);

    return result;
}


double median_dist(const Eigen::MatrixXd& X)
{
    // Preliminaries
    int n_parts = 1;
    int n = X.cols();
    double result;

    if (n > 2000) {
        n_parts = (n + 2000) / 2000;
    }

    if (n_parts == 1) {
        result = partial_median_dist(X, 0, n);
    } else {
        // Medians of the parts
        std::vector<double> medians(n_parts);

        // Size of the parts
        int n_i = n / n_parts + 1;

        // Compute medians
        for (int i = 0; i < n_parts; i++) {
            int start = i * n_i;
            int stop = std::min((i + 1) * n_i, n);
            medians[i] = partial_median_dist(X, start, stop);
        }

        result = median(medians);
    }

    return result;
}


double fusion_threshold(const Eigen::MatrixXd X, const double tau)
{
    double result = tau * median_dist(X);

    return result;
}
