#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "weights.h"


pybind11::dict sparse_weights(const Eigen::MatrixXd& X,
                              const Eigen::MatrixXi& indices,
                              const Eigen::MatrixXd& distances,
                              const double phi,
                              const int k,
                              const bool connected,
                              const bool scale)
{
    // Preliminaries
    int n = int(X.cols());

    // Array of keys and values
    Eigen::ArrayXXi keys;
    Eigen::ArrayXd values;
    if (connected) {
        keys = Eigen::ArrayXXi(2, 2 * (k + 2) * n);
        values = Eigen::ArrayXd(2 * (k + 2) * n);
    } else {
        keys = Eigen::ArrayXXi(2, 2 * (k + 1) * n);
        values = Eigen::ArrayXd(2 * (k + 1) * n);
    }

    // Fill keys
    int key_count = 0;
    for (int i = 0; i < indices.cols(); i++) {
        for (int j = 0; j < indices.rows(); j++) {
            if (i != indices(j, i)) {
                keys(0, key_count) = i;
                keys(1, key_count) = indices(j, i);
                keys(0, key_count + 1) = indices(j, i);
                keys(1, key_count + 1) = i;

                values(key_count) = distances(j, i);
                values(key_count + 1) = distances(j, i);

                key_count += 2;
            }
        }
    }

    // Apply connectedness
    if (connected) {
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            double d_ij = (X.col(i) - X.col(j)).norm();

            keys(0, key_count) = i;
            keys(1, key_count) = j;
            keys(0, key_count + 1) = j;
            keys(1, key_count + 1) = i;

            values(key_count) = d_ij;
            values(key_count + 1) = d_ij;

            key_count += 2;
        }
    }

    // Trim unused key/value pairs
    keys.conservativeResize(2, key_count);
    values.conservativeResize(key_count);

    // Compute mean squared distance
    double msd = 0;
    if (scale) {
        for (int j = 0; j < n; j++) {
            for (int i = j + 1; i < n; i++) {
                msd += (X.col(j) - X.col(i)).squaredNorm();
            }
        }
        msd /= (n * (n - 1) / 2);
    }

    // Compute weights
    values = values.square();
    if (scale) {
        values /= msd;
    }
    values = Eigen::exp(-phi * values);

    // Return result as a dictionary with all relevant variables
    pybind11::dict res;
    res["keys"] = keys;
    res["values"] = values;

    return res;
}
