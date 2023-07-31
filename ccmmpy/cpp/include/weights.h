#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


pybind11::dict sparse_weights(const Eigen::MatrixXd& X,
                              const Eigen::MatrixXi& indices,
                              const Eigen::MatrixXd& distances,
                              const double phi,
                              const int k,
                              const bool sym_circ,
                              const bool scale);
