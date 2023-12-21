#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


pybind11::dict sparse_weights(const Eigen::MatrixXd& X,
                              const Eigen::MatrixXi& indices,
                              const Eigen::MatrixXd& distances,
                              double phi,
                              int k,
                              bool sym_circ,
                              bool scale);
