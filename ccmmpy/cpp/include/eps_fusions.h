#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


double fusion_threshold(const Eigen::MatrixXd X, const double tau);
