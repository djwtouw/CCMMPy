#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


Eigen::MatrixXi find_mst(const Eigen::MatrixXd& G);

Eigen::VectorXi find_subgraphs(const Eigen::MatrixXi& E, int n);
