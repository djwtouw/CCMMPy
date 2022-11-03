#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "ccmm.h"
#include "weights.h"
#include "eps_fusions.h"


namespace py = pybind11;

using namespace pybind11::literals;


PYBIND11_MODULE(_ccmmpy, m)
{
    m.def("_convex_clusterpath", &convex_clusterpath, "Test");
    m.def("_convex_clustering", &convex_clustering, "Test");
    m.def("_fusion_threshold", &fusion_threshold, "Test");
    m.def("_sparse_weights", &sparse_weights, "Test");
}
