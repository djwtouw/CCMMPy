#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "ccmm.h"
#include "weights.h"
#include "eps_fusions.h"
#include "graphs.h"


namespace py = pybind11;

using namespace pybind11::literals;


PYBIND11_MODULE(_ccmmpy, m)
{
    m.def("_convex_clusterpath", &convex_clusterpath, "");
    m.def("_convex_clustering", &convex_clustering, "");
    m.def("_fusion_threshold", &fusion_threshold, "");
    m.def("_sparse_weights", &sparse_weights, "");
    m.def("_find_mst", &find_mst, "");
    m.def("_find_subgraphs", &find_subgraphs, "");
}
