#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


pybind11::dict
convex_clusterpath(const Eigen::MatrixXd& X,
                   const Eigen::Matrix<int, 2, Eigen::Dynamic>& W_idx,
                   const Eigen::VectorXd& W_val,
                   const Eigen::VectorXd& lambdas,
                   const double eps_conv,
                   const double eps_fusions,
                   const bool scale,
                   const bool save_clusterpath,
                   const int burnin_iter,
                   const int max_iter_conv);


pybind11::dict
convex_clustering(const Eigen::MatrixXd& X,
                  const Eigen::Matrix<int, 2, Eigen::Dynamic>& W_idx,
                  const Eigen::VectorXd& W_val,
                  const double eps_conv,
                  const double eps_fusions,
                  const bool scale,
                  const bool save_clusterpath,
                  const int burnin_iter,
                  const int max_iter_conv,
                  const int target_low,
                  const int target_high,
                  const int max_iter_phase_1,
                  const int max_iter_phase_2,
                  const int verbose,
                  const double lambda_init,
                  const double factor);
