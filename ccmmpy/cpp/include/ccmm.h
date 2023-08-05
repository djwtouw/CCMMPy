#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


pybind11::dict
convex_clusterpath(const Eigen::MatrixXd& X,
                   const Eigen::Matrix<int, 2, Eigen::Dynamic>& W_idx,
                   const Eigen::VectorXd& W_val,
                   const Eigen::VectorXd& lambdas,
                   double eps_conv,
                   double eps_fusions,
                   bool scale,
                   bool save_clusterpath,
                   int burnin_iter,
                   int max_iter_conv);


pybind11::dict
convex_clustering(const Eigen::MatrixXd& X,
                  const Eigen::Matrix<int, 2, Eigen::Dynamic>& W_idx,
                  const Eigen::VectorXd& W_val,
                  double eps_conv,
                  double eps_fusions,
                  bool scale,
                  bool save_clusterpath,
                  int burnin_iter,
                  int max_iter_conv,
                  int target_low,
                  int target_high,
                  int max_iter_phase_1,
                  int max_iter_phase_2,
                  int verbose,
                  double lambda_init,
                  double factor);
