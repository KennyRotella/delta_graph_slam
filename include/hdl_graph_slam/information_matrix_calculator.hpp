// SPDX-License-Identifier: BSD-2-Clause

#ifndef INFORMATION_MATRIX_CALCULATOR_HPP
#define INFORMATION_MATRIX_CALCULATOR_HPP

#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <hdl_graph_slam/line_based_scanmatcher.hpp>

namespace hdl_graph_slam {

class InformationMatrixCalculator {
public:
  using PointT = pcl::PointXYZ;

  InformationMatrixCalculator() {}
  InformationMatrixCalculator(ros::NodeHandle& nh);
  ~InformationMatrixCalculator();

  template<typename ParamServer>
  void load(ParamServer& params) {
    use_const_inf_matrix = params.template param<bool>("use_const_inf_matrix", false);
    const_stddev_x = params.template param<double>("const_stddev_x", 0.5);
    const_stddev_q = params.template param<double>("const_stddev_q", 0.1);

    var_gain_a = params.template param<double>("var_gain_a", 20.0);
    min_stddev_x = params.template param<double>("min_stddev_x", 0.1);
    max_stddev_x = params.template param<double>("max_stddev_x", 5.0);
    min_stddev_q = params.template param<double>("min_stddev_q", 0.05);
    max_stddev_q = params.template param<double>("max_stddev_q", 0.2);
    fitness_score_thresh = params.template param<double>("fitness_score_thresh", 2.5);
  }

  static double calc_fitness_score(const pcl::PointCloud<PointT>::ConstPtr& cloud1, const pcl::PointCloud<PointT>::ConstPtr& cloud2, const Eigen::Isometry3d& relpose, double max_range = std::numeric_limits<double>::max());

  Eigen::MatrixXd calc_information_matrix(const pcl::PointCloud<PointT>::ConstPtr& cloud1, const pcl::PointCloud<PointT>::ConstPtr& cloud2, const Eigen::Isometry3d& relpose) const;
  Eigen::MatrixXd calc_information_matrix_buildings_global(double fitness_score) const;
  Eigen::MatrixXd calc_information_matrix_buildings_local(BestFitAlignment fitness_score) const;
  void print_parameters();

private:
  double weight(double a, double max_x, double min_y, double max_y, double x) const {
    double y = (1.0 - std::exp(-a * x)) / (1.0 - std::exp(-a * max_x));
    return min_y + (max_y - min_y) * y;
  }

  double b_weight(double a, double avg_x, double min_y, double max_y, double x) const {
    double y = std::exp(a * (x - avg_x)) / (std::exp(a * (x - avg_x)) + 1.0);
    return min_y + (max_y - min_y) * y;
  }

private:
  bool use_const_inf_matrix;
  double const_stddev_x;
  double const_stddev_q;

  double var_gain_a;
  double min_stddev_x;
  double max_stddev_x;
  double min_stddev_q;
  double max_stddev_q;
  double fitness_score_thresh;

  double b_var_gain_a;
  double b_min_stddev_x;
  double b_max_stddev_x;
  double b_min_stddev_q;
  double b_max_stddev_q;
  double b_avg_fitness_score;

  double b_importance_ratio_global;
  double b_importance_ratio_local;
};

}  // namespace hdl_graph_slam

#endif  // INFORMATION_MATRIX_CALCULATOR_HPP
