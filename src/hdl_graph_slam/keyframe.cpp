// SPDX-License-Identifier: BSD-2-Clause

#include <hdl_graph_slam/keyframe.hpp>

#include <boost/filesystem.hpp>

#include <pcl/io/pcd_io.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam2d/vertex_se2.h>

namespace hdl_graph_slam {

KeyFrame::KeyFrame(const ros::Time& stamp, const Eigen::Isometry3d& odom, const Eigen::Isometry2d& odom2D, double accum_distance, const pcl::PointCloud<PointT>::ConstPtr& cloud, const pcl::PointCloud<PointT>::ConstPtr& flat_cloud) : stamp(stamp), odom(odom), odom2D(odom2D), accum_distance(accum_distance), cloud(cloud), flat_cloud(flat_cloud), node(nullptr) {}

KeyFrame::~KeyFrame() {}

long KeyFrame::id() const {
  return node->id();
}

Eigen::Isometry2d KeyFrame::estimate() const {
  return node->estimate().toIsometry();
}

}  // namespace hdl_graph_slam
