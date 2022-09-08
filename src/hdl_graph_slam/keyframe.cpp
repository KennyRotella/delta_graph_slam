// SPDX-License-Identifier: BSD-2-Clause
#include <hdl_graph_slam/keyframe.hpp>

#include <boost/filesystem.hpp>

#include <pcl/io/pcd_io.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam2d/vertex_se2.h>

namespace hdl_graph_slam {

KeyFrame::KeyFrame(const ros::Time& stamp,
  const Eigen::Isometry3d& odom,
  const Eigen::Isometry2d& odom2D,
  const Eigen::Isometry2d& estimated_odom,
  double accum_distance,
  const pcl::PointCloud<PointT>::ConstPtr& cloud,
  const pcl::PointCloud<PointT>::ConstPtr& flat_cloud,
  BestFitAlignment global_alignment,
  std::vector<Building::Ptr> near_buildings) : stamp(stamp), odom(odom), odom2D(odom2D), estimated_odom(estimated_odom), accum_distance(accum_distance), cloud(cloud), flat_cloud(flat_cloud), global_alignment(global_alignment), near_buildings(near_buildings), node(nullptr) {}

KeyFrame::~KeyFrame() {}

long KeyFrame::id() const {
  return node->id();
}

Eigen::Isometry2d KeyFrame::estimate() const {
  return node->estimate().toIsometry();
}

KeyFrameSnapshot::KeyFrameSnapshot(const Eigen::Isometry3d& pose, const pcl::PointCloud<PointT>::ConstPtr& cloud) : pose(pose), cloud(cloud) {}

KeyFrameSnapshot::KeyFrameSnapshot(const KeyFrame::Ptr& key) : pose(transform2Dto3D(key->estimate().matrix().cast<float>()).cast<double>()), cloud(key->flat_cloud) {}

KeyFrameSnapshot::~KeyFrameSnapshot() {}

}  // namespace hdl_graph_slam
