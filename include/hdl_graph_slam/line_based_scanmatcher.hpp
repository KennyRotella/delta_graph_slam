#ifndef LINE_BASED_SCANMATCHER_HPP
#define LINE_BASED_SCANMATCHER_HPP

#include <hdl_graph_slam/ros_utils.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

namespace hdl_graph_slam {

/**
 * @brief LineFeature extracted feature from pointclouds
 */
struct LineFeature {
  Eigen::Vector2f PointA;
  Eigen::Vector2f PointB;

  // RANSAC line fitting statistics
  double mean_error;
  double std_sigma;
  double max_error;
  double min_error;

  float lenght(){ return (PointA-PointB).norm(); }
  Eigen::Vector2f middlePoint(){ return PointA + (PointB-PointA)/2.f; }
};

class LineBasedScanmatcher {
  // ec.setClusterTolerance (1); // 100cm
  // ec.setMinClusterSize (30);
  // ec.setMaxClusterSize (25000);
  // seg.setMethodType (pcl::SAC_RANSAC);
  // seg.setDistanceThreshold(0.250f);
  // mean_error < 150 && (vt_A-vt_B).norm() > 2.5

  public:
  Eigen::Matrix4f align(pcl::PointCloud<PointT>::Ptr inputSource, pcl::PointCloud<PointT>::Ptr inputTarget);

  private:
  pcl::PointIndices::Ptr extractCluster(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr inliers);
  std::vector<LineFeature> line_extraction(const pcl::PointCloud<PointT>::ConstPtr& cloud);
};

}  // namespace hdl_graph_slam

#endif // LINE_BASED_SCANMATCHER_HPP