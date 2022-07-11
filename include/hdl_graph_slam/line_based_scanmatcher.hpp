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
  using Ptr = std::shared_ptr<LineFeature>;

  Eigen::Vector3f PointA;
  Eigen::Vector3f PointB;

  // RANSAC line fitting statistics
  double mean_error;
  double std_sigma;
  double max_error;
  double min_error;

  float lenght(){ return (PointA-PointB).norm(); }
  Eigen::Vector3f middlePoint(){ return PointA + (PointB-PointA)/2.f; }
};

struct EdgeFeature {
  using Ptr = std::shared_ptr<EdgeFeature>;

  Eigen::Vector3f edgePoint;
  LineFeature::Ptr lineA;
  LineFeature::Ptr lineB;
};

class LineBasedScanmatcher {

  public:
  // Base constructor using default values
  LineBasedScanmatcher():
    min_cluster_size(30),
    max_cluster_size(25000),
    cluster_tolerance(1.5),
    sac_method_type(pcl::SAC_RANSAC),
    sac_distance_threshold(0.250f),
    merror_threshold(150.0),
    line_lenght_threshold(2.5) {}

  // Setter to customize algorithm parameter values
  void setMinClusterSize (double min_cluster_size) {this->min_cluster_size = min_cluster_size;};
  void setMaxClusterSize (double max_cluster_size) {this->max_cluster_size = max_cluster_size;};
  void setClusterTolerance (double cluster_tolerance) {this->cluster_tolerance = cluster_tolerance;};
  void setSACMethodType (int sac_method_type) {this->sac_method_type = sac_method_type;};
  void setSACDistanceThreshold (double sac_distance_threshold) {this->sac_distance_threshold = sac_distance_threshold;};
  
  Eigen::Matrix4f align(pcl::PointCloud<PointT>::Ptr inputSource, pcl::PointCloud<PointT>::Ptr inputTarget);

  public:
  double min_cluster_size;
  double max_cluster_size;
  double cluster_tolerance;
  int sac_method_type;            // SAC_SEGMENTATION_METHOD
  double sac_distance_threshold;  // SAC_SEGMENTATION_DISTANCE_THRESHOLD
  double merror_threshold;        // max mean error acceptance threshold
  double line_lenght_threshold;   // min line lenght acceptance threshold
  
  pcl::PointIndices::Ptr extractCluster(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr inliers);
  std::vector<LineFeature::Ptr> line_extraction(const pcl::PointCloud<PointT>::ConstPtr& cloud);
  std::vector<EdgeFeature::Ptr> edge_extraction(std::vector<LineFeature::Ptr> lines);
  EdgeFeature::Ptr check_edge(LineFeature::Ptr line1, LineFeature::Ptr line2);
};

}  // namespace hdl_graph_slam

#endif // LINE_BASED_SCANMATCHER_HPP