#ifndef LINE_BASED_SCANMATCHER_HPP
#define LINE_BASED_SCANMATCHER_HPP

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
#include <hdl_graph_slam/building.hpp>

namespace hdl_graph_slam {

/**
 * @brief LineFeature extracted feature from pointclouds
 */
struct LineFeature {
  using Ptr = std::shared_ptr<LineFeature>;

  Eigen::Vector3d pointA;
  Eigen::Vector3d pointB;

  // RANSAC line fitting statistics
  double mean_error;
  double std_sigma;
  double max_error;
  double min_error;

  float lenght(){ return (pointA-pointB).norm(); }
  Eigen::Vector3d middlePoint(){ return pointA + (pointB-pointA)/2.f; }
};

struct EdgeFeature {
  using Ptr = std::shared_ptr<EdgeFeature>;

  Eigen::Vector3d edgePoint;
  Eigen::Vector3d pointA;
  Eigen::Vector3d pointB;
};

struct FitnessScore {
  double avg_distance;
  double coverage;
  double coverage_percentage;
};

struct BestFitAlignment {
  std::vector<LineFeature::Ptr> lines;
  Eigen::Matrix4d transformation;
  FitnessScore fitness_score;
};

// forward declaration due to recursive dependencies
class Building;
bool are_buildings_overlapped(std::vector<LineFeature::Ptr> A, Eigen::Vector3d centerA, boost::shared_ptr<Building> B);

class LineBasedScanmatcher {
  typedef pcl::PointXYZ PointT;

  struct NearestNeighbor {
    LineFeature::Ptr nearest_neighbor;
    double distance;
    double coverage;
  };

  public:
  // Base constructor using default values
  LineBasedScanmatcher():
    min_cluster_size(25),
    max_cluster_size(25000),
    cluster_tolerance(1.0),
    sac_method_type(pcl::SAC_RANSAC),
    sac_distance_threshold(0.100f),
    max_iterations(500),
    merror_threshold(150.0),
    line_lenght_threshold(1.0) {}

  // Setter to customize algorithm parameter values
  void setMinClusterSize (int min_cluster_size) {this->min_cluster_size = min_cluster_size;};
  void setMaxClusterSize (int max_cluster_size) {this->max_cluster_size = max_cluster_size;};
  void setClusterTolerance (float cluster_tolerance) {this->cluster_tolerance = cluster_tolerance;};
  void setSACMethodType (int sac_method_type) {this->sac_method_type = sac_method_type;};
  void setSACDistanceThreshold (double sac_distance_threshold) {this->sac_distance_threshold = sac_distance_threshold;};
  void setMax_iterations (float max_iterations) {this->max_iterations = max_iterations;};
  void setMerror_threshold (float merror_threshold) {this->merror_threshold = merror_threshold;};
  void setLine_lenght_threshold (float line_lenght_threshold) {this->line_lenght_threshold = line_lenght_threshold;};
  
  BestFitAlignment align_overlapped_buildings(boost::shared_ptr<Building> A, boost::shared_ptr<Building> B);
  BestFitAlignment align(pcl::PointCloud<PointT>::Ptr inputSource, std::vector<LineFeature::Ptr> linesTarget, bool local_alignment = false, double max_range = std::numeric_limits<double>::max());
  BestFitAlignment align(std::vector<LineFeature::Ptr> linesSource, std::vector<LineFeature::Ptr> linesTarget, bool local_alignment = false, double max_range = std::numeric_limits<double>::max());
  static std::vector<LineFeature::Ptr> transform_lines(std::vector<LineFeature::Ptr> lines, Eigen::Matrix4d transform);

  public:
  int min_cluster_size;
  int max_cluster_size;
  float cluster_tolerance;
  int sac_method_type;            // SAC_SEGMENTATION_METHOD
  float sac_distance_threshold;   // SAC_SEGMENTATION_DISTANCE_THRESHOLD
  int max_iterations;             // max ransac number of iterations
  float merror_threshold;         // max mean error acceptance threshold
  float line_lenght_threshold;    // min line lenght acceptance threshold
  
  double weight(double avg_distance, double coverage_percentage, double transform_distance) const {
    double max_score_distance = 3.5;
    return 0.45 * coverage_percentage - 35 * std::min(max_score_distance, avg_distance) / max_score_distance - 20 * std::min(max_score_distance, transform_distance) / max_score_distance;
  }
  pcl::PointIndices::Ptr extract_cluster(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr inliers);
  std::vector<LineFeature::Ptr> line_extraction(const pcl::PointCloud<PointT>::ConstPtr& cloud);
  std::vector<EdgeFeature::Ptr> edge_extraction(std::vector<LineFeature::Ptr> lines);
  Eigen::Vector3d lines_intersection(LineFeature::Ptr line1, LineFeature::Ptr line2);
  EdgeFeature::Ptr check_edge(LineFeature::Ptr line1, LineFeature::Ptr line2);
  double angle_between_vectors(Eigen::Vector3d A, Eigen::Vector3d B);
  Eigen::Matrix4d align_edges(EdgeFeature::Ptr edge1, EdgeFeature::Ptr edge2, double max_angle=2*M_PI);
  Eigen::Matrix4d align_lines(LineFeature::Ptr line1, LineFeature::Ptr line2);
  double point_to_line_distance(Eigen::Vector3d point, Eigen::Vector3d line_point, Eigen::Vector3d line_direction);
  double point_to_line_distance(Eigen::Vector3d point, LineFeature::Ptr line);
  bool is_point_on_line(Eigen::Vector3d point, LineFeature::Ptr line);
  FitnessScore line_to_line_distance(LineFeature::Ptr line1, LineFeature::Ptr line2);
  FitnessScore calc_fitness_score(std::vector<LineFeature::Ptr> cloud1, std::vector<LineFeature::Ptr> cloud2, double max_range = std::numeric_limits<double>::max());
  NearestNeighbor nearest_neighbor(LineFeature::Ptr line, std::vector<LineFeature::Ptr> cloud);
};

}  // namespace hdl_graph_slam

#endif // LINE_BASED_SCANMATCHER_HPP