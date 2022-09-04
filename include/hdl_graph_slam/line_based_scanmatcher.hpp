#ifndef LINE_BASED_SCANMATCHER_HPP
#define LINE_BASED_SCANMATCHER_HPP

#include <ros/ros.h>
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
  double real_avg_distance;
  double avg_distance;
  double coverage;
  double coverage_percentage;
};

struct BestFitAlignment {
  std::vector<LineFeature::Ptr> not_aligned_lines;
  std::vector<LineFeature::Ptr> aligned_lines;
  Eigen::Matrix4d transformation;
  FitnessScore fitness_score;
  bool isEdgeAligned;
};

// forward declaration due to recursive dependencies
class Building;
bool are_buildings_overlapped(std::vector<LineFeature::Ptr> A, Eigen::Vector3d centerA, std::vector<LineFeature::Ptr> B, Eigen::Vector3d centerB);

class LineBasedScanmatcher {
  typedef pcl::PointXYZ PointT;

  struct NearestNeighbor {
    LineFeature::Ptr nearest_neighbor;
    double real_distance;
    double distance;
    double coverage;
  };

  public:
  // Base constructor using default values
  LineBasedScanmatcher():
    // line fitting params
    min_cluster_size(25),
    max_cluster_size(25000),
    cluster_tolerance(1.0),
    sac_method_type(pcl::SAC_RANSAC),
    sac_distance_threshold(0.100f),
    max_iterations(500),
    merror_threshold(150.0),
    line_lenght_threshold(1.0),
    // global fitness score params
    g_avg_distance_weight(0.6),
    g_coverage_weight(1.0),
    g_transform_weight(0.2),
    g_max_score_distance(5.0),
    g_max_score_translation(5.0),
    l_avg_distance_weight(0.6),
    l_coverage_weight(1.0),
    l_transform_weight(0.2),
    l_max_score_distance(5.0),
    l_max_score_translation(5.0) {}

  // Setter to customize algorithm parameter values
  void setMinClusterSize (int min_cluster_size) {this->min_cluster_size = min_cluster_size;};
  void setMaxClusterSize (int max_cluster_size) {this->max_cluster_size = max_cluster_size;};
  void setClusterTolerance (float cluster_tolerance) {this->cluster_tolerance = cluster_tolerance;};
  void setSACMethodType (int sac_method_type) {this->sac_method_type = sac_method_type;};
  void setSACDistanceThreshold (double sac_distance_threshold) {this->sac_distance_threshold = sac_distance_threshold;};
  void setMax_iterations (float max_iterations) {this->max_iterations = max_iterations;};
  void setMerror_threshold (float merror_threshold) {this->merror_threshold = merror_threshold;};
  void setLine_lenght_threshold (float line_lenght_threshold) {this->line_lenght_threshold = line_lenght_threshold;};

  void setGlobal_avg_distance_weight (double g_avg_distance_weight) {this->g_avg_distance_weight = g_avg_distance_weight;};
  void setGlobal_coverage_weight (double g_coverage_weight) {this->g_coverage_weight = g_coverage_weight;};
  void setGlobal_transform_weight (double g_transform_weight) {this->g_transform_weight = g_transform_weight;};
  void setGlobal_max_score_distance (double g_max_score_distance) {this->g_max_score_distance = g_max_score_distance;};
  void setGlobal_max_score_translation (double g_max_score_translation) {this->g_max_score_translation = g_max_score_translation;};
  void setLocal_avg_distance_weight (double g_avg_distance_weight) {this->g_avg_distance_weight = g_avg_distance_weight;};

  void setlocal_avg_distance_weight (double l_avg_distance_weight) {this->l_avg_distance_weight = l_avg_distance_weight;};
  void setLocal_coverage_weight (double l_coverage_weight) {this->l_coverage_weight = l_coverage_weight;};
  void setLocal_transform_weight (double l_transform_weight) {this->l_transform_weight = l_transform_weight;};
  void setLocal_max_score_distance (double l_max_score_distance) {this->l_max_score_distance = l_max_score_distance;};
  void setLocal_max_score_translation (double l_max_score_translation) {this->l_max_score_translation = l_max_score_translation;};
  void print_parameters();
  
  BestFitAlignment align_overlapped_buildings(boost::shared_ptr<Building> A, boost::shared_ptr<Building> B);
  BestFitAlignment align_global(pcl::PointCloud<PointT>::Ptr cloudSource, std::vector<LineFeature::Ptr> linesTarget, double &line_extraction_time, double &matching_time, bool constrain_angle = false, double max_range = std::numeric_limits<double>::max());
  BestFitAlignment align_local(std::vector<LineFeature::Ptr> linesSource, std::vector<LineFeature::Ptr> linesTarget, double &matching_time, double max_range = std::numeric_limits<double>::max());
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

  // Global fitness score
  double g_avg_distance_weight;     // fitness score weight
  double g_coverage_weight;         // fitness score weight
  double g_transform_weight;        // fitness score weight
  double g_max_score_distance;      // fitness score max avg distance
  double g_max_score_translation;   // fitness score max translation distance

  // Local fitness score
  double l_avg_distance_weight;     // fitness score weight
  double l_coverage_weight;         // fitness score weight
  double l_transform_weight;        // fitness score weight
  double l_max_score_distance;      // fitness score max avg distance
  double l_max_score_translation;   // fitness score max translation distance
  
  double weight_global(double avg_distance, double coverage_percentage, double translation_distance) const {
    return 
      - g_avg_distance_weight * (std::min(g_max_score_distance, avg_distance) / g_max_score_distance) * 100.
      + g_coverage_weight * coverage_percentage
      - g_transform_weight * (std::min(g_max_score_translation, translation_distance) / g_max_score_translation) * 100.;
  }
  double weight_local(double avg_distance, double coverage_percentage, double translation_distance) const {
    return 
      - l_avg_distance_weight * (std::min(l_max_score_distance, avg_distance) / l_max_score_distance) * 100.
      + l_coverage_weight * coverage_percentage
      - l_transform_weight * (std::min(l_max_score_translation, translation_distance) / l_max_score_translation) * 100.;
  }
  pcl::PointIndices::Ptr extract_cluster(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr inliers);
  std::vector<LineFeature::Ptr> line_extraction(const pcl::PointCloud<PointT>::ConstPtr& cloud);
  std::vector<EdgeFeature::Ptr> edge_extraction(std::vector<LineFeature::Ptr> lines, bool only_angular_edges = false, double max_dist_angular_edge = 7.0);
  Eigen::Vector3d lines_intersection(LineFeature::Ptr line1, LineFeature::Ptr line2);
  std::vector<EdgeFeature::Ptr> get_edges(LineFeature::Ptr line1, LineFeature::Ptr line2, bool only_angular_edges, double max_dist_angular_edge);
  double angle_between_vectors(Eigen::Vector3d A, Eigen::Vector3d B);
  Eigen::Matrix4d align_edges(EdgeFeature::Ptr edge1, EdgeFeature::Ptr edge2);
  Eigen::Matrix4d align_lines(LineFeature::Ptr line1, LineFeature::Ptr line2);
  double point_to_line_distance(Eigen::Vector3d point, Eigen::Vector3d line_point, Eigen::Vector3d line_direction);
  double point_to_line_distance(Eigen::Vector3d point, LineFeature::Ptr line);
  bool is_point_on_line(Eigen::Vector3d point, LineFeature::Ptr line);
  FitnessScore line_to_line_distance(LineFeature::Ptr line1, LineFeature::Ptr line2);
  FitnessScore calc_fitness_score(std::vector<LineFeature::Ptr> cloud1, std::vector<LineFeature::Ptr> cloud2, bool is_local, double max_range = std::numeric_limits<double>::max());
  std::vector<NearestNeighbor> nearest_neighbor(LineFeature::Ptr line, std::vector<LineFeature::Ptr> cloud);
  LineFeature::Ptr are_lines_aligned(LineFeature::Ptr line1, LineFeature::Ptr line2);
  std::vector<LineFeature::Ptr> merge_lines(std::vector<LineFeature::Ptr> lines);
};

}  // namespace hdl_graph_slam

#endif // LINE_BASED_SCANMATCHER_HPP