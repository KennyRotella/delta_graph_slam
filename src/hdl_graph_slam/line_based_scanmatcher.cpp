#include <hdl_graph_slam/line_based_scanmatcher.hpp>

namespace hdl_graph_slam {

Eigen::Matrix4f LineBasedScanmatcher::align(pcl::PointCloud<PointT>::Ptr inputSource, pcl::PointCloud<PointT>::Ptr inputTarget) {
  //TODO: To Be Developed
  return Eigen::Matrix4f::Identity();
}

pcl::PointIndices::Ptr LineBasedScanmatcher::extractCluster(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr inliers) {

  pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>);
  for (int i=0; i<inliers->indices.size(); i++){
    PointT pt = cloud->points[inliers->indices[i]];
    cloud_plane->points.push_back(pt);
  }

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
  tree->setInputCloud (cloud_plane);

  std::vector<pcl::PointIndices> clusters;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance (cluster_tolerance);
  ec.setMinClusterSize (min_cluster_size);
  ec.setMaxClusterSize (max_cluster_size);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_plane);
  ec.extract (clusters);

  if(clusters.size() > 0){
    pcl::PointIndices::Ptr cluster_inliers(new pcl::PointIndices);
    cluster_inliers->header = inliers->header;
    for(int i=0; i<clusters[0].indices.size(); i++){
      int cloud_idx = clusters[0].indices[i];
      cluster_inliers->indices.push_back(inliers->indices[cloud_idx]);
    }
    return cluster_inliers;
  } else {
    inliers->indices.clear();
  }

  return inliers;
}

std::vector<LineFeature> LineBasedScanmatcher::line_extraction(const pcl::PointCloud<PointT>::ConstPtr& cloud) {

  pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
  *filtered = *cloud;

  // Get segmentation ready
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ExtractIndices<PointT> extract;
  pcl::SACSegmentation<PointT> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_LINE);
  seg.setMethodType(sac_method_type);
  seg.setDistanceThreshold(sac_distance_threshold);

  std::vector<LineFeature> lines;

  while(true){

    // Fit a line
    seg.setInputCloud(filtered);
    seg.segment(*inliers, *coefficients);

    Eigen::Vector3f vt_line(coefficients->values[0], coefficients->values[1], 0.f);
    Eigen::Vector3f vt_direction(coefficients->values[3], coefficients->values[4], 0.f);
    // All projections will not be scaled
    vt_direction.normalize();

    inliers = extractCluster(filtered, inliers);

    // Check result
    if (!inliers || inliers->indices.size() == 0)
      break;

    // Iterate inliers
    double mean_error(0);
    double max_error(0);
    double min_error(100000);
    std::vector<double> err;
    for (int i=0;i<inliers->indices.size();i++){

      // Get Point
      PointT pt = filtered->points[inliers->indices[i]];

      // Compute distance
      Eigen::Vector3f vt = pt.getVector3fMap();
      Eigen::Vector3f vt_projected = vt_line + vt_direction*((vt - vt_line).dot(vt_direction));
      double d = (vt-vt_projected).norm()*1000; // mm
      err.push_back(d);

      // Update statistics
      mean_error += d;
      if (d>max_error) max_error = d;
      if (d<min_error) min_error = d;

    }
    mean_error /= inliers->indices.size();

    // Compute Standard deviation
    double sigma(0);

    // Initialize line segment end points
    Eigen::Vector3f vt_A, vt_B;
    vt_A = filtered->points[inliers->indices[0]].getVector3fMap();
    vt_A = vt_line + vt_direction*((vt_A - vt_line).dot(vt_direction));
    vt_B = vt_A;

    for (int i=0;i<inliers->indices.size();i++){

      sigma += pow(err[i] - mean_error,2);

      // Get Point
      PointT pt = filtered->points[inliers->indices[i]];
      Eigen::Vector3f vt = pt.getVector3fMap();

      // Projection of the point on the line
      vt = vt_line + vt_direction*((vt - vt_line).dot(vt_direction));
      if((vt - vt_line).dot(vt_direction) < (vt_A - vt_line).dot(vt_direction)){
        vt_A = vt;
      }

      if((vt - vt_line).dot(vt_direction) > (vt_B - vt_line).dot(vt_direction)){
        vt_B = vt;
      }

    }
    sigma = sqrt(sigma/inliers->indices.size());

    // Extract inliers
    extract.setInputCloud(filtered);
    extract.setIndices(inliers);
    extract.setNegative(true);
    pcl::PointCloud<PointT> cloudF;
    extract.filter(cloudF);
    filtered->swap(cloudF);

    if(mean_error < merror_threshold && (vt_A-vt_B).norm() > line_lenght_threshold){
      lines.push_back({
        vt_A,       // PointA
        vt_B,       // PointB
        mean_error, // mean_error
        sigma,      // std_sigma
        max_error,  // max_error
        min_error   // min_error
      });
    }

  }

  return lines;
}

}  // namespace hdl_graph_slam