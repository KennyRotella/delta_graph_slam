#include <hdl_graph_slam/line_based_scanmatcher.hpp>

namespace hdl_graph_slam {

pcl::PointIndices::Ptr extractCluster(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr inliers) {

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
  ec.setClusterTolerance (1); // 100cm
  ec.setMinClusterSize (30);
  ec.setMaxClusterSize (25000);
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

pcl::PointCloud<PointT>::ConstPtr line_extraction(const pcl::PointCloud<PointT>::ConstPtr& cloud) {

  pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
  *filtered = *cloud;

  // Get segmentation ready
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices), line_inliers(new pcl::PointIndices);
  pcl::SACSegmentation<PointT> seg;
  pcl::ExtractIndices<PointT> extract;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_LINE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.250f);

  // Create pointcloud to publish inliers
  pcl::PointCloud<PointT>::Ptr cloud_pub(new pcl::PointCloud<PointT>), cloud_line(new pcl::PointCloud<PointT>);
  int original_size(filtered->height*filtered->width);
  while (filtered->height*filtered->width > original_size*0.05){

    // Fit a line
    seg.setInputCloud(filtered);
    seg.segment(*line_inliers, *coefficients);

    Eigen::Vector3f vt_line(coefficients->values[0], coefficients->values[1], 0.f);
    Eigen::Vector3f vt_direction(coefficients->values[3], coefficients->values[4], 0.f);

    inliers = extractCluster(filtered, line_inliers);

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
      Eigen::Vector3f vt_projected = vt_line + vt_direction.normalized()*((vt - vt_line).dot(vt_direction.normalized()));
      double d = (vt-vt_projected).norm()*1000; // mm
      err.push_back(d);

      // Update statistics
      mean_error += d;
      if (d>max_error) max_error = d;
      if (d<min_error) min_error = d;

    }
    mean_error/=inliers->indices.size();

    // Compute Standard deviation
    double sigma(0);
    PointT pt_A, pt_B;
    pt_A = filtered->points[inliers->indices[0]];
    pt_B = filtered->points[inliers->indices[0]];

    for (int i=0;i<inliers->indices.size();i++){

      sigma += pow(err[i] - mean_error,2);

      // Get Point
      PointT pt = filtered->points[inliers->indices[i]];
      cloud_pub->points.push_back(pt);

      Eigen::Vector3f vt = pt.getVector3fMap();
      Eigen::Vector3f vt_A = pt_A.getVector3fMap();
      Eigen::Vector3f vt_B = pt_B.getVector3fMap();
      if((vt - vt_line).dot(vt_direction) < (vt_A - vt_line).dot(vt_direction)){
        pt_A = pt;
      }

      if((vt - vt_line).dot(vt_direction) > (vt_B - vt_line).dot(vt_direction)){
        pt_B = pt;
      }

    }
    sigma = sqrt(sigma/inliers->indices.size());

    Eigen::Vector3f vt_A = pt_A.getVector3fMap();
    Eigen::Vector3f vt_B = pt_B.getVector3fMap();
    vt_A = vt_line + vt_direction.normalized()*((vt_A - vt_line).dot(vt_direction.normalized()));
    vt_B = vt_line + vt_direction.normalized()*((vt_B - vt_line).dot(vt_direction.normalized()));

    pt_A.x = vt_A.x();
    pt_A.y = vt_A.y();
    pt_B.x = vt_B.x();
    pt_B.y = vt_B.y();

    // Extract inliers
    extract.setInputCloud(filtered);
    extract.setIndices(inliers);
    extract.setNegative(true);
    pcl::PointCloud<PointT> cloudF;
    extract.filter(cloudF);
    filtered->swap(cloudF);

    if(mean_error < 150 && (vt_A-vt_B).norm() > 2.5){
      *cloud_line += *interpolate(pt_A,pt_B);
    }
  }

  cloud_line->header = cloud->header;
  return cloud_line;
}

}  // namespace hdl_graph_slam