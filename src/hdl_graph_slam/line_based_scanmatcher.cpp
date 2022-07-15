#include <hdl_graph_slam/line_based_scanmatcher.hpp>

namespace hdl_graph_slam {

Eigen::Matrix4f LineBasedScanmatcher::align(pcl::PointCloud<PointT>::Ptr inputSource, pcl::PointCloud<PointT>::Ptr inputTarget) {
  //TODO: To Be Developed
  return Eigen::Matrix4f::Identity();
}

pcl::PointIndices::Ptr LineBasedScanmatcher::extract_cluster(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr inliers) {

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

std::vector<LineFeature::Ptr> LineBasedScanmatcher::line_extraction(const pcl::PointCloud<PointT>::ConstPtr& cloud) {

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
  seg.setMaxIterations(max_iterations);

  std::vector<LineFeature::Ptr> lines;

  while(filtered->points.size() > 0){

    // Fit a line
    seg.setInputCloud(filtered);
    seg.segment(*inliers, *coefficients);

    Eigen::Vector3f vt_line(coefficients->values[0], coefficients->values[1], 0.f);
    Eigen::Vector3f vt_direction(coefficients->values[3], coefficients->values[4], 0.f);

    // All projections will not be scaled
    vt_direction.normalize();

    inliers = extract_cluster(filtered, inliers);

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
      double d = point_to_line_distance(pt.getVector3fMap(), vt_line, vt_direction);
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
      LineFeature::Ptr line(new LineFeature());
      *line = {
        vt_A,       // PointA
        vt_B,       // PointB
        mean_error, // mean_error
        sigma,      // std_sigma
        max_error,  // max_error
        min_error   // min_error
      };
      lines.push_back(line);
    }

  }

  return lines;
}

std::vector<EdgeFeature::Ptr> LineBasedScanmatcher::edge_extraction(std::vector<LineFeature::Ptr> lines){

  std::vector<EdgeFeature::Ptr> edges;

  for(int i=0; i<lines.size()-1; i++){
    for(int j=i+1; j<lines.size(); j++){
      EdgeFeature::Ptr edge = check_edge(lines[i], lines[j]);
      if(edge != nullptr){
        edges.push_back(edge);
      }
    }
  }

  return edges;
}

Eigen::Vector3f LineBasedScanmatcher::lines_intersection(LineFeature::Ptr line1, LineFeature::Ptr line2)
{
    // line1 represented as a1x + b1y = c1
    double a1 = line1->pointB.y() - line1->pointA.y();
    double b1 = line1->pointA.x() - line1->pointB.x();
    double c1 = a1*(line1->pointA.x()) + b1*(line1->pointA.y());
 
    // line2 represented as a2x + b2y = c2
    double a2 = line2->pointB.y() - line2->pointA.y();
    double b2 = line2->pointA.x() - line2->pointB.x();
    double c2 = a2*(line2->pointA.x()) + b2*(line2->pointA.y());
 
    double determinant = a1*b2 - a2*b1;
 
    if (determinant == 0)
    {
      ROS_WARN("LineBasedScanmatcher found two parallel lines to intersect!");
    }

    double x = (b2*c1 - b1*c2)/determinant;
    double y = (a1*c2 - a2*c1)/determinant;
    return Eigen::Vector3f(x,y,0);

}

EdgeFeature::Ptr LineBasedScanmatcher::check_edge(LineFeature::Ptr line1, LineFeature::Ptr line2){

  // lines should be almost perpendicular
  double cosine = (line1->pointA - line1->pointB).normalized().dot(
                  (line2->pointA - line2->pointB).normalized());

  if(std::abs(cosine) > 0.5){
    return nullptr;
  }

  EdgeFeature::Ptr edge(nullptr);

  if((line1->pointA - line2->pointA).norm() < 1.0){
    // intersection point
    Eigen::Vector3f edgePoint = lines_intersection(line1, line2);
    edge.reset(new EdgeFeature());
    *edge = {
      edgePoint,
      line1->pointB,
      line2->pointB
    };

  } else if((line1->pointA - line2->pointB).norm() < 1.0){
    // intersection point
    Eigen::Vector3f edgePoint = lines_intersection(line1, line2);
    edge.reset(new EdgeFeature());
    *edge = {
      edgePoint,
      line1->pointB,
      line2->pointA
    };

  } else if((line1->pointB - line2->pointA).norm() < 1.0){
    // intersection point
    Eigen::Vector3f edgePoint = lines_intersection(line1, line2);
    edge.reset(new EdgeFeature());
    *edge = {
      edgePoint,
      line1->pointA,
      line2->pointB
    };

  } else if((line1->pointB - line2->pointB).norm() < 1.0){
    // intersection point
    Eigen::Vector3f edgePoint = lines_intersection(line1, line2);
    edge.reset(new EdgeFeature());
    *edge = {
      edgePoint,
      line1->pointA,
      line2->pointA
    };

  }

  return edge;
}

double LineBasedScanmatcher::angle_between_vectors(Eigen::Vector3f A, Eigen::Vector3f B){

  double dot = A.x()*B.x() + A.y()*B.y(); // dot product between A and B
  double det = A.x()*B.y() - A.y()*B.x(); // determinant
  double angle = std::atan2(det, dot);    // atan2(y, x) or atan2(sin, cos)

  return angle > 0 ?angle :2*M_PI+angle;  // [-π,+π] -> [0,2π]
}

Eigen::Matrix4f LineBasedScanmatcher::align_edges(EdgeFeature::Ptr edge1, EdgeFeature::Ptr edge2){

  Eigen::Vector3f side1A = edge1->pointA-edge1->edgePoint;
  Eigen::Vector3f side1B = edge1->pointB-edge1->edgePoint;
  Eigen::Vector3f side2A = edge2->pointA-edge2->edgePoint;
  Eigen::Vector3f side2B = edge2->pointB-edge2->edgePoint;

  double angle1 = angle_between_vectors(side1A, side2A);
  double angle2 = angle_between_vectors(side1B, side2B);
  double angle3 = angle_between_vectors(side1A, side2B);
  double angle4 = angle_between_vectors(side1B, side2A);

  double angle = 0.0;
  if(std::abs(angle1-angle2) < std::abs(angle3-angle4)){
    angle = (angle1+angle2)/2.0;
  } else {
    angle = (angle3+angle4)/2.0;
  }

  Eigen::Matrix3f rot;
  rot = Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX())
    * Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY())
    * Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ());

  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform.block<3,1>(0,3) = edge2->edgePoint-rot*edge1->edgePoint;
  transform.block<3,3>(0,0) = rot;

  return transform;
}

double LineBasedScanmatcher::point_to_line_distance(Eigen::Vector3f point, Eigen::Vector3f line_point, Eigen::Vector3f line_direction){
  line_direction.normalize();
  Eigen::Vector3f projected_point = line_point + line_direction*((point - line_point).dot(line_direction));
  return (point-projected_point).norm()*1000; // mm
}

double LineBasedScanmatcher::point_to_line_distance(Eigen::Vector3f point, LineFeature::Ptr line){

  Eigen::Vector3f line_point = line->pointA;
  Eigen::Vector3f line_direction = line->pointB - line->pointA;
  line_direction.normalize();

  Eigen::Vector3f projected_point = line_point + line_direction*((point - line_point).dot(line_direction));

  float dot1 = (projected_point - line->pointA).dot(line->pointB - line->pointA);
  float dot2 = (projected_point - line->pointB).dot(line->pointA - line->pointB);

  if(dot1 < 0){
    return (point - line->pointA).norm()*1000;
  }

  if(dot2 < 0){
    return (point - line->pointB).norm()*1000;
  }

  return (point-projected_point).norm()*1000; // mm
}

double LineBasedScanmatcher::line_to_line_distance(LineFeature::Ptr line_src, LineFeature::Ptr line_trg){

  Eigen::Vector3f line_src_point = line_src->pointA;
  Eigen::Vector3f line_src_direction = (line_src->pointB - line_src->pointA).normalized();

  const float sample_step = 0.02;
  double distance = 0;
  int num_points = 0;

  double line_src_lenght = line_src->lenght();
  for(float i=0; i<=line_src_lenght; i=i+sample_step) {
    Eigen::Vector3f point = line_src_point + i*line_src_direction;
    distance += point_to_line_distance(point, line_trg);
    num_points++;
  }

  distance =  distance / num_points;

  return distance;
}

double LineBasedScanmatcher::calc_fitness_score(std::vector<LineFeature::Ptr> cloud1, std::vector<LineFeature::Ptr> cloud2){

}

LineFeature::Ptr LineBasedScanmatcher::nearest_neighbor(LineFeature::Ptr line, std::vector<LineFeature::Ptr> cloud){

  LineFeature::Ptr nearest_line = nullptr;
  double nearest_distance = std::numeric_limits<double>::max();

  for(LineFeature::Ptr cloud_line: cloud){
    double distance = line_to_line_distance(line, cloud_line);
    if(cloud_line != line && distance < nearest_distance){
      nearest_line = cloud_line;
      nearest_distance = distance;
    }
  }

  return nearest_line;
}


}  // namespace hdl_graph_slam