#include <hdl_graph_slam/line_based_scanmatcher.hpp>

namespace hdl_graph_slam {

BestFitAlignment LineBasedScanmatcher::align(pcl::PointCloud<PointT>::Ptr cloudSource, std::vector<LineFeature::Ptr> linesTarget) {
  
  std::vector<LineFeature::Ptr> linesSource = line_extraction(cloudSource);
  return align(linesSource, linesTarget);
}

BestFitAlignment LineBasedScanmatcher::align(std::vector<LineFeature::Ptr> linesSource, std::vector<LineFeature::Ptr> linesTarget, bool local_alignment) {

  double max_range = local_alignment ?3.0 :std::numeric_limits<double>::max();
  double max_distance = local_alignment ?1.0 :3.0;
  double min_cosine = 0.9;

  BestFitAlignment result;
  result.lines = linesSource;
  result.transformation = Eigen::Matrix4d::Identity();
  result.fitness_score = calc_fitness_score(linesSource, linesTarget, max_range);

  std::vector<EdgeFeature::Ptr> edgesSource = edge_extraction(linesSource);
  std::vector<EdgeFeature::Ptr> edgesTarget = edge_extraction(linesTarget);

  std::cout << "START" << std::endl;

  for(EdgeFeature::Ptr edgeSource : edgesSource){
    for(EdgeFeature::Ptr edgeTarget : edgesTarget){

      Eigen::Matrix4d transform;
      if(local_alignment){
        transform = align_edges(edgeSource, edgeTarget, M_PI/9);
      } else {
        transform = align_edges(edgeSource, edgeTarget);
      }

      Eigen::Vector3d traslation = transform.block<3,1>(0,3);

      if(traslation.norm() > max_distance){
        continue;
      }

      double weight = 1 + 0.5 * std::min(max_distance,traslation.norm()) / max_distance;

      std::vector<LineFeature::Ptr> linesSourceTransformed = transform_lines(linesSource, transform);
      double fitness_score = calc_fitness_score(linesSourceTransformed, linesTarget, max_range) * weight;

      if(fitness_score < result.fitness_score){
        result.lines = linesSourceTransformed;
        result.transformation = transform;
        result.fitness_score = fitness_score;

        std::cout << "EDGE FITNESS: " << fitness_score << std::endl;
      }
    }
  }

  for(LineFeature::Ptr lineSource : linesSource){

    NearestNeighbor nn_lineTarget = nearest_neighbor(lineSource, linesTarget);

    Eigen::Vector3d srcLine = (lineSource->pointA - lineSource->pointB).normalized();
    Eigen::Vector3d trgLine = (nn_lineTarget.nearest_neighbor->pointA - nn_lineTarget.nearest_neighbor->pointB).normalized();

    double cosine = srcLine.dot(trgLine);
    if(std::abs(cosine) < min_cosine){
      continue;
    }

    Eigen::Matrix4d transform = align_lines(lineSource, nn_lineTarget.nearest_neighbor);
    Eigen::Vector3d traslation = transform.block<3,1>(0,3);

    if(traslation.norm() > max_distance){
      continue;
    }

    double weight = 1 + 0.3 * std::min(max_distance,traslation.norm()) / max_distance;

    std::vector<LineFeature::Ptr> linesSourceTransformed = transform_lines(linesSource, transform);
    double fitness_score = calc_fitness_score(linesSourceTransformed, linesTarget, max_range);

    if(fitness_score < result.fitness_score){
      result.lines = linesSourceTransformed;
      result.transformation = transform;
      result.fitness_score = fitness_score;

      std::cout << "LINE FITNESS: " << fitness_score << std::endl;
    }
  }

  std::cout << "END" << std::endl;

  return result;
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

  while(filtered->points.size() >= 2){

    // Fit a line
    seg.setInputCloud(filtered);
    seg.segment(*inliers, *coefficients);

    Eigen::Vector3d vt_line(coefficients->values[0], coefficients->values[1], 0.f);
    Eigen::Vector3d vt_direction(coefficients->values[3], coefficients->values[4], 0.f);

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
      double d = point_to_line_distance(pt.getVector3fMap().cast<double>(), vt_line, vt_direction);
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
    Eigen::Vector3d vt_A, vt_B;
    vt_A = filtered->points[inliers->indices[0]].getVector3fMap().cast<double>();
    vt_A = vt_line + vt_direction*((vt_A - vt_line).dot(vt_direction));
    vt_B = vt_A;

    for (int i=0;i<inliers->indices.size();i++){

      sigma += pow(err[i] - mean_error,2);

      // Get Point
      PointT pt = filtered->points[inliers->indices[i]];
      Eigen::Vector3d vt = pt.getVector3fMap().cast<double>();

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

Eigen::Vector3d LineBasedScanmatcher::lines_intersection(LineFeature::Ptr line1, LineFeature::Ptr line2)
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
    return Eigen::Vector3d(x,y,0);

}

EdgeFeature::Ptr LineBasedScanmatcher::check_edge(LineFeature::Ptr line1, LineFeature::Ptr line2){

  // lines should be almost perpendicular
  double cosine = (line1->pointA - line1->pointB).normalized().dot(
                  (line2->pointA - line2->pointB).normalized());

  if(std::abs(cosine) > 0.5){
    return nullptr;
  }

  EdgeFeature::Ptr edge(new EdgeFeature());
  Eigen::Vector3d edgePoint = lines_intersection(line1, line2);
  edge->edgePoint = edgePoint;

  if((line1->pointA - edgePoint).norm() > (line1->pointB - edgePoint).norm()){
    edge->pointA = line1->pointA;
  } else {
    edge->pointA = line1->pointB;
  }
  
  if((line2->pointA - edgePoint).norm() > (line2->pointB - edgePoint).norm()){
    edge->pointB = line2->pointA;
  } else {
    edge->pointB = line2->pointB;
  }

  return edge;
}

double LineBasedScanmatcher::angle_between_vectors(Eigen::Vector3d A, Eigen::Vector3d B){

  double dot = A.x()*B.x() + A.y()*B.y(); // dot product between A and B
  double det = A.x()*B.y() - A.y()*B.x(); // determinant
  double angle = std::atan2(det, dot);    // atan2(y, x) or atan2(sin, cos)

  return angle > 0 ?angle :2*M_PI+angle;  // [-π,+π] -> [0,2π]
}

Eigen::Matrix4d LineBasedScanmatcher::align_edges(EdgeFeature::Ptr edge1, EdgeFeature::Ptr edge2, double max_angle){

  Eigen::Vector3d side1A = edge1->pointA-edge1->edgePoint;
  Eigen::Vector3d side1B = edge1->pointB-edge1->edgePoint;
  Eigen::Vector3d side2A = edge2->pointA-edge2->edgePoint;
  Eigen::Vector3d side2B = edge2->pointB-edge2->edgePoint;

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

  if(angle > max_angle){
    return Eigen::Matrix4d::Identity();
  }

  Eigen::Matrix3d rot;
  rot = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX())
    * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY())
    * Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ());

  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3,1>(0,3) = edge2->edgePoint-rot*edge1->edgePoint;
  transform.block<3,3>(0,0) = rot;

  return transform;
}

Eigen::Matrix4d LineBasedScanmatcher::align_lines(LineFeature::Ptr line1, LineFeature::Ptr line2){

  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  double angle = angle_between_vectors(line1->pointA - line1->pointB, line2->pointA - line2->pointB);

  // [0,2π] -> [-π,+π] 
  angle = angle < M_PI ?angle :angle-2*M_PI;

  // use the smallest angle between two vectors
  if(angle > M_PI / 2){
    angle -= M_PI;
  } else if(angle < -M_PI / 2){
    angle += M_PI;
  }

  Eigen::Vector3d line_point = line2->pointA;
  Eigen::Vector3d line_direction = (line2->pointA - line2->pointB).normalized();
  Eigen::Vector3d projected_point = line_point + line_direction*((line1->pointA - line_point).dot(line_direction));

  Eigen::Matrix3d rot;
  rot = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX())
    * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY())
    * Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ());
  
  transform.block<3,1>(0,3) = projected_point-rot*line1->pointA;
  transform.block<3,3>(0,0) = rot;

  return transform;
}

double LineBasedScanmatcher::point_to_line_distance(Eigen::Vector3d point, Eigen::Vector3d line_point, Eigen::Vector3d line_direction){
  line_direction.normalize();
  Eigen::Vector3d projected_point = line_point + line_direction*((point - line_point).dot(line_direction));
  return (point-projected_point).norm();
}

double LineBasedScanmatcher::point_to_line_distance(Eigen::Vector3d point, LineFeature::Ptr line){

  Eigen::Vector3d line_point = line->pointA;
  Eigen::Vector3d line_direction = line->pointB - line->pointA;
  line_direction.normalize();

  Eigen::Vector3d projected_point = line_point + line_direction*((point - line_point).dot(line_direction));

  double dot1 = (projected_point - line->pointA).dot(line->pointB - line->pointA);
  double dot2 = (projected_point - line->pointB).dot(line->pointA - line->pointB);

  if(dot1 < 0){
    return (point - line->pointA).norm();
  }

  if(dot2 < 0){
    return (point - line->pointB).norm();
  }

  return (point-projected_point).norm(); 
}

double LineBasedScanmatcher::line_to_line_distance(LineFeature::Ptr line_src, LineFeature::Ptr line_trg){

  double distance = 0;

  distance += point_to_line_distance(line_src->pointA, line_trg);
  distance += point_to_line_distance(line_src->pointB, line_trg);

  distance =  distance / 2;

  return distance;
}

double LineBasedScanmatcher::calc_fitness_score(std::vector<LineFeature::Ptr> cloud1, std::vector<LineFeature::Ptr> cloud2, double max_range){

  double distance = 0;
  double total_lenght = 0;

  for(LineFeature::Ptr cloud_line: cloud1){
    NearestNeighbor nn_line = nearest_neighbor(cloud_line, cloud2);
    if(nn_line.distance < max_range){
      distance += nn_line.distance * cloud_line->lenght();
      total_lenght += cloud_line->lenght();
    }
  }

  if(total_lenght > 0)
    distance = distance / total_lenght;
  else
    distance = std::numeric_limits<double>::max();

  return distance;
}

LineBasedScanmatcher::NearestNeighbor LineBasedScanmatcher::nearest_neighbor(LineFeature::Ptr line, std::vector<LineFeature::Ptr> cloud){

  NearestNeighbor nn_line;
  nn_line.nearest_neighbor = nullptr;
  nn_line.distance = std::numeric_limits<double>::max();

  for(LineFeature::Ptr cloud_line: cloud){
    double distance = line_to_line_distance(line, cloud_line);
    if(cloud_line != line && distance < nn_line.distance){
      nn_line.nearest_neighbor = cloud_line;
      nn_line.distance = distance;
    }
  }

  return nn_line;
}

std::vector<LineFeature::Ptr> LineBasedScanmatcher::transform_lines(std::vector<LineFeature::Ptr> lines, Eigen::Matrix4d transform){

  std::vector<LineFeature::Ptr> transformed_lines;
  LineFeature::Ptr transformed_line;

  for(LineFeature::Ptr line : lines){
    transformed_line.reset(new LineFeature());
    *transformed_line = *line;

    Eigen::Matrix4d lineA_trans = Eigen::Matrix4d::Identity();

    lineA_trans.block<3,1>(0,3) = line->pointA;
    lineA_trans = transform*lineA_trans;
    transformed_line->pointA = lineA_trans.block<3,1>(0,3);

    Eigen::Matrix4d lineB_trans = Eigen::Matrix4d::Identity();

    lineB_trans.block<3,1>(0,3) = line->pointB;
    lineB_trans = transform*lineB_trans;
    transformed_line->pointB = lineB_trans.block<3,1>(0,3);

    transformed_lines.push_back(transformed_line);
  }

  return transformed_lines;
}

}  // namespace hdl_graph_slam