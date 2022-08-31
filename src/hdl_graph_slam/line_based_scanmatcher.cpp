#include <hdl_graph_slam/line_based_scanmatcher.hpp>

namespace hdl_graph_slam {

void LineBasedScanmatcher::print_parameters(){
  std::cout << "LINE BASED SCANMATCHER PARAMS" << std::endl;
  std::cout << "min_cluster_size: " << min_cluster_size << std::endl;
  std::cout << "max_cluster_size: " << max_cluster_size << std::endl;
  std::cout << "cluster_tolerance: " << cluster_tolerance << std::endl;
  std::cout << "sac_method_type: " << sac_method_type << std::endl;
  std::cout << "sac_distance_threshold: " << sac_distance_threshold << std::endl;
  std::cout << "max_iterations: " << max_iterations << std::endl;
  std::cout << "merror_threshold: " << merror_threshold << std::endl;
  std::cout << "line_lenght_threshold: " << line_lenght_threshold << std::endl;
  std::cout << "avg_distance_weight: " << avg_distance_weight << std::endl;
  std::cout << "coverage_weight: " << coverage_weight << std::endl;
  std::cout << "transform_weight: " << transform_weight << std::endl;
  std::cout << "max_score_distance: " << max_score_distance << std::endl;
  std::cout << "max_score_translation: " << max_score_translation << std::endl;
}

BestFitAlignment LineBasedScanmatcher::align_overlapped_buildings(Building::Ptr A, Building::Ptr B){
  std::vector<LineFeature::Ptr> linesSource = A->getLines();
  std::vector<LineFeature::Ptr> linesTarget = B->getLines();

  // transform lines in source reference frame
  Eigen::Matrix4d building_pose = transform2Dto3D(A->estimate().matrix().cast<float>()).cast<double>();
  linesSource = transform_lines(linesSource, building_pose.inverse());
  linesTarget = transform_lines(linesTarget, building_pose.inverse());

  // new centers in the source reference frame
  Eigen::Vector3d centerA = Eigen::Vector3d::Zero();
  Eigen::Vector2d centerB_2D = (A->estimate().inverse() * B->estimate()).translation();
  Eigen::Vector3d centerB = Eigen::Vector3d(centerB_2D.x(), centerB_2D.y(), 0.0);

  BestFitAlignment result;
  result.aligned_lines = linesSource;
  result.transformation = Eigen::Matrix4d::Identity();

  double max_angle = M_PI / 3.0;
  double min_translation = std::numeric_limits<double>::max();

  for(LineFeature::Ptr lineSource : linesSource){
    for(LineFeature::Ptr lineTarget : linesTarget){

      Eigen::Vector3d srcLine = (lineSource->pointA - lineSource->pointB).normalized();
      Eigen::Vector3d trgLine = (lineTarget->pointA - lineTarget->pointB).normalized();

      Eigen::Matrix4d transform = align_lines(lineSource, lineTarget);
      Eigen::Vector3d translation = transform.block<3,1>(0,3);
      double angle = Eigen::Rotation2Dd(transform3Dto2D(transform.cast<float>()).cast<double>().block<2,2>(0,0)).angle();

      std::vector<LineFeature::Ptr> linesSourceTransformed = transform_lines(linesSource, transform);

      // take the minimum translation to make buildings not overlapped
      if(translation.norm() < min_translation && std::cos(angle) > std::cos(max_angle)){

        // additional check to make sure buildings are not overlapped
        // remove overlapping transformations from search space
        if(!are_buildings_overlapped(linesSourceTransformed, centerA, linesTarget, centerB)){
          result.aligned_lines = linesSourceTransformed;
          result.transformation = transform;

          min_translation = translation.norm();
        }
      }
    }
  }

  // transform back result in map reference frame
  result.aligned_lines = transform_lines(result.aligned_lines, building_pose);
  result.transformation = building_pose * result.transformation * building_pose.inverse();

  return result;
}

BestFitAlignment LineBasedScanmatcher::align(pcl::PointCloud<PointT>::Ptr cloudSource, std::vector<LineFeature::Ptr> linesTarget, bool local_alignment, double max_range) {
  
  std::vector<LineFeature::Ptr> linesSource = line_extraction(cloudSource);
  return align(linesSource, linesTarget, local_alignment, max_range);
}

BestFitAlignment LineBasedScanmatcher::align(std::vector<LineFeature::Ptr> linesSource, std::vector<LineFeature::Ptr> linesTarget, bool local_alignment, double max_range) {

  double max_distance = 3.5;
  double min_cosine = 0.9;

  BestFitAlignment result;
  result.not_aligned_lines = linesSource;
  result.aligned_lines = linesSource;
  result.transformation = Eigen::Matrix4d::Identity();
  result.fitness_score = calc_fitness_score(linesSource, linesTarget, max_range);
  double result_score = weight(result.fitness_score.avg_distance, result.fitness_score.coverage_percentage, 0.0);

  std::vector<EdgeFeature::Ptr> edgesSource = edge_extraction(linesSource);
  std::vector<EdgeFeature::Ptr> edgesTarget = edge_extraction(linesTarget);

  for(EdgeFeature::Ptr edgeSource : edgesSource){
    for(EdgeFeature::Ptr edgeTarget : edgesTarget){

      Eigen::Matrix4d transform;
      if(local_alignment){
        transform = align_edges(edgeSource, edgeTarget, M_PI/9);
      } else {
        transform = align_edges(edgeSource, edgeTarget);
      }

      Eigen::Vector3d translation = transform.block<3,1>(0,3);

      if(translation.norm() > max_distance || transform == Eigen::Matrix4d::Identity()){
        continue;
      }

      std::vector<LineFeature::Ptr> linesSourceTransformed = transform_lines(linesSource, transform);

      FitnessScore fitness_score = calc_fitness_score(linesSourceTransformed, linesTarget, max_range);
      double score = weight(fitness_score.avg_distance, fitness_score.coverage_percentage, translation.norm());

      if(score > result_score){
        result.aligned_lines = linesSourceTransformed;
        result.transformation = transform;
        result.fitness_score = fitness_score;
        result_score = score;
      }
    }
  }

  // use best transform found so far
  Eigen::Matrix4d best_trans = result.transformation;
  for(LineFeature::Ptr lineSource : result.aligned_lines){

    NearestNeighbor nn_lineTarget = nearest_neighbor(lineSource, linesTarget);

    if(!nn_lineTarget.nearest_neighbor){
      continue;
    }

    Eigen::Vector3d srcLine = (lineSource->pointA - lineSource->pointB).normalized();
    Eigen::Vector3d trgLine = (nn_lineTarget.nearest_neighbor->pointA - nn_lineTarget.nearest_neighbor->pointB).normalized();

    double cosine = srcLine.dot(trgLine);
    if(std::abs(cosine) < min_cosine){
      continue;
    }

    Eigen::Matrix4d transform = align_lines(lineSource, nn_lineTarget.nearest_neighbor);
    Eigen::Vector3d translation = transform.block<3,1>(0,3);

    if(translation.norm() > max_distance){
      continue;
    }

    std::vector<LineFeature::Ptr> linesSourceTransformed = transform_lines(result.aligned_lines, transform);

    FitnessScore fitness_score = calc_fitness_score(linesSourceTransformed, linesTarget, max_range);
    double score = weight(fitness_score.avg_distance, fitness_score.coverage_percentage, translation.norm());

    if(score > result_score){
      result.aligned_lines = linesSourceTransformed;
      result.transformation = best_trans * transform;
      result.fitness_score = fitness_score;
      result_score = score;
    }
  }

  return result;
}

pcl::PointIndices::Ptr LineBasedScanmatcher::extract_cluster(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr inliers) {

  pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>);
  for (int i=0; i<inliers->indices.size(); i++){
    PointT pt = cloud->points[inliers->indices[i]];
    cloud_plane->points.push_back(pt);
  }

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud (cloud_plane);

  std::vector<pcl::PointIndices> clusters;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(cluster_tolerance);
  ec.setMinClusterSize(1);
  ec.setMaxClusterSize(max_cluster_size);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud_plane);
  ec.extract(clusters);

  if(clusters.size() > 0){
    pcl::PointIndices::Ptr cluster_inliers(new pcl::PointIndices);
    cluster_inliers->header = inliers->header;
    // clusters are ordered by their sizes, the first one is the biggest
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

  while(filtered->points.size() >= min_cluster_size){

    // Fit a line
    seg.setInputCloud(filtered);
    seg.segment(*inliers, *coefficients);

    Eigen::Vector3d vt_line(coefficients->values[0], coefficients->values[1], 0.f);
    Eigen::Vector3d vt_direction(coefficients->values[3], coefficients->values[4], 0.f);

    // All projections will not be scaled
    vt_direction.normalize();

    inliers = extract_cluster(filtered, inliers);

    // Check result
    if(!inliers || inliers->indices.size() < min_cluster_size){

      // Remove inliers
      extract.setInputCloud(filtered);
      extract.setIndices(inliers);
      extract.setNegative(true);
      pcl::PointCloud<PointT> cloudF;
      extract.filter(cloudF);
      filtered->swap(cloudF);

      continue;
    }

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

    // Remove inliers
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
  double x, y;

  if (determinant != 0)
  {
    x = (b2*c1 - b1*c2)/determinant;
    y = (a1*c2 - a2*c1)/determinant;
  } else {
    std::cout << "LineBasedScanmatcher found two parallel lines to intersect!" << std::endl;
    x = std::numeric_limits<double>().max();
    y = std::numeric_limits<double>().max();
  }

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

  // sides edge 1
  Eigen::Vector3d side1A = edge1->pointA-edge1->edgePoint;
  Eigen::Vector3d side1B = edge1->pointB-edge1->edgePoint;

  // sides edge 2
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

  // [0,2π] -> [-π,+π] 
  if(angle > M_PI){
    angle -= 2*M_PI;
  }

  if(std::abs(angle) > max_angle){
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
  if(dot1 < 0){
    return (point - line->pointA).norm();
  }

  double dot2 = (projected_point - line->pointB).dot(line->pointA - line->pointB);
  if(dot2 < 0){
    return (point - line->pointB).norm();
  }

  return (point-projected_point).norm(); 
}

bool LineBasedScanmatcher::is_point_on_line(Eigen::Vector3d point, LineFeature::Ptr line){
  double dot1 = (point - line->pointA).dot(line->pointB - line->pointA);
  double dot2 = (point - line->pointB).dot(line->pointA - line->pointB);

  if(dot1 >= 0 && dot2 >= 0){
    return true;
  }

  return false;
}

FitnessScore LineBasedScanmatcher::line_to_line_distance(LineFeature::Ptr line_src, LineFeature::Ptr line_trg){

  double distance1 = 0;
  double distance2 = 0;

  Eigen::Vector3d point1;
  bool point1_found = false;
  FitnessScore score;

  Eigen::Vector3d line_point, line_direction, projected_point;

  line_point = line_trg->pointA;
  line_direction = (line_trg->pointB - line_trg->pointA).normalized();

  // PointA
  projected_point = line_point + line_direction*((line_src->pointA - line_point).dot(line_direction));
  if(is_point_on_line(projected_point, line_trg)){
    point1 = line_src->pointA;
    distance1 = (line_src->pointA-projected_point).norm();
    point1_found = true;
  }

  // PointB
  projected_point = line_point + line_direction*((line_src->pointB - line_point).dot(line_direction));
  if(is_point_on_line(projected_point, line_trg)){
    if(!point1_found){
      point1 = line_src->pointB;
      distance1 = (line_src->pointB-projected_point).norm();
      point1_found = true;
    } else {
      distance2 = (line_src->pointB-projected_point).norm();
      score.avg_distance = (distance1 + distance2) / 2.0;
      score.coverage = (line_src->pointB - point1).norm();
      score.coverage_percentage = score.coverage / line_src->lenght();
      return score;
    }
  }

  // rotate vector direction of 90°
  double x = line_direction.x();
  line_direction.x() = line_direction.y();
  line_direction.y() = -x;

  // PointA_proj
  LineFeature::Ptr lineA(new LineFeature());
  lineA->pointA = line_trg->pointA;
  lineA->pointB = line_trg->pointA + line_direction;

  projected_point = lines_intersection(line_src, lineA);
  if(is_point_on_line(projected_point, line_src)){
    if(!point1_found){
      point1 = projected_point;
      distance1 = (line_trg->pointA-projected_point).norm();
      point1_found = true;
    } else {
      distance2 = (line_trg->pointA-projected_point).norm();
      score.avg_distance = (distance1 + distance2) / 2.0;
      score.coverage = (projected_point - point1).norm();
      score.coverage_percentage = score.coverage / line_src->lenght();
      return score;
    }
  }

  // PointB_proj
  LineFeature::Ptr lineB(new LineFeature());
  lineB->pointA = line_trg->pointB;
  lineB->pointB = line_trg->pointB + line_direction;

  projected_point = lines_intersection(line_src, lineB);
  if(is_point_on_line(projected_point, line_src)){
    if(point1_found){
      distance2 = (line_trg->pointB-projected_point).norm();
      score.avg_distance = (distance1 + distance2) / 2.0;
      score.coverage = (projected_point - point1).norm();
      score.coverage_percentage = score.coverage / line_src->lenght();
      return score;
    }
  }

  score.avg_distance = std::numeric_limits<double>::max();
  score.coverage = 0.0;
  score.coverage_percentage = 0.0;

  return score;
}

FitnessScore LineBasedScanmatcher::calc_fitness_score(std::vector<LineFeature::Ptr> cloud1, std::vector<LineFeature::Ptr> cloud2, double max_range){

  FitnessScore score;
  double distance = 0;
  double coverage_lenght = 0;
  double total_lenght = 0;

  for(LineFeature::Ptr cloud_line: cloud1){
    NearestNeighbor nn_line = nearest_neighbor(cloud_line, cloud2);
    if(nn_line.distance < max_range){
      distance += nn_line.distance * nn_line.coverage;
      coverage_lenght += nn_line.coverage;
    }
    total_lenght += cloud_line->lenght();
  }

  if(coverage_lenght > 0)
    distance = distance / coverage_lenght;
  else
    distance = std::numeric_limits<double>::max();

  score.avg_distance = distance;
  score.coverage = coverage_lenght;

  if(total_lenght > 0){
    score.coverage_percentage = coverage_lenght / total_lenght * 100.0;
  } else {
    score.coverage_percentage = 0.0;
  }

  return score;
}

LineBasedScanmatcher::NearestNeighbor LineBasedScanmatcher::nearest_neighbor(LineFeature::Ptr line, std::vector<LineFeature::Ptr> cloud){

  NearestNeighbor nn_line;
  nn_line.nearest_neighbor = nullptr;
  nn_line.distance = std::numeric_limits<double>::max();
  nn_line.coverage = 0;

  for(LineFeature::Ptr cloud_line: cloud){
    FitnessScore fitness_score = line_to_line_distance(line, cloud_line);

    if(cloud_line != line && fitness_score.avg_distance < nn_line.distance){
      nn_line.nearest_neighbor = cloud_line;
      nn_line.distance = fitness_score.avg_distance;
      nn_line.coverage = fitness_score.coverage;
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