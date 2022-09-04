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

  std::cout << "g_avg_distance_weight: " << g_avg_distance_weight << std::endl;
  std::cout << "g_coverage_weight: " << g_coverage_weight << std::endl;
  std::cout << "g_transform_weight: " << g_transform_weight << std::endl;
  std::cout << "g_max_score_distance: " << g_max_score_distance << std::endl;
  std::cout << "g_max_score_translation: " << g_max_score_translation << std::endl;

  std::cout << "l_avg_distance_weight: " << l_avg_distance_weight << std::endl;
  std::cout << "l_coverage_weight: " << l_coverage_weight << std::endl;
  std::cout << "l_transform_weight: " << l_transform_weight << std::endl;
  std::cout << "l_max_score_distance: " << l_max_score_distance << std::endl;
  std::cout << "l_max_score_translation: " << l_max_score_translation << std::endl << std::endl;
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

  std::vector<EdgeFeature::Ptr> edgesSource = edge_extraction(linesSource);
  std::vector<EdgeFeature::Ptr> edgesTarget = edge_extraction(linesTarget);

  for(EdgeFeature::Ptr edgeSource : edgesSource){
    for(EdgeFeature::Ptr edgeTarget : edgesTarget){

      Eigen::Matrix4d transform = align_edges(edgeSource, edgeTarget);
      Eigen::Vector3d translation = transform.block<3,1>(0,3);
      double angle = Eigen::Rotation2Dd(transform3Dto2D(transform.cast<float>()).cast<double>().block<2,2>(0,0)).angle();

      // take the minimum translation to make buildings not overlapped
      if(translation.norm() < min_translation && std::cos(angle) > std::cos(max_angle)){

        std::vector<LineFeature::Ptr> linesSourceTransformed = transform_lines(linesSource, transform);
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

  for(LineFeature::Ptr lineSource : linesSource){
    for(LineFeature::Ptr lineTarget : linesTarget){

      Eigen::Vector3d srcLine = (lineSource->pointA - lineSource->pointB).normalized();
      Eigen::Vector3d trgLine = (lineTarget->pointA - lineTarget->pointB).normalized();

      Eigen::Matrix4d transform = align_lines(lineSource, lineTarget);
      Eigen::Vector3d translation = transform.block<3,1>(0,3);
      double angle = Eigen::Rotation2Dd(transform3Dto2D(transform.cast<float>()).cast<double>().block<2,2>(0,0)).angle();

      // take the minimum translation to make buildings not overlapped
      if(translation.norm() < min_translation && std::cos(angle) > std::cos(max_angle)){

        std::vector<LineFeature::Ptr> linesSourceTransformed = transform_lines(linesSource, transform);
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

BestFitAlignment LineBasedScanmatcher::align_global(pcl::PointCloud<PointT>::Ptr cloudSource, std::vector<LineFeature::Ptr> linesTarget, double &line_extraction_time, double &matching_time, bool constrain_angle, double max_range) {

  ros::WallTime start_, end_;

  // count line_extraction_time
  start_ = ros::WallTime::now();

  std::vector<LineFeature::Ptr> linesSource = line_extraction(cloudSource);

  end_ = ros::WallTime::now();

  line_extraction_time = (end_ - start_).toNSec() * 1e-6;

  // count matching_time
  start_ = ros::WallTime::now();

  linesTarget = merge_lines(linesTarget);
  
  // constraints on the global transformation
  double max_distance = 2.0;
  double max_angle = M_PI / 9.0;

  BestFitAlignment result;
  result.not_aligned_lines = linesSource;
  result.aligned_lines = linesSource;
  result.transformation = Eigen::Matrix4d::Identity();
  result.fitness_score = calc_fitness_score(linesSource, linesTarget, false, max_range);
  double result_score = weight_global(result.fitness_score.real_avg_distance, result.fitness_score.coverage_percentage, 0.0);

  std::vector<EdgeFeature::Ptr> edgesSource = edge_extraction(linesSource);
  std::vector<EdgeFeature::Ptr> edgesTarget = edge_extraction(linesTarget);

  for(EdgeFeature::Ptr edgeSource : edgesSource){
    for(EdgeFeature::Ptr edgeTarget : edgesTarget){

      Eigen::Matrix4d transform = align_edges(edgeSource, edgeTarget);

      Eigen::Vector3d translation = transform.block<3,1>(0,3);
      if(translation.norm() > max_distance || transform == Eigen::Matrix4d::Identity()){
        continue;
      }

      if(constrain_angle){
        double angle = Eigen::Rotation2Dd(transform3Dto2D(transform.cast<float>()).cast<double>().block<2,2>(0,0)).angle();
        if(std::cos(angle) < std::cos(max_angle)){
          continue;
        }
      }

      std::vector<LineFeature::Ptr> linesSourceTransformed = transform_lines(linesSource, transform);

      FitnessScore fitness_score = calc_fitness_score(linesSourceTransformed, linesTarget, false, max_range);
      double score = weight_global(fitness_score.real_avg_distance, fitness_score.coverage_percentage, translation.norm());

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

    std::vector<NearestNeighbor> neighbors = nearest_neighbor(lineSource, linesTarget);
    if(neighbors.size() == 0){
      continue;
    }

    NearestNeighbor nn_lineTarget = neighbors[0];

    if(!nn_lineTarget.nearest_neighbor){
      continue;
    }

    Eigen::Vector3d srcLine = (lineSource->pointA - lineSource->pointB).normalized();
    Eigen::Vector3d trgLine = (nn_lineTarget.nearest_neighbor->pointA - nn_lineTarget.nearest_neighbor->pointB).normalized();

    double cosine = srcLine.dot(trgLine);
    if(std::abs(cosine) < std::cos(max_angle)){
      continue;
    }

    Eigen::Matrix4d transform = align_lines(lineSource, nn_lineTarget.nearest_neighbor);
    Eigen::Vector3d translation = transform.block<3,1>(0,3);

    if(translation.norm() > max_distance){
      continue;
    }

    std::vector<LineFeature::Ptr> linesSourceTransformed = transform_lines(result.aligned_lines, transform);

    FitnessScore fitness_score = calc_fitness_score(linesSourceTransformed, linesTarget, false, max_range);
    double score = weight_global(fitness_score.real_avg_distance, fitness_score.coverage_percentage, translation.norm());

    if(score > result_score){
      result.aligned_lines = linesSourceTransformed;
      result.transformation = best_trans * transform;
      result.fitness_score = fitness_score;
      result_score = score;
    }
  }

  end_ = ros::WallTime::now();

  matching_time = (end_ - start_).toNSec() * 1e-6;

  return result;
}

BestFitAlignment LineBasedScanmatcher::align_local(std::vector<LineFeature::Ptr> linesSource, std::vector<LineFeature::Ptr> linesTarget, double &matching_time, double max_range) {

  ros::WallTime start_, end_;

  start_ = ros::WallTime::now();

  // constraints on the local transformation
  double max_distance = 2.5;
  double max_angle = M_PI / 9.0;

  BestFitAlignment result;
  result.not_aligned_lines = linesSource;
  result.aligned_lines = linesSource;
  result.transformation = Eigen::Matrix4d::Identity();
  result.fitness_score = calc_fitness_score(linesSource, linesTarget, true, max_range);
  result.isEdgeAligned = false;
  double result_score = weight_local(result.fitness_score.avg_distance, result.fitness_score.coverage_percentage, 0.0);

  std::vector<EdgeFeature::Ptr> edgesSource = edge_extraction(linesSource, true, 0.01);
  std::vector<EdgeFeature::Ptr> edgesTarget = edge_extraction(linesTarget, true);

  for(EdgeFeature::Ptr edgeSource : edgesSource){
    for(EdgeFeature::Ptr edgeTarget : edgesTarget){

      Eigen::Matrix4d transform = align_edges(edgeSource, edgeTarget);

      Eigen::Vector3d translation = transform.block<3,1>(0,3);
      if(translation.norm() > max_distance){
        continue;
      }

      double angle = Eigen::Rotation2Dd(transform3Dto2D(transform.cast<float>()).cast<double>().block<2,2>(0,0)).angle();
      if(std::cos(angle) < std::cos(max_angle)){
        continue;
      }

      std::vector<LineFeature::Ptr> linesSourceTransformed = transform_lines(linesSource, transform);

      FitnessScore fitness_score = calc_fitness_score(linesSourceTransformed, linesTarget, true, max_range);
      double score = weight_local(fitness_score.avg_distance, fitness_score.coverage_percentage, translation.norm());

      if(score > result_score){
        result.aligned_lines = linesSourceTransformed;
        result.transformation = transform;
        result.fitness_score = fitness_score;
        result_score = score;
        result.isEdgeAligned = true;
      }
    }
  }

  // use best transform found so far
  std::vector<LineFeature::Ptr> best_lines = result.aligned_lines;
  Eigen::Matrix4d best_trans = result.transformation;
  for(LineFeature::Ptr lineSource : best_lines){

    std::vector<NearestNeighbor> nn_linesTarget = nearest_neighbor(lineSource, linesTarget);

    // try alignment with the 3 nearest neighbors
    for(int i=0; i<3 || i<nn_linesTarget.size(); i++){
      NearestNeighbor nn_lineTarget = nn_linesTarget[i];

      if(!nn_lineTarget.nearest_neighbor){
        continue;
      }

      Eigen::Vector3d srcLine = (lineSource->pointA - lineSource->pointB).normalized();
      Eigen::Vector3d trgLine = (nn_lineTarget.nearest_neighbor->pointA - nn_lineTarget.nearest_neighbor->pointB).normalized();

      double cosine = srcLine.dot(trgLine);
      if(std::abs(cosine) < std::cos(max_angle)){
        continue;
      }

      Eigen::Matrix4d transform = align_lines(lineSource, nn_lineTarget.nearest_neighbor);
      Eigen::Vector3d translation = transform.block<3,1>(0,3);

      if(translation.norm() > max_distance){
        continue;
      }

      std::vector<LineFeature::Ptr> linesSourceTransformed = transform_lines(best_lines, transform);

      FitnessScore fitness_score = calc_fitness_score(linesSourceTransformed, linesTarget, true, max_range);
      double score = weight_local(fitness_score.avg_distance, fitness_score.coverage_percentage, translation.norm());

      if(score > result_score){
        result.aligned_lines = linesSourceTransformed;
        result.transformation = best_trans * transform;
        result.fitness_score = fitness_score;
        result_score = score;
      }
    }
  }

  end_ = ros::WallTime::now();

  matching_time = (end_ - start_).toNSec() * 1e-6;

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

std::vector<EdgeFeature::Ptr> LineBasedScanmatcher::edge_extraction(std::vector<LineFeature::Ptr> lines, bool only_angular_edges, double max_dist_angular_edge){

  std::vector<EdgeFeature::Ptr> edges;

  for(int i=0; i<lines.size()-1; i++){
    for(int j=i+1; j<lines.size(); j++){
      std::vector<EdgeFeature::Ptr> current_edges = get_edges(lines[i], lines[j], only_angular_edges, max_dist_angular_edge);
      edges.insert(edges.end(), current_edges.begin(), current_edges.end());
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

std::vector<EdgeFeature::Ptr> LineBasedScanmatcher::get_edges(LineFeature::Ptr line1, LineFeature::Ptr line2, bool only_angular_edges, double max_dist_angular_edge){

  std::vector<EdgeFeature::Ptr> edges;

  // lines should be almost perpendicular
  double cosine = (line1->pointA - line1->pointB).normalized().dot(
                  (line2->pointA - line2->pointB).normalized());

  if(std::abs(cosine) > 0.5){
    return edges;
  }

  double min_side_lenght = 1.0;

  Eigen::Vector3d edgePoint, side1A, side1B, side2A, side2B;
  bool same_direction_sides1, same_direction_sides2;
  edgePoint = lines_intersection(line1, line2);

  // sides line1
  side1A = line1->pointA-edgePoint;
  side1B = line1->pointB-edgePoint;
  same_direction_sides1 = side1A.norm() < 0.01 || side1B.norm() < 0.01 || (side1A.normalized()-side1B.normalized()).norm() < 1.;


  // sides line2
  side2A = line2->pointA-edgePoint;
  side2B = line2->pointB-edgePoint;
  same_direction_sides2 = side2A.norm() < 0.01 || side2B.norm() < 0.01 || (side2A.normalized()-side2B.normalized()).norm() < 1. ;

  if(same_direction_sides1 && same_direction_sides2){
    // CASE 1:

    if(std::max(side1A.norm(), side1B.norm()) < min_side_lenght || std::max(side2A.norm(), side2B.norm()) < min_side_lenght){
      return edges;
    }

    if(only_angular_edges && (std::min(side1A.norm(), side1B.norm()) > max_dist_angular_edge || std::min(side2A.norm(), side2B.norm()) > max_dist_angular_edge)){
      return edges;
    }

    EdgeFeature::Ptr edge(new EdgeFeature());
    edge->edgePoint = edgePoint;

    // take the longest side
    if(side1A.norm() > side1B.norm()){
      edge->pointA = line1->pointA;
    } else {
      edge->pointA = line1->pointB;
    }

    // take the longest side
    if(side2A.norm() > side2B.norm()){
      edge->pointB = line2->pointA;
    } else {
      edge->pointB = line2->pointB;
    }

    edges.push_back(edge);
  } else if(same_direction_sides1 && !same_direction_sides2){
    // CASE 2:

    if(std::max(side1A.norm(), side1B.norm()) < min_side_lenght){
      return edges;
    }

    if(only_angular_edges && std::min(side1A.norm(), side1B.norm()) > max_dist_angular_edge){
      return edges;
    }

    // take the longest side
    Eigen::Vector3d edge_pointA;
    if(side1A.norm() > side1B.norm()){
      edge_pointA = line1->pointA;
    } else {
      edge_pointA = line1->pointB;
    }

    if(side2A.norm() > min_side_lenght){
      EdgeFeature::Ptr edge(new EdgeFeature());
      edge->edgePoint = edgePoint;
      edge->pointA = edge_pointA;
      edge->pointB = line2->pointA;

      edges.push_back(edge);
    }

    if(side2B.norm() > min_side_lenght){
      EdgeFeature::Ptr edge(new EdgeFeature());
      edge->edgePoint = edgePoint;
      edge->pointA = edge_pointA;
      edge->pointB = line2->pointB;

      edges.push_back(edge);
    }

  } else if(!same_direction_sides1 && same_direction_sides2){
    // CASE 3:

    if(std::max(side2A.norm(), side2B.norm()) < min_side_lenght){
      return edges;
    }

    if(only_angular_edges && std::min(side2A.norm(), side2B.norm()) > max_dist_angular_edge){
      return edges;
    }

    // take the longest side
    Eigen::Vector3d edge_pointA;
    if(side1A.norm() > side1B.norm()){
      edge_pointA = line2->pointA;
    } else {
      edge_pointA = line2->pointB;
    }

    if(side1A.norm() > min_side_lenght){
      EdgeFeature::Ptr edge(new EdgeFeature());
      edge->edgePoint = edgePoint;
      edge->pointA = edge_pointA;
      edge->pointB = line1->pointA;

      edges.push_back(edge);
    }

    if(side1B.norm() > min_side_lenght){
      EdgeFeature::Ptr edge(new EdgeFeature());
      edge->edgePoint = edgePoint;
      edge->pointA = edge_pointA;
      edge->pointB = line1->pointB;

      edges.push_back(edge);
    }

  } else {
    // CASE 4:

    if(side1A.norm() > min_side_lenght){
      
      if(side2A.norm() > min_side_lenght){
        EdgeFeature::Ptr edge(new EdgeFeature());
        edge->edgePoint = edgePoint;
        edge->pointA = line1->pointA;
        edge->pointB = line2->pointA;

        edges.push_back(edge);
      }

      if(side2B.norm() > min_side_lenght){
        EdgeFeature::Ptr edge(new EdgeFeature());
        edge->edgePoint = edgePoint;
        edge->pointA = line1->pointA;
        edge->pointB = line2->pointB;

        edges.push_back(edge);
      }

    }

    if(side1B.norm() > min_side_lenght){
      
      if(side2A.norm() > min_side_lenght){
        EdgeFeature::Ptr edge(new EdgeFeature());
        edge->edgePoint = edgePoint;
        edge->pointA = line1->pointB;
        edge->pointB = line2->pointA;

        edges.push_back(edge);
      }

      if(side2B.norm() > min_side_lenght){
        EdgeFeature::Ptr edge(new EdgeFeature());
        edge->edgePoint = edgePoint;
        edge->pointA = line1->pointB;
        edge->pointB = line2->pointB;

        edges.push_back(edge);
      }

    }
  }

  return edges;
}

double LineBasedScanmatcher::angle_between_vectors(Eigen::Vector3d A, Eigen::Vector3d B){

  double dot = A.x()*B.x() + A.y()*B.y(); // dot product between A and B
  double det = A.x()*B.y() - A.y()*B.x(); // determinant
  double angle = std::atan2(det, dot);    // atan2(y, x) or atan2(sin, cos)

  return angle;  // [-π,+π] from A to B
}

Eigen::Matrix4d LineBasedScanmatcher::align_edges(EdgeFeature::Ptr edge1, EdgeFeature::Ptr edge2){

  // sides edge 1
  Eigen::Vector3d side1A = edge1->pointA-edge1->edgePoint;
  Eigen::Vector3d side1B = edge1->pointB-edge1->edgePoint;

  // sides edge 2
  Eigen::Vector3d side2A = edge2->pointA-edge2->edgePoint;
  Eigen::Vector3d side2B = edge2->pointB-edge2->edgePoint;

  // use the longest side of edge2 to find the angle
  if(side2A.norm() < side2B.norm()){
    Eigen::Vector3d side2A_tmp = side2A;
    side2A = side2B;
    side2B = side2A_tmp;
  }

  double angle1 = angle_between_vectors(side1A, side2A);
  double angle2 = angle_between_vectors(side1B, side2A);

  Eigen::Matrix3d rot1, rot2;
  rot1 = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX())
    * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY())
    * Eigen::AngleAxisd(angle1, Eigen::Vector3d::UnitZ());

  rot2 = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX())
    * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY())
    * Eigen::AngleAxisd(angle2, Eigen::Vector3d::UnitZ());
  
  Eigen::Vector3d side1B_rot1 = rot1 * side1B;
  Eigen::Vector3d side1A_rot2 = rot2 * side1A;

  double angle3 = angle_between_vectors(side1B_rot1, side2B);
  double angle4 = angle_between_vectors(side1A_rot2, side2B);

  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();

  if(std::abs(angle3) < std::abs(angle4)){
    transform.block<3,1>(0,3) = edge2->edgePoint - rot1 * edge1->edgePoint;
    transform.block<3,3>(0,0) = rot1;
  } else {
    transform.block<3,1>(0,3) = edge2->edgePoint - rot2 * edge1->edgePoint;
    transform.block<3,3>(0,0) = rot2;
  }

  return transform;

}

Eigen::Matrix4d LineBasedScanmatcher::align_lines(LineFeature::Ptr line1, LineFeature::Ptr line2){

  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  double angle = angle_between_vectors(line1->pointA - line1->pointB, line2->pointA - line2->pointB);

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

// this one does not take into account the lenght of the line
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

  if(dot1 >= 0 && dot2 >= 0){
    return (point-projected_point).norm();
  }

  if(dot1 < 0){
    return (point - line->pointA).norm();
  }

  if(dot2 < 0){
    return (point - line->pointB).norm();
  }
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

  FitnessScore score;

  // compute real distance without coverage, used in global matching
  score.real_avg_distance = 0.0;
  score.real_avg_distance += point_to_line_distance(line_src->pointA, line_trg);
  score.real_avg_distance += point_to_line_distance(line_src->pointB, line_trg);

  score.real_avg_distance =  score.real_avg_distance / 2.0;

  // compute distance with coverage
  double distance1 = 0;
  double distance2 = 0;

  Eigen::Vector3d point1;
  bool point1_found = false;
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

FitnessScore LineBasedScanmatcher::calc_fitness_score(std::vector<LineFeature::Ptr> cloud1, std::vector<LineFeature::Ptr> cloud2, bool is_local, double max_range){

  FitnessScore score;
  double real_distance = 0.0;
  double real_distance_lenght = 0.0;
  double distance = 0.0;
  double coverage_lenght = 0.0;
  double total_lenght = 0.0;

  for(LineFeature::Ptr cloud_line: cloud1){

    std::vector<NearestNeighbor> neighbors = nearest_neighbor(cloud_line, cloud2);
    if(neighbors.size() == 0){
      total_lenght += cloud_line->lenght();
      continue;
    }

    NearestNeighbor nn_line = neighbors[0];

    // two different metric are used for local and global alignment
    if((is_local && nn_line.distance < max_range) || (!is_local && nn_line.real_distance < max_range)){
      real_distance += nn_line.real_distance * cloud_line->lenght();
      real_distance_lenght += cloud_line->lenght();
      distance += nn_line.distance * nn_line.coverage;
      coverage_lenght += nn_line.coverage;
    }
    total_lenght += cloud_line->lenght();
  }

  score.coverage = coverage_lenght;

  if(real_distance_lenght > 0){
    score.real_avg_distance = real_distance / real_distance_lenght;
  } else {
    score.real_avg_distance = std::numeric_limits<double>::max();
  }

  if(coverage_lenght > 0){
    score.avg_distance = distance / coverage_lenght;
  } else {
    score.avg_distance = std::numeric_limits<double>::max();
  }

  if(total_lenght > 0){
    score.coverage_percentage = coverage_lenght / total_lenght * 100.0;
  } else {
    score.coverage_percentage = 0.0;
  }

  return score;
}

std::vector<LineBasedScanmatcher::NearestNeighbor> LineBasedScanmatcher::nearest_neighbor(LineFeature::Ptr line, std::vector<LineFeature::Ptr> cloud){

  std::vector<NearestNeighbor> nearest_neighbors;

  for(LineFeature::Ptr cloud_line: cloud){
    FitnessScore fitness_score = line_to_line_distance(line, cloud_line);
    NearestNeighbor nn_line;

    if(cloud_line != line){
      nn_line.nearest_neighbor = cloud_line;
      nn_line.real_distance = fitness_score.real_avg_distance;
      nn_line.distance = fitness_score.avg_distance;
      nn_line.coverage = fitness_score.coverage;

      nearest_neighbors.push_back(nn_line);
    }
  }

  // sort in ascending order
  std::sort(nearest_neighbors.begin(), nearest_neighbors.end(), 
    [](const NearestNeighbor& a, const NearestNeighbor& b) -> bool
  { 
      return a.real_distance < b.real_distance; 
  });

  return nearest_neighbors;
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

LineFeature::Ptr LineBasedScanmatcher::are_lines_aligned(LineFeature::Ptr line1, LineFeature::Ptr line2){
  // lines should be almost parallel
  double cosine = (line1->pointA - line1->pointB).normalized().dot(
                  (line2->pointA - line2->pointB).normalized());

  if(std::abs(cosine) < 0.9995){
    return nullptr;
  }

  double threshold = 0.3;

  // if two lines are identical
  if(((line1->pointA - line2->pointA).norm() < threshold && (line1->pointB - line2->pointB).norm() < threshold) ||
     ((line1->pointA - line2->pointB).norm() < threshold && (line1->pointB - line2->pointA).norm() < threshold)){
    return line1;
  }

  if((line1->pointA - line2->pointA).norm() < threshold){
    // if lines are overlapped
    if(is_point_on_line(line1->pointB, line2) ||
      is_point_on_line(line2->pointB, line1)){
      return nullptr;
    }

    LineFeature::Ptr merged_line(new LineFeature());
    merged_line->pointA = line1->pointB;
    merged_line->pointB = line2->pointB;

    return merged_line;

  } else if((line1->pointA - line2->pointB).norm() < threshold){
    // if lines are overlapped
    if(is_point_on_line(line1->pointB, line2) ||
      is_point_on_line(line2->pointA, line1)){
      return nullptr;
    }

    LineFeature::Ptr merged_line(new LineFeature());
    merged_line->pointA = line1->pointB;
    merged_line->pointB = line2->pointA;

    return merged_line;

  } else if((line1->pointB - line2->pointA).norm() < threshold) {
    // if lines are overlapped
    if(is_point_on_line(line1->pointA, line2) ||
      is_point_on_line(line2->pointB, line1)){
      return nullptr;
    }

    LineFeature::Ptr merged_line(new LineFeature());
    merged_line->pointA = line1->pointA;
    merged_line->pointB = line2->pointB;

    return merged_line;

  } else if((line1->pointB - line2->pointB).norm() < threshold){
    // if lines are overlapped
    if(is_point_on_line(line1->pointA, line2) ||
      is_point_on_line(line2->pointA, line1)){
      return nullptr;
    }

    LineFeature::Ptr merged_line(new LineFeature());
    merged_line->pointA = line1->pointA;
    merged_line->pointB = line2->pointA;

    return merged_line;

  }

  return nullptr;
}

std::vector<LineFeature::Ptr> LineBasedScanmatcher::merge_lines(std::vector<LineFeature::Ptr> lines){

  for(int i=0; i<lines.size(); i++){
  LineFeature::Ptr line1 = lines[i];
    for(int j=i+1; j<lines.size(); j++){
      LineFeature::Ptr line2 = lines[j];
      LineFeature::Ptr merged_lines = are_lines_aligned(line1, line2);
      if(merged_lines != nullptr){
        lines.erase(lines.begin() + j);
        lines[i] = merged_lines;
        i--;
        break;
      }
    }
  }

  return lines;
}

}  // namespace hdl_graph_slam