#include <hdl_graph_slam/building.hpp>

namespace hdl_graph_slam {

std::vector<Eigen::Vector3d> shrink_polygon(std::vector<Eigen::Vector3d> points, Eigen::Vector3d center){
  std::vector<Eigen::Vector3d> shrinked_points;
  double shrink_ratio = 0.99;

  for(Eigen::Vector3d point: points){
    Eigen::Vector3d shrinked_point = center + shrink_ratio*(point-center);
    shrinked_points.push_back(shrinked_point);
  }

  return shrinked_points;
}

bool are_buildings_overlapped(Building::Ptr A, Building::Ptr B){

  Eigen::Vector3d centerA = Eigen::Vector3d::Zero();
  centerA.block<2,1>(0,0) = A->estimate().matrix().block<2,1>(0,2);

  Eigen::Vector3d centerB = Eigen::Vector3d::Zero();
  centerB.block<2,1>(0,0) = B->estimate().matrix().block<2,1>(0,2);

  std::vector<Eigen::Vector3d> pointsA = shrink_polygon(A->getPoints(), centerA);
  std::vector<Eigen::Vector3d> pointsB = shrink_polygon(B->getPoints(), centerB);

  for(int i=0; i < pointsA.size(); i++){
    int count = 0;
    // for each point of polygon A check if none is inside polygon B
    for(int j=0; j < pointsB.size()-1; j++){

      double x = pointsA[i].x();
      double y = pointsA[i].y();

      double x1 = pointsB[j].x();
      double x2 = pointsB[j+1].x();
      double y1 = pointsB[j].y();
      double y2 = pointsB[j+1].y();

      if(y < y1 != y < y2 && x < (x2-x1) / (y2-y1) * (y-y1) + x1){
        count++;
      }

    }

    // if its horizontal raycast intersects an odd number of times it means that it is inside
    if(count % 2 != 0){
      return true;
    }
  }

  return false;
}

}  // namespace hdl_graph_slam