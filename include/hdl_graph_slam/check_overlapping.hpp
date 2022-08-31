#ifndef CHECK_OVERLAPPING_HPP
#define CHECK_OVERLAPPING_HPP

#include <hdl_graph_slam/building.hpp>
#include <hdl_graph_slam/line_based_scanmatcher.hpp>

namespace hdl_graph_slam {

// this function checks if the point of a line is inside a segment of it
bool is_point_on_the_line(LineFeature::Ptr line, Eigen::Vector3d point){

  double x = point.x();
  double y = point.y();

  double x1 = line->pointA.x();
  double x2 = line->pointB.x();
  double y1 = line->pointA.y();
  double y2 = line->pointB.y();

  // this works also if the lines are horizontal or vertical
  return (x < x1 != x < x2 || y < y1 != y < y2);
}

bool are_lines_intersected(LineFeature::Ptr line1, LineFeature::Ptr line2)
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

  if(determinant == 0)
  {
    return false;
  }

  double x = (b2*c1 - b1*c2)/determinant;
  double y = (a1*c2 - a2*c1)/determinant;

  Eigen::Vector3d point(x,y,0);

  return is_point_on_the_line(line1, point) && is_point_on_the_line(line2, point);
}

std::vector<LineFeature::Ptr> shrink_polygon(std::vector<LineFeature::Ptr> lines, Eigen::Vector3d center){
  std::vector<LineFeature::Ptr> shrinked_lines;
  double shrink_ratio = 0.99;

  for(LineFeature::Ptr line: lines){
    Eigen::Vector3d pointA = line->pointA;
    Eigen::Vector3d pointB = line->pointB;

    Eigen::Vector3d shrinked_pointA = center + shrink_ratio*(pointA-center);
    Eigen::Vector3d shrinked_pointB = center + shrink_ratio*(pointB-center);

    LineFeature::Ptr shrinked_line(new LineFeature());
    shrinked_line->pointA = shrinked_pointA;
    shrinked_line->pointB = shrinked_pointB;

    shrinked_lines.push_back(shrinked_line);
  }

  return shrinked_lines;
}

bool are_buildings_overlapped(Building::Ptr A, Building::Ptr B){

  Eigen::Vector3d centerA = Eigen::Vector3d::Zero();
  centerA.block<2,1>(0,0) = A->estimate().matrix().block<2,1>(0,2);

  Eigen::Vector3d centerB = Eigen::Vector3d::Zero();
  centerB.block<2,1>(0,0) = B->estimate().matrix().block<2,1>(0,2);

  std::vector<LineFeature::Ptr> shrinked_linesA;
  shrinked_linesA = shrink_polygon(A->getLines(), centerA);

  std::vector<LineFeature::Ptr> shrinked_linesB;
  shrinked_linesB = shrink_polygon(B->getLines(), centerB);

  for(LineFeature::Ptr lineA: shrinked_linesA){
    for(LineFeature::Ptr lineB: shrinked_linesB){
      if(are_lines_intersected(lineA, lineB)){
        return true;
      }
    }
  }

  return false;
}

bool are_buildings_overlapped(std::vector<LineFeature::Ptr> A, Eigen::Vector3d centerA, std::vector<LineFeature::Ptr> B, Eigen::Vector3d centerB){

  std::vector<LineFeature::Ptr> shrinked_linesA;
  shrinked_linesA = shrink_polygon(A, centerA);

  std::vector<LineFeature::Ptr> shrinked_linesB;
  shrinked_linesB = shrink_polygon(B, centerB);

  for(LineFeature::Ptr lineA: shrinked_linesA){
    for(LineFeature::Ptr lineB: shrinked_linesB){
      if(are_lines_intersected(lineA, lineB)){
        return true;
      }
    }
  }

  return false;
}

}  // namespace hdl_graph_slam

#endif // CHECK_OVERLAPPING_HPP