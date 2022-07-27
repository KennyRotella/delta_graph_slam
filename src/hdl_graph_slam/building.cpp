#include <hdl_graph_slam/building.hpp>

namespace hdl_graph_slam {
	
Building::Building(void) {cloud=nullptr; node=nullptr;}

pcl::PointCloud<Building::PointT>::Ptr Building::getCloud(){
  if(node == nullptr)
    return cloud;

  Eigen::Matrix3d trans = (pose.inverse() * node->estimate().toIsometry()).matrix();

  // transformation is in building frame, this will change it to map frame
  trans.block<2,1>(0,2) += pose.translation() - trans.block<2,2>(0,0) * pose.translation();
  Eigen::Matrix4f trans3D = transform2Dto3D(trans.cast<float>());

  pcl::PointCloud<PointT>::Ptr trans_buildings_cloud(new pcl::PointCloud<PointT>);
  trans_buildings_cloud->header = cloud->header;
  pcl::transformPointCloud(*cloud, *trans_buildings_cloud, trans3D);

  return trans_buildings_cloud;
}

std::vector<LineFeature::Ptr> Building::getLines(){
  if(node == nullptr)
    return lines;

  Eigen::Matrix3d trans = (pose.inverse() * node->estimate().toIsometry()).matrix();

  // transformation is in building frame, this will change it to map frame
  trans.block<2,1>(0,2) += pose.translation() - trans.block<2,2>(0,0) * pose.translation();
  Eigen::Matrix4d trans3D = transform2Dto3D(trans.cast<float>()).cast<double>();

  std::vector<LineFeature::Ptr> buildings_lines;
  buildings_lines = LineBasedScanmatcher::transform_lines(lines, trans3D);

  return buildings_lines;
}

Eigen::Isometry2d Building::estimate() const {
  return node->estimate().toIsometry();
}

}  // namespace hdl_graph_slam