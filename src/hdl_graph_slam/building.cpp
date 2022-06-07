#include <hdl_graph_slam/building.hpp>

namespace hdl_graph_slam {
	
Building::Building(void) {cloud.reset(new pcl::PointCloud<PointT>); node=nullptr;}

}  // namespace hdl_graph_slam