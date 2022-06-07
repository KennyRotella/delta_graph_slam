#ifndef BUILDING_HPP
#define BUILDING_HPP

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>

namespace g2o {
class VertexSE2;
}	 // namespace g2o

namespace hdl_graph_slam {

class Building {
	typedef pcl::PointXYZ PointT;
public:
	typedef boost::shared_ptr<Building> Ptr;

	Building(void);

	std::string id;
	Eigen::Isometry2d pose;					// pose (estimated by OpenStreetMap)
	pcl::PointCloud<PointT>::Ptr cloud;

	g2o::VertexSE2* node;  					// node instance

};

}	 // namespace hdl_graph_slam
#endif