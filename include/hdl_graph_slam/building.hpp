#ifndef BUILDING_HPP
#define BUILDING_HPP

#include <iostream>
#include <string>
#include <Eigen/Dense>

#include <hdl_graph_slam/line_based_scanmatcher.hpp>
#include <g2o/types/slam2d/vertex_se2.h>
#include <pcl/common/transforms.h>
#include <hdl_graph_slam/ros_utils.hpp>

namespace g2o {
class VertexSE2;
}	 // namespace g2o

namespace hdl_graph_slam {

class Building {
	public:
	typedef pcl::PointXYZ PointT;
	typedef boost::shared_ptr<Building> Ptr;

	Building(g2o::VertexSE2* node);
	pcl::PointCloud<PointT>::Ptr getCloud();
	std::vector<LineFeature::Ptr> getLines();
	std::vector<Eigen::Vector3d> getPoints();
	Eigen::Isometry2d estimate() const;

	std::string id;
	Eigen::Isometry2d pose;					// pose (estimated by OpenStreetMap)
	pcl::PointCloud<PointT>::Ptr cloud;
	std::vector<LineFeature::Ptr> lines;
	std::vector<Eigen::Vector3d> points;

	g2o::VertexSE2* node;  					// node instance
};

}	 // namespace hdl_graph_slam
#endif