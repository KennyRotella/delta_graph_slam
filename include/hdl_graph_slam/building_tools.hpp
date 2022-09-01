#ifndef BUILDING_TOOLS
#define BUILDING_TOOLS

#include <geodesy/utm.h>
#include <geodesy/wgs84.h>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/thread/thread.hpp>
#include <hdl_graph_slam/line_based_scanmatcher.hpp>
#include <hdl_graph_slam/building.hpp>
#include <hdl_graph_slam/ros_utils.hpp>
#include <hdl_graph_slam/graph_slam.hpp>
#include <pcl/common/distances.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geographic_msgs/GeoPoint.h>
#include <mutex>
#include <fstream>
#include <regex>
#include <string>
#include <cmath>
#include <sstream>
#include <cstdlib>
#include <iomanip>
#include <map>

#include <curlpp/cURLpp.hpp>
#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/Exception.hpp>

namespace pt = boost::property_tree;

namespace hdl_graph_slam {

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
	int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
	if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
	auto size = static_cast<size_t>( size_s );
	std::unique_ptr<char[]> buf( new char[ size ] );
	std::snprintf( buf.get(), size, format.c_str(), args ... );
	return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

using PointT = pcl::PointXYZ;

class BuildingTools {
public:
	typedef boost::shared_ptr<BuildingTools> Ptr;
	BuildingTools(std::string host, Eigen::Vector3d origin, double scale, GraphSLAM* graph_slam, double radius=35, double buffer_radius=120);
	std::vector<Building::Ptr> getBuildings(geographic_msgs::GeoPoint gps);
	std::vector<Building::Ptr> getBuildings(){ return buildings; };
	std::vector<Building::Ptr> getBuildingNodes();
	
private:
	struct Node {
		std::string id;
		double lat;
		double lon;
	};
	std::string host;
	Eigen::Vector3d origin;
	double scale;
	double radius;
	double buffer_radius;
	Eigen::Vector3d buffer_center;
	std::vector<Node> nodes;
	pt::ptree xml_tree;
	std::mutex xml_tree_mutex;
	boost::thread async_handle;
	std::map<std::string,Building::Ptr> buildings_map;
	std::vector<Building::Ptr> buildings;
	std::unique_ptr<GraphSLAM> graph_slam;

	void downloadBuildings(geographic_msgs::GeoPoint gps);
	std::vector<Building::Ptr> parseBuildings(geographic_msgs::GeoPoint gps);
	Building::Ptr buildPointCloud(std::vector<std::string> nd_refs, Building::Ptr new_building);
	Node getNode(std::string nd_ref);
	Eigen::Vector3d toEnu(geographic_msgs::GeoPoint gps);
	Eigen::Vector3d toEnu(double latitude, double longitude);
	bool isBuildingInRadius(pt::ptree::value_type &tree_node, geographic_msgs::GeoPoint gps);
	Eigen::Isometry2d getBuildingPose(std::vector<std::string> nd_refs);
};

}
#endif