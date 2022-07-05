#ifndef BUILDING_TOOLS
#define BUILDING_TOOLS

#include <geodesy/utm.h>
#include <geodesy/wgs84.h>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/thread/thread.hpp>
#include <hdl_graph_slam/building.hpp>
#include <pcl/common/distances.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
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
	BuildingTools() {}
	BuildingTools(std::string host, Eigen::Vector2d zero_utm, double radius=20, double buffer_radius=100):
		host(host),
		zero_utm(zero_utm),
		radius(radius),
		buffer_radius(buffer_radius) {}
	std::vector<Building::Ptr> getBuildings(double lat, double lon);
	std::vector<Building::Ptr> getBuildings(){ return buildings; };
	std::vector<Building::Ptr> getBuildingNodes();
	
private:
	struct Node {
		std::string id;
		double lat;
		double lon;
	};
	std::string host;
	Eigen::Vector2d zero_utm;
	double radius;
	double buffer_radius;
	Eigen::Vector3f buffer_center;
	std::vector<Node> nodes;
	pt::ptree xml_tree;
	std::mutex xml_tree_mutex;
	boost::thread async_handle;
	std::map<std::string,Building::Ptr> buildings_map;
	std::vector<Building::Ptr> buildings;

	void downloadBuildings(double lat, double lon);
	std::vector<Building::Ptr> parseBuildings(double lat, double lon);
	pcl::PointCloud<PointT>::Ptr buildPointCloud(std::vector<std::string> nd_refs);
	Node getNode(std::string nd_ref);
	pcl::PointCloud<PointT>::Ptr interpolate(PointT a, PointT b);
	PointT toEnu(Eigen::Vector3d lla);
	bool isBuildingInRadius(pt::ptree::value_type &tree_node, double lat, double lon);
	Eigen::Isometry2d getBuildingPose(std::vector<std::string> nd_refs);
};

}
#endif