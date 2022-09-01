#include <hdl_graph_slam/building_tools.hpp>

namespace hdl_graph_slam {

BuildingTools::BuildingTools(std::string host, Eigen::Vector3d origin, double scale, GraphSLAM* graph_slam, double radius, double buffer_radius) {
	this->host = host;
	this->origin = origin;
	this->scale = scale;
	this->radius = radius;
	this->buffer_radius = buffer_radius;
	this->graph_slam.reset(graph_slam);
}

std::vector<Building::Ptr> BuildingTools::getBuildings(geographic_msgs::GeoPoint gps) {

	if(!async_handle.joinable() || async_handle.try_join_for(boost::chrono::milliseconds(1))){
		async_handle = boost::thread(boost::bind(&BuildingTools::downloadBuildings, this, gps));
	}

	// polling with timeout, in case we can get all buildings in the first call
	ros::WallTime timeout = ros::WallTime::now() + ros::WallDuration(2);
	while(xml_tree.empty() && ros::WallTime::now() < timeout){
		ros::WallDuration(0.1).sleep();
	}

	std::vector<Building::Ptr> buildings_in_range;
	buildings_in_range = parseBuildings(gps);
	
	return buildings_in_range;
}

std::vector<Building::Ptr> BuildingTools::getBuildingNodes() {
	std::vector<Building::Ptr> buildingNodes;
	for(Building::Ptr building : buildings){
		if(building->node){
      buildingNodes.push_back(building);
		}
	}
	return buildingNodes;
}

void BuildingTools::downloadBuildings(geographic_msgs::GeoPoint gps) {

	Eigen::Vector3d pointXYZ = toEnu(gps);
	if(!xml_tree.empty() && (pointXYZ - buffer_center).norm() < (buffer_radius / 2.0)){
		return;
	}

	std::string xml_response;
	try {
		std::string url = string_format(
			"%s/api/interpreter?data=way[%27building%27](around:%f,%f,%f);%20(._;%%3E;);out;",
			host.data(),
			buffer_radius,
			gps.latitude,
			gps.longitude
		);

		curlpp::Easy request;
		curlpp::options::Url url_opt(url);
		curlpp::options::Timeout timeout(3);

		// Setting the URL to retrive.
		request.setOpt(url_opt);
		request.setOpt(timeout);

		std::ostringstream os;
		os << request;
		xml_response = os.str();
	}
	catch ( curlpp::LogicError & e ) {
		std::cout << "curlpp logic error: " << e.what() << std::endl;
		return;
	}
	catch ( curlpp::RuntimeError & e ) {
		std::cout << "curlpp runtime error: " << e.what() << std::endl;
		return;
	}

	pt::ptree xml_tree_tmp;
	std::vector<Node> nodes_tmp;
	std::stringstream xml_stream(xml_response);
	read_xml(xml_stream, xml_tree_tmp);

	try {
	BOOST_FOREACH(pt::ptree::value_type &tree_node, xml_tree_tmp.get_child("osm")) {
		if(tree_node.first == "node") {
			Node node;
			node.id = tree_node.second.get<std::string>("<xmlattr>.id");
			node.lat = tree_node.second.get<double>("<xmlattr>.lat");
			node.lon = tree_node.second.get<double>("<xmlattr>.lon");
			nodes_tmp.push_back(node);
		}
	}} catch(pt::ptree_error &e) {
		std::cerr<< "No xml! error:" << e.what() << std::endl;
		return;
	}

	// update xml tree thread safe
	std::lock_guard<std::mutex> lock(xml_tree_mutex);
	nodes = nodes_tmp;
	xml_tree = xml_tree_tmp;
	buffer_center = toEnu(gps);
}

std::vector<Building::Ptr> BuildingTools::parseBuildings(geographic_msgs::GeoPoint gps) {
	std::lock_guard<std::mutex> lock(xml_tree_mutex);
	std::vector<Building::Ptr> buildings_in_range;

	if(xml_tree.empty()){
		return buildings_in_range;
	}

	try {
	BOOST_FOREACH(pt::ptree::value_type &tree_node, xml_tree.get_child("osm")) {
		if(tree_node.first == "way") {
			std::string id = tree_node.second.get<std::string>("<xmlattr>.id");

			if(!isBuildingInRadius(tree_node, gps)){
				continue;
			}

			if(buildings_map.count(id)){
				buildings_in_range.push_back(buildings_map[id]);
				continue;
			}

			std::vector<std::string> nd_refs;

			BOOST_FOREACH(pt::ptree::value_type &tree_node_2, tree_node.second) {
				if(tree_node_2.first == "nd") {
					std::string nd_ref = tree_node_2.second.get<std::string>("<xmlattr>.ref");
					nd_refs.push_back(nd_ref);
				}
			}

			Eigen::Isometry2d pose = getBuildingPose(nd_refs);
			g2o::VertexSE2* node = graph_slam->add_se2_node(pose);
			node->setFixed(false);

			Building::Ptr new_building(new Building(node));

			new_building->id = id;
			new_building->pose = pose;
			buildPointCloud(nd_refs, new_building);

			buildings_in_range.push_back(new_building);
			buildings.push_back(new_building);
			buildings_map[id] = new_building;
		}
	}} catch(pt::ptree_error &e) {
		std::cerr<< "No xml! error:" << e.what() << std::endl;
	}
	return buildings_in_range;
}

Building::Ptr BuildingTools::buildPointCloud(std::vector<std::string> nd_refs, Building::Ptr new_building) {
	new_building->cloud.reset(new pcl::PointCloud<PointT>);

	if(nd_refs.size() == 0){
		return new_building;
	}

	Node node = getNode(nd_refs[0]);
	Eigen::Vector3d previous = toEnu(node.lat, node.lon);
	new_building->points.push_back(previous);

	for(int i = 1; i < nd_refs.size(); i++) {
		node = getNode(nd_refs[i]);
		Eigen::Vector3d pointXYZ = toEnu(node.lat, node.lon);
		new_building->points.push_back(pointXYZ);

		LineFeature::Ptr line(new LineFeature());
		line->pointA = previous;
		line->pointB = pointXYZ;

		new_building->lines.push_back(line);
		*(new_building->cloud) += *interpolate(previous.cast<float>(), pointXYZ.cast<float>());

		previous = pointXYZ;
	}

	new_building->cloud->header.frame_id = "map";
	pcl_conversions::toPCL(ros::Time::now(), new_building->cloud->header.stamp);

	return new_building;
}

BuildingTools::Node BuildingTools::getNode(std::string nd_ref) {
	for(Node &node : nodes) {
		if(nd_ref.compare(node.id) == 0) {
			return node;
		}
	}
	return Node();
}

// toEnu converts to enu coordinates from lla
Eigen::Vector3d BuildingTools::toEnu(geographic_msgs::GeoPoint gps) {

	Eigen::Vector3d xyz = translation_from_gps_msg(gps, scale);
	xyz -= origin;
	xyz.z() = 0.;

	return xyz;
}

Eigen::Vector3d BuildingTools::toEnu(double latitude, double longitude) {

	geographic_msgs::GeoPoint gps;

	gps.latitude = latitude;
	gps.longitude = longitude;
	gps.altitude = 0.;

	Eigen::Vector3d xyz = translation_from_gps_msg(gps, scale);
	xyz -= origin;
	xyz.z() = 0.;

	return xyz;
}

bool BuildingTools::isBuildingInRadius(pt::ptree::value_type &child_tree_node, geographic_msgs::GeoPoint gps){
	try {

	Eigen::Vector3d enu_coords = toEnu(gps);
	BOOST_FOREACH(pt::ptree::value_type &tree_node, child_tree_node.second) {
		if(tree_node.first == "nd") {
			std::string nd_ref = tree_node.second.get<std::string>("<xmlattr>.ref");
			Node node = getNode(nd_ref);

			Eigen::Vector3d pointXYZ = toEnu(node.lat, node.lon);
			
			if((pointXYZ-enu_coords).norm() < radius){
				return true;
			}
		}
	}
	
	} catch(pt::ptree_error &e) {
		std::cerr<< "No xml! error:" << e.what() << std::endl;
	}
	return false;
}

/**
 * @brief this method computes the building position using its middle point
 * @return isometry2d with null orientation
 */
Eigen::Isometry2d BuildingTools::getBuildingPose(std::vector<std::string> nd_refs){
	double x_min = std::numeric_limits<double>::max();
	double x_max = std::numeric_limits<double>::lowest();
	double y_min = std::numeric_limits<double>::max();
	double y_max = std::numeric_limits<double>::lowest();

	for(std::string nd_ref : nd_refs) {
		Node node = getNode(nd_ref);
		Eigen::Vector3d enu_coords = toEnu(node.lat, node.lon);
		
		x_min = enu_coords.x() < x_min ? enu_coords.x() : x_min;
		x_max = enu_coords.x() > x_max ? enu_coords.x() : x_max;
		y_min = enu_coords.y() < y_min ? enu_coords.y() : y_min;
		y_max = enu_coords.y() > y_max ? enu_coords.y() : y_max;
	}

	// take middle point as building pose	
	Eigen::Vector2d PointXY;
	PointXY.x() = (x_min+x_max)/2.f;
	PointXY.y() = (y_min+y_max)/2.f;

	Eigen::Isometry2d buildingPose = Eigen::Isometry2d::Identity();
	buildingPose.translation() = PointXY;

	return buildingPose;
}

}