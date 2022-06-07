#include <mutex>
#include <boost/format.hpp>
#include <Eigen/Dense>

#include <ros/ros.h>
#include <geodesy/utm.h>

#include <visualization_msgs/MarkerArray.h>
#include <pcl_ros/point_cloud.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2_ros/transform_listener.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <geographic_msgs/GeoPoint.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <hdl_graph_slam/ros_utils.hpp>
#include <hdl_graph_slam/registrations.hpp>
#include <hdl_graph_slam/ros_time_hash.hpp>

#include <hdl_graph_slam/graph_slam.hpp>
#include <hdl_graph_slam/keyframe.hpp>
#include <hdl_graph_slam/keyframe_updater.hpp>
#include <hdl_graph_slam/loop_detector.hpp>
#include <hdl_graph_slam/information_matrix_calculator.hpp>
#include <hdl_graph_slam/building_tools.hpp>
#include <fast_gicp/gicp/fast_gicp.hpp>

#include <g2o/types/slam2d/edge_se2.h>
#include <g2o/types/slam2d/vertex_se2.h>
#include <g2o/edge_se3_plane.hpp>
#include <g2o/edge_se3_priorxy.hpp>
#include <g2o/edge_se3_priorxyz.hpp>

namespace hdl_graph_slam {

class DeltaGraphSlamNodelet : public nodelet::Nodelet {
  typedef pcl::PointXYZ PointT;
  typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> ApproxSyncPolicy;
public:
  virtual void onInit() {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    tf_buffer.reset(new tf2_ros::Buffer());
    tf_listener.reset(new tf2_ros::TransformListener(*tf_buffer));

    // init parameters
    map_frame_id = private_nh.param<std::string>("map_frame_id", "map");
    odom_frame_id = private_nh.param<std::string>("odom_frame_id", "odom");
    trans_odom2map.setIdentity();

    max_keyframes_per_update = private_nh.param<int>("max_keyframes_per_update", 10);

    graph_slam.reset(new GraphSLAM(private_nh.param<std::string>("g2o_solver_type", "lm_var")));
    keyframe_updater.reset(new KeyframeUpdater(private_nh));
    loop_detector.reset(new LoopDetector(private_nh));
    inf_calclator.reset(new InformationMatrixCalculator(private_nh));

    points_topic = private_nh.param<std::string>("points_topic", "/velodyne_points");

    // subscribers
    odom_sub.reset(new message_filters::Subscriber<nav_msgs::Odometry>(mt_nh, "/odom", 256));
    cloud_sub.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(mt_nh, "/filtered_points", 32));
    flat_cloud_sub.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(mt_nh, "/flat_filtered_points", 32));
    sync.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(32), *odom_sub, *cloud_sub, *flat_cloud_sub));
    sync->registerCallback(boost::bind(&DeltaGraphSlamNodelet::cloud_callback, this, _1, _2, _3));
    navsat_sub = mt_nh.subscribe("/gps/navsat", 1, &DeltaGraphSlamNodelet::navsat_callback, this); 
    
    // publishers
    buildings_pub = mt_nh.advertise<sensor_msgs::PointCloud2>("/delta_graph_slam/buildings_cloud", 16);
    trans_buildings_pub = mt_nh.advertise<sensor_msgs::PointCloud2>("/delta_graph_slam/trans_buildings_cloud", 16);
    aligned_pub = mt_nh.advertise<sensor_msgs::PointCloud2>("/delta_graph_slam/aligned_cloud", 16);
    markers_pub = mt_nh.advertise<visualization_msgs::MarkerArray>("/delta_graph_slam/markers", 16);
    odom2map_pub = mt_nh.advertise<geometry_msgs::TransformStamped>("/delta_graph_slam/odom2pub", 16);
    read_until_pub = mt_nh.advertise<std_msgs::Header>("/delta_graph_slam/read_until", 32);

    double graph_update_interval = private_nh.param<double>("graph_update_interval", 3.0);
    optimization_timer = mt_nh.createWallTimer(ros::WallDuration(graph_update_interval), &DeltaGraphSlamNodelet::optimization_timer_callback, this);

    registration.reset(new fast_gicp::FastGICP<PointT, PointT>());
    registration->setNumThreads(private_nh.param<int>("delta_num_threads", 0));
    registration->setMaximumIterations(private_nh.param<int>("delta_maximum_iterations", 128));
    registration->setMaxCorrespondenceDistance(private_nh.param<double>("delta_max_correspondece_distance", 2.0));
    registration->setDebugPrint(private_nh.param<bool>("delta_debug_print", true));

    // correct initial pose
    Eigen::Rotation2Df rot(-M_PI*(private_nh.param<float>("delta_init_angle", 31.0f)/180.0f));
    Eigen::Isometry2f trans = Eigen::Isometry2f::Identity();
    trans(0,2) = private_nh.param<float>("delta_init_x", 1.0f);
    trans(1,2) = private_nh.param<float>("delta_init_y", 0.0f);
    trans.linear() = rot.toRotationMatrix();

    trans_odom2map_mutex.lock();
    trans_odom2map = trans.matrix();
    trans_odom2map_mutex.unlock();

    if(odom2map_pub.getNumSubscribers()) {
      geometry_msgs::TransformStamped ts = matrix2transform(ros::Time::now(), trans.matrix().cast<float>(), map_frame_id, odom_frame_id);
      odom2map_pub.publish(ts);
    }
  }

private:
  /**
   * @brief received point clouds are pushed to #keyframe_queue
   * @param odom_msg
   * @param cloud_msg
   */
  void cloud_callback(const nav_msgs::OdometryConstPtr& odom_msg, 
                      const sensor_msgs::PointCloud2::ConstPtr& cloud_msg, 
                      const sensor_msgs::PointCloud2::ConstPtr& flat_cloud_msg) {

    const ros::Time& stamp = cloud_msg->header.stamp;
    Eigen::Isometry3d odom = odom2isometry(odom_msg);
    Eigen::Isometry2d odom2D(transform3Dto2D(odom.matrix().cast<float>()).cast<double>());

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if(base_frame_id.empty()) {
      base_frame_id = cloud_msg->header.frame_id;
    }

    pcl::PointCloud<PointT>::Ptr flat_cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*flat_cloud_msg, *flat_cloud);
    if(base_frame_id.empty()) {
      base_frame_id = flat_cloud_msg->header.frame_id;
    }

    Eigen::Matrix3f map_pose = trans_odom2map*odom2D.matrix().cast<float>();

    // transforming keyframe's odom (ENU) coordinates into UTM coordinates 
    geodesy::UTMPoint utm;
    utm.easting   = zero_utm->x() + map_pose(0,2);
    utm.northing  = zero_utm->y() + map_pose(1,2);
    utm.altitude  = 0;
    utm.zone = zero_utm_zone;
    utm.band = zero_utm_band;

    // convert from utm to lla
    geographic_msgs::GeoPoint lla = geodesy::toMsg(utm); 

    // download and parse buildings
    std::vector<Building::Ptr> buildings; 
    buildings = buildings_manager->getBuildings(lla.latitude, lla.longitude);

    // skip keyframe if there are no buildings within radius
    if(!buildings.empty()){
      
      // buildings_cloud contains all buildings' cloud
      pcl::PointCloud<PointT>::Ptr buildings_cloud(new pcl::PointCloud<PointT>);
      buildings_cloud->header.frame_id = "map";
      for(Building::Ptr building : buildings){
        *(buildings_cloud) += *(building->cloud);
      }

      // building cloud transformed in velo_link frame
      pcl::PointCloud<PointT>::Ptr trans_buildings_cloud(new pcl::PointCloud<PointT>);
      trans_odom2map_mutex.lock();
      pcl::transformPointCloud(*buildings_cloud, *trans_buildings_cloud, transform2Dto3D(map_pose.inverse()));
      trans_odom2map_mutex.unlock();

      trans_buildings_cloud->header = flat_cloud->header;
      trans_buildings_pub.publish(*trans_buildings_cloud);

      registration->setInputSource(flat_cloud);
      registration->setInputTarget(trans_buildings_cloud);

      pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
      registration->align(*aligned);
      Eigen::Matrix4f finalTransformation = registration->getFinalTransformation();
      Eigen::Isometry2d transformation(transform3Dto2D(finalTransformation).cast<double>());

      aligned->header = flat_cloud->header;
      aligned_pub.publish(*aligned);
    }

    if(!keyframe_updater->update(odom2D)) {
      std::lock_guard<std::mutex> lock(keyframe_queue_mutex);
      if(keyframe_queue.empty()) {
        std_msgs::Header read_until;
        read_until.stamp = stamp + ros::Duration(3, 0);
        read_until.frame_id = points_topic;
        read_until_pub.publish(read_until);
        read_until.frame_id = "/filtered_points";
        read_until_pub.publish(read_until);
      }
      return;
    }

    double accum_d = keyframe_updater->get_accum_distance();
    KeyFrame::Ptr keyframe(new KeyFrame(stamp, odom, odom2D, accum_d, cloud, flat_cloud));

    std::lock_guard<std::mutex> lock(keyframe_queue_mutex);
    keyframe_queue.push_back(keyframe);
  }

  /**
   * @brief this method adds all the keyframes in #keyframe_queue to the pose graph (odometry edges)
   * @return if true, at least one keyframe was added to the pose graph
   */
  bool flush_keyframe_queue() {
    std::lock_guard<std::mutex> lock(keyframe_queue_mutex);

    if(keyframe_queue.empty()) {
      return false;
    }

    trans_odom2map_mutex.lock();
    Eigen::Isometry2d odom2map(trans_odom2map.cast<double>());
    trans_odom2map_mutex.unlock();

    int num_processed = 0;
    for(int i = 0; i < std::min<int>(keyframe_queue.size(), max_keyframes_per_update); i++) {
      num_processed = i;

      const auto& keyframe = keyframe_queue[i];
      // new_keyframes will be tested later for loop closure
      new_keyframes.push_back(keyframe);

      // add pose node
      Eigen::Isometry2d odom = odom2map * keyframe->odom2D;
      keyframe->node = graph_slam->add_se2_node(odom);
      keyframe_hash[keyframe->stamp] = keyframe;

      // fix the first node
      if(keyframes.empty() && new_keyframes.size() == 1) {
        if(private_nh.param<bool>("fix_first_node", true)) {
          Eigen::MatrixXd inf = Eigen::MatrixXd::Identity(3, 3);
          std::stringstream sst(private_nh.param<std::string>("fix_first_node_stddev", "1 1 1"));
          for(int i = 0; i < 3; i++) {
            double stddev = 1.0;
            sst >> stddev;
            inf(i, i) = 1.0 / stddev;
          }

          anchor_node = graph_slam->add_se2_node(Eigen::Isometry2d::Identity());
          anchor_node->setFixed(true);
          anchor_edge = graph_slam->add_se2_edge(anchor_node, keyframe->node, Eigen::Isometry2d::Identity(), inf);
        }
      }

      if(i == 0 && keyframes.empty()) {
        continue;
      }

      // add edge between consecutive keyframes
      const auto& prev_keyframe = i == 0 ? keyframes.back() : keyframe_queue[i - 1];

      Eigen::Isometry3d relative_pose = keyframe->odom.inverse() * prev_keyframe->odom;
      Eigen::Isometry2d relative_pose2D = keyframe->odom2D.inverse() * prev_keyframe->odom2D;
      Eigen::MatrixXd information = inf_calclator->calc_information_matrix(keyframe->cloud, prev_keyframe->cloud, relative_pose);
      auto edge = graph_slam->add_se2_edge(keyframe->node, prev_keyframe->node, relative_pose2D, information);
      graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("odometry_edge_robust_kernel", "NONE"), private_nh.param<double>("odometry_edge_robust_kernel_size", 1.0));
    }

    // std_msgs::Header read_until;
    // read_until.stamp = keyframe_queue[num_processed]->stamp + ros::Duration(10, 0);
    // read_until.frame_id = points_topic;
    // read_until_pub.publish(read_until);
    // read_until.frame_id = "/filtered_points";
    // read_until_pub.publish(read_until);

    keyframe_queue.erase(keyframe_queue.begin(), keyframe_queue.begin() + num_processed + 1);
    return true;
  }

  void update_building_nodes() {
    if(!buildings_manager) {
      return;
    }

    for(auto& keyframe : new_keyframes) {
      // transforming keyframe's odom (ENU) coordinates into UTM coordinates 
      geodesy::UTMPoint utm;
      utm.easting   = zero_utm->x() + keyframe->odom2D.translation().x();
      utm.northing  = zero_utm->y() + keyframe->odom2D.translation().y();
      utm.altitude  = 0;
      utm.zone = zero_utm_zone;
      utm.band = zero_utm_band;

      // convert from utm to lla
      geographic_msgs::GeoPoint lla = geodesy::toMsg(utm); 

      // download and parse buildings
      std::vector<Building::Ptr> buildings; 
      buildings = buildings_manager->getBuildings(lla.latitude, lla.longitude);

      // skip keyframe if there are no buildings within radius
      if(buildings.empty()){
        continue;
      }

      // buildings_cloud contains all buildings' cloud
      pcl::PointCloud<PointT>::Ptr buildings_cloud(new pcl::PointCloud<PointT>);
      for(Building::Ptr building : buildings){
        *(buildings_cloud) += *(building->cloud);
      }

      // lidar cloud transformed into odom frame
      // pcl::PointCloud<PointT>::Ptr odom_cloud(new pcl::PointCloud<PointT>);
      // pcl::transformPointCloud(*(keyframe->flat_cloud), *odom_cloud, keyframe->odom.matrix());

      registration->setInputSource(keyframe->flat_cloud);
      registration->setInputTarget(buildings_cloud);

      pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
      registration->align(*aligned);
      Eigen::Isometry2d transformation(transform3Dto2D(registration->getFinalTransformation()).cast<double>());

      aligned->header.stamp = ros::Time::now().nsec/1000;
      aligned->header.frame_id = "map";
      aligned_pub.publish(*aligned);
      
      for(Building::Ptr building : buildings){
        if(building->node == nullptr){
          building->node = graph_slam->add_se2_node(building->pose);
          building->node->setFixed(true);
        }

        Eigen::MatrixXd information = Eigen::MatrixXd::Identity(3, 3);

        // transformation from estimated keyframe pose to building pose
        Eigen::Isometry2d blding_to_kf_trans = building->pose.inverse() * keyframe->odom2D;

        // auto edge = graph_slam->add_se2_edge(keyframe->node, building->node, blding_to_kf_trans, information);
        // graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("building_edge_robust_kernel", "NONE"), private_nh.param<double>("building_edge_robust_kernel_size", 1.0));
      }
    }

    std_msgs::Header read_until;
    read_until.stamp = new_keyframes.back()->stamp + ros::Duration(3, 0);
    read_until.frame_id = points_topic;
    read_until_pub.publish(read_until);
    read_until.frame_id = "/filtered_points";
    read_until_pub.publish(read_until);
  }

  /**
   * @brief publish all buildings clouds
   * @return
   */
  void publish_buildings_clouds() {

    // return if manager was not initialized
    if(!buildings_manager){
      return;
    }

    // download and parse buildings
    std::vector<Building::Ptr> new_buildings = buildings_manager->getBuildings();

    // create pointcloud of all buildings
    pcl::PointCloud<PointT> buildings_cloud;
    for(Building::Ptr building : new_buildings){
      buildings_cloud += *(building->cloud);
    }

    // publish buildings cloud
    sensor_msgs::PointCloud2Ptr buildingsCloud_msg(new sensor_msgs::PointCloud2());
    pcl::toROSMsg(buildings_cloud, *buildingsCloud_msg);
    buildingsCloud_msg->header.frame_id = "map";
    buildingsCloud_msg->header.stamp = ros::Time::now();
    buildings_pub.publish(buildingsCloud_msg);
  }

  /**
   * @brief callback to store zero_utm and visualize gps positions
   * @param navsat_msg
   * @return
   */
  void navsat_callback(const sensor_msgs::NavSatFixConstPtr& navsat_msg) {
    geographic_msgs::GeoPoint gps_msg;
    gps_msg.latitude = navsat_msg->latitude;
    gps_msg.longitude = navsat_msg->longitude;
    gps_msg.altitude = navsat_msg->altitude;
    
    // convert (latitude, longitude, altitude) -> (easting, northing, altitude) in UTM coordinate
    geodesy::UTMPoint utm;
    geodesy::fromMsg(gps_msg, utm);
    Eigen::Vector2d xy(utm.easting, utm.northing);

    if(!zero_utm){
      zero_utm = xy;
      zero_utm_zone = utm.zone;
      zero_utm_band = utm.band;

      buildings_manager.reset(new BuildingTools("https://overpass-api.de", *zero_utm));
      // fetch buildings as soon as possible
      buildings_manager->getBuildings(navsat_msg->latitude, navsat_msg->longitude);
    }

    Eigen::Vector2d xy_enu(xy.x()-zero_utm->x(),xy.y()-zero_utm->y());
    gps_points.push_back(xy_enu);
  }

    /**
   * @brief this methods adds all the data in the queues to the pose graph, and then optimizes the pose graph
   * @param event
   */
  void optimization_timer_callback(const ros::WallTimerEvent& event) {

    publish_buildings_clouds();

    std::lock_guard<std::mutex> lock(main_thread_mutex);

    // add keyframes in the queues to the pose graph
    bool keyframe_updated = flush_keyframe_queue();

    if(!keyframe_updated) {
      std_msgs::Header read_until;
      read_until.stamp = ros::Time::now() + ros::Duration(5, 0);
      read_until.frame_id = points_topic;
      read_until_pub.publish(read_until);
      read_until.frame_id = "/filtered_points";
      read_until_pub.publish(read_until);
      return;
    }

    // add building nodes
    // update_building_nodes();

    // loop detection
    std::vector<Loop::Ptr> loops = loop_detector->detect(keyframes, new_keyframes, *graph_slam);
    for(const auto& loop : loops) {
      Eigen::Isometry3d relpose(loop->relative_pose.cast<double>());
      Eigen::Isometry2d relpose2D(loop->relative_pose2D.cast<double>());
      Eigen::MatrixXd information_matrix = inf_calclator->calc_information_matrix(loop->key1->cloud, loop->key2->cloud, relpose);
      auto edge = graph_slam->add_se2_edge(loop->key1->node, loop->key2->node, relpose2D, information_matrix);
      graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("loop_closure_edge_robust_kernel", "NONE"), private_nh.param<double>("loop_closure_edge_robust_kernel_size", 1.0));
    }

    std::copy(new_keyframes.begin(), new_keyframes.end(), std::back_inserter(keyframes));
    new_keyframes.clear();

    // move the first node anchor position to the current estimate of the first node pose
    // so the first node moves freely while trying to stay around the origin
    if(anchor_node && private_nh.param<bool>("fix_first_node_adaptive", true)) {
      Eigen::Isometry2d anchor_target = static_cast<g2o::VertexSE2*>(anchor_edge->vertices()[1])->estimate().toIsometry();
      anchor_node->setEstimate(anchor_target);
    }

    // optimize the pose graph
    int num_iterations = private_nh.param<int>("g2o_solver_num_iterations", 1024);
    graph_slam->optimize(num_iterations);

    // publish tf
    const auto& keyframe = keyframes.back();
    Eigen::Isometry2d trans = keyframe->estimate() * keyframe->odom2D.inverse();
    trans_odom2map_mutex.lock();
    trans_odom2map = trans.matrix().cast<float>();
    trans_odom2map_mutex.unlock();

    if(odom2map_pub.getNumSubscribers()) {
      geometry_msgs::TransformStamped ts = matrix2transform(keyframe->stamp, trans.matrix().cast<float>(), map_frame_id, odom_frame_id);
      odom2map_pub.publish(ts);
    }

    if(markers_pub.getNumSubscribers()) {
      auto markers = create_marker_array(ros::Time::now());
      markers_pub.publish(markers);
    }
  }

  /**
   * @brief create visualization marker
   * @param stamp
   * @return
   */
  visualization_msgs::MarkerArray create_marker_array(const ros::Time& stamp) const {
    visualization_msgs::MarkerArray markers;
    markers.markers.resize(5);

    // node markers
    visualization_msgs::Marker& traj_marker = markers.markers[0];
    traj_marker.header.frame_id = "map";
    traj_marker.header.stamp = stamp;
    traj_marker.ns = "keyframe_nodes";
    traj_marker.id = 0;
    traj_marker.type = visualization_msgs::Marker::SPHERE_LIST;

    traj_marker.pose.orientation.w = 1.0;
    traj_marker.scale.x = traj_marker.scale.y = traj_marker.scale.z = 0.5;

    traj_marker.points.resize(keyframes.size());
    traj_marker.colors.resize(keyframes.size());
    for(int i = 0; i < keyframes.size(); i++) {
      Eigen::Vector2d pos = keyframes[i]->node->estimate().translation();
      traj_marker.points[i].x = pos.x();
      traj_marker.points[i].y = pos.y();
      traj_marker.points[i].z = 0;

      double p = static_cast<double>(i) / keyframes.size();
      traj_marker.colors[i].r = 1.0 - p;
      traj_marker.colors[i].g = p;
      traj_marker.colors[i].b = 0.0;
      traj_marker.colors[i].a = 1.0;
    }

    // building node markers
    visualization_msgs::Marker& build_marker = markers.markers[1];
    build_marker.header.frame_id = "map";
    build_marker.header.stamp = stamp;
    build_marker.ns = "building_nodes";
    build_marker.id = 0;
    build_marker.type = visualization_msgs::Marker::SPHERE_LIST;

    build_marker.pose.orientation.w = 1.0;
    build_marker.scale.x = build_marker.scale.y = build_marker.scale.z = 0.5;

    std::vector<Building::Ptr> buildings = buildings_manager->getBuildingNodes();
    build_marker.points.resize(buildings.size());
    build_marker.colors.resize(buildings.size());
    for(int i = 0; i < buildings.size(); i++) {
      Eigen::Vector2d pos = buildings[i]->node->estimate().translation();
      build_marker.points[i].x = pos.x();
      build_marker.points[i].y = pos.y();
      build_marker.points[i].z = 0;

      double p = static_cast<double>(i) / buildings.size();
      build_marker.colors[i].r = 1.0 - p;
      build_marker.colors[i].g = p;
      build_marker.colors[i].b = 0.0;
      build_marker.colors[i].a = 1.0;
    }

    // edge markers
    visualization_msgs::Marker& edge_marker = markers.markers[2];
    edge_marker.header.frame_id = "map";
    edge_marker.header.stamp = stamp;
    edge_marker.ns = "edges";
    edge_marker.id = 2;
    edge_marker.type = visualization_msgs::Marker::LINE_LIST;

    edge_marker.pose.orientation.w = 1.0;
    edge_marker.scale.x = 0.05;

    edge_marker.points.resize(graph_slam->graph->edges().size() * 2);
    edge_marker.colors.resize(graph_slam->graph->edges().size() * 2);

    auto edge_itr = graph_slam->graph->edges().begin();
    for(int i = 0; edge_itr != graph_slam->graph->edges().end(); edge_itr++, i++) {
      g2o::HyperGraph::Edge* edge = *edge_itr;
      g2o::EdgeSE2* edge_se2 = dynamic_cast<g2o::EdgeSE2*>(edge);
      if(edge_se2) {
        g2o::VertexSE2* v1 = dynamic_cast<g2o::VertexSE2*>(edge_se2->vertices()[0]);
        g2o::VertexSE2* v2 = dynamic_cast<g2o::VertexSE2*>(edge_se2->vertices()[1]);
        Eigen::Vector2d pt1 = v1->estimate().translation();
        Eigen::Vector2d pt2 = v2->estimate().translation();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = 0;
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = 0;

        double p1 = static_cast<double>(v1->id()) / graph_slam->graph->vertices().size();
        double p2 = static_cast<double>(v2->id()) / graph_slam->graph->vertices().size();
        edge_marker.colors[i * 2].r = 1.0 - p1;
        edge_marker.colors[i * 2].g = p1;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 1.0 - p2;
        edge_marker.colors[i * 2 + 1].g = p2;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        if(std::abs(v1->id() - v2->id()) > 2) {
          edge_marker.points[i * 2].z += 0.5;
          edge_marker.points[i * 2 + 1].z += 0.5;
        }

        continue;
      }

      g2o::EdgeSE3Plane* edge_plane = dynamic_cast<g2o::EdgeSE3Plane*>(edge);
      if(edge_plane) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_plane->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2(pt1.x(), pt1.y(), 0.0);

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z();
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        edge_marker.colors[i * 2].b = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].b = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }

      g2o::EdgeSE3PriorXY* edge_priori_xy = dynamic_cast<g2o::EdgeSE3PriorXY*>(edge);
      if(edge_priori_xy) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_priori_xy->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = Eigen::Vector3d::Zero();
        pt2.head<2>() = edge_priori_xy->measurement();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z() + 0.5;
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z() + 0.5;

        edge_marker.colors[i * 2].r = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }

      g2o::EdgeSE3PriorXYZ* edge_priori_xyz = dynamic_cast<g2o::EdgeSE3PriorXYZ*>(edge);
      if(edge_priori_xyz) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_priori_xyz->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = edge_priori_xyz->measurement();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z() + 0.5;
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        edge_marker.colors[i * 2].r = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }
    }

    // sphere
    visualization_msgs::Marker& sphere_marker = markers.markers[3];
    sphere_marker.header.frame_id = "map";
    sphere_marker.header.stamp = stamp;
    sphere_marker.ns = "loop_close_radius";
    sphere_marker.id = 3;
    sphere_marker.type = visualization_msgs::Marker::SPHERE;

    if(!keyframes.empty()) {
      Eigen::Vector2d pos = keyframes.back()->node->estimate().translation();
      sphere_marker.pose.position.x = pos.x();
      sphere_marker.pose.position.y = pos.y();
      sphere_marker.pose.position.z = 0;
    }
    sphere_marker.pose.orientation.w = 1.0;
    sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = loop_detector->get_distance_thresh() * 2.0;

    sphere_marker.color.r = 1.0;
    sphere_marker.color.a = 0.1;

    // gps markers
    visualization_msgs::Marker& gps_marker = markers.markers[4];
    gps_marker.header.frame_id = "map";
    gps_marker.ns = "gps";
    gps_marker.id = 0;
    gps_marker.type = visualization_msgs::Marker::SPHERE_LIST;

    gps_marker.pose.orientation.w = 1.0;
    gps_marker.scale.x = gps_marker.scale.y = gps_marker.scale.z = 0.5;

    int no_points = gps_points.size();
    gps_marker.header.stamp = stamp;
    for(int i = 0; i < no_points; i++){
      Eigen::Vector2d xy = gps_points[i];
      gps_marker.points.resize(no_points);
      gps_marker.colors.resize(no_points);

      gps_marker.points[i].x = xy.x();
      gps_marker.points[i].y = xy.y();
      gps_marker.points[i].z = 0;

      gps_marker.colors[i].r = 1.0;
      gps_marker.colors[i].g = 1.0;
      gps_marker.colors[i].b = 1.0;
      gps_marker.colors[i].a = 0.2;
    }

    return markers;
  }

  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;
  ros::WallTimer optimization_timer;

  boost::shared_ptr<tf2_ros::Buffer> tf_buffer;
  boost::shared_ptr<tf2_ros::TransformListener> tf_listener;

  std::unique_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> cloud_sub;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> flat_cloud_sub;
  std::unique_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync;

  ros::Subscriber navsat_sub;
  ros::Publisher buildings_pub;
  ros::Publisher trans_buildings_pub;
  ros::Publisher aligned_pub;
  ros::Publisher markers_pub;

  std::string map_frame_id;
  std::string odom_frame_id;

  std::mutex trans_odom2map_mutex;
  Eigen::Matrix3f trans_odom2map;
  ros::Publisher odom2map_pub;

  std::string points_topic;
  ros::Publisher read_until_pub;

  // gps points converted to ENU coordinates
  std::vector<Eigen::Vector2d> gps_points;

  // keyframe queue
  std::string base_frame_id;
  std::mutex keyframe_queue_mutex;
  std::deque<KeyFrame::Ptr> keyframe_queue;

  boost::optional<Eigen::Vector2d> zero_utm;
  int zero_utm_zone;
  char zero_utm_band;

  BuildingTools::Ptr buildings_manager;
  fast_gicp::FastGICP<PointT, PointT>::Ptr registration;
  bool first_guess = true;
  Eigen::Matrix4f prev_guess;

  // graph slam
  // all the below members must be accessed after locking main_thread_mutex
  std::mutex main_thread_mutex;

  int max_keyframes_per_update;
  std::deque<KeyFrame::Ptr> new_keyframes;

  g2o::VertexSE2* anchor_node;
  g2o::EdgeSE2* anchor_edge;
  std::vector<KeyFrame::Ptr> keyframes;
  std::unordered_map<ros::Time, KeyFrame::Ptr, RosTimeHash> keyframe_hash;

  std::unique_ptr<GraphSLAM> graph_slam;
  std::unique_ptr<LoopDetector> loop_detector;
  std::unique_ptr<KeyframeUpdater> keyframe_updater;

  std::unique_ptr<InformationMatrixCalculator> inf_calclator;
};

} // namespace hdl_graph_slam

PLUGINLIB_EXPORT_CLASS(hdl_graph_slam::DeltaGraphSlamNodelet, nodelet::Nodelet)