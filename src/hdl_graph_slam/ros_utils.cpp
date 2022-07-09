#include <hdl_graph_slam/ros_utils.hpp>

namespace hdl_graph_slam {

/**
 * @brief convert Eigen::Matrix to geometry_msgs::TransformStamped
 * @param stamp            timestamp
 * @param pose             Eigen::Matrix to be converted
 * @param frame_id         tf frame_id
 * @param child_frame_id   tf child frame_id
 * @return converted TransformStamped
 */
geometry_msgs::TransformStamped matrix2transform(const ros::Time& stamp, const Eigen::Matrix3f& pose, const std::string& frame_id, const std::string& child_frame_id) {
  // rotation 2D to 3D
  Eigen::Rotation2Df rot2D(pose.block<2,2>(0,0));
  Eigen::Matrix3f rot;
  rot = Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX())
    * Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY())
    * Eigen::AngleAxisf(rot2D.angle(), Eigen::Vector3f::UnitZ());

  Eigen::Quaternionf quat(rot);
  quat.normalize();
  geometry_msgs::Quaternion odom_quat;
  odom_quat.w = quat.w();
  odom_quat.x = quat.x();
  odom_quat.y = quat.y();
  odom_quat.z = quat.z();

  geometry_msgs::TransformStamped odom_trans;
  odom_trans.header.stamp = stamp;
  odom_trans.header.frame_id = frame_id;
  odom_trans.child_frame_id = child_frame_id;

  odom_trans.transform.translation.x = pose(0, 2);
  odom_trans.transform.translation.y = pose(1, 2);
  odom_trans.transform.translation.z = .0f;
  odom_trans.transform.rotation = odom_quat;

  return odom_trans;
}

Eigen::Isometry3d pose2isometry(const geometry_msgs::Pose& pose) {
  Eigen::Isometry3d mat = Eigen::Isometry3d::Identity();
  mat.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  mat.linear() = Eigen::Quaterniond(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z).toRotationMatrix();
  return mat;
}

Eigen::Isometry3d tf2isometry(const tf::StampedTransform& trans) {
  Eigen::Isometry3d mat = Eigen::Isometry3d::Identity();
  mat.translation() = Eigen::Vector3d(trans.getOrigin().x(), trans.getOrigin().y(), trans.getOrigin().z());
  mat.linear() = Eigen::Quaterniond(trans.getRotation().w(), trans.getRotation().x(), trans.getRotation().y(), trans.getRotation().z()).toRotationMatrix();
  return mat;
}

geometry_msgs::Pose isometry2pose(const Eigen::Isometry3d& mat) {
  Eigen::Quaterniond quat(mat.linear());
  Eigen::Vector3d trans = mat.translation();

  geometry_msgs::Pose pose;
  pose.position.x = trans.x();
  pose.position.y = trans.y();
  pose.position.z = trans.z();
  pose.orientation.w = quat.w();
  pose.orientation.x = quat.x();
  pose.orientation.y = quat.y();
  pose.orientation.z = quat.z();

  return pose;
}

Eigen::Isometry3d odom2isometry(const nav_msgs::OdometryConstPtr& odom_msg) {
  const auto& orientation = odom_msg->pose.pose.orientation;
  const auto& position = odom_msg->pose.pose.position;

  Eigen::Quaterniond quat;
  quat.w() = orientation.w;
  quat.x() = orientation.x;
  quat.y() = orientation.y;
  quat.z() = orientation.z; //

  Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity();
  isometry.linear() = quat.toRotationMatrix();
  isometry.translation() = Eigen::Vector3d(position.x, position.y, position.z);
  return isometry;
}

/**
 * @brief object orientations could have multiple combination of roll, pitch and yaw,
 * for example an orientation (ψ,θ,φ) is equal to (ψ,θ,φ) + (i*π,+j*π,k*π) with any combination of integers i,j,k s.t. abs(i)=abs(j)=abs(k),
 * this computation ensures that we choose the euler angles vector with minimum norm.
 * assumption is that all angles are in range [-π,π]
 * @param euler_angs roll, pitch and yaw vector
 */
Eigen::Vector3f normalize_euler_angs(Eigen::Vector3f euler_angs){
  Eigen::Vector3f euler_angs_norm;

  euler_angs_norm(0) = euler_angs(0) - M_PI*(euler_angs(0)>=.0f ?1 :-1);
  euler_angs_norm(1) = euler_angs(1) - M_PI*(euler_angs(1)>=.0f ?1 :-1);
  euler_angs_norm(2) = euler_angs(2) - M_PI*(euler_angs(2)>=.0f ?1 :-1);

  return euler_angs_norm.norm() < euler_angs.norm() ?euler_angs_norm :euler_angs;
}

Eigen::Matrix4f transform2Dto3D(Eigen::Matrix3f trans2D){
  // rotation 2D to 3D
  Eigen::Rotation2Df rot2D(trans2D.block<2,2>(0,0));
  Eigen::Matrix3f rot;
  rot = Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX())
    * Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY())
    * Eigen::AngleAxisf(rot2D.angle(), Eigen::Vector3f::UnitZ());

  if(!rot.isUnitary()){
    ROS_WARN("Error during 2D transform conversion to 3D, matrix should be unitary!");
    std::cout << rot << std::endl << "is unitary: " << rot.isUnitary() << std::endl;
  }

  // translation 2D to 3D
  Eigen::Vector3f translation(trans2D(0,2), trans2D(1,2), 0);
  
  Eigen::Matrix4f trans3D = Eigen::Matrix4f::Identity();
  trans3D.block<3,3>(0,0) = rot;
  trans3D.block<3,1>(0,3) = translation;

  return trans3D;
}

Eigen::Matrix3f transform3Dto2D(Eigen::Matrix4f trans3D){
  // rotation 3D to 2D
  Eigen::Quaternionf quat(trans3D.block<3,3>(0,0));
  Eigen::Vector3f euler_angs = quat.toRotationMatrix().eulerAngles(0,1,2);
  // cannot simply use yaw angle removing the others without normalization
  euler_angs = normalize_euler_angs(euler_angs);
  Eigen::Rotation2Df rot2D(euler_angs.z());

  // translation 3D to 2D
  Eigen::Vector2f translation(trans3D(0,3), trans3D(1,3));

  Eigen::Matrix3f trans2D = Eigen::Matrix3f::Identity();
  trans2D.block<2,2>(0,0) = rot2D.toRotationMatrix();
  trans2D.block<2,1>(0,2) = translation;

  return trans2D;
}

pcl::PointCloud<PointT>::Ptr interpolate(PointT a, PointT b) {
	// linear interpolation: return a line of points between a and b (1 every 2cm)
	const float sample_step = 0.02;
	pcl::PointCloud<PointT>::Ptr building_pointcloud(new pcl::PointCloud<PointT>);
	Eigen::Vector3f AtoB = b.getVector3fMap()-a.getVector3fMap();
	Eigen::Vector3f AtoBnormalized = AtoB.normalized();
	float AtoBnorm = AtoB.norm();

	for(float i=0; i<=AtoBnorm; i=i+sample_step) {
		PointT pointXYZ;

		pointXYZ.x = a.x + i*AtoBnormalized.x();
		pointXYZ.y = a.y + i*AtoBnormalized.y();
		pointXYZ.z = 0;

		building_pointcloud->push_back(pointXYZ);
	}
	return building_pointcloud;
}

}  // namespace hdl_graph_slam
