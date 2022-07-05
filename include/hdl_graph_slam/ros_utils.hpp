// SPDX-License-Identifier: BSD-2-Clause

#ifndef ROS_UTILS_HPP
#define ROS_UTILS_HPP

#include <Eigen/Dense>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_listener.h>

namespace hdl_graph_slam {

/**
 * @brief convert Eigen::Matrix to geometry_msgs::TransformStamped
 * @param stamp            timestamp
 * @param pose             Eigen::Matrix to be converted
 * @param frame_id         tf frame_id
 * @param child_frame_id   tf child frame_id
 * @return converted TransformStamped
 */
geometry_msgs::TransformStamped matrix2transform(const ros::Time& stamp, const Eigen::Matrix3f& pose, const std::string& frame_id, const std::string& child_frame_id);

Eigen::Isometry3d pose2isometry(const geometry_msgs::Pose& pose);

Eigen::Isometry3d tf2isometry(const tf::StampedTransform& trans);

geometry_msgs::Pose isometry2pose(const Eigen::Isometry3d& mat);

Eigen::Isometry3d odom2isometry(const nav_msgs::OdometryConstPtr& odom_msg);

/**
 * @brief object orientations could have multiple combination of roll, pitch and yaw,
 * for example an orientation (ψ,θ,φ) is equal to (ψ,θ,φ) + (i*π,+j*π,k*π) with any combination of integers i,j,k s.t. abs(i)=abs(j)=abs(k),
 * this computation ensures that we choose the euler angles vector with minimum norm.
 * assumption is that all angles are in range [-π,π]
 * @param euler_angs roll, pitch and yaw vector
 */
Eigen::Vector3f normalize_euler_angs(Eigen::Vector3f euler_angs);

Eigen::Matrix4f transform2Dto3D(Eigen::Matrix3f trans2D);

Eigen::Matrix3f transform3Dto2D(Eigen::Matrix4f trans3D);

}  // namespace hdl_graph_slam

#endif  // ROS_UTILS_HPP
