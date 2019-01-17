#include <ros/ros.h>
#include <iostream>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>

float odom_data[13];

void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
	ROS_INFO("I heard: [%i]", msg->header.seq);
	odom_data[0] = msg->pose.pose.position.x;
	odom_data[1] = msg->pose.pose.position.y;
	odom_data[2] = msg->pose.pose.position.z;
	odom_data[3] = msg->pose.pose.orientation.x;
	odom_data[4] = msg->pose.pose.orientation.y;
	odom_data[5] = msg->pose.pose.orientation.z;
	odom_data[6] = msg->pose.pose.orientation.w;
}

int main(int argc, char **argv)
{
	// Set up ROS node
	ros::init(argc, argv, "odometry");
	
	ros::NodeHandle hand;
	ros::Subscriber odom_sub = hand.subscribe("/ransac_slam/odom", 10, odomCallback);
	
	// cam_link to map tree broadcaster
	tf::TransformBroadcaster cam_broadcaster;
	geometry_msgs::TransformStamped cam_trans;
	ros::Rate r(50);
	while(ros::ok())
	{
		ros::spinOnce();

		ros::Time current_time= ros::Time::now();
		cam_trans.header.stamp = current_time;
		cam_trans.header.frame_id = "odom";
		cam_trans.child_frame_id = "cam_link";
		cam_trans.transform.translation.x = odom_data[0];
		cam_trans.transform.translation.y = odom_data[1];
		cam_trans.transform.translation.z = odom_data[2];
		cam_trans.transform.rotation.x = odom_data[3];
		cam_trans.transform.rotation.y = odom_data[4];
		cam_trans.transform.rotation.z = odom_data[5];
		cam_trans.transform.rotation.w = odom_data[6];
		cam_broadcaster.sendTransform(cam_trans);//send the transform
		
		r.sleep();
	}
	return 0;
}
