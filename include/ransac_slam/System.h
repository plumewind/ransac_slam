#ifndef RANSAC_SLAM_SYSTEM_H 
#define RANSAC_SLAM_SYSTEM_H 

//ROS部分
#include <ros/ros.h>
#include <ros/package.h>

// add ros msgs
#include <std_msgs/String.h>
#include <std_msgs/Int32.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

//Use image_transport for publishing and subscribing to images in ROS
#include <image_transport/image_transport.h>
//Use cv_bridge to convert between ROS and OpenCV Image formats
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

//C++部分
#include <iostream>
#include <algorithm>
#include <fstream>
#include <limits>
#include <chrono>
#include <random>
#include <ctime>
#include<cmath>
#include <vector>
#include <string.h>

//OpenCV部分
#include <opencv2/opencv.hpp>
//#include<opencv2/xfeatures2d.hpp>
// #include <opencv2/core/core.hpp>
//#include <opencv2/features2d/features2d.hpp>
// #include <opencv2/highgui/highgui.hpp>

// Eigen 部分
#include <Eigen/Core>
// 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <boost/iterator/iterator_concepts.hpp>

//自定义的类
#include "ransac_slam/ExtendKF.h"
#include "ransac_slam/Map.h"
#include "ransac_slam/Tracking.h"
#include "ransac_slam/Converter.h"

namespace ransac_slam
{
	class ExtendKF;
	class Map;
	class Tracking;
	
	//FIXME: 此处数据结构后续可能进行修改
	//the parameter of carmer
	struct CamParam
	{
		double k1;//K1
		double k2;//k2
		int nRows;
		int nCols;
		double Cx;
		double Cy;
		double f;
		double dx;
		double dy;
		std::string model;
		Eigen::Matrix3d K;
	};
	
	/** system类功能
	 * (1)初始化整个系统，读入相关参数配置文件，准备系统运行环境
	 * (2)启动ExtendKF, Map, Tracking三个类，并进行相应的初始化
	 * (3)调度整体系统资源，协调三个重要类的运行
	 * (4)对相关信息进行可视化
	 * (5)保存系统共享资源，如相机参数CamParam
	 */
	class System
	{
	public:
		System(const std::string &strSettingsFile,  const bool bUseViewer = true);
		~System();
		void TrackRunning(cv::Mat image);
		cv::Mat takeImage(cv::Mat original);
		void InitTopic(void);
		void ShowImages(void);
		void DrawplusSign(cv::Point center, int len, cv::Scalar color);
		void plotUncertainEllip2D(cv::Point center, Eigen::Matrix2d Src, int linewidth, cv::Scalar color);
		void odometry(void);
		void features_publisher(void);
		void state_publisher(void);
		void path_publisher(void);
		void association_publisher(void);
		
	private:
		ros::NodeHandle processor;
		
		cv::FileStorage fsSettings; 
		CamParam cam;//the parameter of carmer
		long long int TrackSteps;
		cv::Mat GobalImage;
		
		//image_transport::ImageTransport it(processor);
		image_transport::ImageTransport it;
		image_transport::Publisher image_pub;
		ros::Publisher state_pub;
		ros::Publisher features_pub, association_pub;
		ros::Publisher odom_pub, path_pub;
		
		nav_msgs::Path cam_trajectory;
		std::vector<Eigen::Matrix<double, 7, 1> > trajectory;
		
		/** ExtendKF类功能
		 * (1)作为最核心的底层类，保存重要数据，特征点集features_info和信息矩阵x_k_k，p_k_k
		 * (2)进行kf预测计算
		 * (3)与Tracking类一起对局内点进行更新
		 */
		ExtendKF* mS_ExtendKF;
		
		/** Map 类功能
		 * (1)构建并整理SLAM地图
		 * (2)检测图片特征点，构建关键帧
		 */
		Map* mS_Map;
		
		/** Tracking功能
		 * (1)进行ransac计算
		 * (2)局内点更新
		 */
		Tracking* mS_Tracking;
	};


}
#endif // RANSAC_SLAM_SYSTEM_H
