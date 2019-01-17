#include "ransac_slam/System.h"

/***************************************************
项目：单点一致性采样定位系统

开始时间：2018年8月18日
最近修改：2018年12月5日

***************************************************/

namespace ransac_slam
{
	System::System(const std::string &strSettingsFile, const bool bUseViewer)
		:it(processor)//构造函数
	{
		 // Output welcome message
		std::cout << std::endl <<
		"RANSAC-SLAM  Copyright (C) 2018.8.20 liuyuqaing, NEU." << std::endl <<
		"This program comes with ABSOLUTELY NO WARRANTY;" << std::endl  <<
		"This is free software, and you are welcome to redistribute it" << std::endl <<
		"under certain conditions. See LICENSE.txt." << std::endl << std::endl;

		//Check settings file
		fsSettings.open(strSettingsFile.c_str(), cv::FileStorage::READ);
		if(!fsSettings.isOpened())
		{
			ros::shutdown();
			std::cerr << "Failed to open settings file at: " << strSettingsFile << std::endl;
			exit(-1);
		}
		
		//set parameter of camera
		//k1, k2径向畸变系数
		cam.k1 = (double)fsSettings["Camera.k1"];
		cam.k2 = (double)fsSettings["Camera.k2"];
		
		//图片的高度row（行）和宽度col（列）
		cam.nRows = (double)fsSettings["Camera.nRows"];
		cam.nCols = (double)fsSettings["Camera.nCols"];
		
		//缩放值
		double d = (double)fsSettings["Camera.d"];
		
		//u0，v0表示图像的中心像素坐标和图像原点像素坐标之间相差的横向和纵向像素数
		cam.Cx = (double)fsSettings["Camera.cx_d"]/d;
		cam.Cy = (double)fsSettings["Camera.cy_d"]/d;
		
		//焦距
		cam.f = (double)fsSettings["Camera.fps"];
		
		//dx和dy表示：x方向和y方向的一个像素分别占多少长度单位，
		//即一个像素代表的实际物理值的大小，其是实现图像物理坐标系与像素坐标系转换的关键。
		cam.dx = (double)fsSettings["Camera.dx"];
		cam.dy = (double)fsSettings["Camera.dy"];
		
		//相机模型：二维径向畸变
		cam.model = (std::string)fsSettings["Camera.model"];
		cam.K << (cam.f/d), 0, cam.Cx, 0, (cam.f/d), cam.Cy, 0, 0, 1;
		
		int min_features = (int)fsSettings["min_number_of_features_in_image"];
		
		//核心kf预测类初始化
		mS_ExtendKF = new ExtendKF(strSettingsFile, &cam, "constant_velocity");
		mS_ExtendKF->initialize_x_and_p();
		
		//地图构建类初始化
		mS_Map = new Map(min_features, mS_ExtendKF);
		
		//ransac更新追踪类初始化
		mS_Tracking = new Tracking(strSettingsFile, mS_ExtendKF);
		
		TrackSteps=0;
		trajectory.clear();
		
		InitTopic();
	}
	System::~System()
	{
		fsSettings.release();  
	}
	void System::InitTopic(void)
	{
// 		cv::namedWindow("FAST feature");
// 		cv::startWindowThread();
		
		features_pub = processor.advertise<visualization_msgs::MarkerArray>("/ransac_slam/features_marker", 80);//因为有35个路标landmarks
		association_pub = processor.advertise<visualization_msgs::MarkerArray>("/ransac_slam/assoc_marker", 1);  // data_association
		state_pub = processor.advertise<visualization_msgs::MarkerArray>("/ransac_slam/state_marker", 1);  // data_association
				
		image_pub = it.advertise("/ransac_slam/image_gray", 1);
		odom_pub = processor.advertise<nav_msgs::Odometry>("/ransac_slam/odom", 5);
		path_pub = processor.advertise<nav_msgs::Path>("/ransac_slam/cam_trajectory",1, true);
		
		// 确保发布标记时，已经打开rviz进行显示
		while (features_pub.getNumSubscribers() < 1)
		{
			if (!ros::ok())	
				return  ;
			ROS_WARN_ONCE("Please create a subscriber to the marker or launch rviz ! ");
			sleep(1);
		}
	}
	void System::TrackRunning(cv::Mat image)
	{
		//Grab image
		image = takeImage(image);
		
		TrackSteps++;
		
		//Map management (adding and deleting features; and converting inverse depth to Euclidean)
		mS_Map->map_management(image, TrackSteps);

		//EKF prediction (state and measurement prediction)
		mS_ExtendKF->ekf_prediction();

		//Search for individually compatible matches
		mS_Tracking->search_IC_matches(image);
		
		//1-Point RANSAC hypothesis and selection of low-innovation inliers
		mS_Tracking->ransac_hypotheses();

		//Partial update using low-innovation inliers
		mS_ExtendKF->ekf_update_li_inliers();

		//"Rescue" high-innovation inliers
		mS_Tracking->rescue_hi_inliers();
	
		//Partial update using high-innovation inliers
		mS_ExtendKF->ekf_update_hi_inliers();
		
		//publish image topic 
		ShowImages();
		odometry();
		features_publisher();
		state_publisher();
		path_publisher();
		//association_publisher();
		std::cout<<"测试"<<(TrackSteps)<<std::endl;
	}
	cv::Mat System::takeImage(cv::Mat original)
	{
		cv::Mat image;

		if ( original.channels() != 1 )
			cv::cvtColor (original, image, CV_BGR2GRAY);  
		else
			image =original.clone();
		
		cv::cvtColor (image, GobalImage, CV_GRAY2BGR);  //保证GobalImage图片是彩色的，用于标识特征点
		
		return image;								//保存返回的image是灰度图，用于检测特征点以及相关处理
	}
	void System::ShowImages()
	{
		sensor_msgs::ImagePtr cv_process;
		Eigen::RowVectorXd im_h;
		Eigen::MatrixXd im_S;
		int line_len = 5;
		int line_wid = 2;
		
		int feat_len = mS_ExtendKF->features_info.size();
		for(int i = 0 ; i < feat_len ; i++)
		{
			im_h = mS_ExtendKF->features_info[i].h;
			im_S = mS_ExtendKF->features_info[i].S;
			if( im_h.cols() && im_S.cols() )
			{//Scalar颜色排序BGR
				cv::Point centerPonts = cv::Point(im_h(0), im_h(1));
				if( mS_ExtendKF->features_info[i].individually_compatible)	//通过相关性找到的匹配
				{
					if( mS_ExtendKF->features_info[i].low_innovation_inlier )//旧局内点匹配
					{//粗红色
						plotUncertainEllip2D(centerPonts, im_S, line_wid*2, cv::Scalar(0, 0, 255));	//绘制椭圆
						DrawplusSign(centerPonts, line_len, cv::Scalar(0, 0, 255));		//绘制加号
					}
					if( mS_ExtendKF->features_info[i].high_innovation_inlier )//细红色，新局内点匹配
					{//细红色
						plotUncertainEllip2D(centerPonts, im_S, line_wid, cv::Scalar(71, 99, 255));
						DrawplusSign(centerPonts, line_len, cv::Scalar(71, 99, 255));
					}
					
					if((mS_ExtendKF->features_info[i].low_innovation_inlier == false) && 
						(mS_ExtendKF->features_info[i].high_innovation_inlier == false) )//被单点RANSAC排除的
					{//蓝色
						plotUncertainEllip2D(centerPonts, im_S, line_wid, cv::Scalar(255, 0, 0));
						DrawplusSign(centerPonts, line_len, cv::Scalar(255, 0, 0));
					}
					
				}else{//通过相关性找不到的匹配
					plotUncertainEllip2D(centerPonts, im_S, line_wid, cv::Scalar(0, 255, 0));//绘制绿色椭圆
					DrawplusSign(centerPonts, line_len, cv::Scalar(0, 255, 0));
				}
			}
		}
		//cv::imshow("FAST feature", GobalImage);
		cv_process = cv_bridge::CvImage(std_msgs::Header(), "bgr8", GobalImage).toImageMsg();
		image_pub.publish(cv_process);
	}
	void System::DrawplusSign(cv::Point center, int len, cv::Scalar color)
	{
		cv::Point centerPoints[4];
		int thickness = 2;
		int lineType = 8;
		
		centerPoints[0] = cv::Point(center.x - len, center.y);
		centerPoints[1] = cv::Point(center.x +len, center.y);
		centerPoints[2] = cv::Point(center.x, center.y + len);
		centerPoints[3] = cv::Point(center.x, center.y - len);
		
		cv::line(GobalImage, centerPoints[0], centerPoints[1], color, thickness, lineType);
		cv::line(GobalImage, centerPoints[2], centerPoints[3], color, thickness, lineType);
	}
	void System::plotUncertainEllip2D(cv::Point center, Eigen::Matrix2d Src, int linewidth, cv::Scalar color)
	{
		//double chi_095_2 = 0.5915;
		double chi_095_2 = 0.9915;
		
		// 特征值
		// 实对称矩阵可以保证对角化成功
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen_solver ( Src );
		Eigen::VectorXd eigen_values = eigen_solver.eigenvalues();
		
		double angle = atan2(eigen_values(0), eigen_values(1)) * 180.0/M_PI;
		cv::Size size = cv::Size(eigen_values(0) * chi_095_2, eigen_values(1) * chi_095_2);

		cv::ellipse(GobalImage, center, size ,angle, 0, 360, color, linewidth);
	}
	void System::odometry(void)
	{
		// cam_link to map tree broadcaster
		tf::TransformBroadcaster cam_broadcaster;
		geometry_msgs::TransformStamped cam_trans;
		
		ros::Rate r(50);
		for(int i = 0 ; i < 20 ; i++)
		{
			ros::Time current_time= ros::Time::now();
			cam_trans.header.stamp = current_time;
			cam_trans.header.frame_id = "odom";
			cam_trans.child_frame_id = "camera_baselink";
			cam_trans.transform.translation.x = mS_ExtendKF->x_k_k(0);
			cam_trans.transform.translation.y = mS_ExtendKF->x_k_k(1);
			cam_trans.transform.translation.z = mS_ExtendKF->x_k_k(2);
			cam_trans.transform.rotation.x = mS_ExtendKF->x_k_k(3);
			cam_trans.transform.rotation.y = mS_ExtendKF->x_k_k(4);
			cam_trans.transform.rotation.z = mS_ExtendKF->x_k_k(5);
			cam_trans.transform.rotation.w = mS_ExtendKF->x_k_k(6);
			cam_broadcaster.sendTransform(cam_trans);//send the transform
			
			//next, we'll publish the odometry message over ROS
			nav_msgs::Odometry cam_odom;
			cam_odom.header.stamp = current_time;
			cam_odom.header.frame_id = "odom";
			cam_odom.child_frame_id = "camera_baselink";
			//set the position and velocity
			cam_odom.pose.pose.position.x = mS_ExtendKF->x_k_k(0);
			cam_odom.pose.pose.position.y = mS_ExtendKF->x_k_k(1);
			cam_odom.pose.pose.position.z = mS_ExtendKF->x_k_k(2);
			cam_odom.pose.pose.orientation.x = mS_ExtendKF->x_k_k(3);
			cam_odom.pose.pose.orientation.y = mS_ExtendKF->x_k_k(4);
			cam_odom.pose.pose.orientation.z = mS_ExtendKF->x_k_k(5);
			cam_odom.pose.pose.orientation.w = mS_ExtendKF->x_k_k(6);
// 			for(int i = 0; i < 36; i++)
// 			odom.pose.covariance[i] = covariance[i];
			cam_odom.twist.twist.linear.x = mS_ExtendKF->x_k_k(7);
			cam_odom.twist.twist.linear.y = mS_ExtendKF->x_k_k(8);
			cam_odom.twist.twist.linear.z = mS_ExtendKF->x_k_k(9);
			cam_odom.twist.twist.angular.x = mS_ExtendKF->x_k_k(10);
			cam_odom.twist.twist.angular.y = mS_ExtendKF->x_k_k(11);
			cam_odom.twist.twist.angular.z = mS_ExtendKF->x_k_k(12);
			
			//publish the message
			odom_pub.publish(cam_odom);
			r.sleep();
		}
	}
	void System::features_publisher(void)
	{
		visualization_msgs::MarkerArray ma;
		visualization_msgs::Marker marker;
		
		marker.header.frame_id = "odom";
		marker.ns = "features";
		marker.action = visualization_msgs::Marker::ADD;
		marker.type = visualization_msgs::Marker::CUBE;// 立方体
		marker.id = 0;
		
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;
		marker.scale.x = 0.2;
		marker.scale.y = 0.2;
		marker.scale.z = 0.2;
		marker.color.r = 0;
		marker.color.g = 255/255.0;
		marker.color.b = 0;
		marker.color.a = 1.0;
		marker.lifetime = ros::Duration(0);
		
		int feat_len = mS_ExtendKF->features_info.size();
		int index = 14 - 1;
		Eigen::VectorXd inverse_coord;
		Eigen::Vector3d carte_coord;
		for(int i = 0 ; i <  feat_len ; i++)
		{
			marker.header.stamp = ros::Time::now();
			marker.id++;
			marker.type = visualization_msgs::Marker::CUBE;
			
			if( strcmp(mS_ExtendKF->features_info[i].type.c_str(), "inversedepth") == 0 )
			{
				inverse_coord = mS_ExtendKF->x_k_k.segment(index, 6);
				carte_coord = mS_ExtendKF->inversedepth2cartesian(inverse_coord);
				index+= 6;
			}else if( strcmp(mS_ExtendKF->features_info[i].type.c_str(), "cartesian") == 0 )
			{
				carte_coord(0) = mS_ExtendKF->x_k_k(index);
				carte_coord(1) = mS_ExtendKF->x_k_k(index + 1);
				carte_coord(2) = mS_ExtendKF->x_k_k(index + 2);
				index+= 3;
			}
			marker.pose.position.x = carte_coord(0);
			marker.pose.position.y = carte_coord(1);
			marker.pose.position.z = carte_coord(2);	
			marker.color.r = 0;
			marker.color.g = 255.0/255.0;
			marker.color.b = 0;
			ma.markers.push_back(marker);
			
			//写入标记数字
			marker.id++;
			marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
			marker.text = std::to_string(i+1);
			marker.pose.position.x = carte_coord(0) + 0.2;//pose.x();
			marker.pose.position.y = carte_coord(1) + 0.2;//pose.y();
			marker.pose.position.z = carte_coord(2) + 0.2;//pose.y();
			marker.color.r = 255.0/255.0;
			marker.color.g = 0;
			marker.color.b = 0;
			ma.markers.push_back(marker);	
		}
		features_pub.publish(ma);
	}
	void System::state_publisher(void)
	{
		visualization_msgs::MarkerArray ma;
		visualization_msgs::Marker marker;
		
		marker.header.frame_id = "odom";
		marker.id = 0;
		marker.ns = "vari_state";
		marker.type = visualization_msgs::Marker::SPHERE;
		marker.action = visualization_msgs::Marker::ADD;
		marker.lifetime = ros::Duration(0);
		
		marker.color.r = 255/255.0;//黄色
		marker.color.g = 255/255.0;
		marker.color.b = 0;
		marker.color.a = 0.5;//设置透明度值
		
		// covariance ellipses
		double quantiles = 5 ;
		double ellipse_scale_= 8;
		tf::Quaternion orientation;
		tf::Matrix3x3 tf3d;
		Eigen::Matrix3d eigenvectors;
		Eigen::Vector3d eigenvalues;
		Eigen::Matrix3d covariance;
		covariance = mS_ExtendKF->p_k_k.topLeftCorner(3, 3);
		if(Converter::computeEllipseOrientationScale3D(eigenvectors, eigenvalues, covariance) == true)
		{
			marker.header.stamp = ros::Time::now();
			tf3d.setValue(eigenvectors(0, 0), eigenvectors(0, 1), eigenvectors(0, 2), 
						eigenvectors(1, 0), eigenvectors(1, 1), eigenvectors(1, 2), 
						eigenvectors(2, 0), eigenvectors(2, 1), eigenvectors(2, 2));
			tf3d.getRotation(orientation);
			marker.pose.position.x = mS_ExtendKF->x_k_k(0);
			marker.pose.position.y = mS_ExtendKF->x_k_k(1);
			marker.pose.position.z = mS_ExtendKF->x_k_k(2);
			marker.pose.orientation.x = orientation.x();
			marker.pose.orientation.y = orientation.y();
			marker.pose.orientation.z = orientation.z();
			marker.pose.orientation.w = orientation.w();
			marker.scale.x = ellipse_scale_ * quantiles * sqrt(eigenvalues[0]);
			marker.scale.y = ellipse_scale_ * quantiles * sqrt(eigenvalues[1]);
			marker.scale.z = ellipse_scale_ * quantiles * sqrt(eigenvalues[2]);
			ma.markers.push_back(marker);
			marker.id++;
		}
		
		int feat_len = mS_ExtendKF->features_info.size();
		int line_len = 10;
		int index = 14 - 1;
		Eigen::VectorXd inverse_coord;
		Eigen::Vector3d carte_coord;
		for(int i = 0 ; i < feat_len ; i++)
		{
			marker.header.stamp = ros::Time::now();
			if( strcmp(mS_ExtendKF->features_info[i].type.c_str(), "inversedepth") == 0 )
			{
				inverse_coord = mS_ExtendKF->x_k_k.segment(index, 6);
				carte_coord = mS_ExtendKF->inversedepth2cartesian(inverse_coord);
				Eigen::MatrixXd p_id= mS_ExtendKF->p_k_k.block(index, index, 6, 6);
				if( inverse_coord(5) - 3 * sqrt(p_id(5, 5)) < 0 )
				{
					if( inverse_coord(5) > 0 )
					{
// 						ray = 8*m(y_id(4),y_id(5));
// 						minimum_distance = inversedepth2cartesian([y_id(1:5); y_id(6)+3*sqrt(p_id(6,6))]);
// 						vectarrow([minimum_distance(1) 0 minimum_distance(3)],[ray(1) 0 ray(3)],'r')
// 						plot3(XYZ(1),XYZ(2),XYZ(3),'r+','Markersize',10)
					}
				}else{
					if( inverse_coord(5) > 0 )
					{
// 						plotUncertainSurfaceXZ( p_id, y_id, 0, [1 0 0], randSphere6D, nPointsRand );
// 						plot3(XYZ(1),XYZ(2),XYZ(3),'r+','Markersize',10)
					}
				}
				index = index + 6;
			}else if( strcmp(mS_ExtendKF->features_info[i].type.c_str(), "cartesian") == 0 )
			{
				covariance = mS_ExtendKF->p_k_k.block(index, index, 3, 3);
				if(Converter::computeEllipseOrientationScale3D(eigenvectors, eigenvalues, covariance) == true)
				{
					tf3d.setValue(eigenvectors(0, 0), eigenvectors(0, 1), eigenvectors(0, 2), 
								eigenvectors(1, 0), eigenvectors(1, 1), eigenvectors(1, 2), 
								eigenvectors(2, 0), eigenvectors(2, 1), eigenvectors(2, 2));
					tf3d.getRotation(orientation);
					marker.pose.position.x = mS_ExtendKF->x_k_k(index);
					marker.pose.position.y = mS_ExtendKF->x_k_k(index+1);
					marker.pose.position.z = mS_ExtendKF->x_k_k(index+2);
					marker.pose.orientation.x = orientation.x();
					marker.pose.orientation.y = orientation.y();
					marker.pose.orientation.z = orientation.z();
					marker.pose.orientation.w = orientation.w();
					marker.scale.x = ellipse_scale_ * quantiles * sqrt(eigenvalues[0]);
					marker.scale.y = ellipse_scale_ * quantiles * sqrt(eigenvalues[1]);
					marker.scale.z = ellipse_scale_ * quantiles * sqrt(eigenvalues[2]);
					ma.markers.push_back(marker);
					marker.id++;
				}
				index = index + 3;
			}
		}
		state_pub.publish(ma);
	}
	void System::path_publisher(void)
	{
		geometry_msgs::PoseStamped path_stamped;
		ros::Time current_time = ros::Time::now();
		
		cam_trajectory.header.stamp = current_time;
		cam_trajectory.header.frame_id="odom";
		path_stamped.header.stamp=current_time;
		path_stamped.header.frame_id="odom";
		path_stamped.pose.position.x = mS_ExtendKF->x_k_k(0);
		path_stamped.pose.position.y = mS_ExtendKF->x_k_k(1);
		path_stamped.pose.position.z = mS_ExtendKF->x_k_k(2);
		path_stamped.pose.orientation.x = mS_ExtendKF->x_k_k(3);
		path_stamped.pose.orientation.y = mS_ExtendKF->x_k_k(4);
		path_stamped.pose.orientation.z = mS_ExtendKF->x_k_k(5);
		path_stamped.pose.orientation.w = mS_ExtendKF->x_k_k(6);
		cam_trajectory.poses.push_back(path_stamped);
	
		path_pub.publish(cam_trajectory);
	}
	void System::association_publisher(void)
	{
		// visualization of the data association
		visualization_msgs::MarkerArray ma;
		visualization_msgs::Marker line_strip;

		line_strip.header.frame_id = "odom";
		line_strip.header.stamp = ros::Time::now();

		line_strip.id = 0;
		line_strip.ns = "data_association";
		line_strip.type = visualization_msgs::Marker::LINE_STRIP;
		line_strip.action = visualization_msgs::Marker::ADD;

		line_strip.lifetime = ros::Duration(0);
		line_strip.pose.position.x = 0;
		line_strip.pose.position.y = 0;
		line_strip.pose.position.z = 0;
		line_strip.pose.orientation.x = 0.0;
		line_strip.pose.orientation.y = 0.0;
		line_strip.pose.orientation.z = 0.0;
		line_strip.pose.orientation.w = 1.0;
		line_strip.scale.x = 1;  // line uses only x component
		line_strip.scale.y = 1;
		line_strip.scale.z = 1;
		line_strip.color.a = 1.0;
		line_strip.color.r = 255/255;
		line_strip.color.g = 0;
		line_strip.color.b = 0;

// 		int ass_num = feature_points.size();
		geometry_msgs::Point point_temp;
		for (int i = 0 ;   i< 0 ; i++)
		{
			line_strip.points.clear();
// 			point_temp.x = camera_points[i](0);
// 			point_temp.y = camera_points[i](1);
// 			point_temp.z = camera_points[i](2);
// 			line_strip.points.push_back(point_temp);
// 			
// 			point_temp.x = feature_points[i](0);
// 			point_temp.y = feature_points[i](1);
// 			point_temp.z = feature_points[i](2);
			line_strip.points.push_back(point_temp);
			ma.markers.push_back(line_strip);
			line_strip.id++;
		}
		association_pub.publish(ma);
	}

	
	
	
	
	
	
}
