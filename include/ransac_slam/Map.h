#ifndef RANSAC_SLAM_MAP_H 
#define RANSAC_SLAM_MAP_H 

 #include "ransac_slam/System.h"
 //#include "ransac_slam/ExtendKF.h"
 
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace ransac_slam
{
	class ExtendKF;
	struct Feature;
	
	class Map
	{
	public:
		Map(const int min_fea,  ExtendKF* m_ExtendKF);
		~Map();
		
		/**
		* @brief  Map management (adding and deleting features; and converting inverse depth to Euclidean)
		* @param image An image of the input
		* @param step Number of image processing steps
		*/
		void map_management(cv::Mat image, int step);
		
		/**
		* @brief  Delete the index id from the list and delete the corresponding id
		*/
		void delete_a_feature(int feature_id);
		
		/**
		* @brief  convert features from inverse depth to cartesian, if necessary
		*/
		void inversedepth_2_cartesian(void);
		
		void initialize_features(int step, int min_features_to_init, cv::Mat image);
		
		/**
		* @brief  Initialize the feature points of a picture
		* @param image An image of the input
		* @param step Number of image processing steps
		 * @param uv Output matrix
		*/
		void initialize_a_features(int step, cv::Mat image, Eigen::MatrixXd& uv);
		
		/**
		* @brief  Detect fast-9 feature points
		* @param image An image of the input
		* @param threshold threshold of detection
		 * @param coords returns the X coordinates in corners(:,1) and the Y coordinares in corners(:,2)
		*/
		void fast_corner_detect_9(cv::Mat image,  double  threshold,  Eigen::Matrix<double, 2, Eigen::Dynamic>& coords);
		
		void add_a_feature_covariance_inverse_depth(Eigen::MatrixXd P, Eigen::VectorXd uvd, Eigen::VectorXd Xv, Eigen::MatrixXd& P_RES);
	private:
		int min_features;//min_number_of_features_in_image
		
		ExtendKF* mM_ExtendKF;
		
	};


}
#endif // RANSAC_SLAM_MAP_H
