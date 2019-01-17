#ifndef RANSAC_SLAM_TRACKING_H 
#define RANSAC_SLAM_TRACKING_H 

#include "ransac_slam/System.h"
#include "ransac_slam/ExtendKF.h"
#include "ransac_slam/Converter.h"

//OpenCV部分
//#include <opencv2/opencv.hpp>
// Eigen 部分
#include <Eigen/Core>
// 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Dense>

namespace ransac_slam
{
	class ExtendKF;
	
	class Tracking
	{
	public:
		Tracking(const std::string &strSettingsFile, ExtendKF* m_ExtendKF);
		~Tracking();
		
		/**
		* @brief Search for individually compatible matches
		*/
		void search_IC_matches(cv::Mat image);
		
		void calculate_derivatives(Eigen::VectorXd xk_km1);
		
		void calculate_Hi_cartesian(Eigen::VectorXd x_v, Eigen::VectorXd yi, int order, Eigen::MatrixXd& Hi);
		void calculate_Hi_inverse_depth(Eigen::VectorXd x_v, Eigen::VectorXd yi, int order, Eigen::MatrixXd& Hi);
		
		void pred_patch_fc(int order, Eigen::Vector3d XYZ_w);
		
		void matching(cv::Mat image);
		
		void ransac_hypotheses(void);
		
		void rescue_hi_inliers(void);
		//void compute_hypothesis_support_fast(Eigen::VectorXd xi, Eigen::MatrixXd state_vector_pattern, Eigen::MatrixXd z_id, Eigen::MatrixXd z_euc);
		
	private:

	
		ExtendKF* mT_ExtendKF;
	};


}
#endif // TRACKING_H
