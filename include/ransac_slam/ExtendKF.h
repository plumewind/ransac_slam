#ifndef RANSAC_SLAM_EXTENDKF_H 
#define RANSAC_SLAM_EXTENDKF_H 

#include "ransac_slam/System.h"
#include <string.h>
//OpenCV部分
#include <opencv2/opencv.hpp>

namespace ransac_slam
{
	struct CamParam;
	
	//FIXME: 此处数据结构后续可能进行修改
	struct Feature
	{
		//Eigen::Matrix<double, 41, 41> patch_when_initialized;
		//Eigen::Matrix<double, 13, 13> patch_when_matching;
		//cv::Mat patch_when_initialized;
		Eigen::MatrixXd patch_when_initialized;//特征点检测时周围的图像块
		Eigen::MatrixXd patch_when_matching;//特征点匹配时周围的图像块
		Eigen::Vector3d r_wc_when_initialized;//初始化时相机的世界坐标
		Eigen::Matrix3d R_wc_when_initialized;//初始化时相机的外参矩阵
		Eigen::RowVector2d uv_when_initialized;//初始化特征点的图像坐标
		int half_patch_size_when_initialized;//初始化时特征图像块的尺寸
		int half_patch_size_when_matching;//特征匹配时图像块的尺寸
		int times_predicted;				//关键帧预测的次数
		int times_measured;				//关键帧的测量次数
		long long int init_frame;			//关键帧序号
		Eigen::Vector2d init_measurement;	//初始观测点的图像坐标
		std::string type;					//关键帧类型
		Eigen::VectorXd yi;					//关键帧的特征信息
		bool individually_compatible;		//独立兼容性标志位
		bool low_innovation_inlier;			//低创新局内点标志位
		bool high_innovation_inlier;			//高创新局内点标志位
		Eigen::VectorXd z;					//最佳相似特征的图像坐标
		Eigen::RowVectorXd h;				//相机位姿估计
		Eigen::MatrixXd H;					//计算衍生值
		Eigen::MatrixXd S;					//关键帧特征点估计位姿矩阵？？
		int state_size;						//状态矩阵尺寸
		int measurement_size;				//观测矩阵尺寸
		Eigen::Matrix2d R;					//相机运动矩阵
	};//the parameter of features
	
	class ExtendKF
	{
	public:
		ExtendKF(const std::string &strSettingsFile, CamParam* param, std::string type);
		~ExtendKF();
		
		/**
		* @brief Initialize state vector and covariance
		*/
		void initialize_x_and_p(void);
		
		void predict_camera_measurements(Eigen::VectorXd xkk);
		
		Eigen::Matrix3d q2r(Eigen::VectorXd q_in);
		Eigen::RowVector4d v2q(Eigen::Vector3d v);
		Eigen::Matrix4d dq3_by_dq1(Eigen::Vector4d q2_in);
		
		/**
		* @brief Calculate commonly used Jacobian part dq(omega * delta_t) by domega
		*/
		Eigen::Matrix<double, 4, 3> dqomegadt_by_domega(Eigen::Vector3d omega, double delta_t);
		double dq0_by_domegaA(double omegaA, double omega, double delta_t);
		double dqA_by_domegaA(double omegaA, double omega, double delta_t);
		double dqA_by_domegaB(double omegaA, double omegaB, double omega, double delta_t);
		
		Eigen::Vector3d q2tr_tr2rpy(Eigen::Vector4d q);
		
		/**
		* @brief  Compute a single measurement
		* @param hrl Points 3D in camera coordinates
		*/
		void hi_cartesian(Eigen::Vector3d hrl, Eigen::MatrixXd& zi);
		
		/**
		* @brief  Inverse depth conversion process
		* @param inverse_depth Input depth value
		* @param cartesian Output inverse depth value
		*/
		Eigen::Vector3d inversedepth2cartesian(Eigen::VectorXd inverse_depth);
		
		/**
		* @brief  Initialize image feature points
		* @param image An image of the input
		* @param step Number of image processing steps
		*/
		void hi_inverse_depth(Eigen::Vector3d hrl, Eigen::MatrixXd& zi);
		
		/**
		* @brief  Convert from world coordinates to image coordinates
		* @param hrl Points 3D in world coordinates
		* @param uv_u Points 2D in image coordinates
		*/
		Eigen::Vector2d hu(Eigen::Vector3d yi);
		
		/**
		* @brief  Distort image coordinates.
		*	The function deals with two models:
		* 	1.- Real-Time 3D SLAM with Wide-Angle Vision, 
		* 	2.- Photomodeler full distortion model.
		* @param uv distorted image points in pixels
		* @param uvd undistorted coordinate points
		*/
		void distort_fm(Eigen::MatrixXd uv, Eigen::MatrixXd& uvd);
		
		void undistort_fm(Eigen::MatrixXd uvd, Eigen::MatrixXd& uvu);
		
		/**
		* @brief  A random number that produces a normal distribution.
		* @param low  The minimum value of the random number.
		* @param high  The maximum value of the random number.
		*/
		double RandomGenerator(const int low, const int high);
		Eigen::MatrixXd rand(int row, int column, double min, double max);
		
		void hinv(Eigen::VectorXd uvd, Eigen::VectorXd Xv, double initial_rho, Eigen::VectorXd& newFeature);
		
		Eigen::Matrix<double, 3, 4> dRq_times_a_by_dq(Eigen::VectorXd q, Eigen::Vector3d aMat);
		
		/**
		* @brief  Jacobian of the undistortion of the image coordinates
		* @param uvd  distorted image points in pixels
		* @param J_dunistor  Jacobian
		*/
		Eigen::Matrix2d jacob_undistor_fm(Eigen::VectorXd uvd);
		
		/**
		* @brief  predict_state_and_covariance
		*/
		void ekf_prediction(void);
		
		/**
		* @brief  Partial update using low-innovation inliers
		*/
		void ekf_update_li_inliers(void);
		
		void update(Eigen::VectorXd x_km_k, Eigen::MatrixXd p_km_k, Eigen::MatrixXd H, Eigen::MatrixXd R, Eigen::VectorXd z, Eigen::VectorXd h);
		
		/**
		* @brief  Partial update using high-innovation inliers
		*/
		void ekf_update_hi_inliers(void);
		
		void fv(double delta_t,  Eigen::VectorXd& X_k_km1);
		
		Eigen::Vector4d qprod(Eigen::Vector4d q, Eigen::Vector3d wW, double delta_t);
		
		void dfv_by_dxv(double delta_t, Eigen::MatrixXd& dfv_by_dxvRES);
		
		
		
		std::vector<Feature> features_info;
		CamParam* cam;//the parameter of camera
		
		//velocity values
		double v_0, std_v_0, w_0, std_w_0;
		
		//state vector and covariance matrix
		Eigen::VectorXd x_k_k;//状态向量，这是动态变化的向量，行数会自动变化
		Eigen::MatrixXd p_k_k;//状态向量对应的协方差矩阵，大小是动态变化的
		
		double std_a;
		double std_alpha;
		double std_z;
		
		Eigen::VectorXd x_k_km1;
		Eigen::MatrixXd p_k_km1;
		
		
	private:
		std::string filter_type;
		
		
		cv::FileStorage fsSettings; 
		double eps;
		
		
	};


}
#endif // RANSAC_SLAM_EXTENDKF_H
