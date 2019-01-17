#ifndef RANSAC_SLAM_CONVERTER_H
#define RANSAC_SLAM_CONVERTER_H


//OpenCV部分
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <Eigen/Dense>

namespace ransac_slam
{
	class Converter
	{
	public:
		static Eigen::Matrix<double, 41, 41> toMatrix41d(const cv::Mat &cvMat41);
		static void  meshgrid(Eigen::VectorXd& MUSrc, Eigen::VectorXd& MVSrc, Eigen::MatrixXd& MU, Eigen::MatrixXd& MV);
		static void meshgrid_opencv(const cv::Range &xgv, const cv::Range &ygv, Eigen::MatrixXd& MU, Eigen::MatrixXd& MV);
		static void reshape(Eigen::MatrixXd in, int rows, int cols, Eigen::MatrixXd& out);
		static cv::Mat toCvMat_i(const Eigen::MatrixXd src);
		static cv::Mat toCvMat_d(const Eigen::MatrixXd src);
		static cv::Mat toCvMat_f(const Eigen::MatrixXd src);
		static Eigen::MatrixXd toMatrixd_atd(const cv::Mat src);
		static Eigen::MatrixXd toMatrixd_ati(const cv::Mat src);
		static Eigen::MatrixXd toMatrixd_atf(const cv::Mat src);
		static Eigen::MatrixXd toMatrixd_atuc(const cv::Mat src);
		static Eigen::MatrixXd corrcoef(const Eigen::MatrixXd &M);
		static Eigen::MatrixXd corrcoef_opencv(const Eigen::MatrixXd &M);
		static Eigen::MatrixXd cov(Eigen::MatrixXd d1, Eigen::MatrixXd d2);
		static Eigen::MatrixXd find(Eigen::MatrixXd src, double m) ;
		static Eigen::VectorXd select(Eigen::VectorXd src, Eigen::VectorXd logical);
		static Eigen::MatrixXd repmat(Eigen::MatrixXd M, int rowM, int colM);
		static void makeRightHanded(Eigen::Matrix2d& eigenvectors, Eigen::Vector2d& eigenvalues);
		static bool computeEllipseOrientationScale3D(Eigen::Matrix3d& eigenvectors, Eigen::Vector3d& eigenvalues, const Eigen::Matrix3d& covariance);
		
		
		
		
		
	};





}

#endif