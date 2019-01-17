#include "ransac_slam/Tracking.h"
#include <boost/concept_check.hpp>

namespace ransac_slam
{
	Tracking::Tracking(const std::string &strSettingsFile, ExtendKF* m_ExtendKF)
		:mT_ExtendKF(m_ExtendKF)
	{
		//Check settings file
// 		fsSettings.open(strSettingsFile.c_str(), cv::FileStorage::READ);
// 		if(!fsSettings.isOpened())
// 		{
// 			std::cerr << "Failed to open settings file at: " << strSettingsFile << std::endl;
// 			exit(-1);
// 		}
// 		
// 		std_a = (double)fsSettings["Sigma.a"];
// 		std_alpha = (double)fsSettings["Sigma.alpha"];
// 		std_z = (double)fsSettings["Sigma.noise"];
// 		
// 		v_0 = (double)fsSettings["Velocity.v0"];
// 		std_v_0 = (double)fsSettings["Velocity.stdv0"];
// 		w_0 = (double)fsSettings["Velocity.w0"];
// 		std_w_0 = (double)fsSettings["Velocity.stdw0"];
		
	}
	Tracking::~Tracking()
	{

		
	}
	void Tracking::search_IC_matches(cv::Mat image)
	{
		//第一步，计算各个特征点的高斯协方差
		mT_ExtendKF->predict_camera_measurements(mT_ExtendKF->x_k_km1);

		calculate_derivatives( mT_ExtendKF->x_k_km1 );			//计算特征点的矩阵h的雅克比矩阵H
		int feat_len = mT_ExtendKF->features_info.size();
		for(int i=0; i < feat_len ; i++)
		{
			if( mT_ExtendKF->features_info[i].h.cols() )
				mT_ExtendKF->features_info[i].S = (mT_ExtendKF->features_info[i].H) * (mT_ExtendKF->p_k_km1) * (mT_ExtendKF->features_info[i].H.transpose()) 
												+ (mT_ExtendKF->features_info[i].R);
		}

		//第二步，根据运动预测来推算下一时刻的特征图像快
		int xk_km1_len = mT_ExtendKF->x_k_km1.rows();
		Eigen::VectorXd x_k_k_rest_of_features = mT_ExtendKF->x_k_km1.tail(xk_km1_len - 13);
		Eigen::Vector3d XYZ_w;
		Eigen::VectorXd y;
		int index = 0;
		for(int i=0; i < feat_len ; i++)
		{
			if( strcmp(mT_ExtendKF->features_info[i].type.c_str(), "cartesian") == 0 )
			{//这里应该只会到达一次
				index = index + 3;
			}else if( strcmp(mT_ExtendKF->features_info[i].type.c_str(), "inversedepth") == 0 )
			{
				y = x_k_k_rest_of_features.segment(index, 6);
				XYZ_w = mT_ExtendKF->inversedepth2cartesian( y); //转换成笛卡尔坐标系
				index = index + 6;
			}
			
			if(  mT_ExtendKF->features_info[i].h.cols() )
				pred_patch_fc(i, XYZ_w);
		}
		
		// 第三步，使用归一化相关系数在搜索区域中查找对应关系
		matching(image);
	}
	void Tracking::calculate_Hi_cartesian(Eigen::VectorXd x_v, Eigen::VectorXd yi, int order, Eigen::MatrixXd& Hi)
	{
		Eigen::RowVectorXd zi = mT_ExtendKF->features_info[order].h;
		int number_of_features = mT_ExtendKF->features_info.size();
		Eigen::VectorXd inverse_depth_features_index = Eigen::MatrixXd::Zero(number_of_features, 1);
		Eigen::VectorXd cartesian_features_index = inverse_depth_features_index;
				
		for(int i=0; i < number_of_features; i++)
		{
			if( strcmp(mT_ExtendKF->features_info[i].type.c_str(), "inversedepth") == 0 )
				inverse_depth_features_index(i) = 1;
			else if( strcmp(mT_ExtendKF->features_info[i].type.c_str(), "cartesian") == 0 )
				cartesian_features_index(i) = 1;
		}
		
		int Hi_cols = 13 +3*cartesian_features_index.sum() + 6*inverse_depth_features_index.sum();
		Hi.setZero(2, Hi_cols);
		
		Eigen::MatrixXd a1 = (mT_ExtendKF->jacob_undistor_fm(zi)).inverse();
		Eigen::MatrixXd Rrw =  ( mT_ExtendKF->q2r( x_v.segment(3, 4) ) ).inverse() ;
		
		double f = mT_ExtendKF->cam->f;
		double ku = 1/(mT_ExtendKF->cam->dx);
		double kv = 1/(mT_ExtendKF->cam->dy);
		Eigen::Vector3d hrl = Rrw  *  (yi - x_v.head(3)) ;
		Eigen::Matrix<double, 2, 3> a2;
		a2 << f*ku/(hrl(2))  ,     0      ,     -hrl(0)*f*ku/(hrl(2)*hrl(2)),
				0   ,     f*kv/(hrl(2))  ,  -hrl(1)*f*kv/(hrl(2)*hrl(2));
		Eigen::MatrixXd a30 = a1 * a2 * (- Rrw);		
		//qconj(Xv_km1_k( 4:7 ))
		Eigen::Vector4d b1 = - x_v.segment(3, 4);
		Eigen::Vector4d b2;
		b1(0) = x_v(3);
		b2 << 1, -1, -1, -1;
		Eigen::MatrixXd b0 = (mT_ExtendKF->dRq_times_a_by_dq(b1, yi - x_v.head(3)) ) * (b2.asDiagonal() );
		Eigen::MatrixXd a31 = a1 * a2 * b0;	
		Eigen::MatrixXd a32 = Eigen::MatrixXd::Zero(2, 6);
		Hi.leftCols(13) << a30, a31, a32;
		
		int index_of_insertion = 13 + 3*(cartesian_features_index.head(order)).sum() + 6*(inverse_depth_features_index.head(order)).sum() + 1;
		Hi.middleCols(index_of_insertion-1, 3) = a1 * a2 * Rrw;
	}
	void Tracking::calculate_Hi_inverse_depth(Eigen::VectorXd x_v, Eigen::VectorXd yi, int order, Eigen::MatrixXd& Hi)
	{
		Eigen::RowVectorXd zi = mT_ExtendKF->features_info[order].h;
		int number_of_features = mT_ExtendKF->features_info.size();
		Eigen::VectorXd inverse_depth_features_index = Eigen::MatrixXd::Zero(number_of_features, 1);
		Eigen::VectorXd cartesian_features_index = inverse_depth_features_index;
		
		for(int i=0; i < number_of_features; i++)
		{
			if( strcmp(mT_ExtendKF->features_info[i].type.c_str(), "inversedepth") == 0 )
				inverse_depth_features_index(i) = 1;
			if( strcmp(mT_ExtendKF->features_info[i].type.c_str(), "cartesian") == 0 )
				cartesian_features_index(i) = 1;
		}
		
		int Hi_cols = 13 +3*cartesian_features_index.sum() + 6*inverse_depth_features_index.sum();
		Hi.setZero(2, Hi_cols);
		
		Eigen::MatrixXd a1 = (mT_ExtendKF->jacob_undistor_fm(zi)).inverse();
		double f = mT_ExtendKF->cam->f;
		double ku = 1/(mT_ExtendKF->cam->dx);
		double kv = 1/(mT_ExtendKF->cam->dy);
		Eigen::Vector3d mi;
		Eigen::MatrixXd Rrw =  ( mT_ExtendKF->q2r( x_v.segment(3, 4) ) ).inverse() ;
		mi << cos(yi(4))*sin(yi(3)), -sin(yi(4)), cos(yi(4))*cos(yi(3));
		Eigen::Vector3d hc = Rrw  *  ( (yi.head(3) - x_v.head(3)) * yi(5) + mi ) ;
		Eigen::Matrix<double, 2, 3> a2;
		a2 << f*ku/(hc(2))  ,     0      ,     -hc(0)*f*ku/(hc(2)*hc(2)),
				0   ,     f*kv/(hc(2))  ,  -hc(1)*f*kv/(hc(2)*hc(2));
		Eigen::MatrixXd a30 = a1 * a2 * (- Rrw) * yi(5);		
		//qconj(Xv_km1_k( 4:7 ))
		Eigen::Vector4d b1 = - x_v.segment(3, 4);
		Eigen::Vector4d b2;
		b1(0) = x_v(3);
		b2 << 1, -1, -1, -1;
		Eigen::MatrixXd b0 = (mT_ExtendKF->dRq_times_a_by_dq(b1, (yi.head(3) - x_v.head(3))*yi(5) + mi ) ) * (b2.asDiagonal() );
		Eigen::MatrixXd a31 = a1 * a2 * b0;	
		Eigen::MatrixXd a32 = Eigen::MatrixXd::Zero(2, 6);
		Hi.leftCols(13) << a30, a31, a32;
		
		int index_of_insertion = 13 + 3 * ( cartesian_features_index.head(order).sum() ) + 6 * ( inverse_depth_features_index.head(order).sum() ) + 1;
		Eigen::MatrixXd c1;
		Eigen::Vector3d c2, c3;
		c1 = yi(5) * Rrw;
		c2 << cos(yi(4))*cos(yi(3)), 0, -cos(yi(4))*sin(yi(3));
		c3 << -sin(yi(4))*sin(yi(3)), -cos(yi(4)), -sin(yi(4))*cos(yi(3));
		Eigen::MatrixXd c4 = Rrw * (yi.head(3) - x_v.head(3));
		Eigen::Matrix<double, 3, 6> c0;
		c0 << c1, Rrw * c2, Rrw * c3, c4;
		Hi.middleCols(index_of_insertion-1, 6) << (a1 * a2 * c0);
	}
	void Tracking::pred_patch_fc(int order, Eigen::Vector3d XYZ_w)
	{
		Eigen::Vector3d r_wc = mT_ExtendKF->x_k_km1.head(3);
		Eigen::Matrix3d R_wc = mT_ExtendKF->q2r(  mT_ExtendKF->x_k_km1.segment(3, 4) );
		Eigen::MatrixXd patch_pred;
		
		Eigen::RowVectorXd uv_p_pred = mT_ExtendKF->features_info[order].h;
		int halfW_pred = mT_ExtendKF->features_info[order].half_patch_size_when_matching;
		int uv_len = halfW_pred * 2 + 1;

		if((uv_p_pred(0) > halfW_pred) && (uv_p_pred(0) < (mT_ExtendKF->cam->nCols - halfW_pred) ) &&
			(uv_p_pred(1) > halfW_pred) && (uv_p_pred(1) < (mT_ExtendKF->cam->nRows - halfW_pred) ) )  
		{
			Eigen::RowVector2d uv_p_f=mT_ExtendKF->features_info[order].uv_when_initialized;
			Eigen::Matrix3d R_Wk_p_f=mT_ExtendKF->features_info[order].R_wc_when_initialized;
			Eigen::Vector3d r_Wk_p_f = mT_ExtendKF->features_info[order].r_wc_when_initialized;
			Eigen::MatrixXd patch_p_f=mT_ExtendKF->features_info[order].patch_when_initialized;
			int halfW_fea=mT_ExtendKF->features_info[order].half_patch_size_when_initialized;
			double dx =mT_ExtendKF->cam->dx;
			double f = mT_ExtendKF->cam->f;
			double cx =mT_ExtendKF->cam->Cx;
			double cy = mT_ExtendKF->cam->Cy;
			Eigen::Matrix3d K = mT_ExtendKF->cam->K;
			
			Eigen::Matrix4d a, b, H_Wk_p_f, H_Wk, H_kpf_k;
			a << R_Wk_p_f, Eigen::MatrixXd::Zero(3, 1), Eigen::MatrixXd::Zero(1, 3), 1;
			b << Eigen::MatrixXd::Identity(3, 3), r_Wk_p_f, Eigen::MatrixXd::Zero(1, 3), 1;
			H_Wk_p_f = a * b;
			a << R_wc, Eigen::MatrixXd::Zero(3, 1), Eigen::MatrixXd::Zero(1, 3), 1;
			b << Eigen::MatrixXd::Identity(3, 3), r_wc, Eigen::MatrixXd::Zero(1, 3), 1;
			H_Wk = a * b ;
			H_kpf_k = H_Wk_p_f.inverse() * H_Wk;
			
			Eigen::VectorXd n2, n_temp;
			Eigen::Vector3d n1, n;
			n1 << (uv_p_f(0) - cx), (uv_p_f(1) - cy), -f/dx;
			n = n1/n1.norm();
			n1 = n;
		
			n2.resize(4);
			n2 << (uv_p_pred(0) - cx), (uv_p_pred(1) - cy), -f/dx, 1;
			n_temp = H_kpf_k * n2;
			n2 = n_temp/n_temp(3);
			n_temp = n2.head(3);
			n = n1 + n_temp/n_temp.norm();
			n1 = n/n.norm();
			n = n1;
		
			Eigen::Vector4d XYZ_kpf;
			Eigen::Vector4d XYZ_temp;
			XYZ_temp << XYZ_w, 1;
			XYZ_kpf = H_Wk_p_f.inverse() * XYZ_temp;
			XYZ_temp = XYZ_kpf/XYZ_kpf(3);
			XYZ_kpf = XYZ_temp;
			double d = -(n.transpose()) * (XYZ_kpf.head(3));
			
		
			//uv_p_pred_patch=rotate_with_dist_fc_c2c1(cam,uv_p_f,H_kpf_k(1:3,1:3),H_kpf_k(1:3,4),n,d);
			Eigen::MatrixXd uv_c1_und, uc_c2_und, uv_c2;
			mT_ExtendKF->undistort_fm(uv_p_f.transpose(), uv_c1_und);
			//uv_c1_und.transposeInPlace();
	
			Eigen::MatrixXd uc_c2_und_a = (K * (H_kpf_k.topLeftCorner(3, 3) - ( H_kpf_k.topRows(3).col(3) * n.transpose()/d)) * K.inverse() ).inverse();
			Eigen::MatrixXd uc_c2_und_b, uv_temp;
			
			uc_c2_und_b.resize(uv_c1_und.rows() + 1, uv_c1_und.cols());
			uc_c2_und_b << uv_c1_und, Eigen::MatrixXd::Ones(1, uv_c1_und.cols());
			uv_temp = uc_c2_und_a * uc_c2_und_b;
			
			uc_c2_und_a.resize(2, uv_temp.cols());
			uc_c2_und_a << uv_temp.row(2), uv_temp.row(2);
			uc_c2_und = uv_temp.topRows(2).array() / uc_c2_und_a.array() ;
			mT_ExtendKF->distort_fm(uc_c2_und, uv_c2);
			//uv_c2.transposeInPlace();
			
			
			Eigen::Vector2d uv_p_pred_patch = uv_c2.col(0).head(2);
			Eigen::MatrixXd u_pred, v_pred,uv_pred;
			Converter::meshgrid_opencv(cv::Range(uv_p_pred_patch(0)-halfW_pred, uv_p_pred_patch(0)+halfW_pred), 
						   cv::Range(uv_p_pred_patch(1)-halfW_pred, uv_p_pred_patch(1)+halfW_pred), u_pred, v_pred);
			u_pred = Eigen::Map<Eigen::MatrixXd>(u_pred.data(), pow(uv_len, 2), 1);//相当于MATLAB里的reshape矩阵重塑函数
			v_pred = Eigen::Map<Eigen::MatrixXd>(v_pred.data(), pow(uv_len, 2), 1);
			uv_pred.resize(u_pred.rows(), u_pred.cols()+v_pred.cols());
			uv_pred << u_pred, v_pred;
			
			//uv_pred_imak_dist=rotate_with_dist_fc_c1c2(cam,uv_pred,H_kpf_k(1:3,1:3),H_kpf_k(1:3,4),n,d);
			Eigen::MatrixXd uv_c2_und, uv_c1_und_a, uv_c1;
			mT_ExtendKF->undistort_fm(uv_pred.transpose(), uv_c2_und);
			
			uv_c1_und_a.resize(uv_c2_und.rows() + 1, uv_c2_und.cols());
			uv_c1_und_a << uv_c2_und, Eigen::MatrixXd::Ones(1, uv_c2_und.cols());
			uv_c1_und = K * (H_kpf_k.topLeftCorner(3, 3) - ( H_kpf_k.topRows(3).col(3) * n.transpose()/d) ) * K.inverse() * uv_c1_und_a;
			
			uv_c1_und_a.resize(2, uv_c1_und.cols());
			uv_c1_und_a << uv_c1_und.row(2), uv_c1_und.row(2);
			uv_c1_und = uv_c1_und.topRows(2).array() / ( uv_c1_und_a.array());
			mT_ExtendKF->distort_fm(uv_c1_und, uv_c1);
			
			Eigen::MatrixXd uv_pred_imak_dist;
			uv_pred_imak_dist = uv_c1.transpose();
			uv_pred_imak_dist.col(0) = uv_pred_imak_dist.col(0).array() - ( uv_p_f(0)-halfW_fea-1 );
			uv_pred_imak_dist.col(1) = uv_pred_imak_dist.col(1).array() - ( uv_p_f(1)-halfW_fea-1 );
			
			Eigen::MatrixXd u_pred_imak_dist, v_pred_imak_dist;
			u_pred_imak_dist = Eigen::Map<Eigen::MatrixXd>(uv_pred_imak_dist.col(0).data(), uv_len, uv_len);
			v_pred_imak_dist = Eigen::Map<Eigen::MatrixXd>(uv_pred_imak_dist.col(1).data(), uv_len, uv_len);
			
			cv::Mat patch_pred_mat;
			cv::remap(Converter::toCvMat_f(patch_p_f), patch_pred_mat, Converter::toCvMat_f(u_pred_imak_dist), Converter::toCvMat_f(v_pred_imak_dist), cv::INTER_LINEAR, 0, cvScalarAll(0));
			patch_pred = Converter::toMatrixd_atf( patch_pred_mat );
			
		}else patch_pred = Eigen::MatrixXd::Zero(uv_len,  uv_len);

		mT_ExtendKF->features_info[order].patch_when_matching = patch_pred;
	}
	void Tracking::matching(cv::Mat image)
	{
		double correlation_threshold = 0.80;
		//double correlation_threshold = 0.70;
		double chi_095_2 = 5.9915;
		Eigen::RowVectorXd h;
		Eigen::MatrixXd S;
		Eigen::MatrixXd predicted_patch;
		int half_patch_size_when_matching;
		int pixels_in_the_matching_patch;

		int feat_len = mT_ExtendKF->features_info.size();
		for(int feature_i = 0 ; feature_i < feat_len ; feature_i++)
		{
			if( mT_ExtendKF->features_info[feature_i].h.cols() )
			{//如果进行了预测，则对这个区域进行搜索
				h = mT_ExtendKF->features_info[feature_i].h;
				S = mT_ExtendKF->features_info[feature_i].S;
				half_patch_size_when_matching = mT_ExtendKF->features_info[feature_i].half_patch_size_when_matching;
				pixels_in_the_matching_patch = pow( (2*half_patch_size_when_matching+1), 2);
				
				//if the ellipse is too big, do not search (something may be wrong)
				Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver ( S );
				Eigen::VectorXd S_eig = eigen_solver.eigenvalues();
				if( ( S_eig.maxCoeff() ) < 100 )
				{
					predicted_patch = mT_ExtendKF->features_info[feature_i].patch_when_matching;
					int half_search_region_size_x=ceil(2*sqrt(S(0, 0)));
					int half_search_region_size_y=ceil(2*sqrt(S(1, 1)));
					
					Eigen::MatrixXd patches_for_correlation = Eigen::MatrixXd::Zero(pixels_in_the_matching_patch, (2*half_search_region_size_x+1)*(2*half_search_region_size_y+1) + 1);
					Eigen::MatrixXd match_candidates = Eigen::MatrixXd::Zero(2, (2*half_search_region_size_x+1)*(2*half_search_region_size_y+1) + 1 );
					patches_for_correlation.col(0) = Eigen::Map<Eigen::MatrixXd>(predicted_patch.data(), pixels_in_the_matching_patch, 1);//reshape矩阵重塑
					
					int index_patches_for_correlation = 0;
					int x_end = round(h(0))+half_search_region_size_x;
					int y_end = round(h(1))+half_search_region_size_y;
					Eigen::Vector2d nu;
					for(int j = (round(h(0))-half_search_region_size_x ); j <= x_end ; j++)
					{
						for(int i = (round(h(1))-half_search_region_size_y) ;i <= y_end ; i++)
						{
							nu << j-h(0),  i-h(1); 
							if( (nu.transpose() * S.inverse() * nu) < chi_095_2 )
							{
								if( (j > half_patch_size_when_matching) && (j < (mT_ExtendKF->cam->nCols-half_patch_size_when_matching) )&&
									(i > half_patch_size_when_matching) && (i < (mT_ExtendKF->cam->nRows-half_patch_size_when_matching) ) )
								{
									cv::Mat image_patch = image( cv::Range(i-half_patch_size_when_matching, i+half_patch_size_when_matching+1), 
															cv::Range(j-half_patch_size_when_matching, j+half_patch_size_when_matching+1 ) );
									index_patches_for_correlation = index_patches_for_correlation + 1;
									Eigen::MatrixXd im_Mat = (Converter::toMatrixd_atuc(image_patch));
									patches_for_correlation.col( index_patches_for_correlation ) = Eigen::Map<Eigen::MatrixXd>(im_Mat.data(), pixels_in_the_matching_patch, 1);//reshape重塑
									match_candidates.col(index_patches_for_correlation-1) << j, i;
								}
							}
						}
					}
					
					Eigen::MatrixXd correlation_matrix = Converter::corrcoef_opencv(patches_for_correlation.leftCols(index_patches_for_correlation+1));//计算归一化相关系数
					int len = correlation_matrix.cols();
					Eigen::VectorXd corr_temp = (correlation_matrix.topRightCorner(1, len - 1)).transpose();
					double maximum_correlation = 0, index = 0;
					maximum_correlation = corr_temp.maxCoeff(&index);//求行向量最大值以及所在行数
					if( maximum_correlation > correlation_threshold )
					{
						mT_ExtendKF->features_info[feature_i].individually_compatible = true;
						mT_ExtendKF->features_info[feature_i].z = match_candidates.col(index);
					}
				}
			}
		}
	}
	void Tracking::ransac_hypotheses(void)
	{
		double p_at_least_one_spurious_free = 0.99; // default value
		//RANSAC threshold should have a low value (less than the standard deviation of the filter measurement noise); as high innovation points  will be later rescued
		double threshold = mT_ExtendKF->std_z;
		int n_hyp = 1000; 				// initial number of iterations, will be updated
		int max_hypothesis_support = 0; // will be updated
		
		//第一步，获得状态向量的模式
		Eigen::MatrixXd state_vector_pattern = Eigen::MatrixXd::Zero(mT_ExtendKF->x_k_km1.rows(), 4);
		Eigen::MatrixXd z_id, z_euc, z_temp;
		z_id.resize(0, 0);
		z_euc.resize(0, 0);
		int position = 14 -  1;			//相机位姿向量是13维
		int feat_len = mT_ExtendKF->features_info.size();
		for(int i = 0 ; i < feat_len ; i++)
		{
			int z_len = mT_ExtendKF->features_info[i].z.rows();
			if( strcmp(mT_ExtendKF->features_info[i].type.c_str(), "inversedepth") == 0 )
			{//应该大部分进入这里
				if( z_len )
				{
					state_vector_pattern.middleRows(position, 3).col(0) = Eigen::MatrixXd::Ones(3, 1);
					state_vector_pattern.middleRows(position+3, 2).col(1) = Eigen::MatrixXd::Ones(2, 1);
					state_vector_pattern(position+5, 2) = 1;
					z_temp.resize(2, z_id.cols() + 1);
					if(z_id.cols())
						z_temp << z_id, mT_ExtendKF->features_info[i].z.head(2);
					else z_temp <<  mT_ExtendKF->features_info[i].z.head(2);
					z_id = z_temp;
				}
				position += 6;
			}else if( strcmp(mT_ExtendKF->features_info[i].type.c_str(), "cartesian") == 0 )
			{
				if( z_len )
				{
					state_vector_pattern.middleRows(position, 3).col(3) = Eigen::MatrixXd::Ones(3, 1);
					z_temp.resize(2, z_euc.cols() + 1);
					if(z_euc.cols())
						z_temp << z_euc, mT_ExtendKF->features_info[i].z.head(2);
					else z_temp << mT_ExtendKF->features_info[i].z.head(2);
					z_euc = z_temp;
				}
				position += 3;
			}
		}
		

		Eigen::VectorXd individually_compatible, zi;
		Eigen::VectorXd xi ;
		int num_IC_matches = 0;
		for(int i = 0 ; i < n_hyp ; i++)
		{
			//第二步，选择任意特征准备匹配，之后进行状态更新
			individually_compatible.setZero(feat_len);
			for(int j = 0 ; j < feat_len ; j++)
			{
				if( mT_ExtendKF->features_info[j].individually_compatible )
					individually_compatible(j) = 1;
			}	
			double t = (mT_ExtendKF->rand(1, 1, 0, 1))(0, 0);
			int random_match_position = floor( t *  individually_compatible.array().sum() ) ;
			Eigen::VectorXd positions_individually_compatible = Converter::find(individually_compatible, 0);
			position = positions_individually_compatible(random_match_position);
			zi = mT_ExtendKF->features_info[position].z;
			num_IC_matches = individually_compatible.array().sum();
			
			Eigen::MatrixXd Hi = mT_ExtendKF->features_info[position].H;
			Eigen::MatrixXd S = Hi * (mT_ExtendKF->p_k_km1) * Hi.transpose() + mT_ExtendKF->features_info[position].R;
			Eigen::MatrixXd K = (mT_ExtendKF->p_k_km1) * Hi.transpose() * S.inverse();
			xi = mT_ExtendKF->x_k_km1 + K * ( zi - (mT_ExtendKF->features_info[position].h).transpose() );

			//第三步，计算假设支持：预测测量并计算阈值下的匹配
			int hypothesis_support = 0;
			Eigen::Matrix<ptrdiff_t, Eigen::Dynamic, Eigen::Dynamic> positions_li_inliers_id, positions_li_inliers_euc;
			
			Eigen::MatrixXd rwc;
			Eigen::Matrix3d rotcw = (mT_ExtendKF->q2r(xi.segment(3, 4))).transpose();
			Eigen::MatrixXd hc, h_norm ;
			double ku =1/(double)(mT_ExtendKF->cam->dx);
			double f = mT_ExtendKF->cam->f;
			double u0 =mT_ExtendKF->cam->Cx;
			double v0 = mT_ExtendKF->cam->Cy;
			Eigen::MatrixXd h_image, h_image_a, h_distorted;
			Eigen::MatrixXd nu, residuals;
			
			int z_id_len = z_id.cols();
			if( z_id_len )
			{
				Eigen::MatrixXd ri, anglesi, rhoi, temp;
				Eigen::VectorXd ri_v,  anglesi_v, rhoi_v;
				ri_v = Converter::select(xi, state_vector_pattern.col(0));
				anglesi_v = Converter::select(xi, state_vector_pattern.col(1));
				rhoi_v = Converter::select(xi, state_vector_pattern.col(2));
				
				ri = Eigen::Map<Eigen::MatrixXd>(ri_v.data(), 3, z_id_len);//相当于MATLAB里的reshape矩阵重塑函数
				anglesi = Eigen::Map<Eigen::MatrixXd>(ri_v.data(), 2, z_id_len);//相当于MATLAB里的reshape矩阵重塑函数
				rhoi = rhoi_v.asDiagonal();
				
				Eigen::MatrixXd mi;
				mi.resize(3, anglesi.cols());
				mi << (anglesi.row(1).array().cos()) * (anglesi.row(0).array().sin()), 
					(-anglesi.row(1).array().sin()), (anglesi.row(1).array().cos()) * (anglesi.row(0).array().cos());
				
				rwc = Converter::repmat(xi.head(3), 1, z_id_len);
				Eigen::MatrixXd ri_minus_rwc = ri - rwc;
				Eigen::RowVectorXd xi_minus_xwc_by_rhoi = ri_minus_rwc.row(0) * rhoi;
				Eigen::RowVectorXd yi_minus_xwc_by_rhoi = ri_minus_rwc.row(1) * rhoi;
				Eigen::RowVectorXd zi_minus_xwc_by_rhoi = ri_minus_rwc.row(2) * rhoi;
				Eigen::MatrixXd ri_minus_rwc_by_rhoi;
				ri_minus_rwc_by_rhoi.resize(3, xi_minus_xwc_by_rhoi.cols());
				ri_minus_rwc_by_rhoi <<  xi_minus_xwc_by_rhoi, yi_minus_xwc_by_rhoi, zi_minus_xwc_by_rhoi;
				
				hc = rotcw*(ri_minus_rwc_by_rhoi + mi);
				h_norm.resize(2, hc.cols());
				h_norm << (hc.row(0).array() / hc.row(2).array()), (hc.row(1).array() / hc.row(2).array());
				
				h_image_a.resize(2, z_id_len);
				h_image_a << u0 * Eigen::MatrixXd::Ones(1, z_id_len), v0 * Eigen::MatrixXd::Ones(1, z_id_len);
				h_image  = f * ku * h_norm + h_image_a;
				mT_ExtendKF->distort_fm( h_image, h_distorted );
				
				nu = z_id - h_distorted;
				residuals = ( ( (nu.row(0).array().pow(2)) + (nu.row(1).array().pow(2)) ).array().sqrt() ).transpose();
				positions_li_inliers_id =  (residuals.array() < threshold).rowwise().count();
				hypothesis_support = hypothesis_support + positions_li_inliers_id.array().sum();
			}else positions_li_inliers_id.resize(0, 0);

			int z_euc_len = z_euc.cols();
			if( z_euc_len )
			{
				Eigen::MatrixXd xyz;
				xyz = Converter::select(xi, state_vector_pattern.col(3));
				xyz = Eigen::Map<Eigen::MatrixXd>(xyz.data(), 3, z_euc_len);//相当于MATLAB里的reshape矩阵重塑函数
				rwc = Converter::repmat(xi.head(3), 1, z_euc_len);
				Eigen::MatrixXd xyz_minus_rwc = xyz - rwc;
	
				hc = rotcw * xyz_minus_rwc;
				h_norm.resize(2, hc.cols());
				h_norm << (hc.row(0).array() / hc.row(2).array()), (hc.row(1).array() / hc.row(2).array());
				
				h_image_a.resize(2, z_euc_len);
				h_image_a << u0 * Eigen::MatrixXd::Ones(1, z_euc_len), v0 * Eigen::MatrixXd::Ones(1, z_euc_len);
				h_image  = f * ku * h_norm + h_image_a;
				mT_ExtendKF->distort_fm( h_image, h_distorted );
				
				nu = z_id - h_distorted;
				residuals = (( (nu.row(0).array().pow(2)) + (nu.row(1).array().pow(2)) ).array().sqrt()).transpose();
				positions_li_inliers_euc = (residuals.array() <threshold).rowwise().count();
				hypothesis_support = hypothesis_support + positions_li_inliers_euc.array().sum();

			}else positions_li_inliers_euc.resize(0, 0);
				
			
			//第四步，判断假设检验的合理性，更新迭代次数
			if(hypothesis_support > max_hypothesis_support)
			{
				max_hypothesis_support = hypothesis_support;
				int j_id = 0, j_euc = 0;
				for(int j = 0 ; j < feat_len ; j++)
				{
					if( mT_ExtendKF->features_info[j].z.rows() )
					{
						if( strcmp(mT_ExtendKF->features_info[j].type.c_str(), "cartesian") == 0 )
						{
							if(positions_li_inliers_euc(j_euc))
								mT_ExtendKF->features_info[j].low_innovation_inlier = true;
							else mT_ExtendKF->features_info[j].low_innovation_inlier = false;
							j_euc = j_euc + 1;
						}else if( strcmp(mT_ExtendKF->features_info[j].type.c_str(), "inversedepth") == 0 )
						{//应该进来这里
							if(positions_li_inliers_id(j_id))
								mT_ExtendKF->features_info[j].low_innovation_inlier = true;
							else mT_ExtendKF->features_info[j].low_innovation_inlier = false;
							j_id = j_id + 1;
						}			
					}
				}
				
				double epsilon = 1 - ( (double)hypothesis_support / (double)num_IC_matches );
				n_hyp = ceil( (log(1-p_at_least_one_spurious_free)) / (log(1 - (1-epsilon) )) );
				if(n_hyp == 0) 
					break; 
			}
			if( i > n_hyp) 
				break; 
		}
	}
	void Tracking::calculate_derivatives(Eigen::VectorXd xk_km1)
	{
		//features_info = calculate_derivatives( get_x_k_km1(filter), cam, features_info );
		Eigen::VectorXd x_v = xk_km1.head(13);
		int xk_km1_rows = xk_km1.rows();
		Eigen::VectorXd x_features = xk_km1.tail(xk_km1_rows - 13);

		int feat_len = mT_ExtendKF->features_info.size();
		int index = 0;
		Eigen::VectorXd y;
		Eigen::MatrixXd Hi;
		for(int i=0 ; i < feat_len ; i++)
		{
			if( mT_ExtendKF->features_info[i].h.cols() )
			{
				if( strcmp(mT_ExtendKF->features_info[i].type.c_str(), "cartesian") == 0 )
				{
					y = x_features.segment(index, 3);
					calculate_Hi_cartesian(x_v, y, i ,Hi);
					index = index + 3;
				}else{
					y = x_features.segment(index, 6);
					calculate_Hi_inverse_depth(x_v, y, i ,Hi);
					index = index + 6;
				}
				mT_ExtendKF->features_info[i].H = Hi;
			}else{
				if( strcmp(mT_ExtendKF->features_info[i].type.c_str(), "cartesian") == 0 )
					index = index + 3;
				else if( strcmp(mT_ExtendKF->features_info[i].type.c_str(), "inversedepth") == 0 )	
					index = index + 6;
			}
		}
	}
	void Tracking::rescue_hi_inliers(void)
	{
		double chi2inv_2_95 = 5.9915;

		mT_ExtendKF->predict_camera_measurements( mT_ExtendKF->x_k_k );
		calculate_derivatives( mT_ExtendKF->x_k_k );
		
		int feat_len = mT_ExtendKF->features_info.size();
		Eigen::VectorXd hi, nui;
		Eigen::MatrixXd Si;
		for(int i = 0 ; i < feat_len ; i++)
		{
			if ( (mT_ExtendKF->features_info[i].individually_compatible ==  true ) && ( mT_ExtendKF->features_info[i].low_innovation_inlier == false) )
			{
				hi = (mT_ExtendKF->features_info[i].h).transpose();
				Si = mT_ExtendKF->features_info[i].H * mT_ExtendKF->p_k_k * ((mT_ExtendKF->features_info[i].H).transpose() );
				nui = mT_ExtendKF->features_info[i].z - hi;
				if( (nui.transpose() * Si.inverse() *nui) < chi2inv_2_95 )
					mT_ExtendKF->features_info[i].high_innovation_inlier = true;
				else mT_ExtendKF->features_info[i].high_innovation_inlier = false;
			}
		}

	}


	
	
	
	
	
	
	
}
