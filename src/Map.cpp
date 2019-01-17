#include "ransac_slam/Map.h"
//#include <ransac_slam/ExtendKF.h>
//#include <opencv2/core/mat.hpp>
//#include <cmath>

namespace ransac_slam
{
	Map::Map( const int min_fea, ExtendKF* m_ExtendKF)
		:min_features(min_fea),  mM_ExtendKF(m_ExtendKF)
	{

	}
	Map::~Map()
	{

	}
	void Map::map_management(cv::Mat image, int step)
	{
		//第一步，筛选并删除不符合条件的特征点
		std::vector<Feature>::iterator feat_it;
		int i = 1;//表示特征关键帧的序号，涉及计算，注意从1开始
		for(feat_it = mM_ExtendKF->features_info.begin() ; feat_it != mM_ExtendKF->features_info.end() ; i++ )
		{
			if( ( (*feat_it).times_measured ) < ( ((*feat_it).times_predicted) * 0.5 )//观测次数小于预测次数的一半
				&& ( (*feat_it).times_predicted ) > 5 )//预测次数太多？？？
			{
				feat_it = mM_ExtendKF->features_info.erase(feat_it);    //删除元素，返回值指向已删除元素的下一个位置
				delete_a_feature( i );//Now, remove them from the state, covariance and features_info vector
			}else
				 ++feat_it;    //指向下一个位置
		}
		
		//第二步，更新特征点信息，重置相关状态
		int measured = 0 ;
		int feat_len = mM_ExtendKF->features_info.size();
		for(int i = 0 ;i < feat_len ; i++)
		{
			//特征点参数更新
			if(mM_ExtendKF->features_info[i].h.cols())
				mM_ExtendKF->features_info[i].times_predicted += 1;//预测次数加1
			if( mM_ExtendKF->features_info[i].low_innovation_inlier || mM_ExtendKF->features_info[i].high_innovation_inlier)
			{
				measured += 1;
				mM_ExtendKF->features_info[i].times_measured += 1;//观测次数加1
			}
			
			//特征点相关参数重置
			mM_ExtendKF->features_info[i].individually_compatible = false;
			mM_ExtendKF->features_info[i].low_innovation_inlier = false;
			mM_ExtendKF->features_info[i].high_innovation_inlier = false;
			mM_ExtendKF->features_info[i].h.resize(0);
			mM_ExtendKF->features_info[i].z.resize(0);
			mM_ExtendKF->features_info[i].H.resize(0, 0);
			mM_ExtendKF->features_info[i].S.resize(0, 0);
		}
	
		//第三步，如果需要的话，就将特征点的逆深度坐标转换为笛卡尔坐标
		inversedepth_2_cartesian();

		//第四步，随机性初始化特征点
		if(measured == 0)
		{//无局内点，以最大数量进行特征点采样
			initialize_features(step, min_features, image);
		}else{
			if(measured < min_features)//保证特征点数量小于阈值
				initialize_features(step, min_features - measured, image);
		}	
	}
	void Map::delete_a_feature(int feature_id)
	{//注意此处feature_id是从1开始编号
		
		//记录特征点状态向量的宽度，不同类型特征点占用宽度不同
		int parToDelete = 0;
		if(strcmp(mM_ExtendKF->features_info[feature_id-1].type.c_str(), "cartesian") == 0 )
			parToDelete = 3;
		else parToDelete = 6;
		
		//用于记录需要删除的特征点的起始位置
		int indexFromWichDelete = 14 - 1;//注意，机器人状态向量0-12, 所以特征点的状态从13开始
		for(int i = 0 ; i < feature_id-1 ; i++)
		{
			if( strcmp(mM_ExtendKF->features_info[i].type.c_str(), "inversedepth") == 0  )
				indexFromWichDelete += 6;
			if( strcmp(mM_ExtendKF->features_info[i].type.c_str(), "cartesian") == 0  )
				indexFromWichDelete += 3;
		}
		
		Eigen::VectorXd x_temp;
		Eigen::MatrixXd p_temp;
		int delete_surplus = mM_ExtendKF->x_k_k.rows() - (indexFromWichDelete + parToDelete);
		x_temp.resize((mM_ExtendKF->x_k_k.rows() - parToDelete));
		p_temp.resize((mM_ExtendKF->p_k_k.rows() - parToDelete), (mM_ExtendKF->p_k_k.rows() - parToDelete));
		
		//删除对应的状态向量
		x_temp <<  mM_ExtendKF->x_k_k.head(indexFromWichDelete), mM_ExtendKF->x_k_k.tail(delete_surplus);
		mM_ExtendKF->x_k_k = x_temp;
		
		//删除对应的协方差
		p_temp << mM_ExtendKF->p_k_k.topLeftCorner(indexFromWichDelete, indexFromWichDelete), 
					mM_ExtendKF->p_k_k.topRightCorner(indexFromWichDelete, delete_surplus),
					mM_ExtendKF->p_k_k.bottomLeftCorner(delete_surplus, indexFromWichDelete), 
					mM_ExtendKF->p_k_k.bottomRightCorner(delete_surplus, delete_surplus);
		mM_ExtendKF->p_k_k = p_temp;
	}
	void Map::inversedepth_2_cartesian(void)
	{
		double linearity_index_threshold = 0.1;
		Eigen::VectorXd X = mM_ExtendKF->x_k_k;
		Eigen::MatrixXd P = mM_ExtendKF->p_k_k;
		
		int feat_len = mM_ExtendKF->features_info.size();
		for(int i = 0; i < feat_len; i++)
		{
			if( strcmp(mM_ExtendKF->features_info[i].type.c_str(), "inversedepth") == 0  )
			{
				int initialPositionOfFeature = 14 - 1;//记录特征点起始位置
				for(int j = 0 ; j < i ; j++)
				{//注意，原程序说每一次step只进行一次转换
					if( strcmp(mM_ExtendKF->features_info[j].type.c_str(), "cartesian") == 0  )
						initialPositionOfFeature += 3;
					if( strcmp(mM_ExtendKF->features_info[j].type.c_str(), "inversedepth") == 0  )
						initialPositionOfFeature += 6;
				}
				
				double std_rho = sqrt(P(initialPositionOfFeature + 5, initialPositionOfFeature + 5));
				double rho = X(initialPositionOfFeature + 5);
				double std_d = std_rho/(rho*rho);
				
				//camera 2 to point distance
				double theta = X(initialPositionOfFeature + 3);
				double phi = X(initialPositionOfFeature + 4);
				
				//double mi = m(theta,phi);
				//Unit vector from azimut-elevation angles
				Eigen::Vector3d mi;
				mi << cos(phi)*sin(theta) ,  -sin(phi) , cos(phi)*cos(theta);
				
				Eigen::Vector3d x_c1 = X.segment(initialPositionOfFeature, 3);
				Eigen::Vector3d x_c2 = X.head(3);
				Eigen::VectorXd X_in = X.segment(initialPositionOfFeature, 6);
				Eigen::Vector3d X_out = mM_ExtendKF->inversedepth2cartesian( X_in);
				double d_c2p = (X_out - x_c2).norm();
				
				//alpha (parallax)
				double a=((X_out - x_c1).transpose() * (X_out - x_c2));
				double b=((X_out - x_c1).norm() * (X_out - x_c2).norm());
				double cos_alpha = a/b;
			
				//Linearity index
				double linearity_index = 4*std_d*cos_alpha/d_c2p;
				if(linearity_index < linearity_index_threshold)
				{
					//更新状态向量
					Eigen::VectorXd X_expansion;
					int size_X_old = X.rows();
					X_expansion.resize( size_X_old - 6 + X_out.rows());
					X_expansion << X.head(initialPositionOfFeature), X_out, X.tail(size_X_old - initialPositionOfFeature - 6);
					mM_ExtendKF->x_k_k = X_expansion;
					
					//更新协方差矩阵
					Eigen::Vector3d dm_dtheta, dm_dphi;
					Eigen::MatrixXd P_expansion;
					Eigen::MatrixXd J, J_all, J_all1, J_all2, J_all3;
			
					dm_dtheta << cos(phi)*cos(theta) , 0 ,  -cos(phi)*sin(theta);
					dm_dphi << -sin(phi)*sin(theta) ,  -cos(phi) ,  -sin(phi)*cos(theta);
			
					J.resize(3, 6);
					J << Eigen::MatrixXd::Identity(3, 3) , (1/rho)*dm_dtheta ,  (1/rho)*dm_dphi, -mi/(rho*rho);
					
					J_all1.resize(initialPositionOfFeature, size_X_old);
					J_all1 << Eigen::MatrixXd::Identity(initialPositionOfFeature, initialPositionOfFeature ), 
							Eigen::MatrixXd::Zero(initialPositionOfFeature, 6), 
							Eigen::MatrixXd::Zero(initialPositionOfFeature, size_X_old - initialPositionOfFeature - 6);
					
					J_all2.resize(J.rows(), size_X_old+J.cols()-6);
					J_all2 << Eigen::MatrixXd::Zero(3, initialPositionOfFeature), J, Eigen::MatrixXd::Zero(3, size_X_old - initialPositionOfFeature - 6);
					
					J_all3.resize(size_X_old - initialPositionOfFeature - 6, size_X_old);
					J_all3 << Eigen::MatrixXd::Zero(size_X_old - initialPositionOfFeature - 6, initialPositionOfFeature), 
							Eigen::MatrixXd::Zero(size_X_old - initialPositionOfFeature - 6,  6), 
							Eigen::MatrixXd::Identity(size_X_old - initialPositionOfFeature-6, size_X_old - initialPositionOfFeature-6);
					
					int J_all_rows = J_all1.rows() + J_all2.rows() + J_all3.rows();
					int J_all_cols = J_all1.cols() ;
					J_all.resize(J_all_rows, J_all_cols);
					J_all << J_all1, J_all2, J_all3;
					P_expansion = J_all * P * J_all.transpose();
					mM_ExtendKF->p_k_k = P_expansion;
					mM_ExtendKF->features_info[i].type = "cartesian";//笛卡尔坐标系
	
					return ; //for the moment, only convert one feature per step (sorry!)
				}
			}	
		}
	}
	void Map::initialize_features(int step, int min_features_to_init, cv::Mat image)
	{
		int max_attempts = 50;//最大采样次数
		int attempts = 0;		//特征点采样次数
		int initialized = 0;		//本次采样获得的特征点数量
		Eigen::MatrixXd uv;	//特征点的图像坐标
		
		while( (initialized < min_features_to_init) &&(attempts < max_attempts))
		{
			attempts+=1;
			initialize_a_features(step, image, uv);//初始化一个特征点
			if(uv.cols() !=0)
				initialized+=1;
		}
	}
	void Map::initialize_a_features(int step, cv::Mat image, Eigen::MatrixXd& uv)
	{
		//numerical values
		int excluded_band = 21;
		int initializing_box_semisize[2] = {30, 20};//采样矩形的大小，列*行
		int initial_rho = 1;
		int std_rho = 1;	
		double std_pxl = mM_ExtendKF->std_z;
		
		//第一步，预测所有特征在下一时刻可能出现的位置，用于检查重复特征
		mM_ExtendKF->predict_camera_measurements(mM_ExtendKF->x_k_k);
		int feat_len = mM_ExtendKF->features_info.size();
		Eigen::MatrixXd uv_pred;
		std::vector<Eigen::Vector2d> h_pred;//特征点
		h_pred.clear();
		for(int i = 0 ; i < feat_len ; i++)
		{
			if(mM_ExtendKF->features_info[i].h.cols())
				h_pred.push_back( mM_ExtendKF->features_info[i].h.head(2).transpose() );
		}
		
		//第二步，随机产生采样窗口，提取fast特征点
		Eigen::Vector2d rand_region_center = mM_ExtendKF->rand(2, 1, 0, 1);//索引，列*行
		rand_region_center(0) = round ( rand_region_center(0) * (mM_ExtendKF->cam->nCols - 2 * excluded_band - 2 * initializing_box_semisize[0] ) ) + excluded_band + initializing_box_semisize[0];
		rand_region_center(1) = round ( rand_region_center(1) * (mM_ExtendKF->cam->nRows - 2 * excluded_band - 2 * initializing_box_semisize[1] ) ) + excluded_band + initializing_box_semisize[1];
		cv::Mat im_k = image( cv::Range(rand_region_center(1) - initializing_box_semisize[1], //图像选取，先行后列
						rand_region_center(1) + initializing_box_semisize[1] + 1), 
					   cv::Range(rand_region_center(0) - initializing_box_semisize[0], rand_region_center(0) + initializing_box_semisize[0]+ 1) );
		Eigen::Matrix<double, 2, Eigen::Dynamic> all_uv, all_uv_cal;
		fast_corner_detect_9(im_k, 100, all_uv);		//阈值可以控制检测特征点的数量
		if( all_uv.cols() )
		{
			all_uv_cal.resize(2, all_uv.cols());
			all_uv_cal.row(0) = all_uv.row(0) + (- initializing_box_semisize[0] + rand_region_center(0) - 1)*Eigen::MatrixXd::Ones(1, all_uv.cols());//列索引
			all_uv_cal.row(1) = all_uv.row(1) + (- initializing_box_semisize[1] + rand_region_center(1) - 1)*Eigen::MatrixXd::Ones(1, all_uv.cols());//行索引
			all_uv = all_uv_cal;
		}
		int are_there_corners = all_uv.cols();	//特征点数量
		
		//第三步，分析采样窗口中是否有之前检测过的特征，确定当前特征点的去留
		int total_features_number = h_pred.size();
		int features_in_the_box = 0;
		for(int i=0 ; i <  total_features_number ; i++ )
		{//检查并统计重复特征点的数量
			if( h_pred[i](0) >  (rand_region_center(0) - initializing_box_semisize[0] )  && 
				 h_pred[i](0) < (rand_region_center(0) + initializing_box_semisize[0] )  && 
				 h_pred[i](1) > (rand_region_center(1) - initializing_box_semisize[1] )  && 
				 h_pred[i](1) < (rand_region_center(1) + initializing_box_semisize[1]) )
				features_in_the_box++;
		}
		uv.resize(0, 0);
		if(are_there_corners && (!features_in_the_box) )//确保不重复添加特征
			uv = all_uv.col(0);
		else uv.resize(0, 0);
		
		//第四步，若该特征点符合条件，则添加该特征点
		if(uv.cols())
		{//注意，这里每次就添加一个特征点
			Eigen::VectorXd X_RES = mM_ExtendKF->x_k_k;
			Eigen::MatrixXd P_RES = mM_ExtendKF->p_k_k;
			Eigen::VectorXd Xv = mM_ExtendKF->x_k_k.head(13);	//获取相机的的位姿状态向量
			Eigen::VectorXd X_REStemp, newFeature;	
			Eigen::MatrixXd P_REStemp;
				
			mM_ExtendKF->hinv(uv.col(0), Xv, initial_rho, newFeature);//将特征点图像坐标转换为特征点状态向量（6维）
			X_REStemp.resize(X_RES.rows()+newFeature.rows());
			X_REStemp << X_RES, newFeature;
			X_RES = X_REStemp;
			add_a_feature_covariance_inverse_depth(P_RES, uv.col(0), Xv, P_REStemp);//计算特征点的协方差矩阵
			P_RES = P_REStemp;
			mM_ExtendKF->x_k_k = X_RES;//更新状态向量和协方差
			mM_ExtendKF->p_k_k = P_RES;
			
			//将特征点信息添加至特征点容器features_info中
			cv::Mat im = image( cv::Range(uv(1, 0) - 20, uv(1, 0) + 20 + 1), cv::Range(uv(0, 0) - 20, uv(0, 0) + 20 + 1) );
			struct Feature new_feature;
			new_feature.patch_when_initialized = Converter::toMatrixd_atuc(im);
			new_feature.patch_when_matching = Eigen::MatrixXd::Zero(2*6+1, 2*6+1);
			new_feature.r_wc_when_initialized = X_RES.head(3);
			new_feature.R_wc_when_initialized = mM_ExtendKF->q2r(X_RES.segment(3, 4));
			new_feature.uv_when_initialized = uv.col(0).head(2).transpose();
			new_feature.half_patch_size_when_initialized = 20;
			new_feature.half_patch_size_when_matching = 6;
			new_feature.times_predicted = 0;
			new_feature.times_measured = 0;
			new_feature.init_frame = step;
			new_feature.init_measurement = uv.col(0).head(2);
			new_feature.type = "inversedepth";
			new_feature.yi = newFeature;
			new_feature.individually_compatible = false;
			new_feature.low_innovation_inlier = false;
			new_feature.high_innovation_inlier = false;
			new_feature.z.resize(0);
			new_feature.h.resize(0);
			new_feature.H.resize(0, 0);
			new_feature.S.resize(0, 0);
			new_feature.state_size = 6;
			new_feature.measurement_size = 2;
			new_feature.R = Eigen::MatrixXd::Identity(2, 2);
			mM_ExtendKF->features_info.push_back(new_feature);
		}
		
		//将所有特征点的预测信息清空
		feat_len = mM_ExtendKF->features_info.size();
		for(int i=0 ; i < feat_len ; i++)
			mM_ExtendKF->features_info[i].h.resize(0);		
	}
	void Map::fast_corner_detect_9(cv::Mat image,  double  threshold, Eigen::Matrix<double, 2, Eigen::Dynamic>& coords)
	{
		//确保传入的图像是灰度图像
		cv::Mat  im_detect;
		if ( image.channels() != 1 )
			cv::cvtColor (image, im_detect, CV_BGR2GRAY);  
		else 
			im_detect =image.clone();
		
		//快速角点检测  
		std::vector<cv::KeyPoint>  keypoints; 
		cv::FAST(im_detect, keypoints, threshold, true);  //默认开启极大值抑制！
		
		coords.resize(2, keypoints.size() );
		for (int i=0;i< keypoints.size();i++)
		{
			coords(0, i) = keypoints[i].pt.x;//列索引
			coords(1, i) = keypoints[i].pt.y;//行索引
		}
	}	
	void Map::add_a_feature_covariance_inverse_depth(Eigen::MatrixXd P, Eigen::VectorXd uvd, Eigen::VectorXd Xv, Eigen::MatrixXd& P_RES)
	{
		double fku = mM_ExtendKF->cam->K(0, 0);
		double fkv = mM_ExtendKF->cam->K(1, 1);
		double U0 = mM_ExtendKF->cam->K(0, 2);
		double V0 = mM_ExtendKF->cam->K(1, 2);
		
		Eigen::VectorXd q_wc = Xv.segment(3, 4);
		Eigen::Matrix3d R_wc = mM_ExtendKF->q2r(q_wc);
		
		Eigen::MatrixXd uvu;
		mM_ExtendKF->undistort_fm(uvd, uvu);
		
		Eigen::Vector3d XYZ_c, XYZ_w;
		XYZ_c <<  -(U0 - uvu(0, 0))/fku, -(V0 - uvu(1, 0))/fkv, 1;
		XYZ_w = R_wc * XYZ_c;
		
		double X_w = XYZ_w(0);
		double Y_w = XYZ_w(1);
		double Z_w = XYZ_w(2);
		
		//Derivatives
		Eigen::RowVector3d dtheta_dgw, dphi_dgw;
		Eigen::Matrix<double, 3, 4> dgw_dqwr = mM_ExtendKF->dRq_times_a_by_dq(q_wc, XYZ_c);		;
		dtheta_dgw << Z_w/(X_w*X_w+Z_w*Z_w), 0, -X_w/(X_w*X_w+Z_w*Z_w);
		dphi_dgw << (X_w*Y_w)/((X_w*X_w+Y_w*Y_w+Z_w*Z_w)*sqrt(X_w*X_w+Z_w*Z_w)),
					-sqrt(X_w*X_w+Z_w*Z_w)/(X_w*X_w+Y_w*Y_w+Z_w*Z_w), (Z_w*Y_w)/((X_w*X_w+Y_w*Y_w+Z_w*Z_w)*sqrt(X_w*X_w+Z_w*Z_w));	
					
		Eigen::Matrix<double, 6, 4> dy_dqwr;
		Eigen::Matrix<double, 6, 3> dy_drw;	
		Eigen::Matrix<double, 6, 13> dy_dxv;	
		dy_dqwr << Eigen::MatrixXd::Zero(3, 4), dtheta_dgw*dgw_dqwr, dphi_dgw*dgw_dqwr, Eigen::MatrixXd::Zero(1, 4);
		dy_drw << Eigen::MatrixXd::Identity(3, 3), Eigen::MatrixXd::Zero(3, 3);
		dy_dxv << dy_drw, dy_dqwr, Eigen::MatrixXd::Zero(6, 6);

		Eigen::Matrix< double, 5, 3 > dyprima_dgw;	
		Eigen::Matrix< double, 3, 2 > dgc_dhu;
		Eigen::Matrix< double, 5, 2 > dyprima_dhd;
		Eigen::Matrix2d dhu_dhd = mM_ExtendKF->jacob_undistor_fm(uvd);
		dyprima_dgw << Eigen::MatrixXd::Zero(3, 3), dtheta_dgw, dphi_dgw;
		dgc_dhu << 1/fku, 0, 0, 0, 1/fkv, 0;
		dyprima_dhd = dyprima_dgw*R_wc*dgc_dhu*dhu_dhd;
		
		Eigen::Matrix< double, 6, 3 > dy_dhd;
		Eigen::Matrix3d Padd;
		dy_dhd << dyprima_dhd, Eigen::MatrixXd::Zero(5, 1), Eigen::MatrixXd::Zero(1, 2), 1;
		Padd << Eigen::MatrixXd::Identity(2, 2) * pow(mM_ExtendKF->std_z, 2), Eigen::MatrixXd::Zero(2, 1), Eigen::MatrixXd::Zero(1, 2), pow(1, 2);
		
		int P_len = P.rows();
		Eigen::MatrixXd P_xv = P.topLeftCorner(13, 13);
		Eigen::MatrixXd P_yxv = P.bottomLeftCorner(P_len - 13, 13);
		Eigen::MatrixXd P_y = P.bottomRightCorner(P_len - 13, P_len - 13);
		Eigen::MatrixXd P_xvy = P.topRightCorner(13, P_len - 13);
		Eigen::MatrixXd P_RES1 = P_xv * dy_dxv.transpose();
		Eigen::MatrixXd P_RES2 = P_yxv * dy_dxv.transpose();
		Eigen::MatrixXd P_RES3 = dy_dxv * P_xv * dy_dxv.transpose() + dy_dhd * Padd * dy_dhd.transpose();
		int P_RES_rows = P_RES1.rows() + P_RES2.rows() + P_RES3.rows();
		int P_RES_cols = P_xv.cols() + P_xvy.cols() + P_RES1.cols();
		P_RES.resize(P_RES_rows, P_RES_cols);
		P_RES.setZero(P_RES_rows, P_RES_cols);
		P_RES << P_xv , P_xvy , P_RES1 , P_yxv , P_y , P_RES2 , dy_dxv*P_xv , dy_dxv*P_xvy , P_RES3;
	}

	
	
	
	
}
