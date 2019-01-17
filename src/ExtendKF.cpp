#include "ransac_slam/ExtendKF.h"

namespace ransac_slam
{
	ExtendKF::ExtendKF(const std::string &strSettingsFile, CamParam* param, std::string type)
		:cam(param), filter_type(type)
	{
		//Check settings file
		fsSettings.open(strSettingsFile.c_str(), cv::FileStorage::READ);
		if(!fsSettings.isOpened())
		{
			std::cerr << "Failed to open settings file at: " << strSettingsFile << std::endl;
			exit(-1);
		}
		
		std_a = (double)fsSettings["Sigma.a"];
		std_alpha = (double)fsSettings["Sigma.alpha"];
		std_z = (double)fsSettings["Sigma.noise"];
		
		v_0 = (double)fsSettings["Velocity.v0"];
		std_v_0 = (double)fsSettings["Velocity.stdv0"];
		w_0 = (double)fsSettings["Velocity.w0"];
		std_w_0 = (double)fsSettings["Velocity.stdw0"];
		eps = std::numeric_limits<double>::epsilon();
		srand(time(NULL));//随机数种子,头文件
		features_info.clear();
	}
	ExtendKF::~ExtendKF()
	{
		fsSettings.release();  
	}
	void ExtendKF::initialize_x_and_p(void)
	{
		x_k_k.resize(13, 1);
		x_k_k.setZero(13, 1);
		x_k_k(3) = 1;
		//FIXME 这里的tail操作不知道是否正确
		x_k_k.tail(6) << v_0, v_0, v_0, w_0, w_0, w_0;
		
		p_k_k.resize(13, 13);
		p_k_k.setZero(13, 13);
		p_k_k(0, 0) = eps;
		p_k_k(1, 1) = eps;
		p_k_k(2, 2) = eps;
		p_k_k(3, 3) = eps;
		p_k_k(4, 4) = eps;
		p_k_k(6, 6) = eps;
		p_k_k(7, 7) = std_v_0*std_v_0;
		p_k_k(8, 8) = std_v_0*std_v_0;
		p_k_k(9, 9) = std_v_0*std_v_0;
		p_k_k(10, 10) = std_w_0*std_w_0;
		p_k_k(11, 11) = std_w_0*std_w_0;
		p_k_k(12, 12) = std_w_0*std_w_0;

	}
	void ExtendKF::predict_camera_measurements(Eigen::VectorXd xkk)
	{
		//单孔相机成像模型
		Eigen::Vector3d t_wc = xkk.head(3);
		Eigen::Matrix3d r_wc = q2r( xkk.segment(3, 4) );
		Eigen::VectorXd features = xkk.tail(xkk.rows() - 13);
		
		Eigen::VectorXd yi;
		Eigen::Vector3d hrl;
		Eigen::MatrixXd hi;
		Eigen::Vector3d mi;
		int index = 0;
		int feat_len = features_info.size();
		for(int i=0 ; i < feat_len ; i++)
		{
			if( strcmp(features_info[i].type.c_str(), "inversedepth") == 0)
			{//大部分情况应该进入这里
				yi = features.segment(index, 6);
				mi << cos(yi[4]) * sin(yi[3]), -sin(yi[4]), cos(yi[4]) * cos(yi[3]);
				hrl = ( r_wc.transpose() ) * ( (yi.head(3) - t_wc) * yi[5] + mi);
				hi_cartesian(hrl, hi);
				if(hi.rows() != 0)
					features_info[i].h = hi.transpose();
				index = index + 6;
			}else if( strcmp(features_info[i].type.c_str(), "cartesian") == 0)
			{
				yi = features.segment(index, 3);
				hrl = r_wc.inverse() * (yi - t_wc);
				hi_cartesian(hrl, hi);
				if(hi.rows() != 0)
					features_info[i].h = hi.transpose();
				index = index + 3;
			}
		}
	}
	Eigen::Matrix3d ExtendKF::q2r(Eigen::VectorXd q_in)
	{
		Eigen::Matrix3d R_out;
		
		double x=q_in(1);
		double y=q_in(2);
		double z=q_in(3);
		double r=q_in(0);
		R_out << r*r+x*x-y*y-z*z , 2*(x*y -r*z) , 2*(z*x+r*y) , 2*(x*y+r*z) , r*r-x*x+y*y-z*z , 2*(y*z-r*x) , 2*(z*x-r*y) , 2*(y*z+r*x) , r*r-x*x-y*y+z*z;
		
		return R_out;
	}
	void ExtendKF::hi_cartesian(Eigen::Vector3d hrl, Eigen::MatrixXd& zi)
	{
		//检查特征点是否在相机前方
		if( (atan2(hrl(0), hrl(2)) * 180 / M_PI < -60) ||
		    (atan2(hrl(0), hrl(2)) * 180 / M_PI > 60)  ||
		    (atan2(hrl(1), hrl(2)) * 180 / M_PI < -60) ||
		    (atan2(hrl(1), hrl(2)) * 180 / M_PI > 60) )
		{
			zi.resize(0, 0);
			return ;
		}
		
		//获得特征点的图像坐标
		Eigen::Vector2d uv_u = hu(hrl);		
		
		//hu(hrl, uv_u);
		//Add distortion
		Eigen::MatrixXd uv_d;
		distort_fm(uv_u, uv_d);

		//Is visible in the image?
		if( ( uv_d(0, 0) > 0 ) && ( uv_d(0, 0) < cam->nCols ) && ( uv_d(1, 0) > 0 ) && ( uv_d(1, 0) < cam->nRows ) )
		{
			zi = uv_d;
			//return ;
		}else{
			zi.resize(0, 0);
			//return ;
		}
	}
	void ExtendKF::hi_inverse_depth(Eigen::Vector3d hrl, Eigen::MatrixXd& zi)
	{//与hi_cartesian函数类似，已经用hi_cartesian函数替代
		
	}
	Eigen::Vector3d ExtendKF::inversedepth2cartesian(Eigen::VectorXd inverse_depth)
	{
		Eigen::Vector3d cartesian, m;
		Eigen::Vector3d rw = inverse_depth.head(3);
		double theta = inverse_depth(3);
		double phi = inverse_depth(4);
		double rho = inverse_depth(5);
		
		m <<cos(phi) * sin(theta) , -sin(phi), cos(phi) * cos(theta);
		
		cartesian(0) = rw(0) + (1.0/rho) * m(0);
		cartesian(1) = rw(1) + (1.0/rho) * m(1);
		cartesian(2) = rw(2) + (1.0/rho) * m(2);
		
		return cartesian;
	}
	Eigen::Vector2d ExtendKF::hu(Eigen::Vector3d yi)
	{
		Eigen::Vector2d uv_u;
		double u0 = cam->Cx;
		double v0 = cam->Cy;
		double f = cam->f;
		double ku = 1.0/cam->dx;
		double kv = 1.0/cam->dy;

// 		uv_u = zeros( 2, size( yi, 2 ) );
// 		
// 		for i = 1:size( yi, 2 )
// 		    uv_u( 1, i ) = u0 + (yi(1,i)/yi(3,i))*f*ku;
// 		    uv_u( 2, i ) = v0 + (yi(2,i)/yi(3,i))*f*kv;
// 		end
		
		//将世界坐标转换为像素坐标
		uv_u(0) = u0 + (yi(0) /(double) yi(2)) * f * ku;
		uv_u(1) = v0 + (yi(1) /(double)yi(2)) * f * kv;

		return uv_u;
	}
	void ExtendKF::distort_fm(Eigen::MatrixXd uv, Eigen::MatrixXd& uvd)
	{
		//消除图像失真
		double Cx = cam->Cx;
		double Cy = cam->Cy;
		double k1 = cam->k1;
		double k2 = cam->k2;
		double dx = cam->dx;
		double dy = cam->dy;
		
		Eigen::MatrixXd xu = ( uv.row(0).array() - Cx ) * dx;
		Eigen::MatrixXd yu = ( uv.row(1).array() - Cy ) * dy;

		Eigen::MatrixXd ru = (xu.array().pow(2) + yu.array().pow(2)).array().sqrt();
		Eigen::MatrixXd rd = ru.array() / (1 + k1 * ru.array().pow(2) + k2 * ru.array().pow(4));
		
		Eigen::MatrixXd f, f_p;
		for(int k=0 ; k<10 ;k++)
		{
			f = rd.array() + k1 * rd.array().pow(3) + k2 * rd.array().pow(5) - ru.array();
			f_p = 1 + 3 * k1 * rd.array().pow(2) + 5 * k2 * rd.array().pow(4);
			rd = rd.array() - f.array()/f_p.array();
		}
		
		Eigen::MatrixXd D = 1 + k1 * rd.array().pow(2) + k2 * rd.array().pow(4);

		uvd.resize(2, xu.cols());
		uvd.row(0) = xu.array()/D.array()/dx + Cx;
		uvd.row(1) = yu.array()/D.array()/dy + Cy;	
	}
	double ExtendKF::RandomGenerator(const int low, const int high)
	{//产生由在(0, 1)之间均匀分布的随机数组成的数组。 
		//创建引擎
// 		static std::random_device rd;
// 		static std::mt19937 engine2(rd());
// 		std::default_random_engine engine2(std::time(0));
		//创建随机数分布标准，第一个是均值，第二个是方差
// 		static std::normal_distribution<double> normal(low, high);
		
		//获取伪随机数
// 		return  (rand()%100)*0.01;
		//srand((unsigned)time(NULL));
		//return (double)std::rand()/RAND_MAX*(high - low) + low;
		return (std::rand())/(RAND_MAX+1.0)*(high-low)+low;
	}
	Eigen::MatrixXd ExtendKF::rand(int row, int column, double min, double max)
	{
		Eigen::MatrixXd p;
		double temp;
		
		//srand((unsigned)time(NULL));//不能用时间作为种子，因为程序快容易使得随机数几乎一样
		p.resize(row, column);
		for (int i = 0 ; i < row ; i++)
			for (int j = 0 ; j < column ; j++)
			{ 
				temp = (double)std::rand()/RAND_MAX*(max-min)+min;
				p(i, j) = temp ;
			}   
		
		return p;
	}
	void ExtendKF::hinv(Eigen::VectorXd uvd, Eigen::VectorXd Xv, double initial_rho, Eigen::VectorXd& newFeature)
	{
		double fku = cam->K(0, 0);
		double fkv = cam->K(1, 1);
		double U0 = cam->K(0, 2);
		double V0 = cam->K(1, 2);

		Eigen::MatrixXd uv;
		undistort_fm(uvd, uv);
		double u = uv(0, 0);
		double v = uv(1, 0);
		
		Eigen::VectorXd r_W = Xv.head(3);
		Eigen::VectorXd q_WR = Xv.segment(3, 4);
		
		double h_LR_x = - (U0-u)/fku;
		double h_LR_y = - (V0-v)/fkv;
		double h_LR_z = 1;
		
		Eigen::Vector3d h_LR; 
		h_LR << h_LR_x, h_LR_y, h_LR_z;
		
		Eigen::Vector3d  n;
		Eigen::Matrix3d n_r = q2r(q_WR);
		n = n_r * h_LR;
		
		double newFeature_2 = atan2(-n(1), sqrt(n(0)*n(0) + n(2)*n(2)));
		newFeature.resize(6);
		newFeature << r_W, atan2(n(0), n(2)), newFeature_2, initial_rho;
	}
	void ExtendKF::undistort_fm(Eigen::MatrixXd uvd, Eigen::MatrixXd& uvu)
	{
		double Cx = cam->Cx;
		double Cy = cam->Cy;
		double k1 = cam->k1;
		double k2 = cam->k2;
		double dx = cam->dx;
		double dy = cam->dy;
		
		Eigen::MatrixXd xd = ( uvd.row(0).array() - Cx ) * dx;
		Eigen::MatrixXd yd = ( uvd.row(1).array() - Cy ) * dy;
			
		Eigen::MatrixXd rd = (xd.array().pow(2) + yd.array().pow(2)).array().sqrt();

		Eigen::MatrixXd D = 1 + k1 * rd.array().pow(2)  + k2 * rd.array().pow(4) ;
		
		uvu.resize(2, D.cols());
		uvu.row(0) = xd.array() * D.array() / dx + Cx;
		uvu.row(1) = yd.array() * D.array() / dy + Cy;
	}
	Eigen::Matrix<double, 3, 4> ExtendKF::dRq_times_a_by_dq(Eigen::VectorXd q, Eigen::Vector3d aMat)
	{
		Eigen::MatrixXd dRq_times_a_by_dqRES;
		dRq_times_a_by_dqRES.setZero(3, 4);
		
		Eigen::Matrix3d TempR;
		Eigen::Vector3d Temp31;
		
		TempR << 2*q(0), -2*q(3), 2*q(2), 2*q(3), 2*q(0), -2*q(1), -2*q(2), 2*q(1), 2*q(0);
		Temp31 = TempR * aMat;
		dRq_times_a_by_dqRES.col(0) << Temp31;
		
		TempR << 2*q(1), 2*q(2), 2*q(3), 2*q(2), -2*q(1), -2*q(0), 2*q(3), 2*q(0), -2*q(1);
		Temp31 = TempR * aMat;
		dRq_times_a_by_dqRES.col(1) << Temp31;
		
		TempR << -2*q(2), 2*q(1), 2*q(0), 2*q(1), 2*q(2), 2*q(3), -2*q(0), 2*q(3), -2*q(2);
		Temp31 = TempR * aMat;
		dRq_times_a_by_dqRES.col(2) << Temp31;
		
		TempR << -2*q(3), -2*q(0), 2*q(1), 2*q(0), -2*q(3), 2*q(2), 2*q(1), 2*q(2), 2*q(3);
		Temp31 = TempR * aMat;
		dRq_times_a_by_dqRES.col(3) << Temp31;
		
		return dRq_times_a_by_dqRES;
	}
	Eigen::Matrix2d ExtendKF::jacob_undistor_fm(Eigen::VectorXd uvd)
	{
		Eigen::Matrix2d J_undistor;
		
		double Cx = cam->Cx;
		double Cy = cam->Cy;
		double k1 = cam->k1;
		double k2 = cam->k2;
		double dx = cam->dx;
		double dy = cam->dy;
		
		double rd2 = pow((uvd(0)-Cx)*dx, 2) + pow((uvd(1)-Cy)*dy, 2);
		
		double uu_ud=(1+k1*rd2+k2*rd2*rd2)+(uvd(0)-Cx)*(k1+2*k2*rd2)*(2*(uvd(0)-Cx)*dx*dx);
		double vu_vd=(1+k1*rd2+k2*rd2*rd2)+(uvd(1)-Cy)*(k1+2*k2*rd2)*(2*(uvd(1)-Cy)*dy*dy);
		double uu_vd=(uvd(0)-Cx)*(k1+2*k2*rd2)*(2*(uvd(1)-Cy)*dy*dy);
		double vu_ud=(uvd(1)-Cy)*(k1+2*k2*rd2)*(2*(uvd(0)-Cx)*dx*dx);
		
		J_undistor << uu_ud, uu_vd, vu_ud, vu_vd;
		return J_undistor;
	}
	void ExtendKF::ekf_prediction(void)
	{
		//第一步，对状态向量进行预测
		double delta_t = 1;
		Eigen::VectorXd Xv_km1_k;
		fv(delta_t, Xv_km1_k);							//预测下一时刻的相机位姿
		x_k_km1.resize( Xv_km1_k.rows() + x_k_k.rows() - 13 );
		x_k_km1 << Xv_km1_k, x_k_k.tail(x_k_k.rows() - 13);	//更新特征点
		
		//第二步，更新协方差矩阵
		//state transition equation derivatives
		Eigen::MatrixXd F;
		dfv_by_dxv(delta_t, F);
		
		double linear_acceleration_noise_covariance = pow(std_a *delta_t, 2);
		double angular_acceleration_noise_covariance = pow(std_alpha * delta_t, 2);//状态噪声
		Eigen::VectorXd Pn_diag;
		Pn_diag.resize(6);
		Pn_diag << linear_acceleration_noise_covariance, linear_acceleration_noise_covariance, linear_acceleration_noise_covariance,
				angular_acceleration_noise_covariance, angular_acceleration_noise_covariance, angular_acceleration_noise_covariance;
		Eigen::MatrixXd Pn = Pn_diag.asDiagonal();
		
		//Q = func_Q( X_k(1:13,:), zeros(6,1), Pn, delta_t, type);
		Eigen::Vector3d omegaOld = x_k_k.segment(10, 3);
		Eigen::Vector4d qOld = x_k_k.segment(3, 4);
		Eigen::MatrixXd G = Eigen::MatrixXd::Zero(13, 6);
		if( strcmp(filter_type.c_str(), "constant_position_and_orientation_location_noise") ==0 )
		{
			G.topLeftCorner(3, 3) = Eigen::MatrixXd::Identity(3, 3) * delta_t;
			Eigen::Vector3d euler_angles = q2tr_tr2rpy( qOld );
			double phi = euler_angles(0);
			double theta = euler_angles(1);
			double psi = euler_angles(2);
			G.block(3, 3, 4, 3) << (0.5)*(-sin(phi/2)+cos(phi/2))  ,   (0.5)*(-sin(theta/2)+cos(theta/2))   ,   (0.5)*(-sin(psi/2)+cos(psi/2)) ,
						(0.5)*(+cos(phi/2)+sin(phi/2))  ,   (0.5)*(-sin(theta/2)-cos(theta/2))  ,    (0.5)*(-sin(psi/2)-cos(psi/2)) ,
						(0.5)*(-sin(phi/2)+cos(phi/2))   ,  (0.5)*(+cos(theta/2)-sin(theta/2))   ,   (0.5)*(-sin(psi/2)+cos(psi/2)) ,
						(0.5)*(-sin(phi/2)-cos(phi/2))   ,  (0.5)*(-sin(theta/2)-cos(theta/2))    ,  (0.5)*(+cos(psi/2)+sin(psi/2)) ;
		}else{//通常进入这里
			G.block(7, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3) ;
			G.block(10, 3, 3, 3)= Eigen::MatrixXd::Identity(3, 3) ;
			G.topLeftCorner(3, 3)=Eigen::MatrixXd::Identity(3, 3) * delta_t;
			G.block(3, 3, 4, 3)=dq3_by_dq1(qOld)*dqomegadt_by_domega(omegaOld, delta_t);
		}
		Eigen::MatrixXd Q=G*Pn*G.transpose();
		
		int size_P_k = p_k_k.rows();
		Eigen::MatrixXd pk_km2 = F*p_k_k.topLeftCorner(13, 13)*F.transpose() + Q;
		Eigen::MatrixXd pk_km3 = F*p_k_k.topRightCorner(13, size_P_k - 13);
		Eigen::MatrixXd pk_km4 = p_k_k.bottomLeftCorner(size_P_k - 13, 13)*F.transpose() ;
		Eigen::MatrixXd pk_km5 = p_k_k.bottomRightCorner(size_P_k - 13, size_P_k - 13);
		
		int pk_km1rows = pk_km2.rows()  + pk_km4.rows();
		int pk_km1cols = pk_km2.cols() + pk_km3.cols() ;
		p_k_km1.resize(pk_km1rows, pk_km1cols);
		p_k_km1 <<  pk_km2 ,  pk_km3 , pk_km4 , pk_km5  ;
	}
	void ExtendKF::fv(double delta_t, Eigen::VectorXd& X_k_km1)
	{
		Eigen::Vector3d rW = x_k_k.head(3);
		Eigen::Vector4d qWR = x_k_k.segment(3, 4);
		Eigen::Vector3d vW = x_k_k.segment(7, 3);
		Eigen::Vector3d wW = x_k_k.segment(10, 3);

		X_k_km1.resize(13);
		X_k_km1.setZero(13);
		if( strcmp(filter_type.c_str(), "constant_velocity") == 0 )
		{
			X_k_km1 << rW+vW*delta_t, qprod(qWR, wW, delta_t), vW, wW;
		}else if(strcmp(filter_type.c_str(), "constant_position") == 0 )
		{
			vW << 0, 0, 0;
			X_k_km1 << rW, qprod(qWR, wW, delta_t),  vW, wW;
		}else if( strcmp(filter_type.c_str(), "constant_orientation") ==0 )
		{
			wW << 0, 0, 0;
			X_k_km1 << rW+vW*delta_t, qWR,  vW, wW;
		}else if( strcmp(filter_type.c_str(), "constant_position_and_orientation") == 0 ||  strcmp(filter_type.c_str(), "constant_position_and_orientation_location_noise") == 0 )
		{
			vW << 0, 0, 0;
			wW << 0, 0, 0;
			X_k_km1 << rW, qWR,  vW, wW;
		}
	}
	Eigen::Vector4d ExtendKF::qprod(Eigen::Vector4d q, Eigen::Vector3d wW, double delta_t)
	{
		Eigen::RowVector4d qp;
		Eigen::Vector3d v = wW * delta_t;
		Eigen::RowVector4d p = v2q(v);
		Eigen::Vector3d q_v = q.segment(1, 3);
		Eigen::Vector3d p_u = p.segment(1, 3).transpose();
		Eigen::Vector3d vCu = q_v.cross(p_u);
		qp << q(0)*p(0)-q_v.transpose()*p_u, (q(0)*p_u+p(0)*q_v).transpose()+vCu.transpose();

		return qp.transpose();
	}
	Eigen::RowVector4d ExtendKF::v2q(Eigen::Vector3d v)
	{// % V2Q(R) converts rotation vector to quaternion.
	// %
	// %     The resultant quaternion(s) 
	// %          v_n=v/norm(v);
	// %          theta=norm(v);
		Eigen::RowVector4d q;
		double theta = v.norm();
		if(theta < eps)
			q << 0, 0, 0, 0;
		else{
			Eigen::Vector3d v_n = v / theta;
			q << cos(theta/2.0), sin(theta/2.0) *(v_n.transpose()/v_n.norm());
		}
		return q;
	}
	void ExtendKF::dfv_by_dxv(double delta_t, Eigen::MatrixXd& dfv_by_dxvRES)
	{
		Eigen::Vector3d omegaOld = x_k_k.segment(10, 3);
		Eigen::Vector4d qOld = x_k_k.segment(3, 4);
		
		dfv_by_dxvRES.resize(13, 13);
		dfv_by_dxvRES.setIdentity(13, 13);
		
		Eigen::RowVector4d qwt = v2q(omegaOld * delta_t);
		Eigen::Matrix<double, 4, 4> qwt_dq;
		qwt_dq << qwt(0), -qwt(1), -qwt(2), -qwt(3),
				qwt(1), qwt(0), qwt(3), -qwt(2),
				qwt(2), -qwt(3), qwt(0), qwt(1),
				qwt(3), qwt(2), -qwt(1), qwt(0);
		dfv_by_dxvRES.block(3, 3, 4, 4) = qwt_dq;
		
		if(strcmp(filter_type.c_str(), "constant_velocity") == 0 )
		{
			dfv_by_dxvRES.block(0, 7, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * delta_t;
			Eigen::MatrixXd a = dq3_by_dq1(qOld);
			Eigen::Matrix<double, 4, 3> b =  dqomegadt_by_domega(omegaOld, delta_t);
			dfv_by_dxvRES.block(3, 10, 4, 3) =  a * b;
		}else if(strcmp(filter_type.c_str(), "constant_orientation") == 0 )
		{
			dfv_by_dxvRES.block(3, 10, 4, 3) = Eigen::MatrixXd::Zero(4, 3);
			dfv_by_dxvRES.block(10, 10, 3, 3) = Eigen::MatrixXd::Zero(3, 3);
		}else if(strcmp(filter_type.c_str(), "constant_position") == 0 )
		{
			dfv_by_dxvRES.block(0, 7, 3, 3) = Eigen::MatrixXd::Zero(3, 3);
			dfv_by_dxvRES.block(7, 7, 3, 3) = Eigen::MatrixXd::Zero(3, 3);
		}else if(strcmp(filter_type.c_str(), "constant_position_and_orientation") == 0 )
		{
			dfv_by_dxvRES.block(3, 10, 4, 3) = Eigen::MatrixXd::Zero(4, 3);
			dfv_by_dxvRES.block(0, 7, 3, 3) = Eigen::MatrixXd::Zero(3, 3);
			dfv_by_dxvRES.block(10, 10, 3, 3) = Eigen::MatrixXd::Zero(3, 3);
			dfv_by_dxvRES.block(7, 7, 3, 3) = Eigen::MatrixXd::Zero(3, 3);
		}
	}
	Eigen::Matrix4d ExtendKF::dq3_by_dq1(Eigen::Vector4d q2_in)
	{
		Eigen::Matrix4d q2_out;
		q2_out << q2_in(0), -q2_in(1), -q2_in(2), -q2_in(3),
				q2_in(1), q2_in(0), -q2_in(3), q2_in(2),
				q2_in(2), q2_in(3), q2_in(0), -q2_in(1),
				q2_in(3), -q2_in(2), q2_in(1), q2_in(0);
		return q2_out;
	}
	Eigen::Matrix< double, 4, 3 > ExtendKF::dqomegadt_by_domega(Eigen::Vector3d omega, double delta_t)
	{
		// Modulus
		double omegamod = omega.norm();
		Eigen::Matrix< double, 4, 3 > dqomegadt_by_domegaRES;
		
		// Use generic ancillary functions to calculate components of Jacobian
		dqomegadt_by_domegaRES(0, 0) = dq0_by_domegaA(omega(0), omegamod, delta_t);
		dqomegadt_by_domegaRES(0, 1) = dq0_by_domegaA(omega(1), omegamod, delta_t);
		dqomegadt_by_domegaRES(0, 2) = dq0_by_domegaA(omega(2), omegamod, delta_t);
		dqomegadt_by_domegaRES(1, 0) = dqA_by_domegaA(omega(0), omegamod, delta_t);
		dqomegadt_by_domegaRES(1, 1) = dqA_by_domegaB(omega(0), omega(1), omegamod, delta_t);
		dqomegadt_by_domegaRES(1, 2) = dqA_by_domegaB(omega(0), omega(2), omegamod, delta_t);
		dqomegadt_by_domegaRES(2, 0) = dqA_by_domegaB(omega(1), omega(0), omegamod, delta_t);
		dqomegadt_by_domegaRES(2, 1) = dqA_by_domegaA(omega(1), omegamod, delta_t);
		dqomegadt_by_domegaRES(2, 2) = dqA_by_domegaB(omega(1), omega(2), omegamod, delta_t);
		dqomegadt_by_domegaRES(3, 0) = dqA_by_domegaB(omega(2), omega(0), omegamod, delta_t);
		dqomegadt_by_domegaRES(3, 1) = dqA_by_domegaB(omega(2), omega(1), omegamod, delta_t);
		dqomegadt_by_domegaRES(3, 2) = dqA_by_domegaA(omega(2), omegamod, delta_t);
		
		return dqomegadt_by_domegaRES;
	}
	double ExtendKF::dq0_by_domegaA(double omegaA, double omega, double delta_t)
	{
		return (-delta_t / 2.0) * (omegaA / omega) * sin(omega * delta_t / 2.0);
	}
	double ExtendKF::dqA_by_domegaA(double omegaA, double omega, double delta_t)
	{
		return (delta_t / 2.0) * omegaA * omegaA / (omega * omega) 
			* cos(omega * delta_t / 2.0) 
			+ (1.0 / omega) * (1.0 - omegaA * omegaA / (omega * omega))
			* sin(omega * delta_t / 2.0);
	}
	double ExtendKF::dqA_by_domegaB(double omegaA, double omegaB, double omega, double delta_t)
	{
		return (omegaA * omegaB / (omega * omega)) * 
			( (delta_t / 2.0) * cos(omega * delta_t / 2.0) 
			- (1.0 / omega) * sin(omega * delta_t / 2.0) );
	}
	Eigen::Vector3d ExtendKF::q2tr_tr2rpy(Eigen::Vector4d q)
	{
		Eigen::Matrix3d r;
		Eigen::Matrix3d t = Eigen::MatrixXd::Identity(4, 4) ;
		double s = q(0);
		double x = q(1);
		double y = q(2);
		double z = q(3);
		
		r << 1-2*(y*y+z*z) , 2*(x*y-s*z) , 2*(x*z+s*y) ,
			2*(x*y+s*z) , 1-2*(x*x+z*z) , 2*(y*z-s*x) ,
			2*(x*z-s*y) , 2*(y*z+s*x) , 1-2*(x*x+y*y) ;
		t.topLeftCorner(3, 3) = r;
		t(3, 3) = 1;
		
		Eigen::Vector3d rpy = Eigen::MatrixXd::Zero(3, 1) ;
		if( abs(t(0, 0)) < eps && abs(t(1, 0)) < eps )
		{
			rpy(0) = 0;
			rpy(1) = atan2(-t(2, 0), t(0, 0));
			rpy(2) = atan2(-t(1, 2), t(1, 1));
		}else{
			rpy(0) = atan2(t(1, 0), t(0, 0));
			rpy(1) = atan2(-t(2, 0), cos(rpy(0)) * t(0, 0) + sin(rpy(0)) * t(1,  0));
			rpy(2) = atan2(sin(rpy(0)) * t(0, 2) - cos(rpy(0)) * t(1, 2), cos(rpy(0)) * t(1, 1) - sin(rpy(0)) * t(0, 1));
		}
		return rpy;
	}
	
	void ExtendKF::ekf_update_li_inliers(void)
	{//mount vectors and matrices for the update
		Eigen::VectorXd z, h, zh_temp;;
		Eigen::MatrixXd H, H_temp;
		int feat_len = features_info.size();
		
		//第一步，查找低创新局内点
		z.resize(0);
		h.resize(0);
		H.resize(0, 0);
		for(int i = 0 ; i < feat_len ; i++)
		{
			if( features_info[i].low_innovation_inlier )
			{
				zh_temp.resize(z.rows() + 2);
				if(z.rows())
					zh_temp << z, features_info[i].z.head(2);
				else zh_temp <<  features_info[i].z.head(2);
				z = zh_temp;

				zh_temp.resize(h.rows() + 2);
				if(h.rows())
					zh_temp << h, features_info[i].h(0), features_info[i].h(1);
				else zh_temp << features_info[i].h(0), features_info[i].h(1);
				h =zh_temp;
				
				H_temp.resize(H.rows() + features_info[i].H.rows(), features_info[i].H.cols());
				if(H.rows())
					H_temp << H, features_info[i].H;
				else H_temp << features_info[i].H;
				H = H_temp;
			}
		}
		
		//第二步，更新局内点
		Eigen::MatrixXd R = Eigen::MatrixXd::Identity(z.rows(), z.rows());
		update( x_k_km1, p_k_km1, H, R, z, h );	
	}
	void ExtendKF::update(Eigen::VectorXd x_km_k, Eigen::MatrixXd p_km_k, Eigen::MatrixXd H, Eigen::MatrixXd R, Eigen::VectorXd z, Eigen::VectorXd h)
	{
		if(z.rows())
		{
			//filter gain
			Eigen::MatrixXd S = H * p_km_k * H.transpose() + R;
			Eigen::MatrixXd K = p_km_k * H.transpose() * S.inverse();
			
			//updated state and covariance
			Eigen::VectorXd xkk = x_km_k + K * ( z - h );
			Eigen::MatrixXd pkk, pkk_temp;
			pkk_temp = p_km_k - K * S * K.transpose();
			pkk = 0.5 * pkk_temp + 0.5 * pkk_temp.transpose();
			
			//normalize the quaternion
			//Jnorm = normJac( x_k_k( 4:7 ) );
			double r = xkk(3);
			double x = xkk(4);
			double y = xkk(5);
			double z = xkk(6);
			
			Eigen::Vector4d xkk_temp = (xkk.segment(3, 4)).array() / xkk.segment(3, 4).norm() ;
			xkk.segment(3, 4) = xkk_temp;
			x_k_k = xkk;
			
			Eigen::Matrix4d Jnorm, temp;
			temp << x*x+y*y+z*z    ,     -r*x     ,    -r*y     ,    -r*z,
					-x*r ,  r*r+y*y+z*z   ,      -x*y    ,     -x*z,
					-y*r    ,     -y*x , r*r+x*x+z*z    ,     -y*z,
					-z*r   ,      -z*x    ,     -z*y ,  r*r+x*x+y*y;
			Jnorm = pow( (r*r+x*x+y*y+z*z), (-3/2) ) * temp;
			
			int size_p_k_k = pkk.rows();
			pkk_temp.resize(pkk.rows(), pkk.cols());
			pkk_temp << pkk.topLeftCorner(3, 3), pkk.block(0, 3, 3, 4) * Jnorm.transpose(), pkk.topRightCorner(3, size_p_k_k - 7),
						Jnorm * pkk.block(3, 0, 4, 3), Jnorm * pkk.block(3, 3, 4, 4) * Jnorm.transpose(), Jnorm * pkk.block(3, 7, 4, size_p_k_k-7),
						pkk.bottomLeftCorner(size_p_k_k - 7, 3), pkk.block(7, 3, size_p_k_k-7, 4)* Jnorm.transpose(), pkk.bottomRightCorner(size_p_k_k - 7, size_p_k_k - 7);
			p_k_k = pkk_temp;
		}else{
			x_k_k = x_km_k;
			p_k_k = p_km_k;
		}
	}
	void ExtendKF::ekf_update_hi_inliers(void)
	{
		//mount vectors and matrices for the update
		Eigen::VectorXd z, h, zh_temp;;
		Eigen::MatrixXd H, H_temp;
		int feat_len = features_info.size();
		
		//第一步，查找高创新局内点
		z.resize(0);
		h.resize(0);
		H.resize(0, 0);
		for(int i = 0 ; i < feat_len ; i++)
		{
			if( features_info[i].high_innovation_inlier )
			{
				zh_temp.resize(z.rows() + 2);
				if(z.rows())
					zh_temp << z, features_info[i].z.head(2);
				else zh_temp << features_info[i].z.head(2);
				z = zh_temp;
				
				zh_temp.resize(h.rows() + 2);
				if(h.rows())
					zh_temp << h, features_info[i].h(0), features_info[i].h(1);
				else zh_temp << features_info[i].h(0), features_info[i].h(1);
				h = zh_temp;
				
				H_temp.resize(H.rows() + features_info[i].H.rows(), features_info[i].H.cols());
				if(H.rows())
					H_temp << H, features_info[i].H;
				else H_temp << features_info[i].H;
				H = H_temp;
			}
		}
		
		//第二步，更新局内点
		Eigen::MatrixXd R = Eigen::MatrixXd::Identity(z.rows(), z.rows());
		update( x_k_k, p_k_k, H, R, z, h );	
	}

	
	
	
	
	
	
	
}
