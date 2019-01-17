#include "ransac_slam/Converter.h"

namespace ransac_slam
{
	Eigen::Matrix< double, 41, 41 > Converter::toMatrix41d(const cv::Mat& cvMat41)
	{
		Eigen::Matrix<double, 41, 41> M;

		for(int i=0; i < 41; i++)
		{
			for(int j=0 ; j < 41 ; j++)
				M(i, j) = cvMat41.at<float>(i, j);
		}
		return M;
	}
	void  Converter::meshgrid(Eigen::VectorXd& MUSrc, Eigen::VectorXd& MVSrc, Eigen::MatrixXd& MU, Eigen::MatrixXd& MV)
	{
		int MU_len = MUSrc.rows(); 
		int MV_len = MVSrc.rows(); 
		
		MU.resize( MU_len,  MU_len );
		Eigen::RowVectorXd MUSrc_row = MUSrc.transpose();
		for (int i =0;i< MV_len ; i++ )
			MU.row(i) = MUSrc_row;

		MV.resize( MV_len,  MV_len );
		for (int i=0;i< MV_len ; i++)
			MV.col(i) = MVSrc;
	}
	void Converter::meshgrid_opencv(const cv::Range &xgv, const cv::Range &ygv, Eigen::MatrixXd& MU, Eigen::MatrixXd& MV)
	{
		std::vector<int> t_x, t_y;
		cv::Mat X, Y;
		
		for(int i = xgv.start; i <= xgv.end; i++) 
			t_x.push_back(i);
		for(int j = ygv.start; j <= ygv.end; j++) 
			t_y.push_back(j);
	
		//需要对x进行一次转置
		cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
		cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
		//cv::repeat(cv::Mat(t_y).t(), 1, t_x.size(), Y);

		MU = toMatrixd_ati(X);
		MV = toMatrixd_ati(Y);
	}
	void Converter::reshape(Eigen::MatrixXd in, int out_rows, int out_cols, Eigen::MatrixXd& out)
	{
		 if ( (out_rows*out_cols) <= (in.rows() *in.cols()) )
		{
			out.resize(out_rows, out_cols);
			for (int j=0 ; j < out_cols ; j++)
				for (int i=0 ; i< out_rows ; i++)
					out(i, j) = in( (j * out_rows + i) % (in.rows()) , (j * out_rows + i ) / (in.rows()) );
		}
		else std::cout<<"错误：矩阵变维时数据不够 ！"<<std::endl;
	}
	cv::Mat Converter::toCvMat_i(const Eigen::MatrixXd src)
	{
		int src_rows = src.rows();
		int src_cols = src.cols();
		cv::Mat cvMat(src_rows, src_cols,CV_32F);
		
		for(int i=0 ; i < src_rows ; i++)
			for(int j=0 ;  j < src_cols ;  j++)
				cvMat.at<int>(i,j) = src(i,j);

		return cvMat.clone();
	}
	cv::Mat Converter::toCvMat_d(const Eigen::MatrixXd src)
	{
		int src_rows = src.rows();
		int src_cols = src.cols();
		cv::Mat cvMat(src_rows, src_cols,CV_32F);
		
		for(int i=0 ; i < src_rows ; i++)
			for(int j=0 ;  j < src_cols ;  j++)
				cvMat.at<double>(i,j) = src(i,j);

		return cvMat.clone();
	}
	cv::Mat Converter::toCvMat_f(const Eigen::MatrixXd src)
	{
		int src_rows = src.rows();
		int src_cols = src.cols();
		cv::Mat cvMat(src_rows, src_cols,CV_32F);
		
		for(int i=0 ; i < src_rows ; i++)
			for(int j=0 ;  j < src_cols ;  j++)
				cvMat.at<float>(i,j) = src(i,j);

		return cvMat.clone();
	}
	Eigen::MatrixXd Converter::toMatrixd_ati(const cv::Mat src)
	{
		Eigen::MatrixXd out;
		int src_rows = src.rows;
		int src_cols =  src.cols;
		out.resize(src_rows, src_cols);

		for(int i=0; i < src_rows; i++)
		{
			for(int j=0 ; j < src_cols ; j++)
				out(i, j) = src.at<int>(i, j);
		}
		return out;
	}
	Eigen::MatrixXd Converter::toMatrixd_atuc(const cv::Mat src)
	{
		Eigen::MatrixXd out;
		int src_rows = src.rows;
		int src_cols =  src.cols;
		out.resize(src_rows, src_cols);

		for(int i=0; i < src_rows; i++)
		{
			for(int j=0 ; j < src_cols ; j++)
				out(i, j) = src.at<uchar>(i, j);
		}
		return out;
	}
	Eigen::MatrixXd Converter::toMatrixd_atd(const cv::Mat src)
	{
		Eigen::MatrixXd out;
		int src_rows = src.rows;
		int src_cols =  src.cols;
		out.resize(src_rows, src_cols);
		
		for(int i=0; i < src_rows; i++)
		{
			for(int j=0 ; j < src_cols ; j++)
				out(i, j) = src.at<double>(i, j);
		}
		return out;
	}
	Eigen::MatrixXd Converter::toMatrixd_atf(const cv::Mat src)
	{
		Eigen::MatrixXd out;
		int src_rows = src.rows;
		int src_cols =  src.cols;
		out.resize(src_rows, src_cols);
		
		for(int i=0; i < src_rows; i++)
		{
			for(int j=0 ; j < src_cols ; j++)
				out(i, j) = src.at<float>(i, j);
		}
		return out;
	}
	Eigen::MatrixXd Converter::cov(Eigen::MatrixXd d1, Eigen::MatrixXd d2)
	{
		Eigen::MatrixXd  CovM(1,1);
		CovM.setZero(1, 1);
		assert(1 ==d1.cols() && 1 ==d2.cols() &&d1.cols()==d2.cols()  );
		
		//求协方差
		double Ex =0, Ey=0;
		for (int i=0;i< d1.rows();++i){
			Ex +=d1(i);
			Ey +=d2(i);
		}
		Ex /=d1.rows();
		Ey /=d2.rows();
		
		for (int i=0;i< d1.rows();++i){
			CovM(0) += (d1(i)-Ex)*(d2(i)-Ey);
		}
		CovM(0) /= d1.rows() -1;
		return CovM;
	}
	//求矩阵的相关系数！
	//返回矩阵A的列向量的相关系数矩阵
	//对行向量求相关系数 , 与行数无关，返回 cols()*cols() 矩阵...
	Eigen::MatrixXd Converter::corrcoef(const Eigen::MatrixXd &M)
	{
		Eigen::MatrixXd Coef;
		int Order= M.cols();;//int Order= (std::max)(Row,Col);
	
		Coef.resize(Order, Order);
		for (int i=0;i<Order;++i){
			for (int j=0;j<Order;++j){
				Coef(i,j)= cov(M.col(i), M.col(j))(0, 0);
			}
		}
		return Coef;
	}
	Eigen::MatrixXd Converter::corrcoef_opencv(const Eigen::MatrixXd& M)
	{
		Eigen::MatrixXd M_out;
		cv::Mat M_mean, M_cov;
		assert(M.cols() > 0);
		
		//计算相关性系数
		cv::calcCovarMatrix(toCvMat_f(M), M_cov, M_mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);//计算协方差,以！行！向量为一个样本
		M_cov = M_cov/(M_cov.cols - 1);

		Eigen::MatrixXd Mt_cov = toMatrixd_atd(M_cov);
		
		//进行归一化
		int len = Mt_cov.rows();
		M_out.resize(len, len);
		M_out.setZero(len, len);
		for(int i = 0 ; i < len ; i++)
			for(int j = 0 ; j < len ; j++)
				M_out(i, j) = Mt_cov(i, j)/sqrt(Mt_cov(i, i)*Mt_cov(j, j));
			
		return M_out;
	}
	Eigen::MatrixXd Converter::find(Eigen::MatrixXd src, double m)  
	{
		Eigen::MatrixXd out;
		int src_rows = src.rows();
		int src_cols = src.cols();
		
		if ( src_cols == 1)
		{//如果是向量
			std::vector<int> location;
			for (int i=0 ; i < (src_cols*src_rows) ; i++)
			{
				if(src(i) > m)
					location.push_back(i);
			}
			
			out.resize(location.size(), 1);
			for (int i=0 ; i < out.rows() ; i++)
				out(i, 0)=location[i];
			return out.col(0);
		}else{//如果是矩阵
			std::vector<int> x,y;
			std::vector<double> value;
			for (int i=0 ; i < src_rows ; i++)
				for (int j=0 ; j < src_cols ; j++)
					if(src(i, j) > m)
					{
						x.push_back(i);
						y.push_back(j);
						value.push_back(src(i,j));
					}
			out.resize(x.size(), 3);
			for (int i=0 ; i < out.rows() ; i++)
			{ 
				out(i,0)=x[i];
				out(i,1)=y[i];
				out(i,2)=value[i];
			}
			return out;
		}
	}
	Eigen::VectorXd Converter::select(Eigen::VectorXd src, Eigen::VectorXd logical)
	{
		Eigen::VectorXd out;
		int src_len =  src.rows();
		assert(src_len == logical.rows());
		
		std::vector<double> item;
		for(int i = 0 ; i < src_len ; i++)
		{
			if(logical(i) > 0)
				item.push_back(src(i));
		}
		
		int out_len = item.size();
		out.resize(out_len, 1);
		for(int i = 0 ; i < out_len ; i++)
			out(i) = item[i];
			
		
		return out;
	}

	//用小矩阵填充成大矩阵
	Eigen::MatrixXd Converter::repmat(Eigen::MatrixXd M, int rowM, int colM)
	{
		Eigen::MatrixXd MD(rowM*M.rows(), colM*M.cols() );
	
		for (int i=0;i< rowM;++i){
			for (int j=0;j< colM;++j){
				for (int m=0;m< M.rows();++m){
					for (int n=0;n<M.cols();++n){
						MD(i*M.rows()+m, j*M.cols()+n)= M(m,n);
					}
				}
			}
		}
		return MD;
	}
	void Converter::makeRightHanded(Eigen::Matrix2d& eigenvectors, Eigen::Vector2d& eigenvalues)
	{
		Eigen::Vector3d c0;
		c0.setZero();
		c0.head(2) = eigenvectors.col(0);
		c0.normalize();
		Eigen::Vector3d c1;
		c1.setZero();
		c1.head(2) = eigenvectors.col(1);
		c1.normalize();
		Eigen::Vector3d cc = c0.cross(c1);
		if (cc(2) < 0)
		{
			eigenvectors << c1.head(2), c0.head(2);
			double e = eigenvalues(0);
			eigenvalues(0) = eigenvalues(1);
			eigenvalues(1) = e;
		}else
			eigenvectors << c0.head(2), c1.head(2);
	}
	bool Converter::computeEllipseOrientationScale3D(Eigen::Matrix3d& eigenvectors, Eigen::Vector3d& eigenvalues, const Eigen::Matrix3d& covariance)
	{
		eigenvectors.setZero(3, 3);
		eigenvalues.setIdentity(3, 1);
		
		// NOTE: The SelfAdjointEigenSolver only references the lower triangular part of the covariance matrix
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(covariance);
		// Compute eigenvectors and eigenvalues
		if (eigensolver.info() == Eigen::Success)
		{
			eigenvalues = eigensolver.eigenvalues();
			eigenvectors = eigensolver.eigenvectors();
		}else{
			eigenvalues = Eigen::Vector3d::Zero();  // Setting the scale to zero will hide it on the screen
			eigenvectors = Eigen::Matrix3d::Identity();
			return false;
		}
		
		// Be sure we have a right-handed orientation system
		//makeRightHanded(eigenvectors, eigenvalues);
		return true;
	}






	
}