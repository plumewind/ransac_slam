#include "ransac_slam/System.h"

int main(int argc, char **argv)
{
	// Set up ROS node
	ros::init(argc,  argv, "mono_slam");
	
	ros::NodeHandle handle_pri("~");
	// Sequence path and initial image	
	std::string Images_Path = "/home/lyq/catkin_slam/src/ransac_slam/data/images_sequences/rawoutput";
	std::string strSettingsFile = "/home/lyq/catkin_slam/src/ransac_slam/examples/Monocular/initialize_param.yaml";
	int initIm = 900;
	//int lastIm = 90;
	int lastIm = 2169;
	
	handle_pri.getParam("image_path", Images_Path);
	handle_pri.getParam("config_file", strSettingsFile);
	handle_pri.getParam("image_start", initIm);
	handle_pri.getParam("image_end", lastIm);

	//Check settings file
	cv::FileStorage fSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
	if(fSettings.isOpened())
	{
		std::cout << "Succeed to open settings file at: " << strSettingsFile << std::endl;
// 		Images_Path = (std::string)fSettings["Images.Path"];
// 		initIm = (int)fSettings["Images.start"];
// 		lastIm = (int)fSettings["Images.end"];
	}
	fSettings.release();  
	
	// Create SLAM system. It initializes all system threads and gets ready to process frames.
	ransac_slam::System SLAM(strSettingsFile, true);
	
	 // Vector for tracking time statistics
	int nImages = lastIm - initIm + 1;
	std::vector<float> vTimesTrack;
	vTimesTrack.resize(nImages);

	std::cout << std::endl << "------------" << std::endl;
	std::cout << "Start processing sequence ..." << std::endl;
	std::cout << "Images in the sequence: " << nImages << std::endl << std::endl;

	// Main loop
	cv::Mat imRGB;
	char filename[40];
	for(int i = initIm; i < lastIm; i++)
	{
		if( ! ros::ok())
			break ;
		// Read image from file
		memset(filename, 0, 40);
		std::sprintf(filename, "%s%04d.pgm", Images_Path.c_str(), i);
		imRGB = cv::imread(std::string(filename), CV_LOAD_IMAGE_UNCHANGED);
		//double tframe = vTimestamps[ni];

		if(imRGB.empty())
		{
			ros::shutdown();
			std::cerr << std::endl << "Failed to load image at: "<< filename << std::endl;
			return 1;
		}
		
		//Pass the image to the SLAM system
		SLAM.TrackRunning( imRGB );
		
		//cv::waitKey(2);//延时5毫秒
		
	}
	
	return 0;
}
