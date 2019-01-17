# ransac_slam
An open source SLAM system (EKFmonocularSLAM) that is rewritten in C++ in combination with ROS, Eigen.

![Image text](https://github.com/plumewind/ransac_slam/blob/master/rviz_screenshot.png)

# Learn from

（1）EKFmonocularSLAM on OpenSLAM : https://openslam-org.github.io/ekfmonoslam.html

（2）EKFmonocularSLAM on github ： https://github.com/OpenSLAM-org/openslam_ekfmonoslam

（3）paper
[1]   Javier Civera, Oscar G. Grasa, Andrew J. Davison, J. M. M. Montiel,
      1-Point RANSAC for EKF Filtering: Application to Real-Time Structure from Motion and Visual Odometry,
      to appear in Journal of Field Robotics, October 2010.
      
# An introduction

I wrote a blog on CSDN : https://blog.csdn.net/qq_36355662/article/details/84938284

# Hardware/Software Requirements
1.ubuntu  
14.04 or 16.04  
2.ROS  
indigo or kinetic  
3.Eigen  
sudo apt-get install libeigen3-dev  
#添加头文件  
include_directories("/usr/include/eigen3")  
4.opencv 3.2.0  
  
5.vision_opencv package  
$ git clone https://github.com/ros-perception/vision_opencv.git  


# How to start ?
$ roalaunch ransac_slam ransac_slam.launch
