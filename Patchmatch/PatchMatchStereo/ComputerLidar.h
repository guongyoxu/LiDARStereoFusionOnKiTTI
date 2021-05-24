#pragma once
#ifndef _COMPUTERLIDAR_
#define _COMPUTERLIDAR_

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
 
#include <opencv2\imgproc\types_c.h>
using namespace cv;
using namespace std;
using namespace Eigen;

void computerlidar(Mat* img, Mat* lidarimg,bool* is_lidar, bool* flag);



#endif