#include "stdafx.h"
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include "PatchMatchStereo.h"
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>  
#include <pcl/point_types.h> 
#include <pcl/io/ply_io.h>
#include "ComputerLidar.h"
#include <cstdio>
#include <cstdint>
using namespace std::chrono;
using namespace Eigen;
using namespace std;
// opencv library

//#ifdef _DEBUG
//#pragma comment(lib,"opencv_world310d.lib")
//#else
//#pragma comment(lib,"opencv_world310.lib")
//#endif

/*显示视差图*/
void ShowDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& name);
/*保存视差图*/
void SaveDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& path);
/*保存视差点云*/
void SaveDisparityCloud(const uint8* img_bytes, const float32* disp_map, const sint32& width, const sint32& height, const std::string& path);

/**
* \brief
* \param argv 3
* \param argc argc[1]:左影像路径 argc[2]: 右影像路径 argc[3]: 最小视差[可选，默认0] argc[4]: 最大视差[可选，默认64]
* \param eg. ..\Data\cone\im2.png ..\Data\cone\im6.png 0 64
* \param eg. ..\Data\Reindeer\view1.png ..\Data\Reindeer\view5.png 0 128
* \return
*/

void on_mouse(int event, int x, int y, int flags, void* ustc)
{
	Mat p;
	p = *(Mat*)ustc;
	Point pt = Point(x, y);
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << x << " " << y << " "<<p.at<ushort>(y, x) << endl;;
	}
}

int main()//path 0 64
{

	printf("Image Loading...");
	//···············································································//
	// 读取影像
	std::string path_left = "../Data/kitti/left1.png";
	std::string path_right = "../Data/kitti/right1.png";
	string path_lidar = "../Data/kitti/0000000059.bin";
	//string path_lidar = "../Data/kitti/LIDARcloud.txt";
	//string path_leftnormals = "../Data/kitti/leftpointnormal.txt";
	//string path_rightnormals = "../Data/kitti/rightpointnormal.txt";
	cv::Mat img_left = cv::imread(path_left, cv::IMREAD_COLOR);
	cv::Mat img_right = cv::imread(path_right, cv::IMREAD_COLOR);
	cv::Mat grayimg_left, grayimg_right;
	cvtColor(img_left, grayimg_left, CV_BGR2GRAY);
	cvtColor(img_right, grayimg_right, CV_BGR2GRAY);
	if (img_left.data == nullptr || img_right.data == nullptr) {
		std::cout << "读取影像失败！" << std::endl;
		return -1;
	}
	if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
		std::cout << "左右影像尺寸不一致！" << std::endl;
		return -1;
	}
	////···············································································//
	printf("Point Cloud Laoding... ");
	//加载点云并对其
	ifstream fp;
	long count = 0;//点云计数
	MatrixXf transform(4, 4), transform_02(4, 4), R_rect(4, 4), P_rect_02(3, 4), P_velo_to_img02(3, 4), cloudpoint(120212, 4), P_velo_to_cam02(4, 4);
	transform << 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03,
		1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02,
		9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01,
		0, 0, 0, 1;//初始化变换矩阵 [R|T]为方便计算 其次表示  R 3x3 T 3X1  [R|T] 4X4   vel_to_cam文件中
	R_rect << 9.999239e-01, 9.837760e-03, -7.445048e-03, 0,
		-9.869795e-03, 9.999421e-01, -4.278459e-03, 0,
		7.402527e-03, 4.351614e-03, 9.999631e-01, 0,
		0, 0, 0, 1;//初始化校正旋转矩阵 R_rect 3X3  为方便计算使用其次表示 4X4   此处是R_rect_00  0号相机坐标系 -> 矫正后的0号相机坐标系  cam_to_cam文件
	//二号相机
	P_rect_02 << 7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01,
		0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01,
		0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03; //投影矩阵 3X4    此处是 P_rect_02： 矫正后的0号相机坐标系  ->  2号相机的图像平面 2号相机是左侧彩色相机  cam_to_cam文件
	P_velo_to_img02 = P_rect_02 * R_rect * transform;  //3X4    投影到2号相机，一共四个相机，
	//cout << P_velo_to_img(1, 1)<<endl;
	//雷达坐标系到相机02的相机坐标系的投影
	transform_02 << 9.999758e-01, -5.267463e-03, -4.552439e-03, 5.956621e-02,
		5.251945e-03, 9.999804e-01, -3.413835e-03, 2.900141e-04,
		4.570332e-03, 3.389843e-03, 9.999838e-01, 2.577209e-03,
		0, 0, 0, 1; //R|T cam00->cam02
	P_velo_to_cam02 = transform_02 * transform;//4X4 雷达坐标系投影到相机坐标系02 此处注意像素坐标系和相机坐标系
	//三号相机
	MatrixXf transform_03(4, 4), P_rect_03(3, 4), P_velo_to_img03(3, 4), P_velo_to_cam03(4, 4);
	P_rect_03 << 7.215377e+02, 0.000000e+00, 6.095593e+02, -3.395242e+02,
		0.000000e+00, 7.215377e+02, 1.728540e+02, 2.199936e+00,
		0.000000e+00, 0.000000e+00, 1.000000e+00, 2.729905e-03; //投影矩阵 3X4    此处是 P_rect_02： 矫正后的0号相机坐标系  ->  3号相机的图像平面 2号相机是左侧彩色相机  cam_to_cam文件
	P_velo_to_img03 = P_rect_03 * R_rect * transform;  //3X4    投影到3号相机，一共四个相机，
	transform_03 << 9.995599e-01, 1.699522e-02, -2.431313e-02, -4.731050e-01,
		-1.704422e-02, 9.998531e-01, -1.809756e-03, 5.551470e-03,
		2.427880e-02, 2.223358e-03, 9.997028e-01, -5.250882e-03,
		0, 0, 0, 1; //R|T cam00->cam03
	P_velo_to_cam03 = transform_03 * transform;//4X4 雷达坐标系投影到相机坐标系03 此处注意像素坐标系和相机坐标系


	fp.open(path_lidar, ios::binary);
	if (!fp.is_open())
		cout << "open file failure" << endl;
	while (fp.peek() != EOF)
	{
		for (int i = 0; i < 4; i++)//提取雷达点云的四个值
		{
			//fp.read((char *)(&b), sizeof(b));   //bin文件已float文件读取
			fp.read((char*)(&cloudpoint(count, i)), sizeof(cloudpoint(count, i)));
			//cout << cloudpoint(count, i) << endl;
			//cloudpoint(count, i) = b;
		}

		if (cloudpoint(count, 0) > 5)//x轴大于5才存进去,不然会被覆盖。在图像平面前方，根据雷达坐标系X朝向前方
			count++;//
	}
	fp.close();
	MatrixXf Y02(count, 3), Y03(count, 3), X(4, 1), Y_cam02(count, 4), Y_cam03(count, 4);//0号相机，X四行一列临时变量，2号相机
	for (int i = 0; i < count; i++)
	{
		X = cloudpoint.block(i, 0, 1, 4).transpose();//
		X(3, 0) = 1;//第四行表示雷达强度的数据为0
		Y02.block(i, 0, 1, 3) = (P_velo_to_img02 * X).transpose();//Y02 = P_velo_to_img * X   Y02的某一列向量怎么表示 雷达坐标系投影到像素坐标系
		Y_cam02.block(i, 0, 1, 4) = (P_velo_to_cam02 * X).transpose();//雷达坐标系投影到相机02坐标系  根据深度Z计算视差D

		Y03.block(i, 0, 1, 3) = (P_velo_to_img03 * X).transpose();
		Y_cam03.block(i, 0, 1, 4) = (P_velo_to_cam03 * X).transpose();
	}//Y02是其次矩阵表示，每一行的第三位不是0

	cv::Mat tmplidarprojection = cv::Mat(375, 1242, CV_8UC1, cv::Scalar(0));
	cv::Mat leftlidarimg = cv::Mat(375, 1242, CV_32FC3, cv::Scalar(0, 0, 0));//视差图三通道（视差，允许最大视差，最小视差）
	cv::Mat rightlidarimg = cv::Mat(375, 1242, CV_32FC3, cv::Scalar(0, 0, 0));
	bool flag_left[375 * 1242] = { false };//雷达点标志
	bool is_leftlidar[375 * 1242] = { false };
	bool flag_right[375 * 1242] = { false };//雷达点标志
	bool is_rightlidar[375 * 1242] = { false };
	for (int i = 0; i < count; i++)
	{
		int x = (int)round((Y02(i, 0) / Y02(i, 2)));
		int y = (int)round((Y02(i, 1) / Y02(i, 2)));
		if (x >= 0 && x <= 1241 && y >= 0 && y <= 374)//	确定像素在在的长宽内
		{
			//leftlidarimg.at<cv::Vec3f>(y, x)[0] = int(389.34 / (Y_cam02(i, 2) / Y_cam02(i, 3)));//整数视差
			leftlidarimg.at<cv::Vec3f>(y, x)[0] = 389.34 / (Y_cam02(i, 2) / Y_cam02(i, 3));//整数视差
			tmplidarprojection.at<uchar>(y,x)= 389.34 / (Y_cam02(i, 2) / Y_cam02(i, 3));//雷达投影
			flag_left[y * 1242 + x] = true;//确认时雷达点
			is_leftlidar[y * 1242 + x] = true;

			//Y02(i, 0) = Y_img(i, 0) / Y_img(i, 2);//特征点在图像中横坐标  因为是其次表示，所以要除一个数。生成的float如何和整数像素对应上
			//Y02(i, 1) = Y_img(i, 1) / Y_img(i, 2);//特征点在图像中纵坐标  投影会失去深度信息
			//Y02(i, 2) = 389.34 / (Y_cam02(i, 2) / Y_cam02(i, 3));//disparity=fb/depth  投影到像素坐标系就失去深度信息了
			//pms_option.number++;//确定有效点云
		}
		x = (int)round((Y03(i, 0) / Y03(i, 2)));
		y = (int)round((Y03(i, 1) / Y03(i, 2)));
		if (x >= 0 && x <= 1241 && y >= 0 && y <= 374)//	确定像素在在的长宽内
		{
			//leftlidarimg.at<cv::Vec3f>(y, x)[0] = int(389.34 / (Y_cam02(i, 2) / Y_cam02(i, 3)));//整数视差
			rightlidarimg.at<cv::Vec3f>(y, x)[0] = -389.34 / (Y_cam03(i, 2) / Y_cam03(i, 3));//整数视差
			flag_right[y * 1242 + x] = true;//确认时雷达点
			is_rightlidar[y * 1242 + x] = true;
		}

	}
	//cv::imshow("leftlidarimg", leftlidarimg);
	cv::equalizeHist(tmplidarprojection, tmplidarprojection);
	cv::imwrite(path_left + "leftlidarimg.svg", tmplidarprojection);
	//左视图增采样
	computerlidar(&grayimg_left, &leftlidarimg, is_leftlidar, flag_left);
	
	//右视图增采样
	computerlidar(&grayimg_right, &rightlidarimg, is_rightlidar, flag_right);



	cv::imshow("leftlidarimg", leftlidarimg);
	cv::imshow("rightlidarimg", rightlidarimg);
	printf("Done!\n");
	//
	//···············································································//
	
	//读取点云法向量并生成法向量图像
	//
	//···············································································//
	//ifstream fp1,fp2;
	//
	//printf("Loading normal...");
	//cv::Mat leftnormalimg = cv::Mat(375, 1242, CV_32FC3,cv::Scalar(0,0,0));
	//cv::Mat rightnormalimg = cv::Mat(375, 1242, CV_32FC3, cv::Scalar(0, 0, 0));
	//fp1.open(path_leftnormals, ios::in);
	//if (!fp1.is_open())
	//	cout << "open file failure" << endl;
	//
	//char s[70] = { 0 };
	//string x = "";
	//string y = "";
	//string z = "";
	//string normal_x = "";
	//string normal_y = "";
	//string normal_z = "";
	//while (!fp1.eof())
	//{
	//	fp1.getline(s, sizeof(s));
	//	stringstream word(s);
	//	word >> x;
	//	word >> y;
	//	word >> z;
	//	word >> normal_x;
	//	word >> normal_y;
	//	word >> normal_z;
	//	leftnormalimg.at<cv::Vec3f>(int(atof(y.c_str())), int(atof(x.c_str()))) = cv::Vec3f(atof(normal_x.c_str()), atof(normal_y.c_str()), atof(normal_z.c_str()));
	//	//cout << atof(x.c_str()) << "  " << atof(normal_z.c_str()) << " " << sum++ << endl;
	//	//point->points[sum].x = int(round(atof(x.c_str()))); //先从string-》char*-》double-》四舍五入-》int
	//	//point->points[sum].y = int(round(atof(y.c_str())));
	//	//point->points[sum].z = int(round(atof(z.c_str())));//整数视差
	//}
	//cout << "Done!" << endl;
	////显示
	////cv::namedWindow("window", cv::WINDOW_AUTOSIZE); // Create a window for display.
	////cv::imshow("window", leftnormalimg);
	////cv::waitKey(10000);
	//fp1.close();

	//fp2.open(path_rightnormals, ios::in);
	//if (!fp2.is_open())
	//	cout << "open file failure" << endl;
	////char s[70] = { 0 };
	///*string x = "";
	//string y = "";
	//string z = "";
	//string normal_x = "";
	//string normal_y = "";
	//string normal_z = "";*/
	//while (!fp2.eof())
	//{
	//	fp2.getline(s, sizeof(s));
	//	stringstream word(s);
	//	word >> x;
	//	word >> y;
	//	word >> z;
	//	word >> normal_x;
	//	word >> normal_y;
	//	word >> normal_z;
	//	rightnormalimg.at<cv::Vec3f>(int(atof(y.c_str())), int(atof(x.c_str()))) = cv::Vec3f(atof(normal_x.c_str()), atof(normal_y.c_str()), atof(normal_z.c_str()));
	//	//cout << atof(x.c_str()) << "  " << atof(normal_z.c_str()) << " " << sum++ << endl;
	//	//point->points[sum].x = int(round(atof(x.c_str()))); //先从string-》char*-》double-》四舍五入-》int
	//	//point->points[sum].y = int(round(atof(y.c_str())));
	//	//point->points[sum].z = int(round(atof(z.c_str())));//整数视差
	//}
	//cout << "Done!" << endl;
	////显示
	////cv::namedWindow("window", cv::WINDOW_AUTOSIZE); // Create a window for display.
	////cv::imshow("window", leftnormalimg);
	////cv::waitKey(10000);
	//fp2.close();

	//保存参数
	PMSOption pms_option;
	pms_option.number = count;//点云数量
	//左视图
	pms_option.flag_left = flag_left;
	pms_option.is_leftlidar = is_leftlidar;
	pms_option.leftlidarimg = &leftlidarimg;
	//pms_option.leftnormalimg = &leftnormalimg;
	//右视图
	pms_option.flag_right = flag_right;
	pms_option.is_rightlidar = is_rightlidar;
	pms_option.rightlidarimg = &rightlidarimg;
	//pms_option.rightnormalimg = &rightnormalimg;
	//cout << "雷达点云数量" << count << endl;
	//···············································································//
	//···············································································//
	const sint32 width = static_cast<uint32>(img_left.cols);
	const sint32 height = static_cast<uint32>(img_right.rows);

	// 左右影像的彩色数据
	auto bytes_left = new uint8[width * height * 3];
	auto bytes_right = new uint8[width * height * 3];
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			bytes_left[i * 3 * width + 3 * j] = img_left.at<cv::Vec3b>(i, j)[0];
			bytes_left[i * 3 * width + 3 * j + 1] = img_left.at<cv::Vec3b>(i, j)[1];
			bytes_left[i * 3 * width + 3 * j + 2] = img_left.at<cv::Vec3b>(i, j)[2];
			bytes_right[i * 3 * width + 3 * j] = img_right.at<cv::Vec3b>(i, j)[0];
			bytes_right[i * 3 * width + 3 * j + 1] = img_right.at<cv::Vec3b>(i, j)[1];
			bytes_right[i * 3 * width + 3 * j + 2] = img_right.at<cv::Vec3b>(i, j)[2];
		}
	}
	

	// PMS匹配参数设计
	//PMSOption pms_option;
	// 聚合路径数
	pms_option.patch_size = 35;//???????
	// 候选视差范围
	pms_option.min_disparity = 0;
	pms_option.max_disparity = 90;
	// gamma 自适应权重
	pms_option.gamma = 10.0f;
	//function of dissimilarity of q and q'
	// alpha 平衡颜色和梯度关系
	pms_option.alpha = 0.9f;//梯度占比大
	// t_col  对遮挡区域鲁棒的截断代价
	pms_option.tau_col = 20.0f;
	// t_grad 对遮挡区域鲁棒的截断代价
	pms_option.tau_grad = 4.0f;
	// 传播迭代次数
	pms_option.num_iters = 3;

	// 一致性检查
	pms_option.is_check_lr = false;//同名点的左视图的视差和右视图的视差相差小于阈值
	pms_option.lrcheck_thres = 2.0f;
	// 视差图填充
	pms_option.is_fill_holes = false;

	// 前端平行窗口
	pms_option.is_fource_fpw =	false;

	// 整数视差精度
	pms_option.is_integer_disp = false;

	// 定义PMS匹配类实例
	PatchMatchStereo pms;

	printf("PatchMatch Initializing...");
	auto start = std::chrono::steady_clock::now();
	//···············································································//
	// 初始化
	if (!pms.Initialize(width, height, pms_option)) {
		std::cout << "PMS初始化失败！" << std::endl;
		return -2;
	}
	auto end = std::chrono::steady_clock::now();
	auto tt = duration_cast<std::chrono::milliseconds>(end - start);
	printf("Done! Timing : %lf s\n", tt.count() / 1000.0);

	printf("PatchMatch Matching...");
	start = std::chrono::steady_clock::now();
	//···············································································//
	// 匹配
	auto disparity = new float32[uint32(width * height)]();
	if (!pms.Match(bytes_left, bytes_right, disparity)) {
		std::cout << "PMS匹配失败！" << std::endl;
		return -2;
	}
	end = std::chrono::steady_clock::now();
	tt = duration_cast<std::chrono::milliseconds>(end - start);
	printf("Done! Timing : %lf s\n", tt.count() / 1000.0);

#if 1
	
	// 显示梯度图
	cv::Mat grad_right = cv::Mat(height, width, CV_8UC1);
	auto* grad_map = pms.GetGradientMap(1);
	if (grad_map) {
		for (sint32 i = 0; i < height; i++) {
			for (sint32 j = 0; j < width; j++) {
				const auto grad = grad_map[i * width + j].x+ grad_map[i * width + j].y;
				grad_right.data[i * width + j] = grad;
			}
		}
	}
	cv::imshow("梯度图-右", grad_right);
	cv::Mat grad_left = cv::Mat(height, width, CV_8UC1);
	auto* grad_map1 = pms.GetGradientMap(0);
	if (grad_map1) {
		for (sint32 i = 0; i < height; i++) {
			for (sint32 j = 0; j < width; j++) {
				const auto grad = grad_map1[i * width + j].y+ grad_map1[i * width + j].x;
				grad_left.data[i * width + j] = grad;
			}
		}
	}
	cv::imshow("梯度图-左", grad_left);
	cv::imwrite(path_left +"gradimage.png",grad_left);

#endif

	//···············································································//
	//···············································································//
	ShowDisparityMap(pms.GetDisparityMap(0), width, height, "disp-left");
	
	
	/*ShowDisparityMap(disp_left_, width, height, "disp-left++");*/
	ShowDisparityMap(pms.GetDisparityMap(1), width, height, "disp-right");
	// 保存视差图
	
	

	SaveDisparityMap(pms.GetDisparityMap(0), width, height, path_left);
	SaveDisparityMap(pms.GetDisparityMap(1), width, height, path_right);
	// 保存视差点云
	SaveDisparityCloud(bytes_left, pms.GetDisparityMap(0), width, height, path_left);
	SaveDisparityCloud(bytes_right, pms.GetDisparityMap(1), width, height, path_left);
	cv::waitKey(0);

	//···············································································//
	// 释放内存
	/*delete[] disparity;
	disparity = nullptr;
	delete[] bytes_left;
	bytes_left = nullptr;
	delete[] bytes_right;
	bytes_right = nullptr;*/
	
	system("pause");
	return 0;
}

void ShowDisparityMap(const float32* disp_map,const sint32& width,const sint32& height, const std::string& name)
{
	// 显示视差图
	const cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
	float32 min_disp = float32(width), max_disp = -float32(width);
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 disp = abs(disp_map[i * width + j]);
			if (disp != Invalid_Float) {
				min_disp = std::min(min_disp, disp);
				max_disp = std::max(max_disp, disp);
			}
		}
	}
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 disp = abs(disp_map[i * width + j]);
			if (disp == Invalid_Float) {
				disp_mat.data[i * width + j] = 0;
			}
			else {
				disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
			}
		}
	}

	cv::imshow(name, disp_mat);
	cv::Mat disp_color;
	applyColorMap(disp_mat, disp_color, cv::COLORMAP_JET);
	cv::imshow(name + "-color", disp_color);

}

void SaveDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& path)
{
	// 保存视差图
	cv::Mat disp_mat = cv::Mat(height, width, CV_16UC1);
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 disp = abs(disp_map[i * width + j]);
				//disp_mat.at<ushort>(i,j) = static_cast<ushort>((disp - min_disp) / (max_disp - min_disp) * 255)*256;
				disp_mat.at<ushort>(i,j) = static_cast<ushort>(disp)*256;
		}
	}
	
	cv::imwrite(path + "-d.png", disp_mat);
	cv::imshow("saved_disp",disp_mat);
	//setMouseCallback("saved_disp", on_mouse, (void*)&disp_map);
	//waitKey(100000000000);
	//cv::Mat disp_color;
	//applyColorMap(disp_mat, disp_color, cv::COLORMAP_JET);
	//cv::imwrite(path + "-c.png", disp_color);
}

void SaveDisparityCloud(const uint8* img_bytes, const float32* disp_map, const sint32& width, const sint32& height, const std::string& path)
{
	// 保存视差点云(x,y,disp,r,g,b)
	FILE* fp_disp_cloud = nullptr;
	fopen_s(&fp_disp_cloud, (path + "-cloud.txt").c_str(), "w");
	if (fp_disp_cloud) {
		for (sint32 i = 0; i < height; i++) {
			for (sint32 j = 0; j < width; j++) {
				const float32 disp = abs(disp_map[i * width + j]);
				if (disp == Invalid_Float) {
					continue;
				}
				fprintf_s(fp_disp_cloud, "%f %f %f %d %d %d\n", float32(j), float32(i),
					disp, img_bytes[i * width * 3 + 3 * j + 2], img_bytes[i * width * 3 + 3 * j + 1], img_bytes[i * width * 3 + 3 * j]);
			}
		}
		fclose(fp_disp_cloud);
	}
}