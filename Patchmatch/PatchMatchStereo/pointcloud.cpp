#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <iostream>
#include <vector>
#include <fstream>  
using namespace std;

int main()
{
	string path_lidar = "../Data/kitti/LIDARcloud.txt";
	ifstream fp;
	int number = 20184;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	
	cloud->width = 20184;
	cloud->height = 1;
	cloud->points.resize(cloud->width * cloud->height);

	printf("Loading Lidarcloud...");
	fp.open(path_lidar, ios::in);
	if (!fp.is_open())
		cout << "open file failure" << endl;
	int sum = 0;
	char s[50] = { 0 };
	string x = "";
	string y = "";
	string z = "";
	while (!fp.eof())
	{
		fp.getline(s, sizeof(s));
		stringstream word(s);
		word >> x;
		word >> y;
		word >> z;
		cloud->points[sum].x = atof(x.c_str());
		cloud->points[sum].y = atof(y.c_str());
		cloud->points[sum].z = atof(z.c_str());
		sum++;
	}
	if (sum != number)
		printf("雷达点和点云个数不匹配");
	fp.close();

	 //初始化kdTree
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	// 设置要搜索的点云
	kdtree.setInputCloud(cloud);

	pcl::PointXYZ searchPoint;

	searchPoint.x = 600;
	searchPoint.y = 149;
	searchPoint.z = 12;

	// K近邻搜索

	int K = 1;

	std::vector<int> pointIdxNKNSearch(K);//索引
	std::vector<float> pointNKNSquaredDistance(K);//距离的平方

	std::cout << "K nearest neighbor search at (" << searchPoint.x
		<< " " << searchPoint.y
		<< " " << searchPoint.z
		<< ") with K=" << K << std::endl;

	if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
	{
		for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
			std::cout << "    " << cloud->points[pointIdxNKNSearch[i]].x
			<< " " << cloud->points[pointIdxNKNSearch[i]].y
			<< " " << cloud->points[pointIdxNKNSearch[i]].z
			<< " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
	}

	// 以radius为半径的范围搜索

	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointRadiusSquaredDistance;

	float radius =4;

	std::cout << "Neighbors within radius search at (" << searchPoint.x
		<< " " << searchPoint.y
		<< " " << searchPoint.z
		<< ") with radius=" << radius << std::endl;


	if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
	{
		for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
			std::cout << "    " << cloud->points[pointIdxRadiusSearch[i]].x
			<< " " << cloud->points[pointIdxRadiusSearch[i]].y
			<< " " << cloud->points[pointIdxRadiusSearch[i]].z
			<< " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
	}

	system("pause");
	return 0;
}