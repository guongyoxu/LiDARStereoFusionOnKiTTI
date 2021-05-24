#include "ComputerLidar.h"

void computerlidar(cv::Mat* img, cv::Mat* lidarimg, bool* is_lidar, bool* flag)
{
	//第一步修补雷达扫描线
	for (int y = 130; y < 374; y++) {
		for (int x = 2; x < 1240; x++) {
			//printf("%d", lidarimg->at<cv::Vec3f>(y, x)) ;
			if (is_lidar[y * 1242 + x])//是雷达点
			{
				if (!is_lidar[y * 1242 + x - 1])//左点
				{
					flag[y * 1242 + x - 1] = true;
					lidarimg->at<cv::Vec3f>(y, x - 1)[0] = lidarimg->at<cv::Vec3f>(y, x)[0];
					lidarimg->at<cv::Vec3f>(y, x - 1)[1] = 0;
					lidarimg->at<cv::Vec3f>(y, x - 1)[2] = 0;
				}
				else if (lidarimg->at<cv::Vec3f>(y, x - 1)[0] < lidarimg->at<cv::Vec3f>(y, x)[0])
					//如果左点存在雷达点，但是比较小。则被替换
					lidarimg->at<cv::Vec3f>(y, x - 1)[0] = lidarimg->at<cv::Vec3f>(y, x)[0];

				if (!is_lidar[y * 1242 + x - 2])//最左点
				{
					flag[y * 1242 + x - 2] = true;
					lidarimg->at<cv::Vec3f>(y, x - 2)[0] = lidarimg->at<cv::Vec3f>(y, x)[0];
					lidarimg->at<cv::Vec3f>(y, x - 2)[1] = 0;
					lidarimg->at<cv::Vec3f>(y, x - 2)[2] = 0;
				}
				else if (lidarimg->at<cv::Vec3f>(y, x - 2)[0] < lidarimg->at<cv::Vec3f>(y, x)[0])//
					lidarimg->at<cv::Vec3f>(y, x - 2)[0] = lidarimg->at<cv::Vec3f>(y, x)[0];

				if (!is_lidar[y * 1242 + x + 1])//右点
				{
					flag[y * 1242 + x + 1] = true;
					lidarimg->at<cv::Vec3f>(y, x + 1)[0] = lidarimg->at<cv::Vec3f>(y, x)[0];
					lidarimg->at<cv::Vec3f>(y, x + 1)[1] = 0;
					lidarimg->at<cv::Vec3f>(y, x + 1)[2] = 0;
				}
				else if (lidarimg->at<cv::Vec3f>(y, x + 1)[0] < lidarimg->at<cv::Vec3f>(y, x)[0])
					lidarimg->at<cv::Vec3f>(y, x + 1)[0] = lidarimg->at<cv::Vec3f>(y, x)[0];

				if (!is_lidar[y * 1242 + x + 2])//最右点
				{
					flag[y * 1242 + x + 2] = true;
					lidarimg->at<cv::Vec3f>(y, x + 2)[0] = lidarimg->at<cv::Vec3f>(y, x)[0];
					lidarimg->at<cv::Vec3f>(y, x + 2)[1] = 0;
					lidarimg->at<cv::Vec3f>(y, x + 2)[2] = 0;
				}
				else if (lidarimg->at<cv::Vec3f>(y, x + 2)[0] < lidarimg->at<cv::Vec3f>(y, x)[0])
					lidarimg->at<cv::Vec3f>(y, x + 2)[0] = lidarimg->at<cv::Vec3f>(y, x)[0];
			}
		}
	}
	//去除重叠点
	for (int y = 130; y < 375; y++) {
		for (int x = 0; x < 1242; x++) {
			//printf("%d", lidarimg->at<cv::Vec3f>(y, x)) ;
			if (lidarimg->at<cv::Vec3f>(y, x)[0] > 0)//是点
			{
				int color = img->at<uchar>(y, x);
				int x_up = x;
				int y_up = y;
				int x_down = x;
				int y_down = y;
				int iterator = 5;
				//int iterator1 = 5;

				while (iterator--)
				{
					y_up++;//向下
					////找到 颜色相差不大，视差相差也不大，跳过,不向下查找
					if (y_up < 375 && lidarimg->at<cv::Vec3f>(y_up, x_up)[0]>0
						//&& abs(img->at<uchar>(y_up, x_up) - color) < 10
						&& abs(lidarimg->at<cv::Vec3f>(y, x)[0] - lidarimg->at<cv::Vec3f>(y_up, x_up)[0]) < 4)
						break;

					if (y_up < 375 && lidarimg->at<cv::Vec3f>(y_up, x_up)[0] > 0
						//&& abs(img->at<uchar>(y_up, x_up) - color) < 8
						&& lidarimg->at<cv::Vec3f>(y, x)[0] - lidarimg->at<cv::Vec3f>(y_up, x_up)[0] > 7)
						//如果找到下一条雷达扫描线,颜色相差不大，视差相差挺大
					{
						lidarimg->at<cv::Vec3f>(y_up, x_up)[0] = lidarimg->at<cv::Vec3f>(y, x)[0];

						//左右占领整条扫描线(存在问题，左右延伸的话会碰到颜色曝光过度的情况)
						//int xx_left = --x_up;//想左
						//int xx_right = ++x_up;//向右
						//int color_left = color;
						//int color_right = color;
						//while (xx_left > 0 && abs(img->at<uchar>(y_up, xx_left) - color ) < 5
						//	&& lidarimg->at<cv::Vec3f>(y_up, xx_left)[0] > 0
						//	)//颜色相差不大（颜色曝光过度）.（视差点不为零0），一直向左扩展
						//{
						//	lidarimg->at<cv::Vec3f>(y_up, xx_left)[0] = lidarimg->at<cv::Vec3f>(y_up, x_up)[0];
						//	//color_left = img->at<uchar>(y_up, xx_left);
						//	xx_left--;
						//}
						//while (xx_right < 375 && abs(img->at<uchar>(y_up, xx_right) - color) < 5
						//	&& lidarimg->at<cv::Vec3f>(y_up, xx_right)[0] > 0
						//	)//颜色相差不大
						//{
						//	lidarimg->at<cv::Vec3f>(y_up, xx_right)[0] = lidarimg->at<cv::Vec3f>(y_up, x_up)[0];
						//	//color_right = img->at<uchar>(y_up, xx_right);
						//	xx_right++;
						//}
					}

				}

			}
		}
	}

	////对雷达进行增采样
	for (int y = 130; y < 375; y++) {
		for (int x = 0; x < 1242; x++) {
			//printf("%d", lidarimg->at<cv::Vec3f>(y, x)) ;
			if (lidarimg->at<cv::Vec3f>(y, x)[0] <= 0)//没有点
			{
				int iterator = 10;//向上向下迭代次数
				int subscript_up = y;//向上移动的数组下标
				int subscript_down = y;
				bool subscript_up_flag = false, subscript_down_flag = false;//是否为目标下标
				while (iterator--)
				{
					if (!subscript_up_flag)
					{
						subscript_up--;
						if (subscript_up > 0 && lidarimg->at<cv::Vec3f>(subscript_up, x)[0] > 0)//是否超出边界,并且找到雷达点
						{
							subscript_up_flag = true;

						}
					}
					if (!subscript_down_flag)
					{
						subscript_down++;
						if (subscript_down < 375 && lidarimg->at<cv::Vec3f>(subscript_down, x)[0] > 0)
						{
							subscript_down_flag = true;
						}
					}
					if (subscript_down_flag && subscript_up_flag)//如果两个都找到，计算视差并停止迭代执行下一次循环
					{
						float alpha = (y - subscript_up) * 1.0 / (subscript_down - subscript_up);
						//cout << y << " " << subscript_down << " " << subscript_down << endl;
						lidarimg->at<cv::Vec3f>(y, x) = alpha * lidarimg->at<cv::Vec3f>(subscript_down, x) + (1 - alpha) * lidarimg->at<cv::Vec3f>(subscript_up, x);
						flag[y * 1242 + x] = true;//确认是增采样雷达点
						//[1]存储比较大 [2]存储小值
						if (lidarimg->at<cv::Vec3f>(subscript_down, x)[0] >= lidarimg->at<cv::Vec3f>(subscript_up, x)[0])//
						{
							lidarimg->at<cv::Vec3f>(y, x)[1] = lidarimg->at<cv::Vec3f>(subscript_down, x)[0];
							lidarimg->at<cv::Vec3f>(y, x)[2] = lidarimg->at<cv::Vec3f>(subscript_up, x)[0];
						}
						else
						{
							lidarimg->at<cv::Vec3f>(y, x)[2] = lidarimg->at<cv::Vec3f>(subscript_down, x)[0];
							lidarimg->at<cv::Vec3f>(y, x)[1] = lidarimg->at<cv::Vec3f>(subscript_up, x)[0];
						}
						break;

					}

				}
			}
		}
	}

}