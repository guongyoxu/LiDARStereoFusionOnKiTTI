#include "ComputerLidar.h"

void computerlidar(cv::Mat* img, cv::Mat* lidarimg, bool* is_lidar, bool* flag)
{
	//��һ���޲��״�ɨ����
	for (int y = 130; y < 374; y++) {
		for (int x = 2; x < 1240; x++) {
			//printf("%d", lidarimg->at<cv::Vec3f>(y, x)) ;
			if (is_lidar[y * 1242 + x])//���״��
			{
				if (!is_lidar[y * 1242 + x - 1])//���
				{
					flag[y * 1242 + x - 1] = true;
					lidarimg->at<cv::Vec3f>(y, x - 1)[0] = lidarimg->at<cv::Vec3f>(y, x)[0];
					lidarimg->at<cv::Vec3f>(y, x - 1)[1] = 0;
					lidarimg->at<cv::Vec3f>(y, x - 1)[2] = 0;
				}
				else if (lidarimg->at<cv::Vec3f>(y, x - 1)[0] < lidarimg->at<cv::Vec3f>(y, x)[0])
					//����������״�㣬���ǱȽ�С�����滻
					lidarimg->at<cv::Vec3f>(y, x - 1)[0] = lidarimg->at<cv::Vec3f>(y, x)[0];

				if (!is_lidar[y * 1242 + x - 2])//�����
				{
					flag[y * 1242 + x - 2] = true;
					lidarimg->at<cv::Vec3f>(y, x - 2)[0] = lidarimg->at<cv::Vec3f>(y, x)[0];
					lidarimg->at<cv::Vec3f>(y, x - 2)[1] = 0;
					lidarimg->at<cv::Vec3f>(y, x - 2)[2] = 0;
				}
				else if (lidarimg->at<cv::Vec3f>(y, x - 2)[0] < lidarimg->at<cv::Vec3f>(y, x)[0])//
					lidarimg->at<cv::Vec3f>(y, x - 2)[0] = lidarimg->at<cv::Vec3f>(y, x)[0];

				if (!is_lidar[y * 1242 + x + 1])//�ҵ�
				{
					flag[y * 1242 + x + 1] = true;
					lidarimg->at<cv::Vec3f>(y, x + 1)[0] = lidarimg->at<cv::Vec3f>(y, x)[0];
					lidarimg->at<cv::Vec3f>(y, x + 1)[1] = 0;
					lidarimg->at<cv::Vec3f>(y, x + 1)[2] = 0;
				}
				else if (lidarimg->at<cv::Vec3f>(y, x + 1)[0] < lidarimg->at<cv::Vec3f>(y, x)[0])
					lidarimg->at<cv::Vec3f>(y, x + 1)[0] = lidarimg->at<cv::Vec3f>(y, x)[0];

				if (!is_lidar[y * 1242 + x + 2])//���ҵ�
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
	//ȥ���ص���
	for (int y = 130; y < 375; y++) {
		for (int x = 0; x < 1242; x++) {
			//printf("%d", lidarimg->at<cv::Vec3f>(y, x)) ;
			if (lidarimg->at<cv::Vec3f>(y, x)[0] > 0)//�ǵ�
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
					y_up++;//����
					////�ҵ� ��ɫ�����Ӳ����Ҳ��������,�����²���
					if (y_up < 375 && lidarimg->at<cv::Vec3f>(y_up, x_up)[0]>0
						//&& abs(img->at<uchar>(y_up, x_up) - color) < 10
						&& abs(lidarimg->at<cv::Vec3f>(y, x)[0] - lidarimg->at<cv::Vec3f>(y_up, x_up)[0]) < 4)
						break;

					if (y_up < 375 && lidarimg->at<cv::Vec3f>(y_up, x_up)[0] > 0
						//&& abs(img->at<uchar>(y_up, x_up) - color) < 8
						&& lidarimg->at<cv::Vec3f>(y, x)[0] - lidarimg->at<cv::Vec3f>(y_up, x_up)[0] > 7)
						//����ҵ���һ���״�ɨ����,��ɫ�����Ӳ����ͦ��
					{
						lidarimg->at<cv::Vec3f>(y_up, x_up)[0] = lidarimg->at<cv::Vec3f>(y, x)[0];

						//����ռ������ɨ����(�������⣬��������Ļ���������ɫ�ع���ȵ����)
						//int xx_left = --x_up;//����
						//int xx_right = ++x_up;//����
						//int color_left = color;
						//int color_right = color;
						//while (xx_left > 0 && abs(img->at<uchar>(y_up, xx_left) - color ) < 5
						//	&& lidarimg->at<cv::Vec3f>(y_up, xx_left)[0] > 0
						//	)//��ɫ������ɫ�ع���ȣ�.���Ӳ�㲻Ϊ��0����һֱ������չ
						//{
						//	lidarimg->at<cv::Vec3f>(y_up, xx_left)[0] = lidarimg->at<cv::Vec3f>(y_up, x_up)[0];
						//	//color_left = img->at<uchar>(y_up, xx_left);
						//	xx_left--;
						//}
						//while (xx_right < 375 && abs(img->at<uchar>(y_up, xx_right) - color) < 5
						//	&& lidarimg->at<cv::Vec3f>(y_up, xx_right)[0] > 0
						//	)//��ɫ����
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

	////���״����������
	for (int y = 130; y < 375; y++) {
		for (int x = 0; x < 1242; x++) {
			//printf("%d", lidarimg->at<cv::Vec3f>(y, x)) ;
			if (lidarimg->at<cv::Vec3f>(y, x)[0] <= 0)//û�е�
			{
				int iterator = 10;//�������µ�������
				int subscript_up = y;//�����ƶ��������±�
				int subscript_down = y;
				bool subscript_up_flag = false, subscript_down_flag = false;//�Ƿ�ΪĿ���±�
				while (iterator--)
				{
					if (!subscript_up_flag)
					{
						subscript_up--;
						if (subscript_up > 0 && lidarimg->at<cv::Vec3f>(subscript_up, x)[0] > 0)//�Ƿ񳬳��߽�,�����ҵ��״��
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
					if (subscript_down_flag && subscript_up_flag)//����������ҵ��������Ӳֹͣ����ִ����һ��ѭ��
					{
						float alpha = (y - subscript_up) * 1.0 / (subscript_down - subscript_up);
						//cout << y << " " << subscript_down << " " << subscript_down << endl;
						lidarimg->at<cv::Vec3f>(y, x) = alpha * lidarimg->at<cv::Vec3f>(subscript_down, x) + (1 - alpha) * lidarimg->at<cv::Vec3f>(subscript_up, x);
						flag[y * 1242 + x] = true;//ȷ�����������״��
						//[1]�洢�Ƚϴ� [2]�洢Сֵ
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