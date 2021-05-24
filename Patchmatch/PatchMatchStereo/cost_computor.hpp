/* -*-c++-*- SemiGlobalMatching - Copyright (C) 2020.
* Author	: Ethan Li <ethan.li.whu@gmail.com>
*			  https://github.com/ethan-li-coding
* Describe	: implement of cost computer
*/

#ifndef PATCH_MATCH_STEREO_COST_HPP_
#define PATCH_MATCH_STEREO_COST_HPP_
#include "pms_types.h"
#include <algorithm>

#define COST_PUNISH 120.0f  // NOLINT(cppcoreguidelines-macro-usage)

#define USE_FAST_EXP
/* ����exp*/
inline double fast_exp(double x) {
	x = 1.0 + x / 1024;
	x *= x; x *= x; x *= x; x *= x;
	x *= x; x *= x; x *= x; x *= x;
	x *= x; x *= x;
	return x;
}

/**
 * \brief ���ۼ���������
 */
class CostComputer {
public:
	/** \brief ���ۼ�����Ĭ�Ϲ��� */
	CostComputer(): img_left_(nullptr), img_right_(nullptr), width_(0), height_(0), patch_size_(0), min_disp_(0),
	                max_disp_(0) {}

	/**
	 * \brief ���ۼ�������ʼ��
	 * \param img_left		��Ӱ������ 
	 * \param img_right		��Ӱ������
	 * \param width			Ӱ���
	 * \param height		Ӱ���
	 * \param patch_size	�ֲ�Patch��С
	 * \param min_disp		��С�Ӳ�
	 * \param max_disp		����Ӳ�
	 */
	CostComputer(const uint8* img_left, const uint8* img_right, const sint32& width,const sint32& height,const sint32& patch_size,const sint32& min_disp, const sint32& max_disp){
		img_left_ = img_left;
		img_right_ = img_right;
		width_ = width;
		height_ = height;
		patch_size_ = patch_size;
		min_disp_ = min_disp;
		max_disp_ = max_disp;
	}

	/** \brief ���ۼ��������� */
	virtual ~CostComputer() = default;

public:

	/**
	 * \brief ������Ӱ��p���Ӳ�Ϊdʱ�Ĵ���ֵ
	 * \param i		p��������
	 * \param j		p�������
	 * \param d		�Ӳ�ֵ
	 * \return ����ֵ
	 */
	virtual float32 Compute(const sint32& i, const sint32& j, const float32& d) = 0;

public:
	/** \brief ��Ӱ������ */
	const uint8* img_left_;
	/** \brief ��Ӱ������ */
	const uint8* img_right_;

	/** \brief Ӱ��� */
	sint32 width_;
	/** \brief Ӱ��� */
	sint32 height_;
	/** \brief �ֲ�����Patch��С */
	sint32 patch_size_;

	/** \brief ��С����Ӳ� */
	sint32 min_disp_;
	sint32 max_disp_;
};


/**
 * \brief ���ۼ�������PatchMatchSteroԭ�Ĵ��ۼ�����
 */
class CostComputerPMS : public CostComputer {
public:

	/** \brief PMS���ۼ�����Ĭ�Ϲ��� */
	CostComputerPMS() : grad_left_(nullptr), grad_right_(nullptr), gamma_(0), alpha_(0), tau_col_(0), tau_grad_(0) {};

	/**
	 * \brief PMS���ۼ��������ι���
	 * \param img_left		��Ӱ������
	 * \param img_right		��Ӱ������
	 * \param grad_left		���ݶ�����
	 * \param grad_right	���ݶ�����
	 * \param width			Ӱ���
	 * \param height		Ӱ���
	 * \param patch_size	�ֲ�Patch��С
	 * \param min_disp		��С�Ӳ�
	 * \param max_disp		����Ӳ�
	 * \param gamma			����gammaֵ
	 * \param alpha			����alphaֵ
	 * \param t_col			����tau_colֵ
	 * \param t_grad		����tau_gradֵ
	 */
	CostComputerPMS(const uint8* img_left, const uint8* img_right, const PGradient* grad_left, const PGradient* grad_right, const sint32& width, const sint32& height, const sint32& patch_size,
		const sint32& min_disp, const sint32& max_disp,
		const float32& gamma, const float32& alpha, const float32& t_col, const float32 t_grad, PMSOption option_) :
		CostComputer(img_left, img_right, width, height, patch_size, min_disp, max_disp) {
		grad_left_ = grad_left;
		grad_right_ = grad_right;
		gamma_ = gamma;
		alpha_ = alpha;
		tau_col_ = t_col;
		tau_grad_ = t_grad;
		option=option_;
	}

	/**
	 * \brief ������Ӱ��p���Ӳ�Ϊdʱ�Ĵ���ֵ��δ���߽��ж�
	 * \param x		p��x����
	 * \param y		p��y����
	 * \param d		�Ӳ�ֵ
	 * \return ����ֵ
	 */
	inline float32 Compute(const sint32& x, const sint32& y, const float32& d) override
	{
		const float32 xr = x - d;
		if (xr < 0.0f || xr >= static_cast<float32>(width_)) {
			return (1 - alpha_) * tau_col_ + alpha_ * tau_grad_;
		}
		// ��ɫ�ռ����
		const auto col_p = GetColor(img_left_, x, y);
		const auto col_q = GetColor(img_right_, xr, y);
		const auto dc = std::min((float)(abs(col_p.b - col_q.x) + abs(col_p.g - col_q.y) + abs(col_p.r - col_q.z)), tau_col_);

		// �ݶȿռ����
		const auto grad_p = GetGradient(grad_left_, x, y);
		const auto grad_q = GetGradient(grad_right_, xr, y);
		const auto dg = std::min((float)(abs(grad_p.x - grad_q.x) + abs(grad_p.y - grad_q.y)), tau_grad_);

		// ����ֵ
		return (1 - alpha_) * dc + alpha_ * dg;
	}

	/**
	 * \brief ������Ӱ��p���Ӳ�Ϊdʱ�Ĵ���ֵ��δ���߽��ж�
	 * \param col_p		p����ɫֵ
	 * \param grad_p	p���ݶ�ֵ
	 * \param x			p��x����
	 * \param y			p��y����
	 * \param d			�Ӳ�ֵ
	 * \return ����ֵ
	 */
	inline float32 Compute(const PColor& col_p, const PGradient& grad_p, const sint32& x, const sint32& y, const float32& d) const
	{
		const float32	xr = x - d;//
		if (xr < 0.0f || xr >= static_cast<float32>(width_)) {//��һ��ͼ������ز���ͼƬ��
			return (1 - alpha_) * tau_col_ + alpha_ * tau_grad_;
		}
		// ��ɫ�ռ����
		const auto col_q = GetColor(img_right_, xr, y);
		const auto dc = std::min(abs(col_p.b - col_q.x) + abs(col_p.g - col_q.y) + abs(col_p.r - col_q.z), tau_col_);

		// �ݶȿռ����
		const auto grad_q = GetGradient(grad_right_, xr, y);
		const auto dg = std::min(abs(grad_p.x - grad_q.x) + abs(grad_p.y - grad_q.y), tau_grad_);

		//���ƾ��루�����Ӳ�ͼ֮����Ӳ�Ĳ
		float dl;
		if (option.direction != true)//����ͼ
		{
			float d2 = option.leftlidarimg->at<cv::Vec3f>(y, x)[0];
			if (d2 > 0)//������Բ���
			{
				dl = std::min(abs(d2 - d), float(4));
				
			}
			else
				dl = 4;
		}
		else
		{
			float d2 = option.rightlidarimg->at<cv::Vec3f>(y, x)[0];
			if (d2 < 0)//������Բ���
			{
				dl = std::min(abs(d2 - d), float(4));
				
			}
			else
				dl = 4;
		}
		//std::cout << dc << " " << dg << std::endl;
		// ����ֵ  �󲿷��ǽضϴ���
		//return (1 - alpha_) * dc + alpha_ * dg+dl;
		return 0.1 * dc + 0.45 * dg +0.45* dl;
		
	}

	
	/**
	 * \brief ������Ӱ��p���Ӳ�ƽ��Ϊpʱ�ľۺϴ���ֵ
	 * \param x		p��x����
	 * \param y 	p��y����
	 * \param p		ƽ�����
	 * \return �ۺϴ���ֵ
	 */
	inline float32 ComputeA(const sint32& x, const sint32& y, const DisparityPlane& p) const
	{
		int sum = 0;//��¼�״�������
		float32 length = 0.0f;//�״�㵽ƽ��ľ���֮�ͣ������ж�����ͬһ��ƽ�棬���ú�w����
		const auto pat = patch_size_ / 2;
		const auto& col_p = GetColor(img_left_, x, y);
		float32 cost = 0.0f;
		for (sint32 r = -pat; r <= pat; r++) {
			const sint32 yr = y + r;
			for (sint32 c = -pat; c <= pat; c++) {
				const sint32 xc = x + c;
				if (yr < 0 || yr > height_ - 1 || xc < 0 || xc > width_ - 1) {
					continue;//����������ͼƬ��
				}
				// �����Ӳ�ֵ
				const float32 d = p.to_disparity(xc,yr);//��������Ӳ���nan
				if (d < min_disp_ || d > max_disp_|| _isnan(d)) {
					cost += COST_PUNISH;
					continue;
				}

				// ����Ȩֵ
				const auto& col_q = GetColor(img_left_, xc, yr);
				const auto dc = abs(col_p.r - col_q.r) + abs(col_p.g - col_q.g) + abs(col_p.b - col_q.b);
				//std::cout << dc << std::endl;//��������
#ifdef USE_FAST_EXP
				auto w = fast_exp(double(-dc / gamma_));
				float w2;
#else
				const auto w = exp(-dc / gamma_);
#endif
				const auto grad_q = GetGradient(grad_left_, xc, yr);

				//if (option.direction!=true)//����ͼ
				//{
				//	float d2 = option.leftlidarimg->at<cv::Vec3f>(y, x)[0];
				//	if (d2)//������Բ���
				//	{
				//		w2=fast_exp(double( abs(d2-d)*-1.0 / gamma_));
				//		w = w * 0.1 + w2 * 0.9;
				//	}
				//}
				//else//����ͼ  �Ӳ��Ǹ���
				//{
				//	float d2 = option.rightlidarimg->at<cv::Vec3f>(y, x)[0];
				//	if (d2<0)//������Բ���
				//	{
				//		w2 = fast_exp(double(abs(d2 - d)*-1.0 / gamma_));
				//		w = w * 0.1 + w2 * 0.9;
				//	}
				//}
				
				// �ۺϴ���
				
				cost += w * (Compute(col_q, grad_q, xc, yr, d));//p���Ӳ�Ϊd�Ĵ���
				//std::cout << w << " " << cost << std::endl;
				//std::cout << col_q << "  " << grad_q << endl;
			}
		}
		//if(sum)
		//std::cout << length << " " << sum << " " << length / sum << std::endl;//��Χ0����ʮ֮��
		//std::cout << cost<<" "<<length/sum << std::endl;//��Χ�ڼ��ٵ���w,�������ƥ��Ĵ��ۡ�
			return cost;
	}

	/**
	* \brief ��ȡ���ص����ɫֵ
	* \param img_data	��ɫ����,3ͨ��
	* \param x			����x����
	* \param y			����y����
	* \return ����(x,y)����ɫֵ
	*/
	inline PColor GetColor(const uint8* img_data, const sint32& x, const sint32& y) const
	{
		auto* pixel = img_data + y * width_ * 3 + 3 * x;
		return { pixel[0], pixel[1], pixel[2] };
	}

	/**
	* \brief ��ȡ���ص����ɫֵ
	* \param img_data	��ɫ����
	* \param x			����x���꣬ʵ���������ڲ�õ���ɫֵ
	* \param y			����y����
	* \return ����(x,y)����ɫֵ
	*/
	inline PVector3f GetColor(const uint8* img_data, const float64& x,const sint32& y) const
	{
		float32 col[3];
		const auto x1 = static_cast<sint32>(x);
		const sint32 x2 = x1 + 1;
		const float32 ofs = x - x1;
		for (sint32 n = 0; n < 3; n++) {
		const auto& g1 = img_data[y * width_ * 3 + 3 * x1 + n];
		const auto& g2 = (x2 < width_) ? img_data[y * width_ * 3 + 3 * x2 + n] : g1;
			col[n] = (1 - ofs) * g1 + ofs * g2;
		}

		return { col[0], col[1], col[2] };
	}

	/**
	* \brief ��ȡ���ص���ݶ�ֵ
	* \param grad_data	�ݶ�����
	* \param x			����x����
	* \param y			����y����
	* \return ����(x,y)���ݶ�ֵ
	*/
	inline PGradient GetGradient(const PGradient* grad_data, const sint32& x, const sint32& y) const
	{
		return grad_data[y * width_ + x];
	}

	/**
	* \brief ��ȡ���ص���ݶ�ֵ
	* \param grad_data	�ݶ�����
	* \param x			����x���꣬ʵ���������ڲ�õ��ݶ�ֵ
	* \param y			����y����
	* \return ����(x,y)���ݶ�ֵ
	*/
	inline PVector2f GetGradient(const PGradient* grad_data, const float32& x, const sint32& y) const
	{
		const auto x1 = static_cast<sint32>(x);
		const sint32 x2 = x1 + 1;
		const float32 ofs = x - x1;

		const auto& g1 = grad_data[y * width_ + x1];
		const auto& g2 = (x2 < width_) ? grad_data[y * width_ + x2] : g1;

		return { (1 - ofs) * g1.x + ofs * g2.x,(1 - ofs) * g1.y + ofs * g2.y };
	}

private:
	/** \brief ��Ӱ���ݶ����� */
	const PGradient* grad_left_;
	/** \brief ��Ӱ���ݶ����� */
	const PGradient* grad_right_;
	
	PMSOption option;


	/** \brief ����gamma */
	float gamma_;
	/** \brief ����alpha */
	float32 alpha_;
	/** \brief ����tau_col */
	float32 tau_col_;
	/** \brief ����tau_grad */
	float32 tau_grad_;
};

// �������ڴ�ͨ��������ķ�ʽʵ������ʵ�ֵĴ��ۼ�����������
class CostCensusComputer : public CostComputer {
public:
	CostCensusComputer() :gray_left_(nullptr), gray_right_(nullptr),lamda_census_(30),lamda_ad_(10) {};
	CostCensusComputer(const uint8* img_left, const uint8* img_right, const uint8* gray_left, const uint8* gray_right, const sint32& width, const sint32& height,
		const sint32& patch_size, const sint32& min_disp, const sint32& max_disp,float gamma
	) :CostComputer(img_left, img_right, width, height, patch_size, min_disp, max_disp), lamda_census_(30), lamda_ad_(10),gamma_(gamma)
	{
		gray_left_ = gray_left;
		gray_right_ = gray_right;
	}
	
	/**
	 * \brief ������Ӱ��p���Ӳ�ƽ��Ϊpʱ�ľۺϴ���ֵ
	 * \param x		p��x����
	 * \param y 	p��y����
	 * \param p		ƽ�����
	 * \return �ۺϴ���ֵ
	 */
	inline float32 ComputeA(const sint32& x, const sint32& y, const DisparityPlane& p) 
	{
		//uint16 count = patch_size_ * patch_size_;//ƽ�����ظ���
		const auto pat = patch_size_ / 2;
		const auto& col_p = GetColor(img_left_, x, y);
		float32 cost = 0.0f;
		for (sint32 r = -pat; r <= pat; r++) {
			const sint32 yr = y + r;
			for (sint32 c = -pat; c <= pat; c++) {
				const sint32 xc = x + c;
				if (yr <= 0 || yr >= height_ - 1 || xc <= 0 || xc >= width_ - 1) {//
					//count--;
					continue;//�߽粻���м���
				}
				// �����Ӳ�ֵ
				const float32 d = p.to_disparity(xc, yr);
				if (d < min_disp_ || d > max_disp_) {
					cost += COST_PUNISH;
					continue;
				}

				// ����Ȩֵ
				const auto& col_q = GetColor(img_left_, xc, yr);
				const auto dc = abs(col_p.r - col_q.r) + abs(col_p.g - col_q.g) + abs(col_p.b - col_q.b);
#ifdef USE_FAST_EXP
				const auto w = fast_exp(double(-dc / gamma_));

#else
				const auto w = exp(-dc / gamma_);
#endif

				// �ۺϴ��� w����ͬһƽ��Ŀ����ԣ�����Խ��wԽС��w��1�� computer����Ĳ������ԡ�
				cost += w * Compute(xc, yr, d);
				/*const auto grad_q = GetGradient(grad_left_, xc, yr);
				cost += w * Compute(col_q, grad_q, xc, yr, d);*/
			}
		}
		return cost;
	}

	//����3x3���ڵ�hanming���룬��������Ĵ�С�������ص����Ƴ̶�
	inline float32 Compute(const sint32& x, const sint32& y, const float32& d)
	{
		const uint8 xr = x - d;
		uint8 cost_ad = abs(gray_left_[y * width_ + x] - gray_right_[y * width_ + xr]);
		const float32 cost_census = static_cast<float32>(Hamming(Census3x3(gray_left_, x, y), Census3x3(gray_right_, x, y)));
		return 1 - exp(-cost_ad / lamda_ad_) + 1 - exp(-cost_census / lamda_census_);
		//����ֵ��[0-2]
	}
	//����ĳ���census����
	sint32 Census3x3(const uint8* img, const uint32 x, const uint32 y)
	{
		if (x <= 0 || y <= 0 ||x>=1241||y>=374)
		{
			printf("Խ��census���۱߽�");
			
		}
		const uint8 gray_center = (img[y * width_ + x+1]+ img[y * width_ + x -1]+ img[(y+1) * width_ + x]+ img[(y-1) * width_ + x])/4;//���ĵ�ĻҶ�ֵ
		uint64 census_val = 0;
		for (sint8 i =  - 1; i <=  1; i++) {
			for (sint8 j =  - 1; j <=   1; j++) {
				census_val <<= 1;
				const uint8 gray = img[(i+y) * width_ + x + j];
				if (gray < gray_center) {
					census_val += 1;
				}
			}
		}
		return census_val;
	}
	//��������census����֮��Ĳ���
	uint8 Hamming(const uint64 xl, const uint64 xr)
	{
		uint64 dist = 0, val = xl ^ xr;

		// Count the number of set bits
		while (val) {
			++dist;//�������λ��ͬ
			val &= val - 1;
		}

		return static_cast<uint64>(dist);
	}
	inline uint8 GetGray(const float * img_gray,const uint8 x,const uint8 y)
	{
		return img_gray[y*width_+x];
	}
	/**
	* \brief ��ȡ���ص����ɫֵ
	* \param img_data	��ɫ����,3ͨ��
	* \param x			����x����
	* \param y			����y����
	* \return ����(x,y)����ɫֵ
	*/
	inline PColor GetColor(const uint8* img_data, const sint32& x, const sint32& y) const
	{
		auto* pixel = img_data + y * width_ * 3 + 3 * x;
		return { pixel[0], pixel[1], pixel[2] };
	}

	/**
	* \brief ��ȡ���ص����ɫֵ
	* \param img_data	��ɫ����
	* \param x			����x���꣬ʵ���������ڲ�õ���ɫֵ
	* \param y			����y����
	* \return ����(x,y)����ɫֵ
	*/
	inline PVector3f GetColor(const uint8* img_data, const float32& x, const sint32& y) const
	{
		float32 col[3];
		const auto x1 = static_cast<sint32>(x);
		const sint32 x2 = x1 + 1;
		const float32 ofs = x - x1;

		for (sint32 n = 0; n < 3; n++) {
			const auto& g1 = img_data[y * width_ * 3 + 3 * x1 + n];
			const auto& g2 = (x2 < width_) ? img_data[y * width_ * 3 + 3 * x2 + n] : g1;
			col[n] = (1 - ofs) * g1 + ofs * g2;
		}

		return { col[0], col[1], col[2] };
	}
private:
	//�Ҷ�ͼ
	const uint8* gray_left_, * gray_right_;
	sint32 lamda_ad_ , lamda_census_;//
	float gamma_;
};

#endif
