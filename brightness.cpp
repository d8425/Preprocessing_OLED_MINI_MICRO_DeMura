#include "brightness.h"

cv::Mat _get_kernel(double mapping) {
	int mid_mapping = int(mapping);

	cv::Mat kernel;
	kernel.create(mid_mapping + 2, mid_mapping + 2, CV_32F);

	kernel = (mapping - mid_mapping)/2; // 外部能量应为一半
	kernel(cv::Range(1, mid_mapping + 1), cv::Range(1, mid_mapping + 1)) = 1;
    cv::Scalar kernel_sum = cv::sum(kernel);
	kernel = kernel * (1/kernel_sum[0]);
	return kernel;
}

void _get_brightness(cv::Mat img, cv::Mat mapx, cv::Mat mapy,
    cv::Mat kernel, cv::Mat& csv)
{
    // 保证连续 & 类型
    img = img.clone();
    mapx = mapx.clone();
    mapy = mapy.clone();
    CV_Assert(img.type() == CV_32FC1);
    CV_Assert(mapx.type() == CV_32FC1 && mapy.type() == CV_32FC1);
    CV_Assert(kernel.type() == CV_32FC1 && kernel.rows == kernel.cols);

    csv.create(mapx.size(), CV_32FC1);

    const int rad = kernel.rows / 2;
    const int ksz = kernel.rows;
    const float* k = kernel.ptr<float>();

    for (int i = 0; i < mapx.rows; ++i) {
        const float* mx = mapx.ptr<float>(i);
        const float* my = mapy.ptr<float>(i);
        float* out = csv.ptr<float>(i);

        for (int j = 0; j < mapx.cols; ++j) {
            int x = static_cast<int>(mx[j]);
            int y = static_cast<int>(my[j]);

            if (x < rad || y < rad || x >= img.rows - rad || y >= img.cols - rad) {
                out[j] = 0.0f;
                continue;
            }

            const float* base = img.ptr<float>(x - rad) + (y - rad);
            float sum = 0.0f;
            for (int dy = 0; dy < ksz; ++dy)
                for (int dx = 0; dx < ksz; ++dx)
                    sum += base[dy * img.cols + dx] * k[dy * ksz + dx];
            out[j] = sum;
        }
    }
}

cv::Mat get_brt(const cv::Mat& img, cv::Mat& mapx, cv::Mat& mapy, double mapping,double panel_rows,double panel_cols) {
	//get brightness kernel
	cv::Mat kernel;
	kernel = _get_kernel(mapping);

	//get brightness
	cv::Mat imgf, csv;
	csv.create(panel_cols, panel_rows, CV_32F);

    img.convertTo(imgf, CV_32FC1, 1.0 / 255.0);
	_get_brightness(imgf, mapy, mapx, kernel, csv);
    return csv;
}