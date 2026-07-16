#include "brightness.h"
#include "get_brightness_cuda.cuh"
//#include "cuda_runtime.h"
#include "tools.h"

cv::Mat _get_kernel(double mapping) {
    int mid_mapping = int(mapping);
    double res = mapping - mid_mapping;

    //if mid_mapping is even
    if (mid_mapping % 2 == 0) {
        //mid_mapping += 1;
        mid_mapping -= 1;
    }

    cv::Mat kernel;
    kernel.create(mid_mapping + 2, mid_mapping + 2, CV_32F);

    kernel = res / 2; // Íâ²¿ÄÜÁ¿Ó¦ÎªÒ»°ë
    kernel(cv::Range(1, mid_mapping + 1), cv::Range(1, mid_mapping + 1)) = 1;
    cv::Scalar kernel_sum = cv::sum(kernel);
    kernel = kernel * (1 / kernel_sum[0]);
    return kernel;
}

void _get_brightness(cv::Mat img, cv::Mat mapx, cv::Mat mapy,
    cv::Mat kernel, cv::Mat& csv)
{
    // ±£Ö¤Á¬Ðø & ÀàÐÍ
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

            if (x < rad || y < rad || x >= img.rows - rad || y >= img.cols - rad) { // È¥³ý±ßÔµµã(mapÖÐµÄ¼«Ð¡Öµµã»ò¸ºÊý»á±»ºöÂÔ)
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

//void _get_brightness_cuda(cv::Mat img, cv::Mat mapx, cv::Mat mapy,
//    cv::Mat kernel, cv::Mat& csv)
//{
//    // ±£Ö¤Á¬Ðø & ÀàÐÍ
//    img = img.clone();
//    mapx = mapx.clone();
//    mapy = mapy.clone();
//    CV_Assert(img.type() == CV_32FC1);
//    CV_Assert(mapx.type() == CV_32FC1 && mapy.type() == CV_32FC1);
//    CV_Assert(kernel.type() == CV_32FC1 && kernel.rows == kernel.cols);
//
//    csv.create(mapx.size(), CV_32FC1);
//
//    const int rad = kernel.rows / 2;
//    const int ksz = kernel.rows;
//    const float* k = kernel.ptr<float>();
//
//    for (int i = 0; i < mapx.rows; ++i) {
//        const float* mx = mapx.ptr<float>(i);
//        const float* my = mapy.ptr<float>(i);
//        float* out = csv.ptr<float>(i);
//
//        for (int j = 0; j < mapx.cols; ++j) {
//            int x = static_cast<int>(mx[j]);
//            int y = static_cast<int>(my[j]);
//
//            if (x < rad || y < rad || x >= img.rows - rad || y >= img.cols - rad) { // È¥³ý±ßÔµµã(mapÖÐµÄ¼«Ð¡Öµµã»ò¸ºÊý»á±»ºöÂÔ)
//                out[j] = 0.0f;
//                continue;
//            }
//
//            const float* base = img.ptr<float>(x - rad) + (y - rad);
//            float sum = 0.0f;
//            for (int dy = 0; dy < ksz; ++dy)
//                for (int dx = 0; dx < ksz; ++dx)
//                    sum += base[dy * img.cols + dx] * k[dy * ksz + dx];
//            out[j] = sum;
//
//        }
//    }
//}

cv::Mat get_brt_curved(const cv::Mat& img, cv::Mat& mapx, cv::Mat& mapy, double mapping, double panel_rows, double panel_cols, std::string color, std::array<double, 3> brightness_setting) {
    //check cuda is available
    int cuda_device_count = 0;
    //cudaError_t err = cudaGetDeviceCount(&cuda_device_count);

    //get brightness kernel
    cv::Mat kernel;
    size_t color_index = std::string("RGB").find(color);
    if (brightness_setting[color_index] != 0) {
        mapping = brightness_setting[color_index]+2; // 修改点
    }

    kernel = _get_kernel(mapping);

    //get brightness
    cv::Mat imgf, csv;
    csv.create(panel_cols, panel_rows, CV_32F);

    img.convertTo(imgf, CV_32FC1, 1.0 / 255.0);

    //main
    csv.create(mapy.rows, mapy.cols, CV_32FC1);


    //if (cuda_device_count) {
    //    //确保连续内存（CUDA 需要）
    //    std::cout << get_time() << ":CUDA" << std::endl;
    //    if (!imgf.isContinuous())  imgf = imgf.clone();
    //    if (!mapy.isContinuous())  mapy = mapy.clone();
    //    if (!mapx.isContinuous())  mapx = mapx.clone();
    //    if (!kernel.isContinuous()) kernel = kernel.clone();

    //    get_brightness_cuda_wrapper(
    //        imgf.ptr<float>(), imgf.rows, imgf.cols,
    //        mapy.ptr<float>(),
    //        mapx.ptr<float>(),
    //        kernel.ptr<float>(), kernel.rows,
    //        csv.ptr<float>(), csv.rows, csv.cols
    //    );
    //}
    //else {
        std::cout << get_time() << ":CPU" << std::endl;
        _get_brightness(imgf, mapy, mapx, kernel, csv);
    //}
    return csv;
}

cv::Mat get_brt(const cv::Mat& img, cv::Mat& mapx, cv::Mat& mapy, double mapping, double panel_rows, double panel_cols, std::string color, std::array<double, 3> brightness_setting) {
    //check cuda is available
    int cuda_device_count = 0;
    //cudaError_t err = cudaGetDeviceCount(&cuda_device_count);

    //get brightness kernel
    cv::Mat kernel;
    size_t color_index = std::string("RGB").find(color);
    if (brightness_setting[color_index] != 0) {
        mapping = brightness_setting[color_index];
    }

    kernel = _get_kernel(mapping);

    //get brightness
    cv::Mat imgf, csv;
    csv.create(panel_cols, panel_rows, CV_32F);

    img.convertTo(imgf, CV_32FC1, 1.0 / 255.0);

    //main
    csv.create(mapy.rows, mapy.cols, CV_32FC1);


    //if (cuda_device_count) {
    //    //确保连续内存（CUDA 需要）
    //    std::cout << get_time() << ":CUDA" << std::endl;
    //    if (!imgf.isContinuous())  imgf = imgf.clone();
    //    if (!mapy.isContinuous())  mapy = mapy.clone();
    //    if (!mapx.isContinuous())  mapx = mapx.clone();
    //    if (!kernel.isContinuous()) kernel = kernel.clone();

    //    get_brightness_cuda_wrapper(
    //        imgf.ptr<float>(), imgf.rows, imgf.cols,
    //        mapy.ptr<float>(),
    //        mapx.ptr<float>(),
    //        kernel.ptr<float>(), kernel.rows,
    //        csv.ptr<float>(), csv.rows, csv.cols
    //    );
    //}
    //else {
        std::cout << get_time() << ":CPU" << std::endl;
        _get_brightness(imgf, mapy, mapx, kernel, csv);
    //}
    return csv;
}