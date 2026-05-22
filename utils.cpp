#include "utils.h"

void plot_map(const cv::Mat src, cv::Mat mapx, cv::Mat mapy,std::string color,int is_save_location_map) {
    cv::Mat img = src.clone();
    if (img.depth() == CV_16U) {
        img /= 256;
        img.convertTo(img, CV_8UC1);
    }
    //mono2color
    cv::Mat color_img;
    cv::cvtColor(img,color_img, cv::COLOR_GRAY2BGR);
    // 定位标记
    for (int r = 0; r < mapx.rows; r++) {
        for (int c = 0; c < mapx.cols; c++) {
            int valx = mapx.at<float>(r, c);
            if (valx > 0) {
                int valy = mapy.at<float>(r, c);
                color_img.at<cv::Vec3b>(valy, valx) = cv::Vec3b(0, 255, 0);
            }
        }
    }
    // 保存
    color_img *= 2;
    if (is_save_location_map != 0) {
        std::vector<int> saving_params;
        saving_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        saving_params.push_back(5); // png压缩率
        cv::imwrite("location_map_"+color+".png", color_img, saving_params);
    }
}