#ifndef BRIGHTNESS_H
#define BRIGHTNESS_H

#include <opencv2/opencv.hpp>
cv::Mat get_brt(const cv::Mat& img, cv::Mat& mapx, cv::Mat& mapy, double mapping, double panel_rows, double panel_cols);

#endif