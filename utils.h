#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

void plot_map(cv::Mat img, cv::Mat mapx, cv::Mat mapy, std::string color, int is_save_location_map); // 定位图展示

#endif