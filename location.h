#ifndef LOCATION_H
#define LOCATION_H

# include <opencv2/opencv.hpp>
# include <string>
# include <iostream>
# include <cmath>
# include <vector>

void get_map(const cv::Mat& location_map, cv::Mat& mapx, cv::Mat& mapy, double& mapping, int panel_res_rows, int panel_res_cols, std::string color, int is_save_location_map, std::array<int, 3> location_setting_pixels);

#endif