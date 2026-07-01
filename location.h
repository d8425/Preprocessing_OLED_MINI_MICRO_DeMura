#ifndef LOCATION_H
#define LOCATION_H

# include <opencv2/opencv.hpp>
# include <string>
# include <iostream>
# include <cmath>
# include <vector>
# include "tools.h"

void get_map(const cv::Mat& location_map, cv::Mat& mapx, cv::Mat& mapy, double& mapping, int panel_res_rows, int panel_res_cols, std::string color, int is_save_location_map, std::array<int,6> location_setting_pixels, bool is_subimage_W, std::string camera_grab_type);
#endif