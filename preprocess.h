#ifndef PREPROCESS_H
#define PREPROCESS_H

# include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
# include <opencv2/opencv.hpp>
# include <vector>
#include <opencv2/core/utils/logger.hpp>

cv::Mat readImage(const std::string& path, std::string);
std::vector<float> parse2Float(const std::string& str);
void img_flip(cv::Mat& img, int position);
std::map<std::string, std::map<std::string, std::string>> read_ini(const std::string& filename);
void string_split(std::string line, std::vector<std::string>& list, char symbol);
void string_split_num(std::string line, std::vector<double>& list, char symbol);
void color2mono(cv::Mat img, std::vector<cv::Mat>& imgc);
cv::Mat preprocess4color_location_map(cv::Mat location_map, std::string single_color);
void img_calibration(cv::Mat& img, cv::Mat& ffc_calibration_coef,  int img_idx);

#endif