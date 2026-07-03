#ifndef POSTPROCESS_H
#define POSTPROCESS_H

# include <string>
# include <opencv2/opencv.hpp>
# include <iostream>
# include <fstream>
# include <iomanip>

void csv_flip(cv::Mat& csv);
void hole_fill(cv::Mat& csv);
void corner_fill(cv::Mat& csv);
double moire_detector_2(cv::Mat& csv);
void de_moire(cv::Mat& csv, double threshold_coef, double blur_strength);
void de_moire_low(cv::Mat& csv, double threshold_coef, double blur_strength);
void curved_corr(cv::Mat& csv, double curved_pixels_num, double curved_pixels_coef, std::string color);
void curved_corr_corner(cv::Mat& csv, double curved_pixels_num, double curved_pixels_coef, std::string color);
void curved_corr_corner_1(cv::Mat& csv, double curved_pixels_num, double curved_pixels_coef, std::string color);
void curved_corr_corner_double_csv(cv::Mat& csv, cv::Mat& csv_curved, double curved_pixels_num, double curved_pixels_coef, double bright_line_coef, std::string color);
void roundedRectFill(cv::Mat& img, cv::Rect rect, int radius, cv::Scalar color);
void roundedRectOutline(cv::Mat& img, cv::Rect rect, int radius, cv::Scalar color, int thickness);
double getTopPercentThresh(cv::Mat src, float percent = 0.2f);
void csv_saving(cv::Mat mat, std::string filename, int precision);
void roi_saving(cv::Mat mat, std::string filename);
void multi_process_csv_saving(cv::Mat mat, std::string filename, int precision);

#endif