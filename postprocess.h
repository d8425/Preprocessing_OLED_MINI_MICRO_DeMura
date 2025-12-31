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
void de_moire(cv::Mat& csv);

void csv_saving(cv::Mat mat, std::string filename, int precision);
void roi_saving(cv::Mat mat, std::string filename);
void multi_process_csv_saving(cv::Mat mat, std::string filename, int precision);

#endif