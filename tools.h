#ifndef TOOLS_H
#define TOOLS_H

#include <opencv2/opencv.hpp>;
#include <fstream>;
#include <iostream>;
#include "postprocess.h";

std::pair<double, double> sandy_quantization(cv::Mat img);
void de_moire_tools(cv::Mat& csv);
void fft_shift_tool(cv::Mat& src, cv::Mat& dst);
bool tools_DeM(cv::Mat img, std::vector<std::vector<double>> tools_setting, std::string name);
double tools_DeM_outer(cv::Mat img, std::vector<std::vector<double>> tools_setting, std::string name);
std::string get_time();
double var(cv::Mat img);
class SimpleLog {
    static std::ofstream fs_;
    static std::streambuf* old_cout_;
    static std::streambuf* old_cerr_;
public:
    static void init(const char* file = "Pre_log.txt");
    static void close();
};

#endif
