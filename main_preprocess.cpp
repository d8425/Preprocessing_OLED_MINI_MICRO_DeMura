#include "postprocess.h";
#include "preprocess.h"


using namespace std;

cv::Mat img = readImage("F:\\ITEMS\\chunyang\\CX1-2-G\\PIEs_AJUN_ORB_W192_green.csv", "csv");
de_moire(img);
