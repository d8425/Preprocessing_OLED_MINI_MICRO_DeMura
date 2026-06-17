#include "dect.h"

cv::Mat dect_aoi(cv::Mat csv) {
	cv::Mat img;
	csv.convertTo(img, CV_32F);

	// Normalize pixel values
	cv::Mat normalized_image;
	// max 2% as base
	img = img.reshape(1, 1);
	std::vector<float> img_line;
	img.copyTo(img_line);
	sort(img_line.begin(), img_line.end());
	int targetIdx = static_cast<int>(img_line.size() * 2.0 / 100.0);


	img.convertTo(normalized_image, CV_32F, 1.0 / 255.0);

	// Resize image
	resize(normalized_image, normalized_image, cv::Size(640, 640));

	cv::Scalar mean(0, 0, 0);
	cv::Scalar std(0.5, 0.5, 0.5);

	normalized_image = (normalized_image - mean) / std;

	//224 for resnet 640 for yolov8
	cv::Mat inputBlob = cv::dnn::blobFromImage(normalized_image, 1.0, cv::Size(), cv::Scalar(), true, false);

	cv::dnn::Net net = cv::dnn::readNetFromONNX("best.onnx");
	net.setInput(inputBlob);

	cv::Mat result = net.forward();
	cv::Point classIdPoint;
	return result;
}