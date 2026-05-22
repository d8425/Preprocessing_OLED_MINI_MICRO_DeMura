// demura tools function
#include "tools.h"

std::ofstream SimpleLog::fs_;
std::streambuf* SimpleLog::old_cout_ = nullptr;
std::streambuf* SimpleLog::old_cerr_ = nullptr;

void SimpleLog::init(const char* file) {
	if (fs_.is_open()) return;
	fs_.open(file, std::ios::app);
	old_cout_ = std::cout.rdbuf();
	old_cerr_ = std::cerr.rdbuf();
	std::cout.rdbuf(fs_.rdbuf());
	std::cerr.rdbuf(fs_.rdbuf());
}

void SimpleLog::close() {
	if (!fs_.is_open()) return;
	std::cout.rdbuf(old_cout_);
	std::cerr.rdbuf(old_cerr_);
	fs_.close();
}

std::string get_time() {
	auto now = std::chrono::system_clock::now();
	std::time_t t = std::chrono::system_clock::to_time_t(now);

	struct tm local_tm = { 0 };

	errno_t err = localtime_s(&local_tm, &t);
	if (err != 0) {
		return "fail to get current time";
	}

	std::ostringstream oss;
	/*oss << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S: ");*/
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		now.time_since_epoch()) % 1000;

	oss << std::put_time(&local_tm, "%H:%M:%S")
		<< '.' << std::setfill('0') << std::setw(3) << ms.count()
		<< ": ";

	return oss.str();
}

bool tools_DeM(cv::Mat img, std::vector<std::vector<double>> tools_setting, std::string name) {
	if (tools_setting[0][0] != 0) { // sandy quantization
		std::pair<double, double> temp = sandy_quantization(img);
		double sq_high = temp.first;
		double sq_middle = temp.second;
		std::cout << name << ": high freq. Sandy quantization:" << sq_high << std::endl;
		std::cout << name << ": middle freq. Sandy quantization:" << sq_middle << std::endl;
		return true;
	}
	if (tools_setting[1][0] != 0) { // csv demoire
		std::cout << name << ": de-moire:" << name << std::endl;
		//bool if_moire_exist = moire_detector_2(img);
		if (1) {
			//de_moire(img, tools_setting[1][1], tools_setting[1][2]);
			de_moire_low(img, tools_setting[1][1], tools_setting[1][2]);
		}
		std::string names = ".//"+ name +".csv";
		csv_saving(img, names, 6);
		return true;
	}
	if (tools_setting[2][0] != 0) { // moire quantization
		std::cout << name << ": moire quantization:" << name << std::endl;
		double moire_quantization_val = moire_detector_2(img);
		std::cout << name << ": moire quantizaion value:" << moire_quantization_val << std::endl;
		return true;
	}
	return false;
}

double tools_DeM_outer(cv::Mat img, std::vector<std::vector<double>> tools_setting, std::string name) {
	if (tools_setting[0][0] != 0) { // sandy quantization
		std::pair<double, double> temp = sandy_quantization(img);
		double sq_high = temp.first;
		double sq_middle = temp.second;
		std::cout << name << ": high freq. Sandy quantization:" << sq_high << std::endl;
		std::cout << name << ": middle freq. Sandy quantization:" << sq_middle << std::endl;
		return sq_high;
	}
	if (tools_setting[1][0] != 0) { // csv demoire
		std::cout << name << ": de-moire:" << name << std::endl;
		de_moire(img, tools_setting[1][1], tools_setting[1][2]);
		std::string names = ".//" + name + ".csv";
		csv_saving(img, names, 6);
		return 0;
	}
	return 1;
}

std::pair<double, double> sandy_quantization(cv::Mat img) {
	img.convertTo(img, CV_32F);
	cv::Scalar mean_val = cv::mean(img);
	img = img / mean_val[0];

	// CALCULATION 1 1*1 DIFF
	cv::Mat sample1(
		(img.rows + 1) / 2,    // 采样后行数：原行数向上取整除以2（如5行→3行，4行→2行）
		(img.cols + 1) / 2,    // 采样后列数：原列数向上取整除以2
		img.type(),            // 保持数据类型一致
		img.data,              // 原矩阵数据起始指针
		img.step * 2           // 行步长×2（隔行），列步长由type自动推导（隔列通过列数+步长实现）
	);
	cv::Mat sample2(
		img.rows / 2,          // 原行数除以2（如5行→2行，4行→2行）
		(img.cols + 1) / 2,
		img.type(),
		img.data + img.step,   // 指针偏移1行（对应MATLAB的第2行）
		img.step * 2
	);
	cv::Mat sample3(
		(img.rows + 1) / 2,
		img.cols / 2,
		img.type(),
		img.data + img.elemSize(),  // 指针偏移1列（elemSize()是单个元素字节数）
		img.step * 2
	);
	cv::Mat sample4(
		img.rows / 2,
		img.cols / 2,
		img.type(),
		img.data + img.step + img.elemSize(),
		img.step * 2
	);
	double val1 = var(sample1) + var(sample2) + var(sample3) + var(sample4);

	// CALCULATION 2 3*3 DIFF
	float kernel_f[] = {
		0.05f,0.2f,0.05f,
		0.2f,-1.0f,0.2f,
		0.05f,0.2f,0.05f
	};
	cv::Mat kernel(3, 3, CV_32F, kernel_f);
	cv::Mat img_conved;
	cv::filter2D(img, img_conved, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	cv::Scalar mean_val1 = cv::mean(cv::abs(img_conved));
	double high_val = 1 - (val1 + mean_val1[0]) / 2;

	// CALCULATION 3 5+ DIFF
	int target_r, target_c;
	target_r = int(img.rows/4);
	target_c = int(img.cols/4);
	cv::Mat img_resized;
	cv::resize(img, img_resized, cv::Size(target_c, target_r), 0, 0, cv::INTER_LINEAR);
	cv::Scalar mean_val2;
	mean_val2 = cv::mean(img_resized);
	cv::Scalar val3 = cv::mean(cv::abs(img_resized - mean_val2[0]));
	double middle_val = 1-val3[0];

	return { high_val ,middle_val };
}

double var(cv::Mat img) {
	cv::Scalar mu = cv::mean(img);
	cv::Mat diff;
	cv::subtract(img, mu[0], diff);
	return cv::mean(diff.mul(diff))[0];
}

void de_moire_tools(cv::Mat& csv) {
	cv::Mat img_fft;
	cv::dft(csv, img_fft, CV_HAL_DFT_COMPLEX_OUTPUT);
	//split
	cv::Mat planes[2];
	cv::split(img_fft, planes);

	//calculation abs
	cv::Mat img_fft_abs;
	cv::magnitude(planes[0], planes[1], img_fft_abs);
	img_fft_abs /= 10;
	img_fft_abs /= 10;

	//fftshift
	cv::Mat img_fft_shift;
	fft_shift_tool(img_fft_abs, img_fft_shift);

	//de-center
	int row, col, mid_row, mid_col, step_row, step_col; // row and col, center row and center col, min search step
	row = int(img_fft_shift.rows);
	col = int(img_fft_shift.cols);
	mid_row = int(row / 2);
	mid_col = int(col / 2);
	step_row = int(row / 8); // changeable
	step_col = int(col / 8);

	cv::Rect center_rect(mid_col - step_col, mid_row - step_row, 2 * step_col, 2 * step_row);
	img_fft_shift(center_rect) = cv::Scalar::all(0);

	//img erode
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)); // changeable
	cv::erode(img_fft_shift, img_fft_shift, kernel);

	//bin
	double threshold;
	cv::minMaxLoc(img_fft_shift, NULL, &threshold, NULL, NULL);
	threshold *= 0.2; //changeable
	cv::Mat mask;
	cv::threshold(img_fft_shift, mask, threshold, 1, cv::THRESH_BINARY);

	//Gass and normalization
	cv::GaussianBlur(mask, mask, cv::Size(5, 5), 1);
	mask = 1 - mask;
	cv::pow(mask, 5, mask); //幂次增强对比度
	//mask ifftshift
	cv::Mat mask_ifft;
	fft_shift_tool(mask, mask_ifft);
	//mask * img
	cv::Mat img_fft_final, mask_ifft_2c;
	cv::Mat mask_ifft_2c_planes[] = { mask_ifft ,mask_ifft };
	cv::merge(mask_ifft_2c_planes, 2, mask_ifft_2c);
	cv::multiply(img_fft, mask_ifft_2c, img_fft_final);

	//ifft
	cv::Mat img_ifft;
	cv::idft(img_fft_final, img_ifft, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
	img_ifft.copyTo(csv);
}

void fft_shift_tool(cv::Mat& src, cv::Mat& dst) {
	dst = src.clone();
	int cx = dst.cols / 2;
	int cy = dst.rows / 2;

	// 分割四个象限
	cv::Mat q1(dst, cv::Rect(0, 0, cx, cy));    // 左上
	cv::Mat q2(dst, cv::Rect(cx, 0, cx, cy));   // 右上
	cv::Mat q3(dst, cv::Rect(0, cy, cx, cy));   // 左下
	cv::Mat q4(dst, cv::Rect(cx, cy, cx, cy));  // 右下

	// 交换象限：左上↔右下，右上↔左下（实现中心对齐）
	cv::Mat temp;
	q1.copyTo(temp);
	q4.copyTo(q1);
	temp.copyTo(q4);

	q2.copyTo(temp);
	q3.copyTo(q2);
	temp.copyTo(q3);
}