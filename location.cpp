# include "location.h"


cv::Mat _fourier_t(const cv::Mat location_map_mono_f32) {
	cv::Mat f_map;

	cv::dft(location_map_mono_f32, f_map, cv::DFT_COMPLEX_OUTPUT);

	cv::Mat magnitude;  // 幅值矩阵（单通道）
	cv::Mat planes[2];  // 用于分离实部和虚部

	cv::split(f_map, planes);

	// magnitude = sqrt(实部² + 虚部²)
	cv::magnitude(planes[0], planes[1], magnitude);

	//shift
	cv::Mat f_map_shift = cv::Mat::zeros(magnitude.size(), magnitude.type());
	int rows = magnitude.rows;
	int cols = magnitude.cols;
	int mid_rows = rows / 2;
	int mid_cols = cols / 2;


	magnitude(cv::Range(0, mid_rows), cv::Range(0, mid_cols)).copyTo(f_map_shift(cv::Range(mid_rows, rows), cv::Range(mid_cols, cols)));
	magnitude(cv::Range(0, mid_rows), cv::Range(mid_cols, cols)).copyTo(f_map_shift(cv::Range(mid_rows, rows), cv::Range(0, mid_cols)));
	magnitude(cv::Range(mid_rows, rows), cv::Range(0, mid_cols)).copyTo(f_map_shift(cv::Range(0, mid_rows), cv::Range(mid_cols, cols)));
	magnitude(cv::Range(mid_rows, rows), cv::Range(mid_cols, cols)).copyTo(f_map_shift(cv::Range(0, mid_rows), cv::Range(0, mid_cols)));

	f_map_shift = f_map_shift / 100000;

	return f_map_shift;
}

double _get_mapping(const cv::Mat& f_map_shift) {
	int rows = f_map_shift.rows;
	int cols = f_map_shift.cols;
	cv::Mat row_roi;
	cv::Mat col_roi;

	// 剔除中心区域 (剔除中心1/3处)
	//int part = 8; //将图像横纵分为n块
	//f_map_shift(cv::Range(0, int(rows/ part)), cv::Range((cols/2)-500, (cols / 2) + 500)).copyTo(row_roi);  // 先验
	//f_map_shift(cv::Range((rows/2)-500, (rows / 2) + 500), cv::Range(0,int(cols/ part))).copyTo(col_roi);   // 先验

	//f_map_shift(0,0,rows,cols/2-50).copyTo(row_roi);  // 先验
	//f_map_shift(cv::Range((rows / 2) - 500, (rows / 2) + 500), cv::Range(0, int(cols / part))).copyTo(col_roi);   // 先验

	//f_map_shift(cv::Range((rows / 2) - 50, (rows / 2) + 50), cv::Range((cols / 2) - 50, (cols / 2) + 50)) = 0;
	f_map_shift(cv::Range(0, rows), cv::Range(0, (cols / 2) - 50)).copyTo(row_roi);
	f_map_shift(cv::Range(0, rows/2-50), cv::Range(0, cols)).copyTo(col_roi);

	// max brightness area
	double unused;
	cv::Point row_roi_max_loc;
	cv::Point col_roi_max_loc;

	cv::minMaxLoc(row_roi, &unused, nullptr, nullptr, &row_roi_max_loc);
	cv::minMaxLoc(col_roi, &unused, nullptr, nullptr, &col_roi_max_loc);

	// 欧式距离计算
	double diff_point_center_row_1 = row_roi_max_loc.y - int(rows / 2);
	double diff_point_center_col_1 = row_roi_max_loc.x - int(cols / 2);
	double diff_point_center_row_2 = col_roi_max_loc.y - int(rows / 2);
	double diff_point_center_col_2 = col_roi_max_loc.x - int(cols / 2);


	int diff_point_center_1 = pow(pow(diff_point_center_row_1, 2) + pow(diff_point_center_col_1, 2), 0.5);
	int diff_point_center_2 = pow(pow(diff_point_center_row_2, 2) + pow(diff_point_center_col_2, 2), 0.5);

	double mapping1 = double(rows) / diff_point_center_1;
	double mapping2 = double(cols) / diff_point_center_2;
	double mapping = (mapping1 + mapping2) / 2;

	return mapping;
}


cv::Mat _erode(cv::Mat img_ori, const double mapping) {
	// 指针遍历，确保内存连续
	cv::Mat dst;
	cv::Mat dst1;
	cv::Mat img;
	dst.create(img.size(), img.type());
	img.create(img_ori.size(), img_ori.type());
	if (img_ori.isContinuous()) {
		// continuous
		img = img_ori;
	}
	else {
		// non continuous
		img = img_ori.clone();
	}
	
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(mapping, mapping));

	dilate(img, dst, kernel); // 膨胀操作等价于最大值滤波

	//最大值差异寻找mask-m2
	cv::Mat mask;
	mask.create(img.size(), img.type());
	img -= 10;  //外部噪声的先验阈值 changeable
	dst -= 10;
	cv::Mat zero_img = cv::Mat::zeros(img.size(), img.type());
	cv::compare(img, zero_img, mask, cv::CMP_EQ);
	cv::compare(img, dst, dst1, cv::CMP_EQ);
	cv::subtract(dst1, mask, dst1);
	return dst1;
}

cv::Mat _erode1(cv::Mat img_ori, const double mapping, std::string color) {
	// 指针遍历，确保内存连续
	cv::Mat dst;
	cv::Mat dst1;
	cv::Mat img;
	dst.create(img.size(), img.type());
	img.create(img_ori.size(), img_ori.type());
	if (img_ori.isContinuous()) {
		// continuous
		img = img_ori;
	}
	else {
		// non continuous
		img = img_ori.clone();
	}


	// 由于出现多点最大值相同导致单像素多特征点，进行腐蚀
	if (color == "R" || color == "B") {
		cv::Mat kernel_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::erode(img, img, kernel_erode);
	}


	//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(mapping+1, mapping+1)); // rect
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(mapping+1, mapping+1)); // circle

	dilate(img, dst, kernel); // 膨胀操作等价于最大值滤波

	//最大值差异寻找mask-m2
	cv::Mat mask,mask1;
	mask.create(img.size(), img.type());
	mask1.create(img.size(), img.type());
	img -= 2560;  //外部噪声的先验阈值(8bit:10,12bit:160,16bit:2560) changeable
	dst -= 2560;
	cv::Mat zero_img = cv::Mat::zeros(img.size(), img.type());

	mask = img.clone();
	cv::compare(img, zero_img, mask, cv::CMP_EQ);
	cv::compare(img, dst, mask1, cv::CMP_EQ);
	cv::subtract(mask1, mask, mask1);

	return mask1;
}


cv::Mat _solo_pixels_color(const cv::Mat location_map_mono, const double mapping, std::string color) {
	// slol_pixels for color camera
	// create kernel
	int kernel_length = ceil(mapping * 2);
	if (kernel_length % 2 == 0) {
		kernel_length += 1;
	}
	//cv::Mat _kernel = cv::Mat::zeros(int(kernel_length), int(kernel_length), CV_32F);
	//double kernel_data = (1 / double(kernel_length * kernel_length - 1));
	//_kernel.setTo(-kernel_data);
	//_kernel.ptr<float>(int(kernel_length/2))[int(kernel_length/2)] = 1.0f;

	// convlution
	cv::Mat img_feature_1;
	cv::Mat img_feature_2;
	cv::Mat img_blur;
	cv::Mat img_feature_3;
	cv::Mat img_feature_4;


	//cv::filter2D(location_map_mono, img_feature_1, -1, _kernel);
	//cv::GaussianBlur(location_map_mono, img_blur, cv::Size(11,11), 2);

	//cv::Mat img_feature_2 = img_feature_1 - 0.5*img_blur;
	//img_feature_2 = location_map_mono - img_blur;

	//连通域收缩
	//img_feature_3 = _erode(img_feature_2, mapping);
	img_feature_4 = _erode1(location_map_mono, mapping, color);


	return img_feature_4;
}

cv::Mat _solo_pixels_mono(const cv::Mat location_map_mono, const double mapping, std::string color) {
	// slol_pixels for mono camera
	// create kernel
	int kernel_length = ceil(mapping * 2);
	if (kernel_length % 2 == 0) {
		kernel_length += 1;
	}
	cv::Mat _kernel = cv::Mat::zeros(int(kernel_length), int(kernel_length), CV_32F);
	double kernel_data = (1 / double(kernel_length * kernel_length - 1));
	_kernel.setTo(-kernel_data);
	_kernel.ptr<float>(int(kernel_length / 2))[int(kernel_length / 2)] = 1.0f;

	// convlution
	//cv::Mat img_feature_1;
	cv::Mat img_feature_2;
	cv::Mat img_blur;
	cv::Mat img_feature_3;
	cv::Mat img_feature_4;


	//cv::filter2D(location_map_mono, img_feature_1, -1, _kernel);
	cv::GaussianBlur(location_map_mono, img_blur, cv::Size(11,11), 2);

	//cv::Mat img_feature_2 = img_feature_1 - 0.5*img_blur;
	img_feature_2 = location_map_mono - img_blur;

	//连通域收缩
	img_feature_3 = _erode(img_feature_2, mapping);
	//img_feature_4 = _erode1(location_map_mono, mapping, color);


	return img_feature_3;
}

void _find_first_pixel(cv::Mat solo_pixels_mat_ori, int& x, int& y) {
	x = 0;
	y = 0;
	//255->1
	cv::Mat solo_pixels_mat;
	solo_pixels_mat.create(solo_pixels_mat_ori.size(), solo_pixels_mat_ori.type());
	cv::threshold(solo_pixels_mat_ori, solo_pixels_mat, 0, 1, cv::THRESH_BINARY);

	// 自上而下寻找第一个像素
	int row = solo_pixels_mat.rows;
	cv::Mat row_mat;
	cv::reduce(solo_pixels_mat, row_mat, 1, cv::REDUCE_SUM, CV_32S);

	// 寻找第一个点
	int record = 0;
	int	idx = 0;

	while (record <= 5) { // record<= n,n为筛选阈值，提升n可以提高鲁棒性
		record = row_mat.at<int>(idx,0);
		idx++;
	}
	y = idx - 1; // 横坐标

	// 第一个点所在行
	cv::Mat line = solo_pixels_mat_ori.row(y);
	// 寻找非0点索引
	cv::Mat non_location;
	cv::findNonZero(line, non_location);

	int idx_x = non_location.rows/2;
	x = non_location.at<int>(idx_x,0);
}


// 迭代指针:left or right
uchar* get_iter_ptr(uchar* current_ptr, int& x, int& y, int step, int elem_size,
	int cols, int search_step,int idx,std::string color) {

	// 计算目标x坐标
	x += search_step;
	// 边界检查
	if (x < 0) {
		return nullptr; // 越界返回空指针
	}

	uchar* new_ptr = 0;
	// RGGB中R or B使用菱形排列，需要向左上、左下、右上或右下进行选择
	if (color == "R" || color == "B") {
		if (idx % 2 == 0) {
			new_ptr = current_ptr + search_step * elem_size + abs(search_step) * step;
			y += abs(search_step);
		}
		else {
			new_ptr = current_ptr + search_step * elem_size - abs(search_step) * step;
			y -= abs(search_step);
		}
	}
	else {
		// 计算左/右侧像素指针（x方向偏移：-search_step * 像素字节数） - 全排
		new_ptr = current_ptr + search_step * elem_size;
	}
	

	return new_ptr;
}

// 迭代指针:down
uchar* get_iter_ptr_down(uchar* current_ptr, int& x, int& y, int step, int elem_size,
	int cols, int search_step,std::string color) {
	if (color == "R" || color == "B") {
		search_step *= 2;
	}
	// 计算目标x坐标
	y += search_step;
	// 边界检查
	if (y < 0) {
		return nullptr; // 越界返回空指针
	}
	// 计算下侧像素指针（y方向偏移：search_step * 像素字节数）
	// 全排 or G画面使用仅向下移动一格
	uchar* new_ptr = current_ptr + search_step * step;

	return new_ptr;
}

bool find_first_non_zeroinROI_center(
	uchar*& center_ptr,
	int& x, int& y,
	int step,
	int elem_size,
	int rows, int cols,
	int n
) {
	if (n <= 0 || n % 2 == 0) {
		std::cerr << "n必须是正奇数（如3,5,7...）" << std::endl;
		return false;
	}

	int half = n / 2;

	// 按距离中心的远近分层遍历
	for (int dist = 0; dist <= half; ++dist) {
		// 遍历当前距离层（曼哈顿距离或欧几里得距离）
		for (int dy = -dist; dy <= dist; ++dy) {
			for (int dx = -dist; dx <= dist; ++dx) {
				// 只处理当前距离层的边界点（距离恰好为dist）
				if (abs(dx) == dist || abs(dy) == dist) {
					int nx = x + dx;
					int ny = y + dy;

					if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
						uchar* pixel_ptr = center_ptr + dy * step + dx * elem_size;
						if (*pixel_ptr != 0) {
							x = nx;
							y = ny;
							center_ptr = pixel_ptr;
							return true;
						}
					}
				}
			}
		}
	}
	return false;
}

bool find_first_non_zeroinROI(
	uchar*& center_ptr,  // 目前像素指针-可迭代用于外部
	int& x, int& y,       // 目前像素坐标-可迭代用于外部
	int step,           // 图像行字节数
	int elem_size,      // 单个像素字节数
	int rows, int cols, // 图像尺寸
	int n               // ROI尺寸半尺寸（n×n，必须为奇数）
) {

	int nx = 0;
	int ny = 0;
	// 检查n的合法性（必须是正奇数）
	if (n <= 0 || n % 2 == 0) {
		std::cerr << "n必须是正奇数（如3,5,7...）" << std::endl;
		return false;
	}

	// 计算半宽（从中心到边缘的距离）
	int half = n/2;

	// 按从上到下、从左到右的顺序遍历n×n范围内的像素
	for (int dy = -half; dy <= half; ++dy) {    // 行方向偏移（-half到+half）
		for (int dx = -half; dx <= half; ++dx) { // 列方向偏移（-half到+half）
			// 计算邻域像素的绝对坐标
			nx = x + dx;
			ny = y + dy;

			// 边界检查：确保在图像范围内
			if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
				// 计算邻域像素的指针
				uchar* pixel_ptr = center_ptr + dy * step + dx * elem_size;

				// 检查像素值是否非0（根据实际数据类型调整判断方式）
				if (*pixel_ptr != 0) {
					// 坐标迭代
					x = nx;
					y = ny;
					// 指针迭代
					center_ptr = pixel_ptr;
					return true; // 找到第一个非0值，立即返回
				}
			}
		}
	}
	// 未找到非0值
	return false;
}

uchar* get_ptr_U8(cv::Mat img, int x, int y) {
	return img.data + y * img.step + x;
}

float* get_ptr_F32(cv::Mat img, int x, int y) { // 这里使用的指针是float类型，外部不能使用图像自身step和elemSize进行指针迭代，需除4
	return reinterpret_cast<float*>(img.data + y * img.step) + x;
}

void _loop_all_pixles(cv::Mat solo_pixels_mat, int x, int y, cv::Mat& mapx, cv::Mat& mapy, double mapping,double panel_res_rows, double panel_res_cols,std::string color, std::array<int,6> location_setting_pixels) {
	//out-location setting
	//location_tolerate_pixels[0] -> 横向，[1]纵向

	//loop step
	//int search_step = 0;
	//if (mapping > 8) {
	//	search_step = round(mapping);
	//}if (mapping > 6) {
	//	search_step = 7;
	//}if (mapping > 4) {
	//	search_step = 5;
	//}if (mapping > 3) {
	//	search_step = 3;
	//}
	int search_step = round(mapping); // 探索步进

	int search_size = 0;
	if (mapping > 5) {
		search_size = 9; // 搜索区域大小
	}
	else {
		search_size = 7;
	}
		
	

	//temp inital
	int current_x_left = x;
	int current_y_left = y;
	int current_x_right = x;
	int current_y_right = y;
	//int x_new, int y_new; // 迭代
	uchar* ptr_left = get_ptr_U8(solo_pixels_mat, x, y);
	uchar* ptr_right = get_ptr_U8(solo_pixels_mat, x, y);
	//stable info
	int step = solo_pixels_mat.step;
	int elem_size = solo_pixels_mat.elemSize();
	int cols = solo_pixels_mat.cols;
	int rows = solo_pixels_mat.rows;
	//map inital
	//mapx.create(panel_res_cols*2, panel_res_rows*2, CV_32F);
	//mapy.create(panel_res_cols*2, panel_res_rows*2, CV_32F);
	int elem_size_map = mapx.elemSize();
	int step_map = mapx.step;
	//record idx
	int record_x = mapx.cols / 2;
	int record_y = 0;
	float* ptr_map_x = get_ptr_F32(mapx, record_x, record_y); //可迭代的map指针
	float* ptr_map_y = get_ptr_F32(mapy, record_x, record_y);

	//中心line起始点
	float* mid_map_x_ptr = ptr_map_x; //用于中心line下向步进的map指针，仅在下向步进时迭代
	float* mid_map_y_ptr = ptr_map_y;
	int mid_map_x = x; //用于中心line下向步进的xy坐标，仅在下向步进时迭代
	int mid_map_y = y;
	uchar* mid_line_ptr;

	//黑点记录标识符
	int pixels_is_empty_num = 0;

	for (int r = 0; r < panel_res_cols; r++) {
		//下向步进
		//map第一个值初始化赋值(仅在每次向下步进时进行)
		*mid_map_x_ptr = mid_map_x;
		*mid_map_y_ptr = mid_map_y;

		//左向loop
		//起始时初始化map指针
		ptr_map_x = mid_map_x_ptr;
		ptr_map_y = mid_map_y_ptr;
		for (int left_idx = 0; left_idx < panel_res_rows; left_idx++) {

			//下一个左边的中心点指针并迭代坐标
			ptr_left = get_iter_ptr(ptr_left, current_x_left, current_y_left, step, elem_size, cols, -search_step, left_idx, color); //由于是左向，故step为负

			//下个ROI第一个非0值并迭代坐标与指针迭代
			bool is_iterable = find_first_non_zeroinROI_center(ptr_left, current_x_left, current_y_left, step, elem_size, rows, cols, search_size); //迭代x,y值，并返回是否ok

			//记录
			ptr_map_x -= elem_size; //左向减法
			ptr_map_y -= elem_size; //左向减法

			//条件
			if (ptr_left == nullptr || is_iterable==false) { //退出条件：1.空指针 2.搜索1次以上无ROI点
				//增强鲁棒性-未拿到时进行跳跃搜索
				pixels_is_empty_num += 1; // 容纳4个像素黑点，向后搜索
				if (pixels_is_empty_num > location_setting_pixels[0]) {
					break;
				}
			}
			else { //is_iterable=true时才赋值，但是每次都会迭代map指针
				pixels_is_empty_num = 0;
				*ptr_map_x = current_x_left;
				*ptr_map_y = current_y_left;
			}
		}

		//右向loop
		//起始时初始化map指针
		ptr_map_x = mid_map_x_ptr;
		ptr_map_y = mid_map_y_ptr;
		for (int right_idx = 0; right_idx < int(panel_res_rows); right_idx++) {

			//下一个左边的中心点指针并迭代坐标
			ptr_right = get_iter_ptr(ptr_right, current_x_right, current_y_right, step, elem_size, cols, search_step, right_idx, color); //由于是右向，故step为正

			//下个ROI第一个非0值并迭代坐标与指针迭代
			bool is_iterable = find_first_non_zeroinROI_center(ptr_right, current_x_right, current_y_right, step, elem_size, rows, cols, search_size); //迭代x,y值，并返回是否ok

			//记录
			ptr_map_x += elem_size; //左向减法
			ptr_map_y += elem_size; //左向减法

			//条件
			if (ptr_right == nullptr || is_iterable == false) { //退出条件：1.空指针 2.搜索2次以上无ROI点
				//增强鲁棒性-未拿到时进行跳跃搜索
				pixels_is_empty_num += 1; 
				if (pixels_is_empty_num > location_setting_pixels[0]) { // 容纳4个像素黑点，向后搜索
					break;
				}
			}
			else { //is_iterable=true时才赋值，但是每次都会迭代map指针
				pixels_is_empty_num = 0;
				*ptr_map_x = current_x_right;
				*ptr_map_y = current_y_right;
			}
		}

		//下向步进
		//xy坐标迭代+原图指针迭代
		mid_line_ptr = get_ptr_U8(solo_pixels_mat, mid_map_x, mid_map_y);
		//坐标重置
		current_x_right = mid_map_x;
		current_y_right = mid_map_y;

		mid_line_ptr = get_iter_ptr_down(mid_line_ptr, current_x_right, current_y_right, step, elem_size, cols, search_step, color);
		//下个ROI第一个非0值并迭代坐标与指针迭代
		bool is_iterable = find_first_non_zeroinROI_center(mid_line_ptr, current_x_right, current_y_right, step, elem_size, rows, cols, search_size); //迭代x,y值，并返回是否ok
		
		if (mid_line_ptr == nullptr || is_iterable == false) {
			//增强鲁棒性-未拿到时进行跳跃搜索
			pixels_is_empty_num += 1;
			if (pixels_is_empty_num > location_setting_pixels[1]) { // 容纳总共4个像素黑点，向后搜索
				break;
			}
		}

		ptr_left = ptr_right = mid_line_ptr;
		//左向+右向迭代完成
		//mid_map_x+mid_map_y临时参数迭代
		mid_map_x = current_x_right;
		mid_map_y = current_y_right;

		//左向指针同步
		current_x_left = current_x_right;
		current_y_left = current_y_right;

		//map指针与xy迭代
		record_y += 1; // 全排是1，如果非全排也在后面处理，此处只做base
		mid_map_x_ptr = get_ptr_F32(mapx, record_x, record_y);
		mid_map_y_ptr = get_ptr_F32(mapy, record_x, record_y);

		//可能存在下向搜索时黑点影响，后期测试--------------------------------------------------------------------------------------------------
	}
}

void _loop_all_pixles_curve(cv::Mat solo_pixels_mat, int x, int y, cv::Mat& mapx, cv::Mat& mapy, double mapping, double panel_res_rows, double panel_res_cols, std::string color, std::array<int, 4> location_setting_pixels) {
	//out-location setting
	//location_tolerate_pixels[0] -> 横向，[1]纵向

	//loop step
	int search_step = 0;
	if (mapping > 8) {
		search_step = round(mapping);
	}if (mapping >= 6) {
		search_step = 7;
	}if (mapping >= 4) {
		search_step = 5;
	}
	else {
		search_step = 3;
	}
	// [NEW] 保存初始步长，用于找到点后恢复
	int initial_search_step = search_step;
	// [NEW] 定义步长上限，防止无限增大
	int max_search_step = initial_search_step * 2;
	//int search_step = round(mapping); // 探索步进
	int search_size = 7; // 搜索区域大小

	//temp inital
	int current_x_left = x;
	int current_y_left = y;
	int current_x_right = x;
	int current_y_right = y;
	//int x_new, int y_new; // 迭代
	uchar* ptr_left = get_ptr_U8(solo_pixels_mat, x, y);
	uchar* ptr_right = get_ptr_U8(solo_pixels_mat, x, y);
	//stable info
	int step = solo_pixels_mat.step;
	int elem_size = solo_pixels_mat.elemSize();
	int cols = solo_pixels_mat.cols;
	int rows = solo_pixels_mat.rows;
	//map inital
	//mapx.create(panel_res_cols*2, panel_res_rows*2, CV_32F);
	//mapy.create(panel_res_cols*2, panel_res_rows*2, CV_32F);
	int elem_size_map = mapx.elemSize();
	int step_map = mapx.step;
	//record idx
	int record_x = mapx.cols / 2;
	int record_y = 0;
	float* ptr_map_x = get_ptr_F32(mapx, record_x, record_y); //可迭代的map指针
	float* ptr_map_y = get_ptr_F32(mapy, record_x, record_y);

	//中心line起始点
	float* mid_map_x_ptr = ptr_map_x; //用于中心line下向步进的map指针，仅在下向步进时迭代
	float* mid_map_y_ptr = ptr_map_y;
	int mid_map_x = x; //用于中心line下向步进的xy坐标，仅在下向步进时迭代
	int mid_map_y = y;
	uchar* mid_line_ptr;

	//黑点记录标识符
	int pixels_is_empty_num = 0;

	for (int r = 0; r < panel_res_cols; r++) {
		//下向步进
		//map第一个值初始化赋值(仅在每次向下步进时进行)
		*mid_map_x_ptr = mid_map_x;
		*mid_map_y_ptr = mid_map_y;

		//左向loop
		//起始时初始化map指针
		ptr_map_x = mid_map_x_ptr;
		ptr_map_y = mid_map_y_ptr;
		for (int left_idx = 0; left_idx < panel_res_rows; left_idx++) {

			//下一个左边的中心点指针并迭代坐标
			ptr_left = get_iter_ptr(ptr_left, current_x_left, current_y_left, step, elem_size, cols, -search_step, left_idx, color); //由于是左向，故step为负

			//下个ROI第一个非0值并迭代坐标与指针迭代
			bool is_iterable = find_first_non_zeroinROI(ptr_left, current_x_left, current_y_left, step, elem_size, rows, cols, search_size); //迭代x,y值，并返回是否ok

			//记录
			ptr_map_x -= elem_size; //左向减法
			ptr_map_y -= elem_size; //左向减法

			//条件
			if (ptr_left == nullptr || is_iterable == false) { //退出条件：1.空指针 2.搜索1次以上无ROI点
				//增强鲁棒性-未拿到时进行跳跃搜索
				pixels_is_empty_num += 1; // 容纳4个像素黑点，向后搜索
				// [NEW] 未找到时，增大步长以跳过空白区域
				search_step = std::min<int>(search_step * 1.2, max_search_step);
				if (pixels_is_empty_num > location_setting_pixels[0]) {
					break;
				}
			}
			else { //is_iterable=true时才赋值，但是每次都会迭代map指针
				pixels_is_empty_num = 0;
				*ptr_map_x = current_x_left;
				*ptr_map_y = current_y_left;
				// [NEW] 找到点时恢复初始步长，保证后续精度
				search_step = initial_search_step;
			}
		}

		//右向loop
		//起始时初始化map指针
		ptr_map_x = mid_map_x_ptr;
		ptr_map_y = mid_map_y_ptr;
		// [NEW] 右向时恢复初始步长，保证后续精度
		search_step = initial_search_step;
		for (int right_idx = 0; right_idx < int(panel_res_rows); right_idx++) {

			//下一个左边的中心点指针并迭代坐标
			ptr_right = get_iter_ptr(ptr_right, current_x_right, current_y_right, step, elem_size, cols, search_step, right_idx, color); //由于是右向，故step为正

			//下个ROI第一个非0值并迭代坐标与指针迭代
			bool is_iterable = find_first_non_zeroinROI(ptr_right, current_x_right, current_y_right, step, elem_size, rows, cols, search_size); //迭代x,y值，并返回是否ok

			//记录
			ptr_map_x += elem_size; //左向减法
			ptr_map_y += elem_size; //左向减法

			//条件
			if (ptr_right == nullptr || is_iterable == false) { //退出条件：1.空指针 2.搜索2次以上无ROI点
				//增强鲁棒性-未拿到时进行跳跃搜索
				pixels_is_empty_num += 1;
				// [NEW] 未找到时，增大步长
				search_step = std::min<int>(search_step * 1.2, max_search_step);
				if (pixels_is_empty_num > location_setting_pixels[0]) { // 容纳4个像素黑点，向后搜索
					break;
				}
			}
			else { //is_iterable=true时才赋值，但是每次都会迭代map指针
				pixels_is_empty_num = 0;
				*ptr_map_x = current_x_right;
				*ptr_map_y = current_y_right;
				// [NEW] 找到时恢复步长
				search_step = initial_search_step;
			}
		}

		//下向步进
		// [NEW] 右向时恢复初始步长，保证后续精度
		search_step = initial_search_step;
		//xy坐标迭代+原图指针迭代
		mid_line_ptr = get_ptr_U8(solo_pixels_mat, mid_map_x, mid_map_y);
		//坐标重置
		current_x_right = mid_map_x;
		current_y_right = mid_map_y;

		mid_line_ptr = get_iter_ptr_down(mid_line_ptr, current_x_right, current_y_right, step, elem_size, cols, search_step, color);
		//下个ROI第一个非0值并迭代坐标与指针迭代
		bool is_iterable = find_first_non_zeroinROI(mid_line_ptr, current_x_right, current_y_right, step, elem_size, rows, cols, search_size); //迭代x,y值，并返回是否ok

		if (mid_line_ptr == nullptr || is_iterable == false) {
			//增强鲁棒性-未拿到时进行跳跃搜索
			pixels_is_empty_num += 1;
			// [NEW] 未找到时，增大步长
			search_step = std::min<int>(search_step * 1.2, max_search_step);
			if (pixels_is_empty_num > location_setting_pixels[1]) { // 容纳总共4个像素黑点，向后搜索
				break;
			}
			//由于存在曲面屏
		}
		else {
			// [NEW] 找到时恢复初始步长
			search_step = initial_search_step;
		}

		ptr_left = ptr_right = mid_line_ptr;
		//左向+右向迭代完成
		//mid_map_x+mid_map_y临时参数迭代
		mid_map_x = current_x_right;
		mid_map_y = current_y_right;

		//左向指针同步
		current_x_left = current_x_right;
		current_y_left = current_y_right;

		//map指针与xy迭代
		record_y += 1; // 全排是1，如果非全排也在后面处理，此处只做base
		mid_map_x_ptr = get_ptr_F32(mapx, record_x, record_y);
		mid_map_y_ptr = get_ptr_F32(mapy, record_x, record_y);

		//可能存在下向搜索时黑点影响，后期测试--------------------------------------------------------------------------------------------------
	}
}

__declspec(noinline) void _line_interpolation(cv::Mat& line) {
	//将line中黑区进行插值填充
	//筛选头尾坐标
	int start_idx = 0;
	int end_idx = 0;
	std::vector<int> start_idx_list;
	std::vector<int> end_idx_list;
	int is_first_up = 0;
	int is_first_down = 0;

	for (int r = 0; r < line.rows; r++) {
		if (r > 0 && r< line.rows-1) {
			//单纯黑点填充
			/*if (line.at<float>(r) < 0 && line.at<float>(r - 1) > 0 && line.at<float>(r + 1) > 0) {
				line.at<float>(r) = std::round((line.at<float>(r-1)+ line.at<float>(r + 1))/2);
			}*/

			if (line.at<float>(r) <= 0 && line.at<float>(r - 1) > 0) {
				start_idx_list.push_back(r - 1);
			}
			if (line.at<float>(r) > 0 && line.at<float>(r - 1) <= 0) {
				end_idx_list.push_back(r); //end_idx 会有多个值，需要的是比start_idx大的值.
			}
		}
	}

	//end_idx select
	if (!end_idx_list.empty()&& !start_idx_list.empty()) {
		for (int idx = 0; idx < end_idx_list.size(); idx++) {
			if (end_idx_list[idx] > start_idx_list[0]) { //认为start_idx_list中第一个点是中心开始点
				end_idx = end_idx_list[idx];
				start_idx = start_idx_list[0];
			}
		}
	}


	//计算y = ax + b
	int val_end = line.at<float>(end_idx);
	int val_start = line.at<float>(start_idx);
	double a = double(val_end - val_start) / double(end_idx - start_idx);
	double b = val_end - a * end_idx;

	//zero 填充(仅中心，外部不纳入)
	for (int r = 0; r < line.rows; r++) {
		if (line.at<float>(r) <= 0 && r> start_idx && r < end_idx) {
			line.at<float>(r) = static_cast<int>(std::round(a * r + b));
		}
	}
}

cv::Mat _line_interpolation_inner(cv::Mat line) {
	//将line中黑区进行插值填充-仅做内插
	std::vector<cv::Point> nonZeroPoints;
	cv::findNonZero(line, nonZeroPoints);
	for (int ind = 0; ind < (int)nonZeroPoints.size() - 1; ind++) {
		// 多个非0点遍历
		int pos1 = nonZeroPoints[ind].y;
		int pos2 = nonZeroPoints[ind + 1].y;
		int diff_between_two_point = pos2 - pos1;
		int val1 = line.at<int>(pos1,0);
		int val2 = line.at<int>(pos2, 0);
		for (int ind_s = 1; ind_s < diff_between_two_point; ind_s++) {
			// 索引n处非0点与n+1处非0点，空值遍历赋值
			float ratio = (float)ind_s / diff_between_two_point;
			line.at<int>(pos1 + ind_s,0) = (int)(val1 * (1.0f - ratio) + val2 * ratio);
		}
	}
	return line;
}

void _post_process_for_map(cv::Mat solo_pixels_mat, cv::Mat& mapx, cv::Mat& mapy, int panel_res_rows, int panel_res_cols, int interval) {
	//初始化ROI坐标
	int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
	cv::Mat map8U;
	mapx.convertTo(map8U, CV_8UC1);
	//最小内切矩形选取
	cv::Mat row_mat;
	cv::Mat col_mat;
	cv::reduce(map8U, row_mat, 1, cv::REDUCE_SUM, CV_32F);
	cv::reduce(map8U, col_mat, 0, cv::REDUCE_SUM, CV_32F);
	for (y1 = 0; y1 < row_mat.rows; y1++) {
		int record = row_mat.at<float>(y1, 0);
		if (record == 0) {
			break;
		}
	}
	for (int c = 0; c < col_mat.cols; c++) {
		if (c > 0) {
			int record = col_mat.at<float>(c);
			if (record > 0 && col_mat.at<float>(0, c - 1) == 0) {
				x0 = c;
			}
			if (c > 0 && record == 0 && col_mat.at<float>(0, c - 1) > 0) {
				x1 = c;
				break;
			}
		}
	}
	//ROI
	cv::Mat map_roi_x;
	cv::Mat map_roi_y;
	mapx(cv::Range(y0, y1), cv::Range(x0, x1)).copyTo(map_roi_x);
	mapy(cv::Range(y0, y1), cv::Range(x0, x1)).copyTo(map_roi_y);
	
	//map匹配分辨率
	if (map_roi_x.rows < panel_res_cols || map_roi_x.cols < panel_res_rows) { // 补充map大小
		int res_gap_row = panel_res_cols - map_roi_x.rows;
		int res_gap_col = panel_res_rows - map_roi_x.cols;
		cv::Mat dstx, dsty;
		dstx.create(map_roi_x.rows + res_gap_row, map_roi_x.cols + res_gap_col,map_roi_x.type());
		dsty.create(dstx.size(), dstx.type());
		map_roi_x.copyTo(dstx(cv::Range(0, map_roi_x.rows), cv::Range(0, map_roi_x.cols)));
		map_roi_y.copyTo(dsty(cv::Range(0, map_roi_y.rows), cv::Range(0, map_roi_y.cols)));
		map_roi_x = dstx;
		map_roi_y = dsty;
	}

	if (map_roi_x.rows > panel_res_cols) { //算法定位像素大于外部像素，需map筛选
		int res_gap = map_roi_x.rows - panel_res_cols;
		cv::Mat dstx, dsty;
		for (int idx = 0; idx < res_gap; idx++) {
			int top_line_num = cv::countNonZero(map_roi_x.row(idx) > 0);
			int down_line_num = cv::countNonZero(map_roi_x.row(map_roi_x.rows - idx - 1) > 0);
			if (top_line_num >= down_line_num) {
				map_roi_x(cv::Range(0, map_roi_x.rows - 1), cv::Range(0, map_roi_x.cols)).copyTo(dstx);
				map_roi_y(cv::Range(0, map_roi_y.rows - 1), cv::Range(0, map_roi_y.cols)).copyTo(dsty);
			}
			else {
				map_roi_x(cv::Range(1, map_roi_x.rows), cv::Range(0, map_roi_x.cols)).copyTo(dstx);
				map_roi_y(cv::Range(1, map_roi_y.rows), cv::Range(0, map_roi_y.cols)).copyTo(dsty);
			}
			map_roi_x = dstx;
			map_roi_y = dsty;
		}
	}
	if (map_roi_x.cols > panel_res_rows) { //算法定位像素大于外部像素，需map筛选
		int res_gap = map_roi_x.cols - panel_res_rows;
		cv::Mat dstx, dsty;
		for (int idx = 0; idx < res_gap; idx++) {
			int left_line_num = cv::countNonZero(map_roi_x.col(idx) > 0);
			int right_line_num = cv::countNonZero(map_roi_x.col(map_roi_x.cols - idx - 1) > 0);
			if (left_line_num >= right_line_num) {
				map_roi_x(cv::Range(0, map_roi_x.rows), cv::Range(0, map_roi_x.cols - 1)).copyTo(dstx);
				map_roi_y(cv::Range(0, map_roi_y.rows), cv::Range(0, map_roi_y.cols - 1)).copyTo(dsty);
			}
			else {
				map_roi_x(cv::Range(0, map_roi_x.rows), cv::Range(1, map_roi_x.cols)).copyTo(dstx);
				map_roi_y(cv::Range(0, map_roi_y.rows), cv::Range(1, map_roi_y.cols)).copyTo(dsty);
			}
			map_roi_x = dstx;
			map_roi_y = dsty;
		}
	}

	//de black hole - method 1
	//map_roi_x 填充
	int rows = map_roi_x.rows;
	int cols = map_roi_x.cols;


	for (int c = 0; c < map_roi_x.cols; c++) {
		cv::Mat line;
		line.create(map_roi_x.rows, 1, map_roi_x.type());
		map_roi_x(cv::Range(0, rows), cv::Range(c,c+1)).copyTo(line);
		_line_interpolation(line);
		line.copyTo(map_roi_x(cv::Range(0, rows), cv::Range(c, c + 1)));
	}

	//map_roi_y 填充
	for (int c = 0; c < map_roi_y.cols; c++) {
		cv::Mat line;
		line.create(map_roi_y.rows, 1, map_roi_y.type());
		map_roi_y(cv::Range(0, rows), cv::Range(c, c + 1)).copyTo(line);
		_line_interpolation(line);
		line.copyTo(map_roi_y(cv::Range(0, rows), cv::Range(c, c + 1)));
	}
	mapx = map_roi_x.clone();
	mapy = map_roi_y.clone();

	//location interval (original point: panel right top)
	int location_interval = interval;
	if (location_interval > 0) {
		map_roi_x = 0;
		map_roi_y = 0;
		location_interval += 1;
		for (int r = 0; r < int(rows/ location_interval); r++) {
			for (int c = 0; c < int(cols / location_interval); c++) {
				if (mapx.at<float>(r, c) > 0 && r * location_interval<=rows && c * location_interval<=cols) {
					map_roi_x.at<float>(r* location_interval, c* location_interval) = mapx.at<float>(r, c);
					map_roi_y.at<float>(r * location_interval, c * location_interval) = mapy.at<float>(r, c);
					r += 0;
					c += 0;
				}
			}
		}

		//map_roi_x row填充
		for (int c = 0; c < map_roi_x.cols; c++) {
			cv::Mat line;
			line.create(map_roi_x.rows, 1, map_roi_x.type());
			map_roi_x(cv::Range(0, rows), cv::Range(c, c + 1)).copyTo(line);
			_line_interpolation_inner(line);
			line.copyTo(map_roi_x(cv::Range(0, rows), cv::Range(c, c + 1)));
		}

		//map_roi_y row填充
		for (int c = 0; c < map_roi_y.cols; c++) {
			cv::Mat line;
			line.create(map_roi_y.rows, 1, map_roi_y.type());
			map_roi_y(cv::Range(0, rows), cv::Range(c, c + 1)).copyTo(line);
			_line_interpolation_inner(line);
			line.copyTo(map_roi_y(cv::Range(0, rows), cv::Range(c, c + 1)));
		}

		//map_roi_x col填充
		for (int c = 0; c < map_roi_x.rows; c++) {
			cv::Mat line;
			line.create(1, map_roi_x.cols, map_roi_x.type());
			map_roi_x(cv::Range(c, c + 1), cv::Range(0, cols)).copyTo(line);
			cv::transpose(line, line);
			_line_interpolation_inner(line);
			cv::transpose(line, line);
			line.copyTo(map_roi_x(cv::Range(c, c + 1), cv::Range(0, cols)));
		}

		//map_roi_y col填充
		for (int c = 0; c < map_roi_y.rows; c++) {
			cv::Mat line;
			line.create(1, map_roi_y.cols, map_roi_y.type());
			map_roi_y(cv::Range(c, c + 1), cv::Range(0, cols)).copyTo(line);
			cv::transpose(line, line);
			_line_interpolation_inner(line);
			cv::transpose(line, line);
			line.copyTo(map_roi_y(cv::Range(c, c + 1), cv::Range(0, cols)));
		}

	}
	mapx = map_roi_x.clone();
	mapy = map_roi_y.clone();
}

void get_map(const cv::Mat& location_map, cv::Mat& mapx, cv::Mat& mapy,double& mapping, int panel_res_rows,int panel_res_cols,std::string color, int is_save_location_map, std::array<int, 6> location_setting_pixels, bool is_subimage_W, std::string camera_grab_type) {
	cv::Mat map;
	cv::Mat location_map_mono_f32_roi;
	cv::Mat solo_pixels_mat;

	// fourier transfrom
	// less roi for dft, speed up!
	std::cout << get_time() << ":Fourier tansform" << std::endl;
	int r = location_map.rows/2;
	int c = location_map.cols/2;
	location_map(cv::Range(r-500,r+500), cv::Range(c - 500, c + 500)).convertTo(location_map_mono_f32_roi, CV_32FC1);

	//cv::Mat kernel_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	//cv::erode(location_map_mono_f32_roi, location_map_mono_f32_roi, kernel_erode);

	cv::Mat f_map_shift = _fourier_t(location_map_mono_f32_roi);
	//location_map_mono_f32_roi = location_map_mono_f32_roi / 80000;
	//f_map_shift = f_map_shift / 100;

	// get mapping
	std::cout << get_time() << ":get mapping" << std::endl;
	mapping = _get_mapping(f_map_shift);
	size_t color_index = std::string("RGB").find(color);
	if (location_setting_pixels[2 + color_index] > 0) {
		mapping = location_setting_pixels[2 + color_index];
	}


	// pixels solo
	std::cout << get_time() << ":pixels solo" << std::endl;
	if (is_subimage_W || camera_grab_type=="C_L_W_G") {
		solo_pixels_mat = _solo_pixels_color(location_map, mapping, color);
	}
	else {
		solo_pixels_mat = _solo_pixels_mono(location_map, mapping, color);
	}
	

	// find the first pixel
	int x, y;
	std::cout << get_time() << ":find the first pixel" << std::endl;
	_find_first_pixel(solo_pixels_mat, x, y);

	// loop in all pixels
	std::cout << get_time() << ":loop in all pixels" << std::endl;
	_loop_all_pixles(solo_pixels_mat, x, y, mapx, mapy, mapping, panel_res_rows, panel_res_cols, color, location_setting_pixels);
	//_loop_all_pixles_curve(solo_pixels_mat, x, y, mapx, mapy, mapping, panel_res_rows, panel_res_cols, color, location_setting_pixels);

	// post process for map
	std::cout << get_time() << ":post process for map" << std::endl;
	_post_process_for_map(solo_pixels_mat, mapx, mapy, panel_res_rows, panel_res_cols, location_setting_pixels[5]);
}