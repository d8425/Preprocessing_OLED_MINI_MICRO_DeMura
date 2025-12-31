# include "preprocess.h"

std::map<std::string, std::map<std::string, std::string>> read_ini(const std::string& filename) {
    // 局部map：仅在函数内部创建，解析完成后返回
    std::map<std::string, std::map<std::string, std::string>> ini_data;
    std::ifstream file(filename);

    // 检查文件是否成功打开
    if (!file.is_open()) {
        throw std::runtime_error("无法打开INI文件: " + filename); // 抛异常提示错误
    }

    std::string line;
    std::string current_section; // 记录当前解析的section（如[Database]）

    // 逐行解析文件
    while (std::getline(file, line)) {
        // 1. 忽略空行、注释行（; 或 # 开头）
        if (line.empty() || line[0] == ';' || line[0] == '#') {
            continue;
        }

        // 2. 解析section（格式：[section_name]）
        if (line.front() == '[' && line.back() == ']') {
            // 提取section名称（去掉前后的[]）
            current_section = line.substr(1, line.size() - 2);
            continue;
        }

        // 3. 解析key=value（跳过无=的无效行）
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            continue;
        }

        // 提取key和value（并去除前后空格）
        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);

        // 去除key前后的空格（避免配置中"key = value"和"key=value"的差异）
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        // 去除value前后的空格（避免value带多余空格）
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        // 将key-value存入当前section对应的map中
        ini_data[current_section][key] = value;
    }

    file.close(); // 关闭文件
    return ini_data; // 返回解析后的配置（无全局变量）
}

cv::Mat readImage(const std::string& path) {
	cv::Mat image = cv::imread(path,cv::COLOR_BGR2GRAY);
	return image;
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
    oss << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S: ");

    return oss.str();
}

void img_flip(cv::Mat& img, int position) {
	// - ROTATE_90_CLOCKWISE：顺时针90°
	// - ROTATE_180：180°
	// - ROTATE_90_COUNTERCLOCKWISE：逆时针90°
	//cv::rotate(img, img, 0);
    if (position == 1) {
        cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    if (position==2){
        cv::rotate(img, img, cv::ROTATE_180);
    }
    if (position == 3) {
        cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
    }
    if (position == 4) {
        //正角度
    }
    else {
        std::cerr << "position需为1-4中数，1为正角度panel，2为顺时针90度，后续依次" << std::endl;
    }
}

void string_split(std::string line, std::vector<std::string>& list, char symbol) {
    std::stringstream ss(line);
    for (std::string t; std::getline(ss, t, symbol); list.push_back(t));
}

void color2mono(cv::Mat img, std::vector<cv::Mat>& imgc) {
    cv::Mat color_img;
    cv::cvtColor(img, color_img, cv::COLOR_BayerRG2BGR);
    /*std::vector<cv::Mat> imgc;*/
    cv::split(color_img, imgc);

    /*imgr = planes[2];
    imgg = planes[1];
    imgb = planes[0];*/
}

void img_calibration(cv::Mat& img, cv::Mat& ffc_calibration_coef,  int img_idx) {
    // 1.FFC(flat filed correction)
    // *校正方法分为两种，1.有标定数据的校准(暂无自用相机，不做) 2.无标准标定的趋近标定(建模相关镜头的衰减模型)
    // assuming FFC model fit the physics formula : I(r) = I(0) * (cos(δ(r)) ^ 4) * V(r)
    // which I(r) : calibrated value, I(0) : original value, r : distance from
    // current pixel to middle point, cos(δ(r)) : physics prior formula for
    // light diffusion, δ(r) : angle distance, V(r) mechine diffusion, also the fitting coef.
    // ->x, the original img

    // FFC SPEED UP --------------------------------------------------------------------------------------------START
    // FFC SPEED UP --------------------------------------------------------------------------------------------END



    // FFC --------------------------------------------------------------------------------------------START
    if (img_idx == 0) {
        std::cout << "calculation" << std::endl;
        int r, c, center_row, center_col;
        r = img.rows;
        c = img.cols;
        center_row = floor(r / 2);
        center_col = floor(c / 2);

        cv::Mat pixel_distance;
        double diff = 0;
        pixel_distance.create(r, c, CV_32FC1);

        cv::Mat row_mat(r, c, CV_32FC1);
        cv::Mat col_mat(r, c, CV_32FC1);
        for (int i = 0; i < r; i++) {
            row_mat.row(i) = i - center_row;
        }
        for (int j = 0; j < c; j++) {
            col_mat.col(j) = j - center_col;
        }

        pow(row_mat, 2, row_mat);
        pow(col_mat, 2, col_mat);
        add(row_mat, col_mat, pixel_distance);
        sqrt(pixel_distance, pixel_distance);

        // pixel distance to angle difference
        double distance_center_target_len, mapping, per_pixel_distance;
        distance_center_target_len = 500; // mm
        mapping = 3.5;
        per_pixel_distance = 65 * 1e-3; // mm * um2mm, 3.76um presents 151M CMOS pixel distance (panel, not CMOS)

        // cos(δ(r))
        double scalar_ratio = per_pixel_distance / mapping;
        cv::Mat middle_mat, distance_target_len, trans_pixel_distance_to_angle;
        cv::Mat temp_mat = pixel_distance * scalar_ratio;
        cv::pow(temp_mat, 2, middle_mat);
        cv::Mat cos_2 = cv::pow(distance_center_target_len, 2) + middle_mat;
        cv::sqrt(cos_2, distance_target_len);
        cv::divide(distance_center_target_len, distance_target_len, trans_pixel_distance_to_angle);

        // V(r)
        cv::Mat zero_mat_v;
        zero_mat_v.create(r, c, CV_32FC1);
        zero_mat_v += 1;

        // simulated_flat_field
        cv::Mat simulated_flat_field;
        cv::pow(trans_pixel_distance_to_angle, 4, trans_pixel_distance_to_angle);
        cv::multiply(trans_pixel_distance_to_angle, zero_mat_v, simulated_flat_field);

        // FFC coef
        simulated_flat_field = simulated_flat_field / cv::mean(simulated_flat_field); // normalization
        ffc_calibration_coef = 1 / simulated_flat_field;
        ffc_calibration_coef = ffc_calibration_coef / cv::mean(ffc_calibration_coef);
    }
    
    // FFC calibration
    img.convertTo(img, CV_32FC1);
    cv::multiply(img, ffc_calibration_coef, img);
    img.convertTo(img, CV_8UC1);

    // FFC --------------------------------------------------------------------------------------------END

    // denoise correction
    // CRF/BLC/畸变校正/响应校正暂不做
}