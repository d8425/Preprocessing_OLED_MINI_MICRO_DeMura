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

cv::Mat readImage(const std::string& path, std::string img_type) {
    if (img_type == "tif" || img_type == "TIF" || img_type == "MIM" || img_type == "mim") {
        cv::Mat image = cv::imread(path, cv::COLOR_BGR2GRAY);
        return image;
    }
    if (img_type == "csv" || img_type == "CSV") {
        std::vector<std::vector<float>> g;
        std::string l;
        for (std::ifstream s(path); std::getline(s, l);) {
            std::stringstream t(l);
            std::vector<float> r;
            std::string v;
            while (std::getline(t, v, ',')) r.push_back(std::stof(v));
            g.emplace_back(r);
        }
        cv::Mat m(g.size(), g[0].size(), CV_32F);
        for (int i = 0; i < m.rows; ++i) std::memcpy(m.ptr(i), g[i].data(), m.cols * sizeof(float));
        return m;
    }
}

std::vector<float> parse2Float(const std::string& str) {
    // convert '1,0.9,0.8' to [1,0.9,0.8]
    std::vector<float> values;
    std::stringstream ss(str);
    std::string item;

    // 按逗号分割
    while (std::getline(ss, item, ',')) {
        // 去掉空格（防止 "1, 0.9 , 0.8" 这种格式报错）
        item.erase(std::remove_if(item.begin(), item.end(), isspace), item.end());
        if (!item.empty()) {
            values.push_back(std::stof(item)); // 转成浮点数
        }
    }
    return values;
}

void img_flip(cv::Mat& img, int position) {
    // right side finding first
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
        //stable
    }
    else {
        std::cerr << "position selection error" << std::endl;
    }

    //// left side finding first
    //// - ROTATE_90_CLOCKWISE：顺时针90°
    //// - ROTATE_180：180°
    //// - ROTATE_90_COUNTERCLOCKWISE：逆时针90°
    ////cv::rotate(img, img, 0);
    //if (position == 1) {
    //    cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
    //}
    //if (position == 2) {
    //    //stable
    //}
    //if (position == 3) {
    //    cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
    //}
    //if (position == 4) {
    //    cv::rotate(img, img, cv::ROTATE_180);
    //}
    //else {
    //    std::cerr << "position selection error" << std::endl;
    //}
}

void string_split(std::string line, std::vector<std::string>& list, char symbol) {
    std::stringstream ss(line);
    for (std::string t; std::getline(ss, t, symbol); list.push_back(t));
}

void string_split_num(std::string line, std::vector<double>& list, char symbol) {
    std::stringstream ss(line);
    for (std::string t; std::getline(ss, t, symbol); list.push_back(std::stod(t)));
}

void color2mono(cv::Mat img, std::vector<cv::Mat>& imgc) {
    cv::Mat color_img;
    cv::cvtColor(img, color_img, cv::COLOR_BayerRGGB2RGB);
    /*std::vector<cv::Mat> imgc;*/
    cv::split(color_img, imgc);

    /*imgr = planes[2];
    imgg = planes[1];
    imgb = planes[0];*/
}

cv::Mat preprocess4color_location_map(cv::Mat location_map, std::string single_color) {
    cv::Mat img_low, img_high;
    // get high freq. info.
    cv::GaussianBlur(location_map, img_low, cv::Size(11, 11),3);
    img_high = location_map - 0.5*img_low;
    // erode
    if (single_color != "G") { // G画面像素密集，故不进行腐蚀
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::erode(img_high, img_high, kernel);
    }
    
    // nonzeros value should be in a suitable value range
    double ratio = cv::sum(img_high)[0] / cv::countNonZero(img_high);
    if (ratio<50){ // changeable
        img_high *= 128 / ratio;
    }

    return img_high;
}

void img_calibration(cv::Mat& img, cv::Mat& ffc_calibration_coef,  int img_idx) {
    // 1.FFC(flat filed correction)
    // 1: exist calibration data(unavailable)  2: physics model for ffc(availavle)
    // assuming FFC model fit the physics formula : I(r) = I(0) * (cos(δ(r)) ^ 4) * V(r)
    // which I(r) : calibrated value, I(0) : original value, r : distance from
    // current pixel to middle point, cos(δ(r)) : physics prior formula for
    // light diffusion, δ(r) : angle distance, V(r) mechine diffusion, also the fitting coef.
    // ->x, the original img

    // FFC SPEED UP --------------------------------------------------------------------------------------------START
    // FFC SPEED UP --------------------------------------------------------------------------------------------END

    bool is_16UC1 = 0;
    if (img.type() == CV_16UC1) {
        is_16UC1 = 1;
    }

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
    cv::Mat dst_16U;
    if (is_16UC1){
        img.convertTo(dst_16U, CV_16UC1,1.0);
        img = dst_16U;
    }
    else {
        img.convertTo(img, CV_8UC1,1.0,0.0);
    }
    

    // FFC --------------------------------------------------------------------------------------------END

    // denoise correction
    // CRF/BLC/畸变校正/响应校正暂不做
}