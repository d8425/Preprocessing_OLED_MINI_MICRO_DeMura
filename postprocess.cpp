# include "postprocess.h";

void csv_flip(cv::Mat& csv) {
	// - ROTATE_90_CLOCKWISE：顺时针90°
	// - ROTATE_180：180°
	// - ROTATE_90_COUNTERCLOCKWISE：逆时针90°
	cv::rotate(csv, csv, cv::ROTATE_90_CLOCKWISE);
}

void hole_fill(cv::Mat& csv) {
    cv::Mat roi, mask;
	csv(cv::Range(0, int(csv.rows/10)), cv::Range(3*int(csv.cols / 10), 7*int(csv.cols / 10))).copyTo(roi); // roi select, changeable
	cv::Scalar mean_val = cv::mean(roi);
	float mean_val_u = mean_val[0];
	mask = roi <= 0.1* mean_val; //changeable

    cv::GaussianBlur(mask, mask, cv::Size(3, 3), 1);
    mask = (mask > 0);
    
    cv::Mat temp_mat = csv(cv::Range(0, int(csv.rows / 10)), cv::Range(3 * int(csv.cols / 10), 7 * int(csv.cols / 10)));
    temp_mat.setTo(cv::Scalar(mean_val_u), mask);
}

void corner_fill(cv::Mat& csv) {
	cv::Mat roi, mask;
	cv::Scalar mean_val = cv::mean(csv);
	float mean_val_u = mean_val[0];
	mask = csv <= 0.1 * mean_val; // changeable


    // 连通域个数卡控-mask
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    
    int minArea = 50; // min pixels num
    cv::Mat filtered = cv::Mat::zeros(mask.size(), CV_8U);
    for (int idx = 1; idx < n; ++idx) { // n=0, is background
        int area = stats.at<int>(idx, cv::CC_STAT_AREA);
        if (area >= minArea) {
            filtered.setTo(1, labels == idx);
        }
    }

    // get value
    cv::Mat down_sample_img, up_sample_img;
    cv::resize(csv, down_sample_img, cv::Size(), 0.01, 0.01);
    cv::resize(down_sample_img, up_sample_img, cv::Size(csv.cols, csv.rows));

    // compensation
    cv::Mat coef_mat;
    filtered.convertTo(filtered, CV_32F);
    cv::multiply(filtered, up_sample_img, coef_mat);
    csv += coef_mat;
}

void fft_shift(cv::Mat& src, cv::Mat& dst){
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

double moire_detector_2(cv::Mat& csv) //中频最高值方法
{
    double moire_num = 0;
    // check if moire exist
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
    fft_shift(img_fft_abs, img_fft_shift);

    //de-center
    int row, col, mid_row, mid_col, step_row, step_col; // row and col, center row and center col, min search step
    row = int(img_fft_shift.rows);
    col = int(img_fft_shift.cols);
    mid_row = int(row / 2);
    mid_col = int(col / 2);
    step_row = int(row / 10); // changeable
    step_col = int(col / 10);

    cv::Rect center_rect(mid_col - step_col, mid_row - step_row, 2 * step_col, 2 * step_row);
    img_fft_shift(center_rect) = cv::Scalar::all(0);

    //denoise
    cv::Mat img_fft_shift_show = img_fft_shift / 50;
    int count_non_img = cv::countNonZero(img_fft_shift);
    double noise_val = cv::sum(img_fft_shift)[0] / count_non_img;
    //img_fft_shift -= 20 * noise_val;

    //////erode and imdilter
    //cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));
    //cv::morphologyEx(img_fft_shift, img_fft_shift, cv::MORPH_CLOSE, kernel);

    //count
    int high_val_num = cv::countNonZero(img_fft_shift > noise_val);
    double moire_ratio = (double)high_val_num/ count_non_img;

    ////decide
    //bool if_demoire_exist = false;
    //if (high_val_num > 30) {
    //    if_demoire_exist = true;
    //}

    return moire_ratio;
}

void de_moire(cv::Mat& csv, double threshold_coef, double blur_strength) {
    cv::Mat img_fft;
    csv.convertTo(csv, CV_32FC1);
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
    fft_shift(img_fft_abs, img_fft_shift);
    cv::Mat csv1 = csv / 3;
    cv::Mat img_fft_shift1 = img_fft_shift / 10;

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
    //threshold *= 0.05; //changeable - important changebale point
    threshold *= threshold_coef;
    cv::Mat mask;
    cv::threshold(img_fft_shift, mask, threshold, 1, cv::THRESH_BINARY);

    //Gass and normalization - def a blur strength: blur_strength
    int ksize = blur_strength * 2 + 1;
    cv::GaussianBlur(mask, mask, cv::Size(ksize, ksize), 0); // important changebale point (5,5) 1
    mask = 1 - mask;
    cv::pow(mask, 5, mask); //幂次增强对比度
    //mask ifftshift
    cv::Mat mask_ifft;
    fft_shift(mask, mask_ifft);
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

void de_moire_low(cv::Mat& csv, double threshold_coef, double blur_strength) {
    cv::Mat img_fft;
    csv.convertTo(csv, CV_32FC1);
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
    fft_shift(img_fft_abs, img_fft_shift);
    cv::Mat csv1 = csv / 700;
    cv::Mat img_fft_shift1 = img_fft_shift / 300;

    ////de-center
    //int row, col, mid_row, mid_col, step_row, step_col; // row and col, center row and center col, min search step
    //row = int(img_fft_shift.rows);
    //col = int(img_fft_shift.cols);
    //mid_row = int(row / 2);
    //mid_col = int(col / 2);
    //step_row = int(row / 12); // changeable
    //step_col = int(col / 12);

    //cv::Rect center_rect(mid_col - step_col, mid_row - step_row, 2 * step_col, 2 * step_row);
    //cv::Mat mask1 = cv::Mat::zeros(img_fft_shift.size(), CV_8U);
    //cv::rectangle(mask1, center_rect, cv::Scalar(255), -1);  // -1 表示填充
    //cv::bitwise_not(mask1, mask1);
    //img_fft_shift.setTo(cv::Scalar::all(0), mask1);

    ////img erode
    ////cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)); // changeable
    ////cv::Mat small_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    ////cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    ////cv::erode(img_fft_shift, img_fft_shift, kernel);

    //cv::Mat kernel = (cv::Mat_<uchar>(3, 3) <<
    //    0, 1, 0,
    //    1, 1, 1,
    //    0, 1, 0
    //    );

    //// 执行最小腐蚀
    //cv::erode(img_fft_shift, img_fft_shift, kernel);

    ////bin
    //double threshold;
    //cv::minMaxLoc(img_fft_shift, NULL, &threshold, NULL, NULL);
    ////threshold *= 0.05; //changeable - important changebale point
    //threshold *= threshold_coef;
    //cv::Mat mask;
    //cv::threshold(img_fft_shift, mask, threshold, 1, cv::THRESH_BINARY);

    ////Gass and normalization - def a blur strength: blur_strength
    //int ksize = blur_strength * 2 + 1;
    //cv::GaussianBlur(mask, mask, cv::Size(ksize, ksize), 0); // important changebale point (5,5) 1
    //mask = 1 - mask;
    //cv::pow(mask, 5, mask); //幂次增强对比度
    // 
    // 
    

    cv::Mat mask = cv::Mat::ones(img_fft_shift1.size(), CV_32FC1);

    ////左右
    //cv::Rect r1(320, 1371, 6, 6);
    //mask(r1) = 0.1f;

    //cv::Rect r2(304, 1372, 6, 6);
    //mask(r2) = 0.1f;

    ////上下
    //cv::Rect r3(312, 1354, 7, 8);
    //mask(r3) = 0.1f;

    //cv::Rect r4(314, 1388, 7, 8);
    //mask(r4) = 0.1f;

    //左右
    cv::Rect r1(294, 1317, 6, 6);
    mask(r1) = 0.1f;

    cv::Rect r2(311, 1318, 6, 6);
    mask(r2) = 0.1f;

    //上下
    cv::Rect r3(301, 1297, 6, 6);
    mask(r3) = 0.1f;

    cv::Rect r4(301, 1337, 6, 6);
    mask(r4) = 0.1f;

    //mask ifftshift
    cv::Mat mask_ifft;
    fft_shift(mask, mask_ifft);
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

void csv_saving(cv::Mat mat,std::string filename, int precision) {
    if (mat.empty()) {
        std::cerr << "输入矩阵为空！" << std::endl;
        return;
    }
    if (mat.channels() != 1) {
        std::cerr << "仅支持单通道矩阵！" << std::endl;
        return;
    }

    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "无法打开文件：" << filename << std::endl;
        return;
    }

    // 设置输出精度（保留小数位数）
    outFile << std::fixed << std::setprecision(precision);

    // 遍历矩阵的每一行
    for (int row = 0; row < mat.rows; ++row) {
        // 遍历当前行的每一列
        for (int col = 0; col < mat.cols; ++col) {
            // 根据矩阵元素类型获取值（这里以float为例，其他类型需调整）
            float value = mat.at<float>(row, col);  // 若为uchar，用mat.at<uchar>(row, col)
            outFile << value;

            // 最后一列后不加逗号
            if (col != mat.cols - 1) {
                outFile << ",";
            }
        }
        // 每行结束后换行
        outFile << "\n";
    }
    outFile.close();
}

void roi_saving(cv::Mat mat, std::string filename) {
    cv::Scalar mean_val = cv::mean(mat);
    mat = mat / mean_val[0];
    mat *= 128;
    mat.convertTo(mat, CV_8U);
    cv::imwrite(filename, mat);
}

void multi_process_csv_saving(cv::Mat mat, std::string filename, int precision) {
    // used for threadpool, without log print
    if (mat.empty() || mat.channels() != 1) return;
    std::ofstream out(filename);
    if (!out.is_open()) return;
    out << std::fixed << std::setprecision(precision);
    for (int r = 0; r < mat.rows; ++r) {
        for (int c = 0; c < mat.cols; ++c) {
            out << mat.at<float>(r, c);
            if (c + 1 < mat.cols) out << ",";
        }
        out << "\n";
    }
}