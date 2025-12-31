#include "preprocess.h"
#include "location.h"
#include "brightness.h"
#include "postprocess.h"
#include "thread_pool.h"

int main()
{
    try {
        // warning log off
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

        // param in
        std::map<std::string, std::map<std::string, std::string>> ini_data;
        auto param = read_ini("./pre_param.ini");
        //std::string time = get_time();
        std::cout << get_time() << "param readin" << std::endl;

        // data type
        std::string img_path = param["Database"]["img_path"];
        std::string single_color = param["Database"]["img_color"]; //单图颜色
        std::string color = "N"; //多图中单图颜色
        std::string camera_index = param["Database"]["camera_index"];
        std::string camera_index_exp = param["Database"]["camera_index_exp"];
        std::string camera_index_gain = param["Database"]["camera_index_gain"];
        std::string files_path = param["Database"]["data_path"];
        std::string img_type = param["Database"]["pattern_type"];
        std::string color_list = param["Database"]["location_color"];

        // funcation switch
        int panel_res_rows_all = std::stoi(param["Panel"]["panel_res_rows"]);
        int panel_res_cols_all = std::stoi(param["Panel"]["panel_res_cols"]);
        int position = std::stoi(param["Database"]["position"]);

        // location setting
        int location_mapping = stoi(param["Database"]["location_mapping"]);
        int location_rows_tolerate_pixels = stoi(param["Database"]["location_rows_tolerate_pixels"]);
        int location_cols_tolerate_pixels = stoi(param["Database"]["location_cols_tolerate_pixels"]);
        std::array<int, 3> location_setting_pixels = { location_rows_tolerate_pixels, location_cols_tolerate_pixels, location_mapping };

        // setting transformation
        std::vector<std::string> files_index, color_list_sub, single_img_name, exp_list,gain_list;
        string_split(camera_index, files_index,',');
        string_split(color_list, color_list_sub, ',');
        string_split(img_path, single_img_name, '\\'); // 图像索引集
        string_split(camera_index_exp, exp_list, ','); // 图像对应曝光集
        string_split(camera_index_gain, gain_list, ','); // 图像对应增益集

        // preprocess
        int is_ffc = std::stoi(param["Preprocess"]["ffc"]);
        
        // postprocess
        int is_hole_corner_filling = std::stoi(param["Process"]["hole_corner_filling"]); // hole与corner填充
        int is_demoire = std::stoi(param["Process"]["de_moire"]); // 摩尔纹去除

        // saving
        int is_save_csv = std::stoi(param["Saving"]["save_csv"]);
        int is_save_location_map = std::stoi(param["Saving"]["save_location_map"]);
        int is_save_roi = std::stoi(param["Saving"]["save_roi"]);
        
        // inital
        int location_idx = 0; // 定位图索引
        bool is_location_map = true; // 定位图标志位
        cv::Mat mapx, mapy; // 定位数据
        double mapping = 0; // 外部mapping数值
        std::string name; // 单图名称
        cv::Mat ffc_calibration_coef; // ffc校正矩阵
        int subimg_num = 1; // 区分mono与color
        std::vector<cv::Mat> imgc; // color img 初始化
        std::string RGB = { "RGB" };
        ThreadPool pool(4); // 多线程初始化
        std::vector<std::future<void>> fut; // future 捕获，防止多线程冲突

        // main start ---------------------------------------------------------------------------------------------
        for (int img_idx = 0; img_idx < size(files_index); img_idx++) {
            // 1.image readin
            std::ifstream f(img_path);
            cv::Mat img;
            img = readImage(files_path + "/" + files_index[img_idx] + "." + img_type); // 多图提取
            if (img_idx == 0 || files_index[img_idx - 1][0] != files_index[img_idx][0]) {
                is_location_map = true;
                color = color_list_sub[location_idx];
                location_idx++;
                single_color = color;
            }
            else {
                is_location_map = false;
                single_color = files_index[img_idx][0];
            }
            name = files_index[img_idx];

            std::cout << get_time() << name+":img readin" << std::endl;

            // 2.image preprocess
            if (is_ffc != 0) {
                std::cout << get_time() << name + ":img calibration" << std::endl;
                img_calibration(img, ffc_calibration_coef, img_idx);
            }
            img_flip(img, position);

            // 3.color image transformer
            if (single_color == "W") {
                img /= 16; // 12bit to 8bit
                img.convertTo(img, CV_8UC1);
                color2mono(img, imgc);
            }
            int panel_res_rows = panel_res_rows_all;
            int panel_res_cols = panel_res_cols_all;

            for (int sub_img_idx = 0; sub_img_idx < subimg_num; sub_img_idx++) {
                if (single_color == "W") { // COLOR
                    img = imgc[2 - sub_img_idx];
                    single_color = RGB[sub_img_idx];
                }

                //MONO
                if (single_color == "R" || single_color == "B") { // 如果是RB则使用delta
                    panel_res_cols = panel_res_cols_all / 2;
                }

                // 4.image location
                if (is_location_map == true) {
                    mapx.create(panel_res_cols * 2, panel_res_rows * 2, CV_32FC1);
                    mapy.create(panel_res_cols * 2, panel_res_rows * 2, CV_32FC1);
                    std::cout << get_time() << name + ":location start" << std::endl;
                    get_map(img, mapx, mapy, mapping, panel_res_rows, panel_res_cols, single_color, is_save_location_map, location_setting_pixels);
                    std::cout << get_time() << name + ":location end" << std::endl;
                }

                // 5.image get brt
                cv::Mat csv;
                std::cout << get_time() << name + ":getting brightness" << std::endl;
                csv = get_brt(img, mapx, mapy, mapping, panel_res_rows, panel_res_cols);
                csv = csv / (std::stoi(exp_list[img_idx]) * std::stoi(gain_list[img_idx]));
                std::cout << get_time() << name + ":getting brightness end" << std::endl;

                // 6.image postprocess
                csv_flip(csv);
                if (is_hole_corner_filling != 0) {
                    hole_fill(csv);
                    std::cout << get_time() << name + ":hole fill" << std::endl;
                    corner_fill(csv);
                    std::cout << get_time() << name + ":corner fill" << std::endl;
                }
                if (is_demoire != 0) {
                    de_moire(csv);
                    std::cout << get_time() << name + ":de-moire" << std::endl;
                }

                // 7.image saving
                if (is_save_csv != 0) {
                    std::cout << get_time() << name + ":csv saving start" << std::endl;
                    std::string name = "./" + files_index[img_idx] + ".csv";
                    // csv_saving(csv, name, 6);
                    // thread_pool saving
                    fut.emplace_back(
                        pool.enqueue([csv, name]() mutable {
                            multi_process_csv_saving(csv, name, 6);
                        })
                    );
                }
                if (is_save_roi != 0) {
                    std::cout << get_time() << name + ":roi saving start" << std::endl;
                    std::string name = "./" + files_index[img_idx] + ".bmp";
                    roi_saving(csv, name);
                }
                std::cout << get_time() << name + ":csv saving end" << std::endl;
            }

        }

        // thread pool delay
        std::cout << get_time() << "waiting for the thread pool" << std::endl;
        for (auto& f : fut) f.wait();
        std::cout << get_time() << "Preprocess done" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << get_time() << "Error:  " << e.what() << "\n";
    }
}


//Mat image = imread("84.jpg", IMREAD_COLOR);

//cv::Mat img;
//img = image;
//img.convertTo(img, CV_32F);

//// Normalize pixel values
//cv::Mat normalized_image;
//img.convertTo(normalized_image, CV_32F, 1.0 / 255.0);

//// Resize image
//resize(normalized_image, normalized_image, cv::Size(640, 640));

//cv::Scalar mean(0, 0, 0);
//cv::Scalar std(0.5, 0.5, 0.5);

//normalized_image = (normalized_image - mean) / std;
//
////224 for resnet 640 for yolov8
//cv::Mat inputBlob = cv::dnn::blobFromImage(normalized_image, 1.0, cv::Size(), cv::Scalar(), true, false);

//cv::dnn::Net net = cv::dnn::readNetFromONNX("best.onnx");
//net.setInput(inputBlob);

//cv::Mat result = net.forward();
//cv::Point classIdPoint;