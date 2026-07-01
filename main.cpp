#include "preprocess.h"
#include "location.h"
#include "brightness.h"
#include "postprocess.h"
#include "dect.h"
#include "thread_pool.h"
#include "tools.h"
#include "utils.h"


int main()
{
    try {
        // work flow log inital
        SimpleLog::init();
        std::cout << get_time() << "W COLOR start: -------------------------------------------------------------" << std::endl;
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
        std::string gray_color = param["Database"]["gray_color"];

        // funcation switch
        int panel_res_rows_all = std::stoi(param["Panel"]["panel_res_rows"]);
        int panel_res_cols_all = std::stoi(param["Panel"]["panel_res_cols"]);
        int position = std::stoi(param["Database"]["position"]);

        // location setting
        std::vector<double> location_mapping;
        string_split_num(param["Database"]["location_mapping"], location_mapping, ',');
        int location_rows_tolerate_pixels = stoi(param["Database"]["location_rows_tolerate_pixels"]);
        int location_cols_tolerate_pixels = stoi(param["Database"]["location_cols_tolerate_pixels"]);
        int location_interval = stoi(param["Database"]["location_interval"]);
        std::array<int, 6> location_setting_pixels = { location_rows_tolerate_pixels, location_cols_tolerate_pixels, location_mapping[0], location_mapping[1], location_mapping[2], location_interval};

        // brightness setting
        std::vector<double> brightness_mapping;
        string_split_num(param["Database"]["brightness_mapping"], brightness_mapping, ',');
        std::array<double, 3> brightness_setting = { brightness_mapping[0], brightness_mapping[1], brightness_mapping[2] };

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
        std::vector<double> is_demoire, is_curved_corr;
        int is_hole_corner_filling = std::stoi(param["Process"]["hole_corner_filling"]); // hole与corner填充
        string_split_num(param["Process"]["de_moire"], is_demoire, ',');
        string_split_num(param["Process"]["curved_corr"], is_curved_corr, ',');


        // tools
        /*int is_sandy_quantization = std::stoi(param["Tool"]["sandy_quantization"]);
        int is_csv_demoire = std::stoi(param["Tool"]["csv_demoire"]);*/
        std::vector<double> is_sandy_quantization, is_csv_demoire, is_moire_quantization;
        string_split_num(param["Tool"]["sandy_quantization"], is_sandy_quantization, ',');
        string_split_num(param["Tool"]["csv_demoire"], is_csv_demoire, ',');
        string_split_num(param["Tool"]["moire_quantization"], is_moire_quantization, ',');

        std::vector<std::vector<double>> tools_setting = { is_sandy_quantization, is_csv_demoire, is_moire_quantization };

        // aoi
        int is_aoi = std::stoi(param["AOI"]["aoi"]);

        // saving
        int is_save_csv = std::stoi(param["Saving"]["save_csv"]);
        int is_save_location_map = std::stoi(param["Saving"]["save_location_map"]);
        int is_save_roi = std::stoi(param["Saving"]["save_roi"]);

        // camera grab type C_L_W_G: color location white gray
        std::string camera_grab_type = "";
        if (color_list.find("W") == cv::String::npos && gray_color.find("W") != cv::String::npos) {
            camera_grab_type = "C_L_W_G";
        }
        if (color_list.find("W") == cv::String::npos && gray_color.find("W") == cv::String::npos) {
            camera_grab_type = "W_L_W_G";
        }
        if (color_list.find("W") != cv::String::npos && gray_color.find("W") != cv::String::npos) {
            camera_grab_type = "C_L_C_G";
        }
        
        // inital
        int location_idx = 0; // 定位图索引
        bool is_location_map = true; // 定位图标志位
        cv::Mat location_map; // 定位图句柄
        cv::Mat mapx, mapy; // 定位数据
        double mapping = 0; // 外部mapping数值
        std::string name; // 单图名称
        std::string sub_name; // 子图名称
        cv::Mat ffc_calibration_coef; // ffc校正矩阵
        int subimg_num = 1; // 区分mono与color
        std::vector<cv::Mat> imgc; // color img 初始化
        std::string RGB = "RGB";
        ThreadPool pool(4); // 多线程初始化
        std::vector<std::future<void>> fut; // future 捕获，防止多线程冲突
        bool is_subimage_W = 0; // 是否是彩色相机W图中RGB图
        std::vector<cv::Mat> map_store;

        // main start ---------------------------------------------------------------------------------------------
        for (int img_idx = 0; img_idx < size(files_index); img_idx++) {
            // 1.image readin
            std::ifstream f(img_path);
            cv::Mat img;

            name = files_index[img_idx];
            std::cout << get_time() << name + ":img readin" << std::endl;
            img = readImage(files_path + "/" + files_index[img_idx] + "." + img_type, img_type); // 多图提取
            std::cout << get_time() << name + ":img readin end" << std::endl;

            // 1.5.tools
            if (tools_DeM(img, tools_setting, name)) {
                continue;
            }

            if ((img_idx == 0 || files_index[img_idx - 1][0] != files_index[img_idx][0]) && (camera_grab_type!="C_L_W_G" || name[0]!='W')) {
                is_location_map = true;
                color = color_list_sub[location_idx];
                location_idx++;
                single_color = color;
            }
            else {
                is_location_map = false;
                single_color = files_index[img_idx][0];
            }

            // 2.image preprocess
            if (is_ffc != 0) {
                std::cout << get_time() << name + ":img calibration" << std::endl;
                img_calibration(img, ffc_calibration_coef, img_idx);
            }
            img_flip(img, position);

            // 3.color image transformer
            is_subimage_W = 0; // mono图初始化
            if (single_color == "W" && camera_grab_type != "C_L_W_G") {
                img *= 16; // 12bit to 16bit for higher acc.
                //img.convertTo(img, CV_8UC1);
                color2mono(img, imgc);
                subimg_num = 3;
                is_subimage_W = 1;
            }
            if (camera_grab_type == "C_L_W_G" && single_color != "W") {
                img *= 12;
                color2mono(img, imgc);
                size_t ind = RGB.find(single_color);
                img = imgc[ind];
            }
            if (camera_grab_type == "C_L_W_G" && single_color == "W") {
                color2mono(img, imgc);
                subimg_num = 3;
                is_subimage_W = 1;
            }

            for (int sub_img_idx = 0; sub_img_idx < subimg_num; ++sub_img_idx) { // main start --------------------------------------
                int panel_res_rows = panel_res_rows_all;
                int panel_res_cols = panel_res_cols_all;
                

                if (single_color == "W"|| is_subimage_W==1) { // COLOR
                    img = imgc[sub_img_idx];
                    single_color = RGB[sub_img_idx];
                    sub_name = single_color + "_" + files_index[img_idx];
                }
                else { // MONO
                    sub_name = files_index[img_idx];
                    sub_img_idx = location_idx-1; //与开头定位索引统一，保持RGB灰阶图一致性
                }

                //MONO
                if (single_color == "R" || single_color == "B") { // 如果是RB则使用delta
                    panel_res_cols = panel_res_cols_all / 2;
                }

                // 4.image location
                if (is_location_map == true) {
                    mapx.create(panel_res_cols * 2, panel_res_rows * 2, CV_32FC1);
                    mapy.create(panel_res_cols * 2, panel_res_rows * 2, CV_32FC1);
                    std::cout << get_time() << sub_name + ":location start" << std::endl;
                    //if (is_subimage_W == 1) {
                    //    //location_map = preprocess4color_location_map(img, single_color);
                    //    location_map = img;
                    //}
                    //else {
                    //    location_map = img;
                    //}
                    location_map = img.clone();
                    get_map(location_map, mapx, mapy, mapping, panel_res_rows, panel_res_cols, single_color, is_save_location_map, location_setting_pixels, is_subimage_W, camera_grab_type);
                    plot_map(img, mapx, mapy, single_color, is_save_location_map);
                    std::cout << get_time() << sub_name + ":location end" << std::endl;
                    // mono+color map store
                    map_store.push_back(mapx);
                    map_store.push_back(mapy);
                }

                // 5.image get brt
                cv::Mat csv;
                std::cout << get_time() << sub_name + ":getting brightness" << std::endl;
                csv = get_brt(img, map_store[sub_img_idx*2], map_store[sub_img_idx*2+1], mapping, panel_res_rows, panel_res_cols, single_color, brightness_setting);
                csv = csv / (std::stoi(exp_list[img_idx]) * std::stoi(gain_list[img_idx]));
                std::cout << get_time() << sub_name + ":getting brightness end" << std::endl;

                // 6.image postprocess
                csv_flip(csv);
                if (is_hole_corner_filling != 0) {
                    hole_fill(csv);
                    std::cout << get_time() << sub_name + ":hole fill" << std::endl;
                    corner_fill(csv);
                    std::cout << get_time() << sub_name + ":corner fill" << std::endl;
                }
                if (is_demoire[0] != 0) {
                    de_moire(csv, is_demoire[1], is_demoire[2]);
                    std::cout << get_time() << sub_name + ":de-moire" << std::endl;
                }
                if (is_curved_corr[0] != 0) {
                    //csv /= 80;
                    curved_corr(csv, is_curved_corr[0], is_curved_corr[1], single_color);
                    std::cout << get_time() << sub_name + ":curved corr" << std::endl;
                }

                // 7.AOI
                if (is_aoi != 0) {
                    cv::Mat abnormal_img;
                    abnormal_img = dect_aoi(csv);
                }

                // 8.image saving
                if (is_save_csv != 0) {
                    std::cout << get_time() << sub_name + ":csv saving start" << std::endl;
                    std::string sub_name_concat = "./" + sub_name + ".csv";
                    // csv_saving(csv, name, 6);
                    // thread_pool saving
                    fut.emplace_back(
                        pool.enqueue([csv, sub_name_concat]() mutable {
                            multi_process_csv_saving(csv, sub_name_concat, 6);
                        })
                    );
                }
                if (is_save_roi != 0) {
                    std::cout << get_time() << sub_name + ":roi saving start" << std::endl;
                    std::string sub_name_concat = "./" + sub_name + ".bmp";
                    roi_saving(csv, sub_name_concat);
                }
                std::cout << get_time() << sub_name + ":csv saving end" << std::endl;
            }
        }

        // thread pool delay
        std::cout << get_time() << "waiting for the thread pool" << std::endl;
        for (auto& f : fut) f.wait();
        std::cout << get_time() << "Preprocess done: -------------------------------------------------------------" << std::endl;
        // work flow log end
        SimpleLog::close();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << get_time() << "Error:  " << e.what() << "\n";
    }
}