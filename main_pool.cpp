#include "preprocess.h"
#include "location.h"
#include "brightness.h"
#include "postprocess.h"
#include "dect.h"
#include "thread_pool.h"
#include "tools.h"
#include "utils.h"
#include <future>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <memory>

// 定位结果存储结构
struct LocationResult {
    cv::Mat mapx, mapy;
    double mapping = 0;
    std::atomic<bool> ready{ false };
    std::mutex mtx;
    std::condition_variable cv;

    // 禁用拷贝，允许移动
    LocationResult() = default;
    LocationResult(const LocationResult&) = delete;
    LocationResult& operator=(const LocationResult&) = delete;
    LocationResult(LocationResult&&) = default;
    LocationResult& operator=(LocationResult&&) = default;
};

// 任务类型
enum class TaskType { LOCATION, PROCESSING };

struct ImageTask {
    int img_idx;
    std::string filename;
    std::string color_channel; // R/G/B
    bool is_location_map;
    std::string sub_name;
    TaskType type;
    int sub_idx = 0; // 用于W图的RGB子索引 0=R, 1=G, 2=B
};

int main()
{
    try {
        SimpleLog::init();
        std::cout << get_time() << "RGB MONO start: -------------------------------------------------------------" << std::endl;
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

        // 参数读取
        auto param = read_ini("./pre_param.ini");
        std::cout << get_time() << "param readin" << std::endl;

        // 基础配置
        std::string files_path = param["Database"]["data_path"];
        std::string img_type = param["Database"]["pattern_type"];
        std::string camera_index = param["Database"]["camera_index"];
        std::string camera_index_exp = param["Database"]["camera_index_exp"];
        std::string camera_index_gain = param["Database"]["camera_index_gain"];
        std::string color_list = param["Database"]["location_color"];
        int position = std::stoi(param["Database"]["position"]);

        int panel_res_rows_all = std::stoi(param["Panel"]["panel_res_rows"]);
        int panel_res_cols_all = std::stoi(param["Panel"]["panel_res_cols"]);

        int location_rows_tolerate_pixels = stoi(param["Database"]["location_rows_tolerate_pixels"]);
        int location_cols_tolerate_pixels = stoi(param["Database"]["location_cols_tolerate_pixels"]);
        int location_mapping = stoi(param["Database"]["location_mapping"]);
        std::array<int, 3> location_setting_pixels = { location_rows_tolerate_pixels, location_cols_tolerate_pixels, location_mapping };

        int brightness_mapping = stoi(param["Database"]["brightness_mapping"]);
        std::array<int, 1> brightness_setting = { brightness_mapping };

        std::vector<std::string> files_index, color_list_sub, exp_list, gain_list;
        string_split(camera_index, files_index, ',');
        string_split(color_list, color_list_sub, ',');
        string_split(camera_index_exp, exp_list, ',');
        string_split(camera_index_gain, gain_list, ',');

        int is_ffc = std::stoi(param["Preprocess"]["ffc"]);
        int is_hole_corner_filling = std::stoi(param["Process"]["hole_corner_filling"]);
        //int is_demoire = std::stoi(param["Process"]["de_moire"]);
        int is_aoi = std::stoi(param["AOI"]["aoi"]);
        int is_save_csv = std::stoi(param["Saving"]["save_csv"]);
        int is_save_location_map = std::stoi(param["Saving"]["save_location_map"]);
        int is_save_roi = std::stoi(param["Saving"]["save_roi"]);

        // tools
        int is_sandy_quantization = std::stoi(param["Tool"]["sandy_quantization"]);
        int is_csv_demoire = std::stoi(param["Tool"]["csv_demoire"]);

        std::array<int, 2> tools_setting = { is_sandy_quantization, is_csv_demoire };

        // postprocess
        std::vector<double> is_demoire;
        int is_hole_corner_filling = std::stoi(param["Process"]["hole_corner_filling"]); // hole与corner填充
        string_split_num(param["Process"]["de_moire"], is_demoire, ',');

        cv::Mat ffc_calibration_coef;
        std::mutex log_mutex;

        // 存储R/G/B三色的定位结果 - 使用unique_ptr避免拷贝问题
        std::map<std::string, std::unique_ptr<LocationResult>> loc_results;
        loc_results["R"] = std::make_unique<LocationResult>();
        loc_results["G"] = std::make_unique<LocationResult>();
        loc_results["B"] = std::make_unique<LocationResult>();

        ThreadPool pool(8);
        std::vector<std::future<void>> save_futures;
        std::vector<std::future<void>> task_futures;

        // 1 任务分类
        std::vector<ImageTask> location_tasks;
        std::vector<ImageTask> processing_tasks;

        int location_idx = 0;
        for (int img_idx = 0; img_idx < files_index.size(); img_idx++) {
            std::string name = files_index[img_idx];
            std::string first_char = name.substr(0, 1);
            bool is_new_sequence = (img_idx == 0) || (files_index[img_idx - 1][0] != name[0]);

            if (is_new_sequence) {
                std::string loc_color = color_list_sub[location_idx];
                location_tasks.push_back({ img_idx, name, loc_color, true, name, TaskType::LOCATION });
                location_idx++;
            }
            else {
                if (first_char == "W") {
                    processing_tasks.push_back({ img_idx, name, "R", false, "R_" + name, TaskType::PROCESSING, 2 });
                    processing_tasks.push_back({ img_idx, name, "G", false, "G_" + name, TaskType::PROCESSING, 1 });
                    processing_tasks.push_back({ img_idx, name, "B", false, "B_" + name, TaskType::PROCESSING, 0 });
                }
                else {
                    processing_tasks.push_back({ img_idx, name, first_char, false, name, TaskType::PROCESSING });
                }
            }
        }

        // 2 并行处理定位图
        std::cout << get_time() << "Phase 1: Parallel location processing..." << std::endl;

        for (auto& task : location_tasks) {
            task_futures.push_back(std::async(std::launch::async, [&, task]() {
                std::string color = task.color_channel;

                cv::Mat img = readImage(files_path + "/" + task.filename + "." + img_type, img_type);

                if (tools_DeM(img, tools_setting, task.filename)) return;

                if (is_ffc != 0) {
                    img_calibration(img, ffc_calibration_coef, task.img_idx);
                }
                img_flip(img, position);

                cv::Mat location_map;
                if (color == "W") {
                    location_map = preprocess4color_location_map(img, color);
                }
                else {
                    location_map = img;
                }
                
                int panel_res_cols = (color == "R" || color == "B") ? panel_res_cols_all / 2 : panel_res_cols_all;

                cv::Mat mapx(panel_res_cols * 2, panel_res_rows_all * 2, CV_32FC1);
                cv::Mat mapy(panel_res_cols * 2, panel_res_rows_all * 2, CV_32FC1);
                double mapping = 0;

                get_map(location_map, mapx, mapy, mapping, panel_res_rows_all, panel_res_cols,
                    color, is_save_location_map, location_setting_pixels);
                plot_map(img, mapx, mapy, color, is_save_location_map);

                {
                    std::lock_guard<std::mutex> lock(loc_results[color]->mtx);
                    loc_results[color]->mapx = mapx.clone();
                    loc_results[color]->mapy = mapy.clone();
                    loc_results[color]->mapping = mapping;
                    loc_results[color]->ready = true;
                }
                loc_results[color]->cv.notify_all();

                std::lock_guard<std::mutex> lock(log_mutex);
                std::cout << get_time() << task.filename + ": location completed" << std::endl;
                }));
        }

        std::cout << get_time() << "Phase 2: Parallel processing all images..." << std::endl;

        for (auto& task : processing_tasks) {
            auto& loc_res_ptr = loc_results[task.color_channel];

            task_futures.push_back(pool.enqueue([&, task]() {
                std::string color = task.color_channel;

                {
                    std::unique_lock<std::mutex> lock(loc_res_ptr->mtx);
                    loc_res_ptr->cv.wait(lock, [&]() { return loc_res_ptr->ready.load(); });
                }

                cv::Mat img = readImage(files_path + "/" + task.filename + "." + img_type, img_type);

                if (tools_DeM(img, tools_setting, task.filename)) return;

                if (is_ffc != 0) {
                    img_calibration(img, ffc_calibration_coef, task.img_idx);
                }
                img_flip(img, position);

                if (task.sub_name[0] == 'R' || task.sub_name[0] == 'G' || task.sub_name[0] == 'B') {
                    if (task.filename[0] == 'W') {
                        img /= 16;
                        img.convertTo(img, CV_8UC1);
                        std::vector<cv::Mat> imgc;
                        color2mono(img, imgc);
                        img = imgc[task.sub_idx];
                    }
                }

                int panel_res_cols = (color == "R" || color == "B") ? panel_res_cols_all / 2 : panel_res_cols_all;

                cv::Mat csv;
                {
                    std::lock_guard<std::mutex> lock(loc_res_ptr->mtx);
                    csv = get_brt(img, loc_res_ptr->mapx, loc_res_ptr->mapy, loc_res_ptr->mapping,
                        panel_res_rows_all, panel_res_cols, brightness_setting);
                }
                csv = csv / (std::stod(exp_list[task.img_idx]) * std::stod(gain_list[task.img_idx]));

                csv_flip(csv);
                if (is_hole_corner_filling != 0) {
                    hole_fill(csv);
                    corner_fill(csv);
                }
                if (is_demoire != 0) {
                    de_moire(csv, is_demoire[1], is_demoire[2]);
                }

                if (is_aoi != 0) {
                    cv::Mat abnormal_img = dect_aoi(csv);
                }

                {
                    std::lock_guard<std::mutex> lock(log_mutex);
                    if (is_save_csv != 0) {
                        std::string csv_path = "./" + task.sub_name + ".csv";
                        cv::Mat csv_copy = csv.clone();
                        save_futures.push_back(pool.enqueue([csv_copy, csv_path]() mutable {
                            multi_process_csv_saving(csv_copy, csv_path, 6);
                            }));
                        std::cout << get_time() << task.sub_name + ": csv saving queued" << std::endl;
                    }
                    if (is_save_roi != 0) {
                        std::string roi_path = "./" + task.sub_name + ".bmp";
                        roi_saving(csv, roi_path);
                        std::cout << get_time() << task.sub_name + ": roi saved" << std::endl;
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(log_mutex);
                    std::cout << get_time() << task.sub_name + ": processing completed" << std::endl;
                }
                }));
        }

        std::cout << get_time() << "Waiting for all processing tasks..." << std::endl;
        for (auto& f : task_futures) {
            if (f.valid()) f.wait();
        }

        std::cout << get_time() << "Waiting for saving tasks..." << std::endl;
        for (auto& f : save_futures) {
            if (f.valid()) f.wait();
        }

        std::cout << get_time() << "All done -------------------------------------------------------------" << std::endl;
        SimpleLog::close();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << get_time() << "Error: " << e.what() << "\n";
        return -1;
    }
}