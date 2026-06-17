# include "tools.h"
# include "preprocess.h"
# include <filesystem>
# include <iostream>
# include <fstream>
# include <string>

namespace fs = std::filesystem;
// interface for sandy quantization using for outer funcation 

std::vector<std::string> splitByDoubleSlash(const std::string& str) {
    std::vector<std::string> result;
    size_t start = 0;
    size_t end = str.find(".");

    while (end != std::string::npos) {
        result.push_back(str.substr(start, end - start));
        start = end + 2;
        end = str.find(".", start);
    }
    result.push_back(str.substr(start));
    return result;
}

int main() {
    // tools
    std::array<int, 2> tools_setting = { 1, 0 }; // first: sandy quantization; second: 

    // get name
    fs::path current_path = fs::current_path();
    std::string csv_path = current_path.string() + std::string("\\Input CSV");
    std::vector<std::string> csv_files;
    std::vector<std::string> name_list;

    std::ofstream outFile(current_path.string() + "\\sandy_Quantization.txt");

    for (const auto& entry : fs::directory_iterator(csv_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
            csv_files.push_back(entry.path().string());
            
            std::string name = entry.path().filename().string();
            auto parts = splitByDoubleSlash(name);
            name_list.push_back(parts[0]);
        }
    }

    for (int img_idx = 0; img_idx < size(csv_files); img_idx++) {
        // 1.image readin
        cv::Mat img;
        img = readImage(csv_files[img_idx], "csv"); // ¶àÍ¼ÌáÈ¡


        // 2.tools
        double sq_q = tools_DeM_outer(img, tools_setting, name_list[img_idx]);

        // 3.saving
        std::string content = name_list[img_idx] + "," + std::to_string(sq_q);
        outFile << content << std::endl;

    }
}

