#include "common/config.h"
#include <yaml-cpp/yaml.h>
#include <iostream>

Config::Config(const std::string& filename){
    try {
        YAML::Node config = YAML::LoadFile(filename);
        if (!config["Dataset"]) {
            std::cerr << "[Config] Missing 'Dataset' section in YAML file: " << filename << std::endl;
            return;
        }
        left_gray_dir_  = config["Dataset"]["left_gray_dir"].as<std::string>();
        right_gray_dir_ = config["Dataset"]["right_gray_dir"].as<std::string>();
        calib_dir_      = config["Dataset"]["calib_dir"].as<std::string>();

        std::cout << "[Config] Loaded dirs:" << std::endl;
        std::cout << " Left: "  << left_gray_dir_  << std::endl;
        std::cout << " Right: " << right_gray_dir_ << std::endl;
        std::cout << " Calib: " << calib_dir_      << std::endl;
    }catch (const std::exception& e) {
        std::cerr << "[Config] Failed to load YAML file: " << filename << std::endl;
        std::cerr << "Error: " << e.what() << std::endl;
    }
}