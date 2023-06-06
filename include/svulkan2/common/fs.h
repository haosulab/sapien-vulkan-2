#pragma once
#include <filesystem>
#include <string>
#include <vector>

namespace svulkan2 {

namespace fs = std::filesystem;

std::vector<char> readFile(std::filesystem::path const &filename);

bool inline check_file(fs::path const &path) {
  return fs::is_regular_file(path);
}

bool inline check_dir(fs::path const &path) { return fs::is_directory(path); }

void inline check_file_required(fs::path const &path, std::string const &err) {
  if (!check_file(path)) {
    throw std::runtime_error(err);
  }
}

void inline check_dir_required(fs::path const &path, std::string const &err) {
  if (!check_dir(path)) {
    throw std::runtime_error(err);
  }
}

} // namespace svulkan2
